"""P5 DOE: 설계변수 LHS 샘플링 → 경량 성능 평가 → 데이터셋 생성.

설계변수 (5): a_m, T_m, T_m2_ratio(=T_m2/T_m), W_t, MagnetR
응답 (5)    : T_avg[mNm], ripple_pct, EMF_rms[V@1000rpm],
              magnet_area[mm²](체적 프록시), B_tooth[T](철심 95퍼센타일)

변수 변경은 variables_raw 오버라이드 → resolve_variables 재해석으로
theta_one 등 종속 수식까지 일관 전파한다.
"""
from __future__ import annotations

import json
import math
import warnings
from typing import Dict

import numpy as np

warnings.filterwarnings("ignore")

from .expressions import resolve_variables
from .geometry import build_motor
from .sliding import SlidingBandMesh
from .solver_ms import Magnetostatic2D
from .postproc import (torque_arkkio, coenergy, build_winding_map,
                       flux_linkages)

DELTA_E_DEG = 290.0          # P4 캘리브레이션 MTPA 전기위상 (권선/극배치 고정)
RPM_EMF = 1000.0

# 설계 평가에서 "정상적인 설계 실패"로 취급할 예외 — 형상 불완전(ValueError),
# 메시 폭주(ValueError), 솔버 발산/특이행렬(RuntimeError·LinAlgError) 등.
# 이 밖의 예외(NameError·KeyError 등 하네스 버그)는 별도 태그로 노출한다.
_DESIGN_FAIL = (ValueError, RuntimeError, ArithmeticError,
                np.linalg.LinAlgError)

BOUNDS = {
    "a_m":        (0.80, 0.95),
    "T_m":        (1.8, 2.6),
    "T_m2_ratio": (0.60, 0.92),
    "W_t":        (3.0, 4.2),
    "MagnetR":    (0.40, 1.10),
}


def bounds_for_model(v: dict) -> dict:
    """현재 설계값 기준 상대 범위 — 400W에서 기존 BOUNDS와 일치하도록 보정.

    (T_m 2.2→1.80~2.60, W_t 3.5→3.00~4.20, MagnetR 0.8→0.40~1.10)
    """
    tm, wt, mr = v["T_m"] * 1e3, v["W_t"] * 1e3, v["MagnetR"] * 1e3
    return {
        "a_m":        (0.80, 0.95),
        "T_m":        (round(tm * 0.818, 3), round(tm * 1.182, 3)),
        "T_m2_ratio": (0.60, 0.92),
        "W_t":        (round(wt * 0.857, 3), round(wt * 1.2, 3)),
        "MagnetR":    (round(max(0.2, mr * 0.5), 3), round(mr * 1.375, 3)),
    }


def baseline_design(v: dict) -> Dict[str, float]:
    """현재 변수의 설계점 (DOE 0번 샘플·SAC 시작점용)."""
    return {"a_m": float(v["a_m"]), "T_m": v["T_m"] * 1e3,
            "T_m2_ratio": float(v["T_m2"] / v["T_m"]),
            "W_t": v["W_t"] * 1e3, "MagnetR": v["MagnetR"] * 1e3}


def calibrate_delta(model: dict, style: str, *, I_rms: float,
                    steel_name: str | None = None,
                    magnet_name: str | None = None,
                    n_band: int = 2880) -> float:
    """DOE 전류각 δ* 캘리브레이션 (로터 0° 고정, 4점 사인 피팅).

    evaluate_design과 동일 컨벤션(te = pp·θ + δ)을 쓴다."""
    from .aedt_parser import detect_material_names
    if steel_name is None or magnet_name is None:
        steel_name, magnet_name = detect_material_names(model)
    v = model["variables"]
    geo = build_motor(v, style)
    sbm = SlidingBandMesh(geo, n_band=n_band)
    mesh = sbm.merge(0.0)
    L = v["L_stk"]
    Zc = int(round(v["Zc"]))
    Ia = I_rms * math.sqrt(2.0)
    wmap = None
    T = {}
    for d in (0.0, 90.0, 180.0, 270.0):
        s = Magnetostatic2D(mesh, model["materials"], steel_name, magnet_name)
        if wmap is None:
            wmap = build_winding_map(s)
        te = math.radians(d)
        iph = {"A": Ia * math.sin(te),
               "B": Ia * math.sin(te - 2 * math.pi / 3),
               "C": Ia * math.sin(te + 2 * math.pi / 3)}
        at = {}
        for ph, sides in wmap.items():
            for ci, sgn in sides:
                at[ci] = sgn * Zc * iph[ph]
        s.set_coil_currents(at)
        res = s.solve(tol=1e-5)
        T[d] = torque_arkkio(s, res, sbm.r_i + 0.005, sbm.r_o - 0.005, L)
    A = (T[90.0] - T[270.0]) / 2.0
    B = (T[0.0] - T[180.0]) / 2.0
    return math.degrees(math.atan2(A, B)) % 360.0


def vary(model: dict, x: Dict[str, float]) -> Dict[str, float]:
    """설계점 x를 원시 수식에 주입해 전체 변수 재해석 (SI)."""
    raw = dict(model["variables_raw"])
    raw["a_m"] = repr(x["a_m"])
    raw["T_m"] = f"{x['T_m']}mm"
    raw["T_m2"] = f"{x['T_m2_ratio'] * x['T_m']:.4f}mm"
    raw["W_t"] = f"{x['W_t']}mm"
    raw["MagnetR"] = f"{x['MagnetR']}mm"
    return resolve_variables(raw)


def evaluate_design(model: dict, style: str, x: Dict[str, float],
                    n_emf: int = 6, n_load: int = 8,
                    n_band: int = 2880, *,
                    I_rms: float | None = None,
                    delta_e_deg: float = DELTA_E_DEG,
                    steel_name: str | None = None,
                    magnet_name: str | None = None,
                    with_efficiency: bool = False,
                    with_cogging: bool = False,
                    with_current_min: bool = False,
                    n_cog: int = 24,
                    rpm: float | None = None,
                    d_cu_mm: float = 0.25,
                    strands: int = 11,
                    T_cu_C: float = 80.0,
                    R_ph_ohm: float | None = None) -> dict:
    """단일 설계 평가. 실패 시 status='fail'.

    I_rms 미지정 시 모델 변수값 사용(무부하 설계는 0이므로 명시 권장).
    재질 미지정 시 자동 감지. delta_e_deg는 calibrate_delta()로 모델별 산출.

    with_efficiency=True면 전기 1주기 부하 스윕(검증된 sweep_loss 경로)을
    추가로 돌려 efficiency·P_fe·P_cu를 출력에 더한다 — 솔브 ~50회 추가
    소요. rpm/권선 사양(d_cu_mm·strands·T_cu_C) 또는 R_ph_ohm 직접 지정.
    """
    out = {"x": x, "status": "ok"}
    try:
        if steel_name is None or magnet_name is None:
            from .aedt_parser import detect_material_names
            steel_name, magnet_name = detect_material_names(model)
        v = vary(model, x)
        geo = build_motor(v, style)
        n_slot = int(round(v["N_slot"]))
        if len(geo.coils) != 2 * n_slot \
                or len(geo.magnets) != int(round(v["N_pole"])):
            raise ValueError("형상 불완전")
        sbm = SlidingBandMesh(geo, n_band=n_band)
        L = v["L_stk"]
        Zc = int(round(v["Zc"]))
        Ia = (I_rms if I_rms is not None else v["I_rms"]) * math.sqrt(2.0)
        pp = int(round(v["N_pole"])) // 2
        wmap = None

        def solve(theta, load):
            nonlocal wmap
            s = Magnetostatic2D(sbm.merge(theta), model["materials"],
                                steel_name, magnet_name)
            if wmap is None:
                wmap = build_winding_map(s)
            if load:
                te = pp * math.radians(theta) + math.radians(delta_e_deg)
                iph = {"A": Ia * math.sin(te),
                       "B": Ia * math.sin(te - 2 * math.pi / 3),
                       "C": Ia * math.sin(te + 2 * math.pi / 3)}
                at = {}
                for ph, sides in wmap.items():
                    for ci, d in sides:
                        at[ci] = d * Zc * iph[ph]
                s.set_coil_currents(at)
            else:
                iph = None
                s.set_coil_currents({})
            res = s.solve(tol=1e-5)
            return s, res, iph

        # ---- 무부하: EMF (λ_A 푸리에, 전기 1주기) ---------------------
        angs_e = np.linspace(0, 360.0 / pp, n_emf, endpoint=False)
        lamA = []
        for a in angs_e:
            s, res, _ = solve(a, False)
            lamA.append(flux_linkages(s, res, wmap, Zc, L)["A"])
        lamA = np.asarray(lamA)
        # 전기 1주기(45°) 등간격 → FFT로 고조파, e_k = k·ω_e·Λ_k
        F = np.fft.rfft(lamA) / n_emf
        w_e = RPM_EMF / 60 * 2 * math.pi * pp
        e_rms = math.sqrt(sum(0.5 * (k * w_e * 2 * abs(F[k])) ** 2
                              for k in range(1, len(F))))
        out["emf_rms"] = e_rms

        # ---- 부하: 평균토크 + 리플 (전기 60° = 리플 1주기 + 가드 2점) --
        step = (60.0 / pp) / n_load
        angs_l = np.arange(-1, n_load + 1) * step      # 가드 포함 n+2점
        Wc, lam3, i3, Ta = [], [], [], []
        Bt = Bt_tooth = Bt_yoke = 0.0
        r_yi = v["D_so"] / 2.0 - v["T_Yoke"]      # 스테이터 요크 내반경[m]
        for a in angs_l:
            s, res, iph = solve(a, True)
            Wc.append(coenergy(s, res, L))
            lm = flux_linkages(s, res, wmap, Zc, L)
            lam3.append([lm["A"], lm["B"], lm["C"]])
            i3.append([iph["A"], iph["B"], iph["C"]])
            Ta.append(torque_arkkio(s, res, sbm.r_i + 0.005,
                                    sbm.r_o - 0.005, L))
            st = res.Bmag[s.is_steel]
            Bt = max(Bt, float(np.percentile(st, 95)))
            # 치/요크 분리: 스테이터 강판을 반경으로 (요크=바깥 링, 치=안쪽)
            rad = np.hypot(s.centroid[:, 0], s.centroid[:, 1])
            stator = (s.kind == "stator")
            tooth = stator & (rad < r_yi)
            yoke = stator & (rad >= r_yi)
            if tooth.any():
                Bt_tooth = max(Bt_tooth,
                               float(np.percentile(res.Bmag[tooth], 95)))
            if yoke.any():
                Bt_yoke = max(Bt_yoke,
                              float(np.percentile(res.Bmag[yoke], 95)))
        th = np.radians(angs_l)
        Wc = np.asarray(Wc)
        lam3 = np.asarray(lam3)
        i3 = np.asarray(i3)
        Tvw = (np.gradient(Wc, th)
               - np.sum(lam3 * np.gradient(i3, th, axis=0), axis=1))
        Tvw = Tvw[1:-1]                       # 가드 제거 → 정확히 1주기
        Ta_in = np.asarray(Ta)[1:-1]          # Arkkio 파형 (리플용)
        out["T_avg"] = float(np.mean(Tvw) * 1e3)         # 평균은 가상일(정확)
        out["T_arkkio"] = float(np.mean(Ta_in) * 1e3)
        # 리플은 Arkkio 직접적분 파형에서 — 코에너지 미분(Tvw)은 노이즈가 커
        # 서로게이트 학습 불가(R²<0). Arkkio 파형이 매끄러워 학습성이 좋다.
        out["ripple_pct"] = float(
            (Ta_in.max() - Ta_in.min()) / np.mean(Ta_in) * 100)
        out["ripple_pct_vw"] = float(
            (Tvw.max() - Tvw.min()) / np.mean(Tvw) * 100)   # 참고용
        out["B_tooth"] = Bt              # 전체 강판 95% (기존 호환)
        out["B_tooth_st"] = Bt_tooth     # 스테이터 치 95% (포화 ≤1.8T)
        out["B_yoke"] = Bt_yoke          # 스테이터 요크 95% (포화 ≤1.6T)
        out["magnet_area"] = float(sum(p.area for p, _, _ in geo.magnets))

        # ---- (옵션) 코깅: 무부하 1주기 가상일토크 → FFT 저차 pk-pk -------
        # 코깅 기본주기(기계각)=360/LCM(슬롯,극). 코깅=−dW_co/dθ(가상일)을
        # 무부하 1주기에서 구한 뒤, 저차(1~3)만 남겨 슬라이딩밴드 이산화
        # 노이즈(고차 alias)를 제거한다. raw Arkkio pk-pk는 밴드노이즈로
        # 3~6배 과대(검증: 400W raw 16~37 vs 코에너지FFT 4.1 ≈ Maxwell 5.95).
        if with_cogging:
            npole = int(round(v["N_pole"]))
            cog_period = 360.0 / int(np.lcm(n_slot, npole))
            angs = np.linspace(0, cog_period, n_cog, endpoint=False)
            Wc = []
            for a in angs:
                sc, rc, _ = solve(a, False)
                Wc.append(coenergy(sc, rc, L))
            T = -np.gradient(np.asarray(Wc), np.radians(angs))   # 코깅 [N·m]
            F = np.fft.rfft(T); F[4:] = 0.0       # 저차(코깅 1~3차)만 유지
            Tc = np.fft.irfft(F, n_cog)
            out["cogging_pp"] = float((Tc.max() - Tc.min()) * 1e3)   # mNm

        # ---- (옵션) 전류 최소화 = 동손 최소화 --------------------------
        # 목표 토크 T를 내는 데 필요한 동손 = 3·I_req²·R_ph, I_req=I·T/T_avg
        #  → Pcu(T) = (3·I²·R_ph)/T_avg² · T² = Pcu_per_Nm2 · T².
        # 즉 Pcu_per_Nm2(=동손/토크²)를 최소화하면 임의의 목표 토크에 대해
        # 필요 전류와 동손이 동시에 최소가 된다(전류최소화 ≡ 동손최소화).
        # T_avg(학습됨)·R_ph(기하, FEM불필요)로 계산 → 추가 솔브 0.
        if with_current_min:
            from .winding import phase_resistance
            Iph = float(I_rms if I_rms is not None else v.get("I_rms") or 0.0)
            rph = R_ph_ohm
            if not rph or rph <= 0:
                rph = phase_resistance(v, d_cu_mm=d_cu_mm, strands=strands,
                                       T_cu_C=T_cu_C)["R_ph"]
            T_Nm = out["T_avg"] / 1000.0
            if Iph > 0 and T_Nm > 0:
                pcu = 3.0 * Iph ** 2 * rph                  # 현 전류 동손 [W]
                out["Pcu_W"] = float(pcu)                     # 운전점 동손 [W]
                out["Pcu_per_Nm2"] = float(pcu / T_Nm ** 2)  # 동손/토크² [W/Nm²]참고
                out["I_per_Nm"] = float(Iph / T_Nm)          # 전류/토크 [A/Nm]참고
                out["R_ph_ohm"] = float(rph)

        # ---- (옵션) 효율: 검증된 전기1주기 스윕 + 손실 ------------------
        if with_efficiency:
            Irms_ph = float(I_rms if I_rms is not None
                            else v.get("I_rms") or 0.0)
            if Irms_ph > 0:
                from .sweep_loss import (sweep_load_with_fields,
                                         compute_responses, calibrate_gamma)
                from .winding import phase_resistance
                m2 = {**model, "variables": v}
                rpm_use = float(rpm or v.get("BaseRPM") or 1000.0)
                ip = v.get("ini_pos", 0.0)
                ini = math.degrees(ip) if isinstance(ip, (int, float)) else 0.0
                cal = calibrate_gamma(m2, style, rpm=rpm_use, I_rms=Irms_ph,
                                      n_steps=4, init_pos_deg=ini)  # γ*는 강건,4로 단축
                sw = sweep_load_with_fields(
                    m2, style, rpm=rpm_use, I_rms=Irms_ph,
                    gamma_deg=cal["gamma_max_deg"], n_steps=36,
                    init_pos_deg=ini, steel_name=steel_name,
                    magnet_name=magnet_name, compute_vw=False)  # DOE는 기존동작 유지(빠르게)
                rph = R_ph_ohm
                if not rph or rph <= 0:
                    rph = phase_resistance(v, d_cu_mm=d_cu_mm, strands=strands,
                                           T_cu_C=T_cu_C)["R_ph"]
                rr = compute_responses(sw, m2, R_ph_ohm=rph)
                out["efficiency"] = float(rr["efficiency"])
                out["P_fe"] = float(rr["P_fe"])
                out["P_cu"] = float(rr["P_cu"])
                # 리플은 36스텝 전주기 스윕값으로 대체(8스텝 base보다 정밀, 공짜)
                out["ripple_pct"] = float(rr["T_ripple_pct"])
    except _DESIGN_FAIL as e:
        # 예상되는 형상/메시/솔버 실패 = 설계 탈락 (정상)
        out["status"] = f"fail: {type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001
        # 예상 못한 예외 = 하네스 버그(NameError/KeyError 등) — 설계 실패와
        # 구분되게 별도 태그. KeyboardInterrupt는 Exception이 아니라 전파됨.
        out["status"] = f"ERROR(bug): {type(e).__name__}: {e}"
    return out


# ---------------------------------------------------------------- 러너

_W: dict = {}


def _init(aedt_path):
    from .aedt_parser import parse_aedt, detect_magnet_style
    m = parse_aedt(aedt_path)
    _W["m"] = m
    _W["style"] = detect_magnet_style(m)


def _eval(x):
    return evaluate_design(_W["m"], _W["style"], x)


def run_doe(aedt_path: str, n: int = 200, out_path: str = "doe_results.jsonl",
            nproc: int = 1, seed: int = 7, time_budget: float = None,
            resume: bool = True):
    import os, time as _time
    from scipy.stats import qmc
    keys = list(BOUNDS)
    lo = np.array([BOUNDS[k][0] for k in keys])
    hi = np.array([BOUNDS[k][1] for k in keys])
    X = qmc.LatinHypercube(d=len(keys), seed=seed).random(n) * (hi - lo) + lo
    designs = [dict(zip(keys, map(float, row))) for row in X]
    # 기준 설계도 포함 (코너 케이스 검증용) — BOUNDS 안에 있는지 확인 후 클립
    base = {"a_m": 0.89, "T_m": 2.2, "T_m2_ratio": 2.02 / 2.2,
            "W_t": 3.5, "MagnetR": 0.8}
    for k in keys:
        klo, khi = BOUNDS[k]
        if not (klo <= base[k] <= khi):
            print(f"경고: 기준 설계 {k}={base[k]}가 BOUNDS[{klo},{khi}] 밖 "
                  f"— 클립함", flush=True)
            base[k] = min(max(base[k], klo), khi)
    designs.insert(0, base)

    done_keys = set()
    if resume and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:    # cp949 회피
            for line in f:
                try:
                    done_keys.add(tuple(round(val, 9) for val in
                                        json.loads(line)["x"].values()))
                except Exception:
                    pass
    designs = [d for d in designs
               if tuple(round(val, 9) for val in d.values()) not in done_keys]
    print(f"남은 설계 {len(designs)}개 (완료 {len(done_keys)}개 스킵)",
          flush=True)
    # 벽시계(wall clock) — 멀티프로세싱에선 워커 CPU가 부모 process_time에
    # 안 잡혀 예산판정이 무의미. monotonic으로 단일/병렬 동작을 일치시킨다.
    t_start = _time.monotonic()

    done = 0
    with open(out_path, "a", encoding="utf-8") as f:
        if nproc <= 1:
            _init(aedt_path)
            it = map(_eval, designs)
            for r in it:
                f.write(json.dumps(r) + "\n"); f.flush()
                done += 1
                if done % 5 == 0:
                    print(f"{done}/{len(designs)} "
                          f"{_time.monotonic()-t_start:.0f}s", flush=True)
                if time_budget and _time.monotonic() - t_start > time_budget:
                    print("시간 예산 도달 — 체크포인트 후 종료", flush=True)
                    return
        else:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(
                    nproc, initializer=_init,
                    initargs=(aedt_path,)) as pool:
                it = pool.imap_unordered(_eval, designs, chunksize=1)
                for r in it:
                    f.write(json.dumps(r) + "\n"); f.flush()
                    done += 1
                    if done % 10 == 0:
                        print(f"{done}/{len(designs)} "
                              f"{_time.monotonic()-t_start:.0f}s", flush=True)
                    # 예산 초과 시 신규 결과 수신 중단 — 진행 중인 작업은
                    # 받지 않고 풀을 종료(with-블록 __exit__가 terminate).
                    if time_budget and \
                            _time.monotonic() - t_start > time_budget:
                        print("시간 예산 도달 — 체크포인트 후 종료", flush=True)
                        pool.terminate()
                        return
    print("DOE DONE")
