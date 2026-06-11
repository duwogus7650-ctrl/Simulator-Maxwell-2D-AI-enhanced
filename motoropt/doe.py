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

BOUNDS = {
    "a_m":        (0.80, 0.95),
    "T_m":        (1.8, 2.6),
    "T_m2_ratio": (0.60, 0.92),
    "W_t":        (3.0, 4.2),
    "MagnetR":    (0.40, 1.10),
}


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
                    n_band: int = 2880) -> dict:
    """단일 설계 평가. 실패 시 status='fail'."""
    out = {"x": x, "status": "ok"}
    try:
        v = vary(model, x)
        geo = build_motor(v, style)
        if len(geo.coils) != 36 or len(geo.magnets) != int(round(v["N_pole"])):
            raise ValueError("형상 불완전")
        sbm = SlidingBandMesh(geo, n_band=n_band)
        L = v["L_stk"]
        Zc = int(round(v["Zc"]))
        Ia = v["I_rms"] * math.sqrt(2.0)
        pp = int(round(v["N_pole"])) // 2
        wmap = None

        def solve(theta, load):
            nonlocal wmap
            s = Magnetostatic2D(sbm.merge(theta), model["materials"],
                                "20PNX1200F_20C",
                                "Arnold_Magnetics_N45UH_80C")
            if wmap is None:
                wmap = build_winding_map(s)
            if load:
                te = pp * math.radians(theta) + math.radians(DELTA_E_DEG)
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

        # ---- 무부하: EMF (λ_A 푸리에) --------------------------------
        angs_e = np.linspace(0, 45, n_emf, endpoint=False)
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

        # ---- 부하: 평균토크 + 리플 (리플 1주기 = 7.5° + 가드 2점) ------
        step = 7.5 / n_load
        angs_l = np.arange(-1, n_load + 1) * step      # 가드 포함 n+2점
        Wc, lam3, i3, Ta = [], [], [], []
        Bt = 0.0
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
        th = np.radians(angs_l)
        Wc = np.asarray(Wc)
        lam3 = np.asarray(lam3)
        i3 = np.asarray(i3)
        Tvw = (np.gradient(Wc, th)
               - np.sum(lam3 * np.gradient(i3, th, axis=0), axis=1))
        Tvw = Tvw[1:-1]                       # 가드 제거 → 정확히 1주기
        out["T_avg"] = float(np.mean(Tvw) * 1e3)
        out["ripple_pct"] = float((Tvw.max() - Tvw.min()) / np.mean(Tvw) * 100)
        out["T_arkkio"] = float(np.mean(np.asarray(Ta)[1:-1]) * 1e3)
        out["B_tooth"] = Bt
        out["magnet_area"] = float(sum(p.area for p, _, _ in geo.magnets))
    except Exception as e:  # noqa: BLE001
        out["status"] = f"fail: {type(e).__name__}: {e}"
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
    # 기준 설계도 포함 (코너 케이스 검증용)
    designs.insert(0, {"a_m": 0.89, "T_m": 2.2, "T_m2_ratio": 2.02 / 2.2,
                       "W_t": 3.5, "MagnetR": 0.8})

    done_keys = set()
    if resume and os.path.exists(out_path):
        with open(out_path) as f:
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
    t_start = _time.process_time()   # 컨테이너 정지 무관 CPU 시간

    done = 0
    with open(out_path, "a") as f:
        if nproc <= 1:
            _init(aedt_path)
            it = map(_eval, designs)
            for r in it:
                f.write(json.dumps(r) + "\n"); f.flush()
                done += 1
                if done % 5 == 0:
                    print(f"{done}/{len(designs)} "
                          f"cpu {_time.process_time()-t_start:.0f}s", flush=True)
                if time_budget and _time.process_time() - t_start > time_budget:
                    print("시간 예산 도달 — 체크포인트 후 종료", flush=True)
                    return
        else:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(
                    nproc, initializer=_init,
                    initargs=(aedt_path,)) as pool:
                for r in pool.imap_unordered(_eval, designs, chunksize=1):
                    f.write(json.dumps(r) + "\n"); f.flush()
                    done += 1
                    if done % 10 == 0:
                        print(f"{done}/{len(designs)}", flush=True)
    print("DOE DONE")
