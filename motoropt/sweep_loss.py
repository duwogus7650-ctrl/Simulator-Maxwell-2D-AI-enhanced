"""부하 회전 스윕 + 손실·효율 응답 (P4 확장).

SlidingBandMesh를 사용해 스테이터/로터 메시를 고정하고 전기 1주기를
등간격 스윕한다. 메시 위상이 각도 간 동일하므로 요소별 B(t) 시계열을
그대로 추적할 수 있고, 이를 coreloss.harmonic 분해에 넘겨 철손을 구한다.

3상 여자(Maxwell 등가):
    i_A = √2·I_rms · sin(θe + γ)
    i_B = √2·I_rms · sin(θe − 120° + γ)
    i_C = √2·I_rms · sin(θe + 120° + γ)
θe = 전기각(스윕 진행), γ = 전류 위상각. 로터 기계각은
ini_pos + θe/pp 로 진행해 aedt PeakLoad 정렬을 재현한다.

반환 응답: T_avg, T_ripple(pp·%), P_fe(스테이터/로터, 성분 분해),
P_cu, P_out, efficiency.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .geometry import build_motor
from .sliding import SlidingBandMesh
from .solver_ms import Magnetostatic2D
from .postproc import torque_arkkio, build_winding_map, coenergy
from .coreloss import core_loss, copper_loss_dc


def sweep_load_with_fields(model: dict, style: str, *,
                           rpm: float, I_rms: float,
                           gamma_deg: float = 0.0,
                           n_steps: int = 36,
                           init_pos_deg: float | None = None,
                           h: Dict[str, float] | None = None,
                           n_band: int = 2880,
                           steel_name: str | None = None,
                           magnet_name: str | None = None,
                           compute_vw: bool = True,
                           verbose: bool = False) -> dict:
    """전기 1주기 부하 스윕 — 토크 시계열 + 강 요소 B(t) 수집."""
    v = model["variables"]
    if steel_name is None or magnet_name is None:
        for name, mat in model["materials"].items():
            if steel_name is None and "core_loss_kh" in mat:
                steel_name = name
            if magnet_name is None and "coercivity_A_per_m" in mat:
                magnet_name = name
        if steel_name is None or magnet_name is None:
            raise ValueError("강판/자석 재질 자동감지 실패 — 명시 필요")
    pp = int(round(v["N_pole"])) // 2
    span_mech = 360.0 / pp                       # 전기 1주기의 기계각
    if init_pos_deg is None:
        init_pos_deg = float(v.get("ini_pos", 0.0)) \
            if isinstance(v.get("ini_pos", 0.0), (int, float)) else 0.0

    geo = build_motor(v, style, rotor_angle_deg=0.0)
    band = SlidingBandMesh(geo, n_band=n_band, h=h)

    L = v["L_stk"]
    r_mag = (v["D_ro"] / 2) * 1e3                # [mm]
    r_bore = geo.bore_radius
    Zc = int(round(v["Zc"]))
    I_pk = math.sqrt(2.0) * I_rms
    g_rad = math.radians(gamma_deg)

    torque = np.empty(n_steps)            # Arkkio (리플 파형용)
    torque_vw = np.empty(n_steps)         # 가상일 coenergy (정확 평균용)
    d_vw_deg = 0.25                       # 가상일 dθ 섭동 [기계각 °]
    lam = {ph: np.empty(n_steps) for ph in "ABC"}
    sel_st = sel_rt = None
    Bx_st = By_st = Bx_rt = By_rt = None
    wmap = None
    areas_st = areas_rt = None

    for i in range(n_steps):
        th_e = 2.0 * math.pi * i / n_steps                  # 전기각 [rad]
        ang = init_pos_deg + span_mech * i / n_steps        # 로터 기계각 [deg]
        mesh = band.merge(ang)
        s = Magnetostatic2D(mesh, model["materials"], steel_name, magnet_name)

        if wmap is None:                                    # 1회만 (외측 고정)
            wmap = build_winding_map(s, n_slot=int(round(v["N_slot"])))
            kind = s.kind
            sel_st = np.where(kind == "stator")[0]
            sel_rt = np.where(kind == "rotor")[0]
            areas_st = s.area[sel_st].copy()                # [m²]
            areas_rt = s.area[sel_rt].copy()
            Bx_st = np.empty((n_steps, len(sel_st)))
            By_st = np.empty_like(Bx_st)
            Bx_rt = np.empty((n_steps, len(sel_rt)))
            By_rt = np.empty_like(Bx_rt)

        i_ph = {"A": I_pk * math.sin(th_e + g_rad),
                "B": I_pk * math.sin(th_e + g_rad - 2 * math.pi / 3),
                "C": I_pk * math.sin(th_e + g_rad + 2 * math.pi / 3)}
        cur = {}
        for ph, sides in wmap.items():
            for ci, d in sides:
                cur[ci] = d * Zc * i_ph[ph]
        s.set_coil_currents(cur)
        res = s.solve()

        torque[i] = torque_arkkio(s, res, r_mag + 0.03, r_bore - 0.03, L)
        # 가상일 토크: 전류 고정·로터만 Δθ 회전 → T = dW_co/dθ (전진차분).
        # Arkkio는 메시·적분반경에 따라 ±수% 바이어스가 있어, 절대 평균토크는
        # 가상일이 더 정확(400W 가상일 +1.0% vs Arkkio +8% 검증).
        if compute_vw:
            W0 = coenergy(s, res, L)
            s2 = Magnetostatic2D(band.merge(ang + d_vw_deg),
                                 model["materials"], steel_name, magnet_name)
            s2.set_coil_currents(cur)
            res2 = s2.solve()
            torque_vw[i] = (coenergy(s2, res2, L) - W0) / math.radians(d_vw_deg)
        else:
            torque_vw[i] = torque[i]      # 캘리브레이션 등 각도탐색엔 불필요
        from .postproc import flux_linkages
        for ph, val in flux_linkages(s, res, wmap, Zc=Zc, L_stk_m=L).items():
            lam[ph][i] = val

        Bx_st[i] = res.Bx[sel_st]
        By_st[i] = res.By[sel_st]
        # 로터: 재질 좌표계(−ang 회전)로 변환 — 강체 회전분 제거
        c, sn = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        Bx_rt[i] = c * res.Bx[sel_rt] + sn * res.By[sel_rt]
        By_rt[i] = -sn * res.Bx[sel_rt] + c * res.By[sel_rt]

        if verbose:
            print(f"  step {i + 1:2d}/{n_steps}  ang={ang:7.3f}°  "
                  f"T={torque[i]:8.4f} N·m  NR={res.iterations} "
                  f"res={res.residual:.2e}")

    return {"torque": torque, "torque_vw": torque_vw, "lam": lam,
            "Bx_st": Bx_st, "By_st": By_st, "areas_st": areas_st,
            "Bx_rt": Bx_rt, "By_rt": By_rt, "areas_rt": areas_rt,
            "rpm": rpm, "I_rms": I_rms, "pp": pp, "L_stk": L,
            "n_steps": n_steps, "steel_name": steel_name,
            "coil_areas": _coil_side_areas(s),
            "n_coil_sides": int(np.sum([len(x) for x in wmap.values()]))}


def _coil_side_areas(solver) -> float:
    """코일사이드 평균 단면적 [m²] (동손 기하 추정용)."""
    sel = solver.is_coil
    coil_idx = np.array([solver.table[a].get("index", -1)
                         for a in solver.attr])
    uniq = np.unique(coil_idx[sel])
    areas = [solver.area[sel & (coil_idx == ci)].sum() for ci in uniq]
    return float(np.mean(areas))


def compute_responses(sw: dict, model: dict, *,
                      R_ph_ohm: float | None = None,
                      L_end_m: float = 0.0,
                      fill_factor: float = 0.45,
                      temp_C: float = 80.0) -> dict:
    """스윕 결과 → 응답 dict (Objective 탭 연동용).

    R_ph_ohm 제공 시 동손은 3·I²·R(권장), 미제공 시 기하 추정.
    """
    mat = model["materials"][sw["steel_name"]]
    kh, kc, ke = (mat["core_loss_kh"], mat["core_loss_kc"],
                  mat["core_loss_ke"])
    ks = mat.get("stacking_factor", 1.0)
    f1 = sw["rpm"] / 60.0 * sw["pp"]

    fe_st = core_loss(sw["Bx_st"], sw["By_st"], sw["areas_st"], f1,
                      sw["L_stk"], kh, kc, ke, stacking_factor=ks)
    fe_rt = core_loss(sw["Bx_rt"], sw["By_rt"], sw["areas_rt"], f1,
                      sw["L_stk"], kh, kc, ke, stacking_factor=ks)

    cu = copper_loss_dc(sw["I_rms"], R_ph_ohm,
                        coil_side_area_m2=sw["coil_areas"],
                        n_coil_sides=sw["n_coil_sides"],
                        Zc=int(round(model["variables"]["Zc"])),
                        L_stk_m=sw["L_stk"], L_end_m=L_end_m,
                        fill_factor=fill_factor, temp_C=temp_C)

    T = sw["torque"]                               # Arkkio (리플 파형)
    T_vw = sw.get("torque_vw", T)                  # 가상일 (정확 평균)
    T_avg = float(T_vw.mean())                     # ← 평균토크는 가상일
    T_avg_arkkio = float(T.mean())
    T_pp = float(T.max() - T.min())                # 리플 진폭은 Arkkio 파형에서
    w_m = sw["rpm"] * 2 * math.pi / 60.0
    P_out = T_avg * w_m
    P_fe = fe_st.P_total + fe_rt.P_total
    P_loss = P_fe + cu["P_cu"]
    eta = P_out / (P_out + P_loss) if P_out > 0 else 0.0

    return {
        "T_avg": T_avg,
        "T_avg_arkkio": T_avg_arkkio,
        "T_ripple_pp": T_pp,
        "T_ripple_pct": 100.0 * T_pp / abs(T_avg) if T_avg else float("inf"),
        "P_out": P_out,
        "P_fe_stator": fe_st.as_dict(),
        "P_fe_rotor": fe_rt.as_dict(),
        "P_fe": P_fe,
        "P_cu": cu["P_cu"],
        "R_ph": cu["R_ph"],
        "R_ph_estimated": cu["estimated"],
        "efficiency": eta,
        "f1_Hz": f1,
        "note": "자석 와류손·기계손 미포함. "
                "η = P_out/(P_out+P_cu+P_fe).",
    }


def calibrate_gamma(model: dict, style: str, *, rpm: float, I_rms: float,
                    n_steps: int = 6, init_pos_deg: float | None = None,
                    h: Dict[str, float] | None = None) -> dict:
    """전류 위상각 γ 캘리브레이션 — 4점 프로브 + 사인 피팅.

    SPM에서 T(γ) = A sin γ + B cos γ 이므로 0/90/180/270° 평균토크로
    A, B를 구하고 γ* = atan2(A, B)에서 최대토크(MTPA, q축 전류)가 된다.
    내부 권선 패턴과 aedt 상 명칭의 위상 오프셋도 여기서 흡수된다.
    """
    T = {}
    for g in (0.0, 90.0, 180.0, 270.0):
        sw = sweep_load_with_fields(model, style, rpm=rpm, I_rms=I_rms,
                                    gamma_deg=g, n_steps=n_steps,
                                    init_pos_deg=init_pos_deg, h=h,
                                    compute_vw=False)
        T[g] = float(sw["torque"].mean())
    A = (T[90.0] - T[270.0]) / 2.0
    B = (T[0.0] - T[180.0]) / 2.0
    g_star = math.degrees(math.atan2(A, B))
    return {"gamma_max_deg": g_star % 360.0,
            "T_max_est": math.hypot(A, B), "probes": T}
