"""회전 스윕: 각도별 형상 재생성 → 무부하 정자기 해석 → 토크·λ 수집."""
from __future__ import annotations

import multiprocessing as mp
import warnings
from typing import Dict, List

import numpy as np

warnings.filterwarnings("ignore")

_CTX: dict = {}


def _init(model, style, h_override):
    _CTX["model"] = model
    _CTX["style"] = style
    _CTX["h"] = h_override


def _solve_one(angle_deg: float) -> dict:
    from .geometry import build_motor
    from .meshing import build_mesh
    from .solver_ms import Magnetostatic2D
    from .postproc import torque_arkkio, build_winding_map, flux_linkages

    model, style, h = _CTX["model"], _CTX["style"], _CTX["h"]
    v = model["variables"]
    geo = build_motor(v, style, rotor_angle_deg=angle_deg)
    mesh = build_mesh(geo, h=h)
    s = Magnetostatic2D(mesh, model["materials"],
                        "20PNX1200F_20C", "Arnold_Magnetics_N45UH_80C")
    s.set_coil_currents({})
    res = s.solve()
    L = v["L_stk"]
    r_mag = (v["D_ro"] / 2) * 1e3        # 자석 크라운 정점 [mm]
    r_bore = geo.bore_radius
    T = torque_arkkio(s, res, r_mag + 0.03, r_bore - 0.03, L)
    wmap = build_winding_map(s)
    lam = flux_linkages(s, res, wmap, Zc=int(round(v["Zc"])), L_stk_m=L)
    return {"angle": angle_deg, "torque": T, **{f"lam_{k}": v2 for k, v2 in lam.items()},
            "iters": res.iterations, "res": res.residual,
            "Bmax": float(res.Bmag.max()), "Amax": float(np.abs(res.A).max())}


def sweep(model: dict, style: str, angles_deg: List[float],
          h_override: Dict[str, float] | None = None,
          nproc: int = 4) -> list:
    args = (model, style, h_override)
    if nproc <= 1:
        _init(*args)
        return [_solve_one(a) for a in angles_deg]
    with mp.get_context("spawn").Pool(nproc, initializer=_init,
                                      initargs=args) as pool:
        return pool.map(_solve_one, angles_deg)
