"""P4: 부하 해석 — 전류각 캘리브레이션 + 전기 1주기 토크 스윕."""
import sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, ".")
from motoropt.aedt_parser import parse_aedt, detect_magnet_style
from motoropt.geometry import build_motor
from motoropt.sliding import SlidingBandMesh
from motoropt.solver_ms import Magnetostatic2D
from motoropt.postproc import (torque_arkkio, coenergy,
                               build_winding_map, flux_linkages)

AEDT = "/mnt/user-data/uploads/400W.aedt"
m = parse_aedt(AEDT)
v = m["variables"]
geo = build_motor(v, detect_magnet_style(m))
sbm = SlidingBandMesh(geo, n_band=5760, gap_frac=(0.35, 0.65),
                      h={"air_gap_in": 0.18, "air_gap_out": 0.18,
                         "magnet": 0.6})
L = v["L_stk"]
Zc = int(round(v["Zc"]))
Ia = v["I_rms"] * np.sqrt(2.0)
pp = int(round(v["N_pole"])) // 2          # 극쌍수 8

_wmap_cache = {}

def solve_at(theta_deg, delta_e_rad):
    """로터각 theta, 전류 전기위상 delta에서 해석."""
    mesh = sbm.merge(theta_deg)
    s = Magnetostatic2D(mesh, m["materials"],
                        "20PNX1200F_20C", "Arnold_Magnetics_N45UH_80C")
    if "w" not in _wmap_cache:
        _wmap_cache["w"] = build_winding_map(s)
    wmap = _wmap_cache["w"]
    th_e = pp * np.radians(theta_deg) + delta_e_rad
    i_ph = {"A": Ia * np.sin(th_e),
            "B": Ia * np.sin(th_e - 2 * np.pi / 3),
            "C": Ia * np.sin(th_e + 2 * np.pi / 3)}
    at = {}
    for ph, sides in wmap.items():
        for ci, d in sides:
            at[ci] = d * Zc * i_ph[ph]
    s.set_coil_currents(at)
    res = s.solve(tol=1e-6)
    T_ak = torque_arkkio(s, res, sbm.r_i + 0.003, sbm.r_o - 0.003, L)
    Wc = coenergy(s, res, L)
    lam = flux_linkages(s, res, wmap, Zc, L)
    return T_ak, Wc, lam, i_ph, float(res.Bmag.max()), res.iterations

if sys.argv[1] == "calib":
    # 로터 0° 고정, 전류 전기각 스캔 → 최대토크점(beta=0)
    print("delta_e[deg]  T_arkkio[mNm]")
    for de in np.arange(0, 360, 15):
        T, *_ = solve_at(0.0, np.radians(de))
        print(f"{de:8.1f}  {T*1e3:10.2f}", flush=True)

elif sys.argv[1] == "sweep":
    d0 = float(sys.argv[2])                 # 캘리브레이션된 delta_e [deg]
    angs = np.arange(0, 45 + 1e-9, 0.625)
    rows = []
    t0 = time.time()
    for i, a in enumerate(angs):
        T, Wc, lam, iph, Bmax, it = solve_at(a, np.radians(d0))
        rows.append([a, T, Wc, lam["A"], lam["B"], lam["C"],
                     iph["A"], iph["B"], iph["C"], Bmax, it])
        if i % 12 == 0:
            print(f"{i}/{len(angs)} {time.time()-t0:.0f}s "
                  f"T={T*1e3:.1f}mNm NR{it}", flush=True)
    arr = np.array(rows)
    np.savez("p4_load.npz", data=arr, delta0=d0)
    print("DONE", time.time() - t0)
