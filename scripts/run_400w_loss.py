# 400W 회귀 + 손실·효율 검증 (PDF Rev1 p.7 기준, 4500rpm / 4.9Arms)
import sys, time, math, warnings, json
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, ".")
from motoropt.aedt_parser import parse_aedt, detect_magnet_style
from motoropt.geometry import build_motor
from motoropt.sliding import SlidingBandMesh
from motoropt.solver_ms import Magnetostatic2D
from motoropt.postproc import torque_arkkio, coenergy, build_winding_map, flux_linkages
from motoropt.sweep_loss import sweep_load_with_fields, compute_responses, calibrate_gamma

AEDT = r"C:\Users\user\Desktop\aedt파일\400W.aedt"
t0 = time.time()

m = parse_aedt(AEDT)
v = m["variables"]
style = detect_magnet_style(m)
print(f"design={m['design_name']} style={style} vars={len(v)}", flush=True)

pp_vw = pp_ark = None
if "--skip-cog" in sys.argv:
    print("[코깅 회귀] 건너뜀 (이전 실행: 가상일 6.31 / Arkkio 6.63 mNm)", flush=True)

# ── 1) 코깅 회귀 (weld 패치 후 4.93/5.12 mNm 재현 확인, 144차 폴딩 2.5°) ──
if "--skip-cog" not in sys.argv:
    geo = build_motor(v, style)
    sbm = SlidingBandMesh(geo, n_band=5760, gap_frac=(0.35, 0.65),
                          h={"air_gap_in": 0.18, "air_gap_out": 0.18,
                             "magnet": 0.6})
    L = v["L_stk"]; Zc = int(round(v["Zc"]))
    wm = {}
    rows = []
    angs = np.arange(0.0, 2.5 + 1e-9, 0.125)
    for a in angs:
        mesh = sbm.merge(a)
        s = Magnetostatic2D(mesh, m["materials"], "20PNX1200F_20C",
                            "Arnold_Magnetics_N45UH_80C")
        if "w" not in wm:
            wm["w"] = build_winding_map(s)
            print(f"  mesh elems={len(s.area)}", flush=True)
        s.set_coil_currents({})
        res = s.solve(tol=1e-6)
        lam = flux_linkages(s, res, wm["w"], Zc, L)
        rows.append([a, torque_arkkio(s, res, sbm.r_i + 0.003, sbm.r_o - 0.003, L),
                     coenergy(s, res, L), lam["A"], lam["B"], lam["C"]])
    arr = np.array(rows)
    th = np.radians(arr[:, 0])
    T_ark = arr[:, 1] * 1e3
    T_vw = np.gradient(arr[:, 2], th) * 1e3   # 무전류: dW'/dθ
    pp_ark = T_ark.max() - T_ark.min()
    pp_vw = T_vw[2:-2].max() - T_vw[2:-2].min()
    print(f"[코깅 회귀] 가상일 pk2pk {pp_vw:.2f} | Arkkio {pp_ark:.2f} mNm "
          f"(이전 4.93/5.12, Maxwell PDF 5.95) | {time.time()-t0:.0f}s", flush=True)
    np.savez("p400_cog_regression.npz", data=arr)

# ── 2) 전류각 캘리브레이션 (기본 메시, 4점 프로브) ──
ini = math.degrees(v["ini_pos"])
cal = calibrate_gamma(m, style, rpm=4500, I_rms=4.9, n_steps=6,
                      init_pos_deg=ini)
print(f"[γ 캘리브레이션] γ*={cal['gamma_max_deg']:.1f}° "
      f"T_max≈{cal['T_max_est']*1e3:.0f} mNm probes={ {k: round(t*1e3) for k, t in cal['probes'].items()} } "
      f"| {time.time()-t0:.0f}s", flush=True)

# ── 3) 부하 손실 스윕 (전기 1주기 36스텝, 정밀 메시) ──
sw = sweep_load_with_fields(m, style, rpm=4500, I_rms=4.9,
                            gamma_deg=cal["gamma_max_deg"], n_steps=36,
                            init_pos_deg=ini, n_band=5760,
                            h={"air_gap_in": 0.18, "air_gap_out": 0.18,
                               "magnet": 0.6}, verbose=True)
r_est = compute_responses(sw, m, R_ph_ohm=None)          # 기하 추정 동손
R_PDF = 11.4 / (3 * 4.9**2)                              # PDF 동손 역산 상저항
r_pdf = compute_responses(sw, m, R_ph_ohm=R_PDF)

def fmt(r):
    return (f"T_avg {r['T_avg']*1e3:.1f} mNm | 리플 {r['T_ripple_pct']:.2f}% "
            f"({r['T_ripple_pp']*1e3:.1f} mNm pp) | P_fe {r['P_fe']:.1f} W "
            f"(st {r['P_fe_stator']['P_total']:.1f}/rt {r['P_fe_rotor']['P_total']:.2f}) | "
            f"P_cu {r['P_cu']:.1f} W (R_ph {r['R_ph']*1e3:.0f} mΩ"
            f"{', 추정' if r['R_ph_estimated'] else ''}) | η {r['efficiency']*100:.2f}%")

print("\n=== PDF 기준: T 850 mNm | 리플 2.97% | P_cu 11.4 | P_fe 15.8 | 자석 2.2 | η 93.1% ===")
print("[기하 R_ph]", fmt(r_est))
print(f"[PDF  R_ph] {fmt(r_pdf)}  (R_ph={R_PDF*1e3:.0f} mΩ, PDF 동손 역산)")
eta_with_pm = r_pdf["P_out"] / (r_pdf["P_out"] + r_pdf["P_cu"] + r_pdf["P_fe"] + 2.2)
print(f"[자석손 2.2W 가산 시] η {eta_with_pm*100:.2f}% | 총 {time.time()-t0:.0f}s")
json.dump({"resp_est": r_est, "resp_pdfR": r_pdf,
           "gamma": cal["gamma_max_deg"], "cog_pp_vw": pp_vw,
           "cog_pp_ark": pp_ark},
          open("p400_loss_results.json", "w"), indent=1, default=str)
