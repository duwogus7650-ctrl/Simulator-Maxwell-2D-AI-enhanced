# 내전형 전 모델 회귀 테스트 — 파싱→형상→메시→무부하 솔브→부하 스윕(6스텝)
# 사용: venv\Scripts\python scripts\run_regression.py [aedt 폴더]
import sys, os, glob, json, time, math, traceback, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
from motoropt.aedt_parser import (parse_aedt, detect_magnet_style,
                                  detect_material_names)
from motoropt.expressions import resolve_variables
from motoropt.geometry import build_motor
from motoropt.sliding import SlidingBandMesh
from motoropt.solver_ms import Magnetostatic2D
from motoropt.sweep_loss import (sweep_load_with_fields, compute_responses,
                                 calibrate_gamma)
from motoropt.winding import phase_resistance

FOLDER = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\user\Desktop\aedt파일"
results = []

for path in sorted(glob.glob(os.path.join(FOLDER, "*.aedt"))):
    name = os.path.basename(path)
    rec = {"file": name}
    t0 = time.time()
    try:
        m = parse_aedt(path)
        v = m["variables"]
        if v["D_ro"] > v["D_so"]:
            rec["status"] = "skip(외전형)"
            results.append(rec)
            print(f"⏭  {name}: 외전형 — 건너뜀", flush=True)
            continue
        style = detect_magnet_style(m)
        steel, mag = detect_material_names(m)
        rec.update(style=style, steel=steel, magnet=mag)

        # 형상 + 메시 + 무부하 솔브
        geo = build_motor(v, style)
        sbm = SlidingBandMesh(geo, n_band=2880)
        s = Magnetostatic2D(sbm.merge(0.0), m["materials"], steel, mag)
        s.set_coil_currents({})
        res = s.solve(tol=1e-5)
        rec.update(elems=int(len(s.area)), NR=res.iterations,
                   Az=float(np.abs(res.A).max()), Bmax=float(res.Bmag.max()))
        print(f"   {name}: 무부하 OK ({rec['elems']}요소, NR {res.iterations}, "
              f"|B|max {rec['Bmax']:.2f}T, {time.time()-t0:.0f}s)", flush=True)

        # 부하 스윕 (정격전류 없으면 1A 스모크)
        I = v.get("I_rms", 0) or 1.0
        rpm = v.get("BaseRPM", 0) or 1000.0
        ini = math.degrees(v.get("ini_pos", 0.0))
        cal = calibrate_gamma(m, style, rpm=rpm, I_rms=I, n_steps=4,
                              init_pos_deg=ini)
        sw = sweep_load_with_fields(m, style, rpm=rpm, I_rms=I,
                                    gamma_deg=cal["gamma_max_deg"],
                                    n_steps=6, init_pos_deg=ini,
                                    steel_name=steel, magnet_name=mag)
        w = phase_resistance(v, d_cu_mm=0.3, strands=11)   # 스모크용 사양
        r = compute_responses(sw, m, R_ph_ohm=w["R_ph"])
        rec.update(gamma=round(cal["gamma_max_deg"], 1),
                   T_avg=round(r["T_avg"], 4),
                   ripple_pct=round(r["T_ripple_pct"], 2),
                   P_fe=round(r["P_fe"], 2), eff=round(r["efficiency"], 4),
                   sec=round(time.time() - t0, 1))
        rec["status"] = "PASS"
        print(f"✅ {name}: γ*={rec['gamma']}° T={rec['T_avg']} N·m "
              f"리플 {rec['ripple_pct']}% P_fe {rec['P_fe']}W "
              f"η {rec['eff']*100:.1f}% | {rec['sec']}s", flush=True)
    except Exception:
        rec["status"] = "FAIL"
        rec["error"] = traceback.format_exc().splitlines()[-1]
        rec["sec"] = round(time.time() - t0, 1)
        print(f"❌ {name}: {rec['error']} | {rec['sec']}s", flush=True)
        traceback.print_exc()
    results.append(rec)

json.dump(results, open("regression_results.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
n_pass = sum(1 for r in results if r["status"] == "PASS")
n_fail = sum(1 for r in results if r["status"] == "FAIL")
print(f"\n=== 회귀 결과: PASS {n_pass} / FAIL {n_fail} / "
      f"스킵 {len(results)-n_pass-n_fail} ===")
sys.exit(1 if n_fail else 0)
