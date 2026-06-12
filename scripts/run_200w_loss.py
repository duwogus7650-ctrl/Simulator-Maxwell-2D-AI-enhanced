# 200W 손실·효율 검증 — 400W 형상에서 L_stk=15*0.93mm, I_rms=3.7A만 변경
# (PDF Rev1 p.7: T 480 mNm | 리플 2.73% | P_cu 5.5 | P_fe 11.2 | 자석 1.2 | η 92.4%)
import sys, time, math, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from motoropt.aedt_parser import parse_aedt, detect_magnet_style
from motoropt.sweep_loss import sweep_load_with_fields, compute_responses

AEDT = r"C:\Users\user\Desktop\aedt파일\400W.aedt"
t0 = time.time()
m = parse_aedt(AEDT)
v = m["variables"]
v["L_stk"] = 0.015 * 0.93          # 적층 15mm (400W와 동일 비율 0.93 적용)
ini = math.degrees(v["ini_pos"])

# γ*는 권선 패턴 종속 — 400W 캘리브레이션 값 재사용
sw = sweep_load_with_fields(m, detect_magnet_style(m), rpm=4500, I_rms=3.7,
                            gamma_deg=190.88, n_steps=36, init_pos_deg=ini,
                            n_band=5760,
                            h={"air_gap_in": 0.18, "air_gap_out": 0.18,
                               "magnet": 0.6}, verbose=False)
r_est = compute_responses(sw, m, R_ph_ohm=None)
R_PDF = 5.5 / (3 * 3.7**2)
r_pdf = compute_responses(sw, m, R_ph_ohm=R_PDF)

def fmt(r):
    return (f"T_avg {r['T_avg']*1e3:.1f} mNm | 리플 {r['T_ripple_pct']:.2f}% "
            f"({r['T_ripple_pp']*1e3:.1f} mNm pp) | P_fe {r['P_fe']:.1f} W | "
            f"P_cu {r['P_cu']:.1f} W (R_ph {r['R_ph']*1e3:.0f} mΩ"
            f"{', 추정' if r['R_ph_estimated'] else ''}) | η {r['efficiency']*100:.2f}%")

print("=== PDF 기준: T 480 mNm | 리플 2.73% | P_cu 5.5 | P_fe 11.2 | 자석 1.2 | η 92.4% ===")
print("[기하 R_ph]", fmt(r_est))
print("[PDF  R_ph]", fmt(r_pdf))
eta_pm = r_pdf["P_out"] / (r_pdf["P_out"] + r_pdf["P_cu"] + r_pdf["P_fe"] + 1.2)
print(f"[자석손 1.2W 가산 시] η {eta_pm*100:.2f}% | 총 {time.time()-t0:.0f}s")
json.dump({"resp_est": r_est, "resp_pdfR": r_pdf},
          open("p200_loss_results.json", "w"), indent=1, default=str)
