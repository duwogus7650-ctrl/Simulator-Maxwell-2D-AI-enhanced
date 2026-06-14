# 솔버 자가검증 — 검증된 기준값(Maxwell/PDF)과 자동 대조해 드리프트 감지.
# 코드를 고친 뒤 이걸 돌려서 솔버 정확도가 틀어지지 않았는지 확인한다.
# 사용: venv\Scripts\python scripts\run_validation.py [aedt폴더]
#   (기본 폴더: C:\Users\user\Desktop\aedt파일)
import sys, os, math, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
from motoropt.aedt_parser import (parse_aedt, detect_magnet_style,
                                  detect_material_names)
from motoropt.sweep_loss import (sweep_load_with_fields, compute_responses,
                                  calibrate_gamma)

FOLDER = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\user\Desktop\aedt파일"

# ── 기준값: (motoropt 검증값, 허용오차, Maxwell/PDF 실측, 설명) ────────────
# motoropt 값에서 벗어나면 = 코드가 솔버 출력을 바꿈(드리프트) → FAIL.
checks = []


def emf_rms(model, style, steel, mag, rpm):
    """무부하 상EMF rms [V] — 자속쇄교 기본파."""
    v = model["variables"]; pp = int(round(v["N_pole"])) // 2
    ini = math.degrees(v.get("ini_pos", 0.0)); n = 36
    sw = sweep_load_with_fields(model, style, rpm=rpm, I_rms=0.0,
                                gamma_deg=0.0, n_steps=n, init_pos_deg=ini,
                                steel_name=steel, magnet_name=mag,
                                compute_vw=False)
    lam = np.asarray(sw["lam"]["A"]); k = np.fft.rfft(lam) / n
    return 2 * abs(k[1]) * (rpm / 60 * 2 * math.pi * pp) / math.sqrt(2)


def load_pt(model, style, steel, mag, *, I_ph, rpm, d_cu, strands, tcu):
    """부하: 가상일 평균토크[N·m]·효율 — γ 캘리브 후 1주기 스윕."""
    from motoropt.winding import phase_resistance
    v = model["variables"]; ini = math.degrees(v.get("ini_pos", 0.0))
    cal = calibrate_gamma(model, style, rpm=rpm, I_rms=I_ph, n_steps=6,
                          init_pos_deg=ini)
    sw = sweep_load_with_fields(model, style, rpm=rpm, I_rms=I_ph,
                                gamma_deg=cal["gamma_max_deg"], n_steps=36,
                                init_pos_deg=ini, steel_name=steel,
                                magnet_name=mag)
    rph = phase_resistance(v, d_cu_mm=d_cu, strands=strands,
                           T_cu_C=tcu)["R_ph"]
    r = compute_responses(sw, model, R_ph_ohm=rph)
    return r["T_avg"], r["efficiency"]


def run():
    # ── 400W (경부하 정격, 자석 일반형상) ─────────────────────────────
    p = os.path.join(FOLDER, "400W.aedt")
    if os.path.exists(p):
        m = parse_aedt(p); style = detect_magnet_style(m)
        steel, mag = detect_material_names(m)
        e = emf_rms(m, style, steel, mag, 1000.0)
        checks.append(("400W 무부하 EMF@1000rpm", e, 6.16, 0.02, "rel",
                       "Maxwell 6.166 V (검증 -0.1%)"))
        T, eff = load_pt(m, style, steel, mag, I_ph=4.9, rpm=4500,
                         d_cu=0.3, strands=11, tcu=80)
        checks.append(("400W 부하토크(가상일)@4.9A", T, 0.86, 0.05, "rel",
                       "PDF 0.85 N·m (검증 +1~2%)"))
        checks.append(("400W 효율@정격", eff, 0.947, 0.02, "abs",
                       "PDF 93.1% (자석와류·기계손 제외분)"))
    else:
        print(f"⏭  400W.aedt 없음 ({p}) — 건너뜀")

    # ── KRO80 V3 (순시 과부하, spline 자석, 14턴) ─────────────────────
    p = os.path.join(FOLDER, "InnerType_KRO80_120Nm_GearRatio_13_4.aedt")
    if os.path.exists(p):
        m = parse_aedt(p); style = detect_magnet_style(m)
        steel, mag = detect_material_names(m)
        m["variables"]["Zc"] = 14.0            # 설계 턴수(aedt는 15)
        m["variables_raw"]["Zc"] = "14"
        # KRO80 코일선경 0.3mm(MotorCAD 실제). ⚠ MLT 동손추정이 KRO80
        # 엔드와인딩을 ~25% 과소(MLT 84 vs 보고서함의 107mm) → 동손 400 vs
        # 533W → 효율 74% vs 보고서 67.8%(+6%p). 400W/1250W는 MLT가 맞지만
        # KRO80은 안 맞음(geometry 특이). 정확한 효율은 GUI 상저항에 실측 R_ph
        # (≈245mΩ) 입력. 여기선 motoropt 실측(MLT추정)을 드리프트 기준으로 둠.
        T, eff = load_pt(m, style, steel, mag, I_ph=46.59 / math.sqrt(3),
                         rpm=1478, d_cu=0.3, strands=11, tcu=80)
        checks.append(("KRO80 부하토크(가상일)@26.9A", T, 7.66, 0.05, "rel",
                       "Maxwell 7.13 N·m (극한포화 +7.5%)"))
        checks.append(("KRO80 효율@순시(0.3mm,MLT추정)", eff, 0.741, 0.02, "abs",
                       "보고서 67.8% — MLT가 KRO80 엔드와인딩 과소(동손-25%)·"
                       "실측 R_ph 입력 시 일치"))
    else:
        print(f"⏭  KRO80 aedt 없음 ({p}) — 건너뜀")

    n_fail = 0
    print("\n=== 솔버 자가검증 (motoropt 기준값 대비 드리프트) ===")
    for name, got, ref, tol, mode, note in checks:
        dev = (got - ref) / ref if mode == "rel" else (got - ref)
        ok = abs(dev) <= tol
        n_fail += 0 if ok else 1
        dtxt = (f"{dev*100:+.1f}%" if mode == "rel"
                else f"{dev:+.3f}")
        flag = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {flag} {name}: {got:.4g} (기준 {ref:g}, 편차 {dtxt}, "
              f"허용 {'±%g%%' % (tol*100) if mode=='rel' else '±%g' % tol}) "
              f"| {note}")
    print(f"\n=== 검증 결과: {len(checks)-n_fail} PASS / {n_fail} FAIL ===")
    if n_fail:
        print("⚠ 솔버 출력이 검증값에서 벗어남 — 최근 코드 변경을 점검하세요.")
    return n_fail


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
