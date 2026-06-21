#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""정격 설계결과 산출 — 권선 레이아웃이 올바른 16P18S 모터들만, 정격 전류·정밀 스윕.

postproc.build_winding_map 이 18슬롯/16극 패턴(PATTERN_18S16P)을 하드코딩하므로
12슬롯(14P12S/10P12S)·외전형은 권선이 틀려 부하토크가 비물리적이다. 따라서 여기서는
권선이 검증된 16P18S + 정격전류(I_rms>0) 모터만 정밀(n_steps=36) 해석해 신뢰 가능한
설계결과(정격 토크·리플)를 낸다. 효율은 제네릭 권선(d_cu=0.3,strands=11) 기준 근사.
"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8")
    except Exception: pass
from run_cli import analyze_one

BASE = r"C:/Users/user/Desktop/aedt파일"
# 권선 검증된 16P18S + 정격전류 보유 모터
MOTORS = ["400W.aedt", "750W,1200W.aedt",
          "InnerType_KRO80_120Nm_GearRatio_13_4.aedt", "InwheelMotor.aedt",
          "SH_Reducer_QDD_14_17.aedt", "SH_Reducer_QDD_20.aedt"]

out = []
for f in MOTORS:
    rec = analyze_one(os.path.join(BASE, f), n_steps=36)  # 정격 I_rms 사용
    out.append(rec)
    if rec["status"] == "PASS":
        print("✅ %-42s %2dP%2dS  I=%5.1fA  γ*=%6.1f°  T=%7.4f N·m  "
              "리플 %5.1f%%  η≈%4.1f%%" % (
                  f, rec["N_pole"], rec["N_slot"], rec["I_rms_A"],
                  rec["gamma_deg"], rec["T_avg_Nm"], rec["ripple_pct"],
                  rec["efficiency"]*100), flush=True)
    else:
        print("⚠ %-42s %s" % (f, rec["status"]), flush=True)

json.dump(out, open("design_results_rated.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
ts = {(round(r["T_avg_Nm"], 4)) for r in out if r["status"] == "PASS"}
print("\n=== %d개 PASS, 고유 토크 %d개 ===" % (
    sum(1 for r in out if r["status"] == "PASS"), len(ts)))
