#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""오프라인 헤드리스 러너 — GUI 없이 .aedt 한 개 또는 폴더를 해석한다.

디스플레이/Qt 없이 순수 Python으로 동작하므로 서버·오프라인·자동검증에 쓴다.
GUI(gui/app.py)와 동일한 motoropt 파이프라인을 사용한다:
    파싱 → 형상 → 슬라이딩밴드 메시 → 무부하 솔브 → 부하 스윕 → 응답 지표

사용법:
    python run_cli.py <파일.aedt | 폴더>            # 폴더면 *.aedt 전부
    python run_cli.py <...> --current 27.1          # 상전류[A] 지정(없으면 파일값/1A)
    python run_cli.py <...> --rpm 4500              # 회전수 지정(없으면 파일값/1000)
    python run_cli.py <...> --steps 12             # 부하 스윕 스텝 수(기본 6)
    python run_cli.py <...> --json out.json         # 결과를 JSON으로 저장

예:
    python run_cli.py "C:/Users/user/Desktop/aedt파일/400W.aedt"
    python run_cli.py "C:/Users/user/Desktop/aedt파일" --json results.json

각 모터는 자신의 형상·변수로 독립 해석되므로 결과는 모터마다 다르게 나온다.
외전형(D_ro > D_so)은 현재 2D 내전형 솔버 범위 밖이라 건너뛴다(명시적으로 보고).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
import traceback

# 루트를 import 경로에 추가(어느 cwd에서 실행해도 motoropt를 찾도록)
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Windows cp949 콘솔에서 한글/기호 출력 시 깨지지 않도록 UTF-8 재구성
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from motoropt.aedt_parser import (parse_aedt, detect_magnet_style,
                                  detect_material_names)
from motoropt.geometry import build_motor
from motoropt.sliding import SlidingBandMesh
from motoropt.solver_ms import Magnetostatic2D
from motoropt.sweep_loss import (sweep_load_with_fields, compute_responses,
                                 calibrate_gamma)
from motoropt.winding import phase_resistance


def analyze_one(path: str, current: float | None = None,
                rpm: float | None = None, n_steps: int = 6) -> dict:
    """단일 .aedt를 끝까지 해석해 결과 dict를 돌려준다.

    실패하면 status=="FAIL"+error, 외전형이면 status=="skip(외전형)"."""
    name = os.path.basename(path)
    rec: dict = {"file": name}
    t0 = time.time()
    m = parse_aedt(path)
    v = m["variables"]
    rec["design_name"] = m.get("design_name", "")
    if v["D_ro"] > v["D_so"]:
        rec["status"] = "skip(외전형)"
        return rec
    style = detect_magnet_style(m)
    steel, mag = detect_material_names(m)
    rec.update(style=style, steel=steel, magnet=mag,
               D_so_mm=round(v.get("D_so", 0) * 1000, 2),
               D_ro_mm=round(v.get("D_ro", 0) * 1000, 2),
               N_pole=int(v.get("N_pole", 0)), N_slot=int(v.get("N_slot", 0)))

    geo = build_motor(v, style)
    sbm = SlidingBandMesh(geo, n_band=2880)
    s = Magnetostatic2D(sbm.merge(0.0), m["materials"], steel, mag)
    s.set_coil_currents({})
    res = s.solve(tol=1e-5)
    rec.update(elems=int(len(s.area)), NR=res.iterations,
               Bmax=round(float(res.Bmag.max()), 3))

    I = current if current is not None else (v.get("I_rms", 0) or 1.0)
    rpm_use = rpm if rpm is not None else (v.get("BaseRPM", 0) or 1000.0)
    ini = math.degrees(v.get("ini_pos", 0.0))
    cal = calibrate_gamma(m, style, rpm=rpm_use, I_rms=I, n_steps=4,
                          init_pos_deg=ini)
    sw = sweep_load_with_fields(m, style, rpm=rpm_use, I_rms=I,
                                gamma_deg=cal["gamma_max_deg"], n_steps=n_steps,
                                init_pos_deg=ini, steel_name=steel,
                                magnet_name=mag)
    w = phase_resistance(v, d_cu_mm=0.3, strands=11)
    r = compute_responses(sw, m, R_ph_ohm=w["R_ph"])
    rec.update(I_rms_A=round(I, 2), rpm=round(rpm_use, 1),
               gamma_deg=round(cal["gamma_max_deg"], 1),
               T_avg_Nm=round(r["T_avg"], 4),
               ripple_pct=round(r["T_ripple_pct"], 2),
               P_fe_W=round(r["P_fe"], 2),
               efficiency=round(r["efficiency"], 4),
               sec=round(time.time() - t0, 1))
    rec["status"] = "PASS"
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="오프라인 모터 해석 러너")
    ap.add_argument("target", help="*.aedt 파일 또는 .aedt들이 든 폴더")
    ap.add_argument("--current", type=float, default=None, help="상전류[A]")
    ap.add_argument("--rpm", type=float, default=None, help="회전수[rpm]")
    ap.add_argument("--steps", type=int, default=6, help="부하 스윕 스텝 수")
    ap.add_argument("--json", dest="json_out", default=None, help="결과 JSON 경로")
    args = ap.parse_args(argv)

    if os.path.isdir(args.target):
        paths = sorted(glob.glob(os.path.join(args.target, "*.aedt")))
    elif os.path.isfile(args.target):
        paths = [args.target]
    else:
        print(f"[오류] 경로를 찾을 수 없음: {args.target}", file=sys.stderr)
        return 2
    if not paths:
        print(f"[오류] .aedt 파일이 없습니다: {args.target}", file=sys.stderr)
        return 2

    results = []
    for p in paths:
        name = os.path.basename(p)
        try:
            rec = analyze_one(p, current=args.current, rpm=args.rpm,
                              n_steps=args.steps)
        except Exception:
            rec = {"file": name, "status": "FAIL",
                   "error": traceback.format_exc().splitlines()[-1]}
            print(f"❌ {name}: {rec['error']}", flush=True)
            traceback.print_exc()
            results.append(rec)
            continue
        if rec["status"] == "PASS":
            print(f"✅ {name} [{rec['design_name']}]  "
                  f"{rec['N_pole']}P{rec['N_slot']}S  "
                  f"γ*={rec['gamma_deg']}°  T={rec['T_avg_Nm']} N·m  "
                  f"리플 {rec['ripple_pct']}%  η {rec['efficiency']*100:.1f}%  "
                  f"|B|max {rec['Bmax']}T  | {rec['sec']}s", flush=True)
        else:
            print(f"⏭  {name}: {rec['status']}", flush=True)
        results.append(rec)

    n_pass = sum(1 for r in results if r.get("status") == "PASS")
    n_fail = sum(1 for r in results if r.get("status") == "FAIL")
    n_skip = len(results) - n_pass - n_fail
    print(f"\n=== 결과: PASS {n_pass} / FAIL {n_fail} / 스킵 {n_skip} ===")

    # 결과가 모터마다 다른지 한눈에: 고유 (T_avg, η) 조합 수
    sigs = {(r.get("T_avg_Nm"), r.get("efficiency"))
            for r in results if r.get("status") == "PASS"}
    if n_pass > 1:
        print(f"고유 결과(서로 다른 T·η 조합): {len(sigs)} / {n_pass} "
              f"{'✓ 모두 다름' if len(sigs) == n_pass else '⚠ 중복 있음'}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"JSON 저장: {args.json_out}")

    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
