"""aedt 내보내기: 변수 오버라이드를 원본 .aedt 텍스트에 주입.

Maxwell .aedt의 변수 정의 라인
  VariableProp('T_m', 'UD', '', '2.2mm')
의 마지막 값 필드만 교체한다. 파일 구조·작도 히스토리·재질·설정은
원본 그대로 보존되므로 Maxwell에서 즉시 열어 해석 가능하다.
"""
from __future__ import annotations

import re
from typing import Dict


def export_aedt(src_path: str, dst_path: str,
                overrides: Dict[str, str]) -> Dict[str, tuple]:
    """overrides: {'T_m': '1.8mm', 'a_m': '0.8831', ...} (원시 수식 문자열).

    반환: {변수: (이전 수식, 새 수식)} — 실제 교체된 항목.
    """
    with open(src_path, encoding="utf-8", errors="replace") as f:
        text = f.read()

    replaced: Dict[str, tuple] = {}
    for name, new_expr in overrides.items():
        pat = re.compile(
            r"(VariableProp\('"
            + re.escape(name)
            + r"', '[^']*', '[^']*', ')([^']*)('\))")

        def sub(mm, _new=new_expr, _name=name):
            replaced[_name] = (mm.group(2), _new)
            return mm.group(1) + _new + mm.group(3)

        text, n = pat.subn(sub, text)
        if n == 0:
            raise KeyError(f"변수 {name} 를 aedt에서 찾지 못함")

    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(text)
    return replaced


def overrides_from_design(x: Dict[str, float]) -> Dict[str, str]:
    """DOE/최적화 설계점(x) → aedt 변수 오버라이드 문자열."""
    out = {}
    if "a_m" in x:
        out["a_m"] = f"{x['a_m']:.6g}"
    if "T_m" in x:
        out["T_m"] = f"{x['T_m']:.6g}mm"
    if "T_m2_ratio" in x and "T_m" in x:
        out["T_m2"] = f"{x['T_m2_ratio'] * x['T_m']:.6g}mm"
    if "W_t" in x:
        out["W_t"] = f"{x['W_t']:.6g}mm"
    if "MagnetR" in x:
        out["MagnetR"] = f"{x['MagnetR']:.6g}mm"
    return out
