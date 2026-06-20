"""aedt 내보내기: 변수 오버라이드를 원본 .aedt 텍스트에 주입.

Maxwell .aedt의 변수 정의 라인
  VariableProp('T_m', 'UD', '', '2.2mm')
의 마지막 값 필드만 교체한다. 파일 구조·작도 히스토리·재질·설정은
원본 그대로 보존되므로 Maxwell에서 즉시 열어 해석 가능하다.

다중 설계 파일: 파서(parse_aedt)는 첫 Maxwell2D 설계만 읽는다. 일관성을
위해 본 모듈도 교체를 *첫 설계 텍스트 영역에 한정*한다(나머지 설계는 원본
그대로 보존 — 무차별 치환으로 인한 손상 방지). 내보낸 파일은 재파싱·재해석
하여 의도한 SI 값과 일치하는지 검증한다(round-trip validation).
"""
from __future__ import annotations

import re
import warnings
from typing import Dict, Tuple

from .aedt_parser import parse_aedt
from .expressions import eval_expr

_BEGIN_DESIGN_RE = re.compile(r"^\s*\$begin '(Maxwell2DModel)'\s*$")


def _first_design_span(text: str) -> Tuple[int, int]:
    """첫 Maxwell2DModel 블록의 [start, end) 문자 오프셋을 반환.

    파서가 읽는 설계와 동일한(첫) 설계 영역만 치환 대상으로 삼기 위함.
    설계가 하나뿐이면 사실상 전체 텍스트 범위를 반환한다.
    """
    lines = text.splitlines(keepends=True)
    start_off = end_off = None
    offset = 0
    depth = 0          # Maxwell2DModel 진입 후 $begin/$end 중첩 깊이
    in_design = False
    for line in lines:
        stripped = line.strip()
        if not in_design and _BEGIN_DESIGN_RE.match(line):
            in_design = True
            start_off = offset
            depth = 1
        elif in_design:
            if stripped.startswith("$begin '"):
                depth += 1
            elif stripped.startswith("$end '"):
                depth -= 1
                if depth == 0:
                    end_off = offset + len(line)
                    break
        offset += len(line)
    if start_off is None:
        raise ValueError("Maxwell2D 디자인을 찾을 수 없습니다")
    if end_off is None:        # 짝이 안 맞으면 첫 설계 시작부터 파일 끝까지
        end_off = len(text)
    return start_off, end_off


def export_aedt(src_path: str, dst_path: str,
                overrides: Dict[str, str],
                validate: bool = True,
                tol: float = 1e-9) -> Dict[str, tuple]:
    """overrides: {'T_m': '1.8mm', 'a_m': '0.8831', ...} (원시 수식 문자열).

    교체는 첫 Maxwell2D 설계 영역에만 적용한다(다중 설계 파일 손상 방지).
    validate=True(기본): 내보낸 파일을 재파싱하여 각 오버라이드 변수가
    의도한 SI 값과 tol 이내로 일치하는지 검증한다(불일치 시 ValueError).

    반환: {변수: (이전 수식, 새 수식)} — 실제 교체된 항목.
    """
    with open(src_path, encoding="utf-8", errors="replace") as f:
        text = f.read()

    start, end = _first_design_span(text)
    head, region, tail = text[:start], text[start:end], text[end:]

    replaced: Dict[str, tuple] = {}
    for name, new_expr in overrides.items():
        if "'" in new_expr:
            raise ValueError(
                f"변수 {name} 의 새 수식에 작은따옴표(')가 포함됨: {new_expr!r} "
                f"— VariableProp 따옴표 구조를 깨뜨리므로 거부함")
        pat = re.compile(
            r"(VariableProp\('"
            + re.escape(name)
            + r"', '[^']*', '[^']*', ')([^']*)('\))")

        def sub(mm, _new=new_expr, _name=name):
            replaced[_name] = (mm.group(2), _new)
            return mm.group(1) + _new + mm.group(3)

        region, n = pat.subn(sub, region)
        if n == 0:
            raise KeyError(f"변수 {name} 를 aedt(첫 설계 영역)에서 찾지 못함")

    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(head + region + tail)

    if validate:
        _validate_roundtrip(dst_path, overrides, tol)
    return replaced


def _validate_roundtrip(dst_path: str, overrides: Dict[str, str],
                        tol: float = 1e-9) -> None:
    """내보낸 파일을 재파싱·재해석하여 의도한 SI 값과 일치하는지 검증.

    의도값: 다른 변수에 의존하지 않는 단순 수식이면 직접 평가,
    의존 수식이면 재파싱된 변수 컨텍스트로 평가한다.
    """
    model = parse_aedt(dst_path)
    resolved = model["variables"]
    for name, new_expr in overrides.items():
        if name not in resolved:
            raise ValueError(
                f"round-trip 검증 실패: 변수 {name} 가 재파싱 결과에 없음")
        try:
            intended = eval_expr(new_expr, resolved)
        except Exception:
            # new_expr 가 의존/비표준 수식이면 독립 기대값을 못 구함 →
            # 자기 자신과 비교하는 동어반복(항상 통과)을 피해 검증을 생략하고
            # 큰소리로 알린다. (이 내보내기가 만드는 오버라이드는 모두 단순
            # 수치+단위라 정상 경로에선 여기 도달하지 않음.)
            warnings.warn(
                f"round-trip: 변수 {name} 수식 {new_expr!r} 독립 평가 불가 "
                f"(의존/비표준 수식) — 수치 검증 생략")
            continue
        actual = resolved[name]
        scale = max(abs(intended), abs(actual), 1.0)
        if abs(actual - intended) > tol * scale:
            raise ValueError(
                f"round-trip 검증 실패: 변수 {name} 기대 SI 값 {intended!r} "
                f"≠ 재파싱 값 {actual!r} (수식 {new_expr!r})")


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
