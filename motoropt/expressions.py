"""Maxwell 변수 수식 평가기.

AEDT 변수 값은 '62mm', '180deg/N_pole*a_m', 'asin(...)' 같은 단위 포함
수식 문자열이다. 내부 표준 단위는 SI(m, rad, s, A, rad/s)로 통일한다.
"""
from __future__ import annotations

import ast
import math
import re
from typing import Dict

# 단위 → SI 변환 계수
_UNIT_FACTORS = {
    "mm": 1e-3, "cm": 1e-2, "m": 1.0, "um": 1e-6,
    "deg": math.pi / 180.0, "rad": 1.0,
    "rpm": 1.0, "RPM": 1.0,  # rpm은 회전수 단위 그대로 보존 (omega 수식에서 60RPM으로 나눔)
    "A": 1.0, "mA": 1e-3, "kA": 1e3,
    "s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9,
    "V": 1.0, "ohm": 1.0, "nH": 1e-9, "mV": 1e-3,
    "Hz": 1.0, "cel": 1.0,
}

_UNIT_ALT = "|".join(sorted(_UNIT_FACTORS, key=len, reverse=True))

_NUM_UNIT_RE = re.compile(
    r"(?<![\w.])(\d+\.?\d*(?:[eE][+-]?\d+)?)\s*(" + _UNIT_ALT + r")\b"
)
# '(12.16*0.97) mm' — 괄호 닫힌 수식 뒤에 단위가 붙는 형태
_PAREN_UNIT_RE = re.compile(r"\)\s*(" + _UNIT_ALT + r")\b")

_ALLOWED_FUNCS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "atan2": math.atan2, "sqrt": math.sqrt, "abs": abs,
    "exp": math.exp, "log": math.log, "ln": math.log,
    "pow": pow, "min": min, "max": max,
}
_ALLOWED_CONSTS = {"PI": math.pi, "pi": math.pi, "E": math.e}


def _replace_units(expr: str) -> str:
    """'62mm' → '(62*0.001)', '180deg' → '(180*0.017453...)'"""
    def sub(m: re.Match) -> str:
        val, unit = m.group(1), m.group(2)
        return f"({val}*{_UNIT_FACTORS[unit]!r})"
    expr = _NUM_UNIT_RE.sub(sub, expr)
    return _PAREN_UNIT_RE.sub(
        lambda m: f")*{_UNIT_FACTORS[m.group(1)]!r}", expr)


class _SafeEval(ast.NodeVisitor):
    """화이트리스트 AST 평가기 — eval() 직접 사용 금지."""

    def __init__(self, names: Dict[str, float]):
        self.names = names

    def visit(self, node):  # noqa: D102
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"허용되지 않는 상수: {node.value!r}")
        if isinstance(node, ast.Name):
            if node.id in self.names:
                return float(self.names[node.id])
            if node.id in _ALLOWED_CONSTS:
                return _ALLOWED_CONSTS[node.id]
            raise KeyError(node.id)
        if isinstance(node, ast.BinOp):
            left, right = self.visit(node.left), self.visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
            if isinstance(node.op, ast.Mod):
                return left % right
            raise ValueError(f"허용되지 않는 연산: {node.op}")
        if isinstance(node, ast.UnaryOp):
            v = self.visit(node.operand)
            if isinstance(node.op, ast.USub):
                return -v
            if isinstance(node.op, ast.UAdd):
                return +v
            raise ValueError(f"허용되지 않는 단항 연산: {node.op}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
                raise ValueError(f"허용되지 않는 함수: {ast.dump(node.func)}")
            args = [self.visit(a) for a in node.args]
            return _ALLOWED_FUNCS[node.func.id](*args)
        raise ValueError(f"허용되지 않는 구문: {ast.dump(node)}")


def eval_expr(expr: str, names: Dict[str, float]) -> float:
    """Maxwell 수식 문자열 하나를 평가해 SI 값으로 반환."""
    expr = expr.strip()
    expr = _replace_units(expr)
    expr = expr.replace("^", "**")
    tree = ast.parse(expr, mode="eval")
    return _SafeEval(names).visit(tree)


def resolve_variables(raw: Dict[str, str], max_passes: int = 12) -> Dict[str, float]:
    """변수 정의 dict(이름→수식 문자열)를 위상정렬 없이 반복 평가로 해석.

    rpm 단위는 보존된다: BaseRPM=4500rpm → 4500.0 (omega 수식이
    '2*PI*BaseRPM/60RPM*...' 형태라 rpm/RPM 계수가 1로 상쇄됨 — Maxwell과 동일 동작).
    """
    resolved: Dict[str, float] = {}
    pending = dict(raw)
    for _ in range(max_passes):
        progressed = False
        for name in list(pending):
            try:
                resolved[name] = eval_expr(pending[name], resolved)
                del pending[name]
                progressed = True
            except (KeyError, NameError):
                continue
        if not pending:
            break
        if not progressed:
            raise ValueError(f"순환 또는 미정의 변수: {sorted(pending)}")
    return resolved
