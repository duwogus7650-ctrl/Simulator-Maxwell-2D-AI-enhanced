"""P6 목적함수: Derringer-Suich 만족도 함수.

시나리오 (합의): 자석 원가 절감
  T_avg       : larger-is-better, L=848.7(기준 유지), U=875 — L 미만 d=0
  emf_rms     : target=6.17, ±5% 양측 (L=5.8615, U=6.4785)
  magnet_area : smaller-is-better, U=339.22(기준), L=250 — U 초과 d=0
종합 D = (d_T · d_E · d_A)^(1/3)
"""
from __future__ import annotations

import numpy as np

from .surrogate import X_KEYS, Y_KEYS


def d_larger(y, L, U, w=1.0):
    return np.clip((y - L) / (U - L), 0, 1) ** w


def d_smaller(y, L, U, w=1.0):
    return np.clip((U - y) / (U - L), 0, 1) ** w


def d_target(y, L, T, U, w=1.0):
    y = np.asarray(y, float)
    d = np.where(y < T, (y - L) / (T - L), (U - y) / (U - T))
    return np.clip(d, 0, 1) ** w


SPEC = {
    "T_avg":       ("larger", 848.7, 875.0),
}

# 추가 응답 (GUI에서 체크 시 사용). ripple_pct는 노이즈가 커 참고용.
# B_tooth_st(치)·B_yoke(요크)는 FEM 검증 단계서 평가되는 포화 제약 — 필수(🔒)로
# 체크해 한계(치 1.8T·요크 1.6T)를 못 넘게 하는 용도. (emf·자석·효율은 사용자
# 요청으로 목표서 제외 — 후보표 참고값으로만 표시.)
SPEC_EXTRA = {
    "ripple_pct": ("smaller", 1.0, 5.0),
    "cogging_pp": ("smaller", 5.0, 30.0),   # mNm, 모델별로 L/U 조정 권장
    "B_tooth_st": ("smaller", 1.5, 1.8),    # 스테이터 치 포화 한계 1.8T
    "B_yoke":     ("smaller", 1.2, 1.6),    # 스테이터 요크 포화 한계 1.6T
}
# 동손[W]은 최적화 목표에서 제외(2026-06-15). 고정 운전전류에서 동손=3·I²·R_ph는
# 자석/형상 설계변수를 바꿔도 ±3%만 변해(전류·턴수·선경이 고정) 사실상 상수 →
# 목표로 쓰면 상한을 그 값 밑으로 두는 순간 d=0→D=0. 물리적으로 "고정전류 동손
# 최소화"는 "평균토크 최대화"와 거의 같음(같은 전류로 토크↑ = 토크당 동손↓).
# 동손 절대값은 Solve 탭 부하해석 P_cu로 확인. (동손/토크² 효율지표가 필요하면
# doe.py가 Pcu_per_Nm2를 계속 출력 — 변동 41%로 최적화 가능하나 토크와 중복.)

_D_FUNCS = {"larger": d_larger, "smaller": d_smaller, "target": d_target}


def _hard_pass(y, typ: str, *bnds) -> np.ndarray:
    """하드(필수) 제약 통과여부 → 0/1. 만족하면 1, 어기면 0(설계 탈락).

    larger: y≥L(하한 필수) / smaller: y≤U(상한 필수) /
    target: L≤y≤U(범위 필수). bnds=(L,U) 또는 (L,T,U)."""
    y = np.asarray(y, float)
    if typ == "larger":
        return (y >= bnds[0]).astype(float)
    if typ == "smaller":
        return (y <= bnds[-1]).astype(float)
    if typ == "target":
        return ((y >= bnds[0]) & (y <= bnds[-1])).astype(float)
    return np.ones_like(y)


def desirability(Y: np.ndarray, spec: dict | None = None,
                 y_keys: list | None = None,
                 hard_keys: set | None = None) -> np.ndarray:
    """Y: (n, len(y_keys)) 응답 행렬 → D (n,).

    spec의 키 중 y_keys(기본 Y_KEYS)에 있는 응답만 참여.
    hard_keys: 하드 제약 키 — 만족하면 1, 어기면 0(곱)으로 들어가 탈락시킴.
    나머지(소프트)는 만족도 램프의 기하평균. D = 소프트기하평균 × Π(하드 0/1).
    """
    spec = spec or SPEC
    y_keys = y_keys or Y_KEYS
    hard_keys = hard_keys or set()
    Y = np.atleast_2d(np.asarray(Y, float))
    soft = np.ones(Y.shape[0])
    hard = np.ones(Y.shape[0])
    n_soft = n_hard = 0
    for j, k in enumerate(y_keys):
        if k not in spec:
            continue
        s = spec[k]
        if k in hard_keys:
            hard = hard * _hard_pass(Y[:, j], s[0], *s[1:]); n_hard += 1
        else:
            soft = soft * _D_FUNCS[s[0]](Y[:, j], *s[1:]); n_soft += 1
    if n_soft + n_hard == 0:
        raise ValueError("스펙에 서로게이트 응답(y_keys)이 하나도 없음")
    D_soft = soft ** (1.0 / n_soft) if n_soft else np.ones(Y.shape[0])
    return D_soft * hard


def desirability_from_dict(resp: dict, spec: dict,
                           hard_keys: set | None = None) -> float:
    """응답 dict + spec → 종합 만족도 D.

    spec 키 중 resp에 실제로 존재하는 응답만 참여(efficiency 등 FEM 응답 포함).
    hard_keys: 하드 제약 — 어기면 D=0(탈락).
    """
    hard_keys = hard_keys or set()
    ds = []
    hard = 1.0
    for k, s in spec.items():
        if k in resp and resp[k] is not None:
            if k in hard_keys:
                hard *= float(_hard_pass([resp[k]], s[0], *s[1:])[0])
            else:
                ds.append(float(_D_FUNCS[s[0]](np.array([resp[k]], float),
                                               *s[1:])[0]))
    if not ds and not hard_keys:
        return 0.0
    D_soft = float(np.prod(ds) ** (1.0 / len(ds))) if ds else 1.0
    return D_soft * hard


class SurrogateObjective:
    """정규화 입력 u∈[0,1]^5 → D. RL/GA 공용 평가기."""

    def __init__(self, bundle_path: str, bounds: dict,
                 spec: dict | None = None, hard_keys: set | None = None):
        import joblib
        b = joblib.load(bundle_path)
        self.model, self.mu, self.sd = b["model"], b["mu"], b["sd"]
        self.keys = b["x_keys"]
        self.y_keys = b.get("y_keys", Y_KEYS)
        self.spec = spec
        self.hard_keys = hard_keys or set()
        self.lo = np.array([bounds[k][0] for k in self.keys])
        self.hi = np.array([bounds[k][1] for k in self.keys])

    def x_of(self, u: np.ndarray) -> np.ndarray:
        return self.lo + np.clip(u, 0, 1) * (self.hi - self.lo)

    def predict(self, u: np.ndarray) -> np.ndarray:
        u = np.atleast_2d(u)
        return self.model.predict(self.x_of(u)) * self.sd + self.mu

    def D(self, u: np.ndarray) -> np.ndarray:
        return desirability(self.predict(u), spec=self.spec,
                            y_keys=self.y_keys, hard_keys=self.hard_keys)
