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
    "emf_rms":     ("target", 6.17 * 0.95, 6.17, 6.17 * 1.05),
    "magnet_area": ("smaller", 250.0, 339.22),
}

# 추가 응답 (GUI에서 체크 시 사용) — ripple_pct는 서로게이트 응답에 있으나
# 경량평가 노이즈가 커 참고용, efficiency는 부하 스윕(FEM)에서만 평가됨.
SPEC_EXTRA = {
    "ripple_pct": ("smaller", 1.0, 5.0),
    "efficiency": ("larger", 0.90, 0.95),
}

_D_FUNCS = {"larger": d_larger, "smaller": d_smaller, "target": d_target}


def desirability(Y: np.ndarray, spec: dict | None = None,
                 y_keys: list | None = None) -> np.ndarray:
    """Y: (n, len(y_keys)) 응답 행렬 → D (n,).

    spec의 키 중 y_keys(기본 Y_KEYS)에 있는 응답만 기하평균에 참여한다.
    """
    spec = spec or SPEC
    y_keys = y_keys or Y_KEYS
    Y = np.atleast_2d(np.asarray(Y, float))
    D = np.ones(Y.shape[0])
    n = 0
    for j, k in enumerate(y_keys):
        if k not in spec:
            continue
        s = spec[k]
        D = D * _D_FUNCS[s[0]](Y[:, j], *s[1:])
        n += 1
    if n == 0:
        raise ValueError("스펙에 서로게이트 응답(y_keys)이 하나도 없음")
    return D ** (1.0 / n)


def desirability_from_dict(resp: dict, spec: dict) -> float:
    """응답 dict + spec → 종합 만족도 D.

    spec 키 중 resp에 실제로 존재하는 응답만 기하평균에 참여한다
    (서로게이트 Y_KEYS에 없는 efficiency 등 FEM 응답도 포함 가능).
    """
    ds = []
    for k, s in spec.items():
        if k in resp and resp[k] is not None:
            ds.append(float(_D_FUNCS[s[0]](np.array([resp[k]], float),
                                           *s[1:])[0]))
    if not ds:
        return 0.0
    return float(np.prod(ds) ** (1.0 / len(ds)))


class SurrogateObjective:
    """정규화 입력 u∈[0,1]^5 → D. RL/GA 공용 평가기."""

    def __init__(self, bundle_path: str, bounds: dict,
                 spec: dict | None = None):
        import joblib
        b = joblib.load(bundle_path)
        self.model, self.mu, self.sd = b["model"], b["mu"], b["sd"]
        self.keys = b["x_keys"]
        self.y_keys = b.get("y_keys", Y_KEYS)
        self.spec = spec
        self.lo = np.array([bounds[k][0] for k in self.keys])
        self.hi = np.array([bounds[k][1] for k in self.keys])

    def x_of(self, u: np.ndarray) -> np.ndarray:
        return self.lo + np.clip(u, 0, 1) * (self.hi - self.lo)

    def predict(self, u: np.ndarray) -> np.ndarray:
        u = np.atleast_2d(u)
        return self.model.predict(self.x_of(u)) * self.sd + self.mu

    def D(self, u: np.ndarray) -> np.ndarray:
        return desirability(self.predict(u), spec=self.spec,
                            y_keys=self.y_keys)
