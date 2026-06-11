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


def desirability(Y: np.ndarray) -> np.ndarray:
    """Y: (n, len(Y_KEYS)) 응답 행렬 → D (n,)"""
    iT = Y_KEYS.index("T_avg")
    iE = Y_KEYS.index("emf_rms")
    iA = Y_KEYS.index("magnet_area")
    dT = d_larger(Y[:, iT], *SPEC["T_avg"][1:])
    dE = d_target(Y[:, iE], *SPEC["emf_rms"][1:])
    dA = d_smaller(Y[:, iA], *SPEC["magnet_area"][1:])
    return (dT * dE * dA) ** (1 / 3)


class SurrogateObjective:
    """정규화 입력 u∈[0,1]^5 → D. RL/GA 공용 평가기."""

    def __init__(self, bundle_path: str, bounds: dict):
        import joblib
        b = joblib.load(bundle_path)
        self.model, self.mu, self.sd = b["model"], b["mu"], b["sd"]
        self.keys = b["x_keys"]
        self.lo = np.array([bounds[k][0] for k in self.keys])
        self.hi = np.array([bounds[k][1] for k in self.keys])

    def x_of(self, u: np.ndarray) -> np.ndarray:
        return self.lo + np.clip(u, 0, 1) * (self.hi - self.lo)

    def predict(self, u: np.ndarray) -> np.ndarray:
        u = np.atleast_2d(u)
        return self.model.predict(self.x_of(u)) * self.sd + self.mu

    def D(self, u: np.ndarray) -> np.ndarray:
        return desirability(self.predict(u))
