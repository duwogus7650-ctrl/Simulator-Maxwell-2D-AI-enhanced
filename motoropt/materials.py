"""재질 자기 모델.

- NuCurve : 비선형 BH 테이블 → ν(B²)=H/B 와 dν/dB² (Newton-Raphson용)
- PMLinear: 영구자석 데마그 곡선 → 선형 리코일 모델 (Br, μ_rec)
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.interpolate import PchipInterpolator

MU0 = 4e-7 * np.pi
NU0 = 1.0 / MU0


class NuCurve:
    """비선형 연자성체의 ν(B²) 모델.

    BH 테이블(H[A/m], B[T])을 PCHIP 보간한 H(B)에서
    ν = H/B, dν/dB² = (H'B - H) / (2B³) 를 계산한다.
    테이블 범위를 넘으면 dB/dH = μ0 기울기로 외삽한다.
    """

    def __init__(self, bh: list[list[float]], stacking_factor: float = 1.0):
        arr = np.asarray(bh, float)
        arr = arr[arr[:, 1] >= 0]          # B>=0 분기만 사용
        if stacking_factor < 1.0:           # 적층(z) 유효 BH 변환
            ks = stacking_factor
            arr = arr.copy()
            arr[:, 1] = ks * arr[:, 1] + (1 - ks) * MU0 * arr[:, 0]
        # B 단조 증가 보장
        order = np.argsort(arr[:, 1])
        B, H = arr[order, 1], arr[order, 0]
        keep = np.concatenate([[True], np.diff(B) > 1e-12])
        self.B, self.H = B[keep], H[keep]
        if len(self.B) < 2:
            # BH 테이블이 단일 점으로 축약 — PCHIP·B[1] 모두 불가.
            # 공기 등가(ν0) 선형 모델로 안전 폴백한다.
            warnings.warn(
                "BH 테이블 유효점 < 2 — 비선형 곡선 불가, 공기 등가 선형 폴백")
            self._h_of_b = None
            self._dh_db = None
            self.B_max = float(self.B[0]) if len(self.B) else 0.0
            self.H_max = float(self.H[0]) if len(self.H) else 0.0
            self.b_small = self.B_max if self.B_max > 0 else 1.0
            self.nu_init = NU0
            return
        self._h_of_b = PchipInterpolator(self.B, self.H, extrapolate=False)
        self._dh_db = self._h_of_b.derivative()
        self.B_max = self.B[-1]
        self.H_max = self.H[-1]
        self.b_small = self.B[1]
        # 초기 투자율 (B→0 극한): ν0 = dH/dB(0)
        self.nu_init = float(self._dh_db(self.B[1] * 0.5))

    def nu_and_dnu(self, b2: np.ndarray):
        """b2 = |B|² 배열 → (ν, dν/dB²)."""
        b2 = np.maximum(b2, 0.0)
        B = np.sqrt(b2)
        nu = np.full_like(B, self.nu_init)
        dnu = np.zeros_like(B)
        if self._h_of_b is None:        # 단일 점 폴백 — 전 구간 공기 등가
            return nu, dnu

        small = B < self.b_small * 0.5
        inside = (~small) & (B <= self.B_max)
        beyond = B > self.B_max

        if inside.any():
            Bi = B[inside]
            Hi = self._h_of_b(Bi)
            dHi = self._dh_db(Bi)
            nu[inside] = Hi / Bi
            dnu[inside] = (dHi * Bi - Hi) / (2.0 * Bi ** 3)
        if beyond.any():
            Bb = B[beyond]
            Hb = self.H_max + (Bb - self.B_max) * NU0
            nu[beyond] = Hb / Bb
            dnu[beyond] = (NU0 * Bb - Hb) / (2.0 * Bb ** 3)
        return nu, dnu


class PMLinear:
    """영구자석 선형 리코일: B = μ0·μ_rec·H + Br."""

    def __init__(self, bh: list[list[float]] | None = None,
                 br: float | None = None, mu_rec: float | None = None):
        if bh is not None:
            arr = np.asarray(bh, float)
            i0 = int(np.argmin(np.abs(arr[:, 0])))      # H=0 근방
            self.Br = float(arr[i0, 1])
            # 리코일 기울기 dB/dH = μ0·μ_rec 를 H=0 근방 선형 구간에서
            # 최소제곱 피팅한다. 고정 4점 후방차분은 희소·비균일 데마그
            # 테이블에서 불안정 → H=0에 가장 가까운 몇 점(기본 5점)으로 회귀.
            Harr, Barr = arr[:, 0], arr[:, 1]
            order = np.argsort(np.abs(Harr - Harr[i0]))  # H=0 근접순
            win = order[:min(5, len(order))]
            if len(win) >= 2 and np.ptp(Harr[win]) > 0:
                slope = float(np.polyfit(Harr[win], Barr[win], 1)[0])
            else:                                        # 폴백: 2점 차분
                i1 = max(i0 - 4, 0)
                slope = ((arr[i0, 1] - arr[i1, 1])
                         / (arr[i0, 0] - arr[i1, 0]))
            self.mu_rec = float(slope / MU0)
            # 고유(intrinsic) J-H 곡선 감지: 무릎 전 평탄 → 기울기≈0.
            # 노멀 곡선 B=J+μ0H 이므로 μ_rec(normal)=μ_rec(intr)+1.
            if self.mu_rec < 0.5:
                warnings.warn(
                    f"데마그 기울기 μ_rec={self.mu_rec:.3f}<0.5 — "
                    f"고유(J-H) 곡선으로 가정하고 +1.0 보정(노멀 B-H 변환)")
                self.mu_rec += 1.0
        else:
            self.Br, self.mu_rec = br, mu_rec
        self.nu = 1.0 / (MU0 * self.mu_rec)
