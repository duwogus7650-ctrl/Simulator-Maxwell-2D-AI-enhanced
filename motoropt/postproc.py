"""후처리: 토크(Arkkio법) · 권선 매핑 · 상 쇄교자속.

Arkkio법: 공극 환형역 전체에서 Maxwell 응력의 반경 평균
    T = L/(μ0 (r2−r1)) ∫_annulus r · B_r · B_t dS
선형요소의 윤곽선 MST보다 메시 노이즈에 훨씬 강건하다.

권선 매핑(18슬롯/16극 치집중권, 더블레이어):
    치 패턴 [+A,−A,+A, +B,−B,+B, +C,−C,+C] × 2
    슬롯 s의 right half(치 s 인접) ← 치 s 코일,
    슬롯 s의 left  half(치 s+1 인접) ← 치 s+1 코일(반대 방향).
"""
from __future__ import annotations

import math
import warnings
from typing import Dict, List

import numpy as np

from .materials import MU0

PATTERN_18S16P = ["A+", "A-", "A+", "B+", "B-", "B+", "C+", "C-", "C+"] * 2


def torque_arkkio(solver, res, r1_mm: float, r2_mm: float,
                  L_stk_m: float) -> float:
    """공극 환형역 [r1, r2](mm)의 공기 요소로 토크[N·m] 계산."""
    cx, cy = solver.centroid[:, 0], solver.centroid[:, 1]
    r = np.hypot(cx, cy)
    r1, r2 = r1_mm * 1e-3, r2_mm * 1e-3
    sel = (~solver.is_steel) & (~solver.is_magnet) & (~solver.is_coil) \
        & (r > r1) & (r < r2)
    if not sel.any():
        warnings.warn("Arkkio 적분 영역에 요소 0개 — 토크 0 반환(반경 r1/r2 확인)")
    Br = (res.Bx[sel] * cx[sel] + res.By[sel] * cy[sel]) / r[sel]
    Bt = (-res.Bx[sel] * cy[sel] + res.By[sel] * cx[sel]) / r[sel]
    dS = solver.area[sel]
    return float(L_stk_m / (MU0 * (r2 - r1))
                 * np.sum(r[sel] * Br * Bt * dS))


def build_winding_map(solver, n_slot: int = 18) -> Dict[str, List[tuple]]:
    """코일사이드 인덱스 → (상, 방향) 매핑.

    반환: {'A': [(coil_idx, ±1), ...], 'B': [...], 'C': [...]}
    """
    # 코일사이드별 (슬롯 번호, 사이드) 판정
    slot_pitch = 2 * math.pi / n_slot
    sides = {}
    for e_kind, e_attr in [(None, None)]:
        pass
    coil_idx_of_elem = np.array([solver.table[a].get("index", -1)
                                 for a in solver.attr])
    out = {"A": [], "B": [], "C": []}
    for ci in range(n_slot * 2):
        sel = solver.is_coil & (coil_idx_of_elem == ci)
        if not sel.any():
            continue
        w = solver.area[sel]
        cx = np.average(solver.centroid[sel, 0], weights=w)
        cy = np.average(solver.centroid[sel, 1], weights=w)
        ang = math.atan2(cy, cx) % (2 * math.pi)
        s = int(ang // slot_pitch)              # 슬롯 번호 (치 s ~ 치 s+1 사이)
        slot_center = (s + 0.5) * slot_pitch
        side = "right" if (ang - slot_center) < 0 else "left"
        tooth = s if side == "right" else (s + 1) % n_slot
        ph, sgn = PATTERN_18S16P[tooth][0], PATTERN_18S16P[tooth][1]
        d = 1 if sgn == "+" else -1
        if side == "left":                       # 코일 반대편 사이드
            d = -d
        out[ph].append((ci, d))
    return out


def flux_linkages(solver, res, winding_map: Dict[str, List[tuple]],
                  Zc: int, L_stk_m: float) -> Dict[str, float]:
    """상별 쇄교자속 λ[Wb] (직렬 Zc턴, 병렬 1)."""
    coil_idx_of_elem = np.array([solver.table[a].get("index", -1)
                                 for a in solver.attr])
    A_e = res.A[solver.T].mean(axis=1)           # 요소 평균 Az
    lam = {}
    for ph, sides in winding_map.items():
        total = 0.0
        for ci, d in sides:
            sel = solver.is_coil & (coil_idx_of_elem == ci)
            S = solver.area[sel].sum()
            A_avg = float((A_e[sel] * solver.area[sel]).sum() / S)
            total += d * Zc * L_stk_m * A_avg
        lam[ph] = total
    return lam


def coenergy(solver, res, L_stk_m: float) -> float:
    """전계 코에너지 W' [J] — 가상일법 토크 T = dW'/dθ 용.

    철심: w' = B·H(B) − ∫H dB (BH 테이블 적분)
    자석(선형 리코일): w' = ν_m |B|²/2  (상수항 소거)
    공기/코일: w' = ν0 |B|²/2
    """
    import numpy as np
    from .materials import NU0
    b2 = res.Bx ** 2 + res.By ** 2
    B = np.sqrt(b2)
    w = np.empty_like(B)
    # 공기류
    other = (~solver.is_steel) & (~solver.is_magnet)
    w[other] = 0.5 * NU0 * b2[other]
    # 자석
    w[solver.is_magnet] = 0.5 * solver.pm.nu * b2[solver.is_magnet]
    # 철심: 코에너지 밀도 테이블 보간
    st = solver.steel
    if not hasattr(st, "_wco"):
        from scipy.interpolate import PchipInterpolator
        from scipy.integrate import cumulative_trapezoid
        Wint = cumulative_trapezoid(st.H, st.B, initial=0.0)  # ∫H dB
        wco = st.B * st.H - Wint                              # 코에너지밀도
        st._wco = PchipInterpolator(st.B, wco, extrapolate=False)
        st._wco_end = (st.B_max, st.H_max, float(wco[-1]))
    sel = solver.is_steel
    Bs = B[sel]
    ws = np.empty_like(Bs)
    inside = Bs <= st.B_max
    ws[inside] = st._wco(Bs[inside])
    if (~inside).any():
        B0, H0, w0 = st._wco_end
        dB = Bs[~inside] - B0
        # 선형 외삽 구간: H = H0 + dB/μ0 → w' 증가분 = B·H − ∫H dB 적분
        ws[~inside] = (w0 + (B0 + dB) * (H0 + dB * NU0)
                       - (H0 * dB + 0.5 * NU0 * dB ** 2) - B0 * H0)
    w[sel] = ws
    return float(L_stk_m * np.sum(w * solver.area))
