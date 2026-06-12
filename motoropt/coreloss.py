"""철손(코어손실) 계산 — Maxwell 'Electrical Steel' 모델 등가.

요소별 B(t) 시계열(전기 1주기, 등간격 N스텝)을 고조파 분해해
Bertotti 3항 모델로 손실 밀도를 적분한다.

    p = Σ_k [ K_h f_k B_k² + K_c (f_k B_k)² + K_e (f_k B_k)^1.5 ]  [W/m³]
        f_k = k·f1,  B_k = 고조파 진폭

- 히스테리시스·와류항(B² 비례)은 x/y 성분별 합산이 정확하다
  (회전 자계 포함). 과잉손항(지수 1.5)은 합성 진폭
  B_k = √(B_xk²+B_yk²) 근사를 쓴다.
- 적층 보정: 솔버는 적층 유효 BH(ks 반영)로 풀므로 해석된 B는
  기하 단면 평균값이다. 실제 강판 내 B ≈ B/ks, 강판 체적 = V·ks 로
  보정한다(correct_stacking=True 기본).

계수 출처: aedt 재질 정의(core_loss_kh/kc/ke) — aedt_parser가 추출.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CoreLossResult:
    P_total: float                       # [W]
    P_hyst: float
    P_eddy: float
    P_excess: float
    p_elem: np.ndarray = field(repr=False)   # 요소별 손실밀도 [W/m³] (플롯용)

    def as_dict(self) -> dict:
        return {"P_total": self.P_total, "P_hyst": self.P_hyst,
                "P_eddy": self.P_eddy, "P_excess": self.P_excess}


def harmonic_amplitudes(B_t: np.ndarray) -> np.ndarray:
    """B_t (n_steps, n_elem) — 전기 1주기 등간격(끝점 제외) 시계열.

    반환: (n_harm, n_elem) 고조파 진폭 (k=1..N//2).
    """
    n = B_t.shape[0]
    X = np.fft.rfft(B_t, axis=0)
    amp = 2.0 * np.abs(X[1:]) / n            # k>=1
    if n % 2 == 0:                            # 나이퀴스트 빈은 ×1
        amp[-1] *= 0.5
    return amp


def core_loss(Bx_t: np.ndarray, By_t: np.ndarray, areas_m2: np.ndarray,
              f1: float, L_stk_m: float, kh: float, kc: float, ke: float,
              stacking_factor: float = 1.0,
              correct_stacking: bool = True) -> CoreLossResult:
    """요소 집합의 철손 [W].

    Bx_t, By_t : (n_steps, n_elem) — 해당 재질 좌표계의 B 시계열
                 (스테이터=전역, 로터=로터 회전 좌표계)
    areas_m2   : (n_elem,) 요소 면적 [m²]
    f1         : 전기 기본 주파수 [Hz]
    """
    ks = float(stacking_factor)
    b_scale = 1.0 / ks if (correct_stacking and ks < 1.0) else 1.0
    v_scale = ks if (correct_stacking and ks < 1.0) else 1.0

    Ax = harmonic_amplitudes(Bx_t) * b_scale       # (nh, ne)
    Ay = harmonic_amplitudes(By_t) * b_scale
    nh = Ax.shape[0]
    k = np.arange(1, nh + 1, dtype=float)[:, None]
    fk = k * f1

    B2 = Ax * Ax + Ay * Ay                         # 성분합 B_k²
    Bk = np.sqrt(B2)

    p_h = kh * np.sum(fk * B2, axis=0)             # [W/m³]
    p_c = kc * np.sum((fk ** 2) * B2, axis=0)
    p_e = ke * np.sum((fk * Bk) ** 1.5, axis=0)

    vol = areas_m2 * L_stk_m * v_scale             # 강판 실체적
    Ph = float(np.sum(p_h * vol))
    Pc = float(np.sum(p_c * vol))
    Pe = float(np.sum(p_e * vol))
    return CoreLossResult(P_total=Ph + Pc + Pe, P_hyst=Ph,
                          P_eddy=Pc, P_excess=Pe,
                          p_elem=p_h + p_c + p_e)


def copper_loss_dc(I_rms: float, R_ph_ohm: float | None = None, *,
                   n_phase: int = 3,
                   # --- R_ph 미지 시 기하 추정 파라미터 ---
                   coil_side_area_m2: float | None = None,
                   n_coil_sides: int | None = None,
                   Zc: int | None = None,
                   L_stk_m: float | None = None,
                   L_end_m: float = 0.0,
                   fill_factor: float = 0.45,
                   sigma_cu: float = 5.8e7,
                   temp_C: float = 80.0) -> dict:
    """DC 동손 [W].

    1) R_ph_ohm 제공 시: P = n_phase · I_rms² · R_ph  (권장 — 실측/Motor-CAD값)
    2) 미제공 시 기하 추정: 코일사이드 단면 = 영역면적 × fill_factor,
       도체 길이 = (L_stk + L_end) × 2 × Zc / 사이드.
       ※ 추정치 — 특히 박형(L_stk 작은) 모터는 L_end 영향이 커서
         L_end_m 미입력 시 오차 큼. 결과 dict에 'estimated' 플래그 포함.
    """
    sigma = sigma_cu / (1.0 + 0.00393 * (temp_C - 20.0))     # 구리 온도보정
    if R_ph_ohm is not None:
        return {"P_cu": n_phase * I_rms ** 2 * R_ph_ohm,
                "R_ph": R_ph_ohm, "estimated": False}
    if None in (coil_side_area_m2, n_coil_sides, Zc, L_stk_m):
        raise ValueError("R_ph 미지정 시 기하 파라미터 4종 필요")
    sides_per_phase = n_coil_sides / n_phase
    A_cond = coil_side_area_m2 * fill_factor / Zc            # 도체 1가닥 단면
    L_turn = 2.0 * (L_stk_m + L_end_m)                       # 1턴 (양 사이드)
    R_ph = (sides_per_phase / 2) * Zc * L_turn / (sigma * A_cond)
    return {"P_cu": n_phase * I_rms ** 2 * R_ph,
            "R_ph": R_ph, "estimated": True}
