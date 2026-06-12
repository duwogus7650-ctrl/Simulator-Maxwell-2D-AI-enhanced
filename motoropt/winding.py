"""상저항 계산 — Motor-CAD MLT(평균 턴 길이) 방식.

mini-motorcad(바탕화면 mini-motorcad-main)에서 포팅. 1250W-jk 기준
Motor-CAD FEA 대비 MLT 92.96/92.99 mm, R_ph 52.59/52.58 mΩ 재현 확인,
400W CoBot 모터는 80°C에서 159 mΩ (Maxwell PDF 동손 역산 158 mΩ, +0.9%).

집중권(FSCW, throw=1 전치권선) 기준. MLT = 2·L_stk + π·코일피치.
주의: v["L_stk"]가 적층계수 포함 유효길이(예 20×0.93)면 동선 길이가
~수 % 과소 — 기하 적층(철심 물리 길이)을 L_stk_m으로 넘기면 보정됨.
"""
from __future__ import annotations

import math


def phase_resistance(v: dict, *, d_cu_mm: float, strands: int,
                     T_cu_C: float = 80.0, throw: int = 1,
                     L_stk_m: float | None = None) -> dict:
    """모델 변수 + 권선 사양 → 상저항 [Ω].

    v        : 해석된 변수 dict (N_slot, Zc, D_si, d_1, d_2, H_t, W_t,
               L_stk[, a=병렬회로수])
    d_cu_mm  : 나동선(도체) 지름 [mm]
    strands  : 가닥수 (strands in hand)
    """
    Ns = int(round(v["N_slot"]))
    Nc = int(round(v["Zc"]))
    paths = int(round(v.get("a", 1.0)))
    bore = v["D_si"] * 1e3                                    # [mm]
    slot_depth = (v["d_1"] + v["d_2"] + v["H_t"]) * 1e3
    tooth_w = v["W_t"] * 1e3
    L = (L_stk_m if L_stk_m is not None else v["L_stk"]) * 1e3

    taus_mid = math.pi * (bore + slot_depth) / Ns             # 슬롯피치(중간 반경)
    coil_pitch = throw * taus_mid - (taus_mid - tooth_w) / 2.0
    MLT = 2.0 * L + math.pi * coil_pitch                      # [mm]

    turn_csa = strands * math.pi / 4.0 * d_cu_mm ** 2         # [mm²]
    n_series = (Ns / 3.0) * Nc / paths                        # 직렬 턴수/상
    rho = 1.724e-8 * (1.0 + 0.003862 * (T_cu_C - 20.0))       # [Ω·m]
    R_ph = rho * (MLT * 1e-3 * n_series) / (turn_csa * 1e-6) / paths

    return {"R_ph": R_ph, "MLT_mm": MLT, "coil_pitch_mm": coil_pitch,
            "turn_csa_mm2": turn_csa, "n_series": n_series}
