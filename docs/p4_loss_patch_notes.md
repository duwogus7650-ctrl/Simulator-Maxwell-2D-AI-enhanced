# P4 확장 v2: 철손·효율·토크리플 응답 — 패치 노트 (2026-06-12)

## ⚠ v1 폐기 — 반드시 v2 사용
v1의 미세 세그먼트 필터가 정상 코일 경계까지 제거해 코일 영역 36개 중
11개가 이웃 영역에 흡수되는 버그(토크 −33% 원인)가 있었음. v2에서
필터 제거 + weld 꼭짓점 스냅 방식으로 교체, 추가로 materials.py의
자석 intrinsic(J-H) 곡선 감지 보정 포함(N48UH mu_rec 0.05→1.05).

## Maxwell 대조 검증 (QDD-20 PeakLoad, 2200rpm/13.6A)
| 항목 | motoropt v2 | Maxwell | 편차 |
|---|---|---|---|
| T_avg | 1.1747 N·m | 1.1185 N·m | +5.0% (γ*=MTPA vs beta=0 차이 포함) |
| 토크리플 | 2.50% pp | ~2.2% pp | ✓ |
| 철손 | 1.66 W | ~1.35 W(정상상태) | +23% (고조파 합산 모델의 보수성) |
| 자석 와류손 | 미포함 | 0.062 W | 무시 가능 확인 |


## 새 파일 (motoropt/에 복사)
- **coreloss.py** — Bertotti 3항 철손(고조파 분해) + DC 동손.
  계수는 aedt 재질에서 자동(`core_loss_kh/kc/ke`, aedt_parser가 이미 추출).
  적층 보정: B/ks, 체적×ks (ks=0.97).
- **sweep_loss.py** — 슬라이딩 밴드 부하 스윕(전기 1주기) + 요소별 B(t)
  수집 → `compute_responses()`로 T_avg / T_ripple_pp·pct / P_fe(분해) /
  P_cu / efficiency 산출. `calibrate_gamma()`로 MTPA 전류각 자동 탐색.
  강판/자석 재질은 모델에서 자동 감지.

## 수정 파일 (기존 교체 — diff 확인 권장)
- **materials.py** — PMLinear: intrinsic J-H 곡선 감지 시 μ_rec += 1 (노멀 변환)
- **sliding.py** — `_pslg_from_lines`에 미세 세그먼트(<2µm)·슬리버 면
  (<1e-5mm²) 필터 + 코일 경계를 스테이터 벽에 용접(weld).
- **meshing.py** — `weld_boundary()` 추가, build_mesh에도 동일 용접 적용.

## 🔴 중요: 버그 수정 포함
QDD-20(SH_Reducer_QDD_20.aedt) 형상은 코일 꼭짓점 1개/코일이 슬롯 벽에서
**1e-7mm 비공선** → 노딩 시 극소각 → q25 품질 세분화 폭주로
**기존 build_mesh도 1,800만 요소**가 됨(48초+메모리 폭증). 이 패치 없이는
GUI Solve 탭도 QDD-20에서 사실상 멈춤. weld 패치로 4만 요소/0.2초 정상화.
→ 400W 모델 회귀 확인 필요(코깅 4.93 재현되는지 한 번 돌려볼 것).

## QDD-20 검증 결과 (2200rpm, I_rms 13.6A, γ*=199°, 24스텝, 기본메시, 28s)
- T_avg +0.754 N·m, 리플 7.6%pp
- P_fe 2.28W (히 1.21 / 와류 0.52 / 과잉 0.55; 로터 0.002W)
- P_cu 18.0W(기하 추정, L_end=0 — 과소), η 89.5%
- 재질 검증: 계수로 W10/400 = 11.31 W/kg (등급 보증 ≤12.0 ✓)

## 사용 예
```python
from motoropt.aedt_parser import parse_aedt, detect_magnet_style
from motoropt.sweep_loss import sweep_load_with_fields, compute_responses, calibrate_gamma
import math

m = parse_aedt("SH_Reducer_QDD_20.aedt")
ini = math.degrees(m["variables"]["ini_pos"])          # ini_pos는 rad로 파싱됨!
cal = calibrate_gamma(m, "spline", rpm=2200, I_rms=13.6, init_pos_deg=ini)
sw = sweep_load_with_fields(m, "spline", rpm=2200, I_rms=13.6,
                            gamma_deg=cal["gamma_max_deg"],
                            n_steps=36, init_pos_deg=ini)   # Maxwell도 36스텝
r = compute_responses(sw, m, R_ph_ohm=None)            # 실측 R_ph 있으면 입력!
print(r["efficiency"], r["T_ripple_pct"], r["P_fe"])
```

## GUI(Objective 탭) 연동 가이드
- 응답 키 추가: `efficiency`(larger), `T_ripple_pct`(smaller) —
  objective.py의 SPEC dict와 Y_KEYS에 두 키 추가, GUI 테이블 행 추가.
- Solve 탭에 부하 해석 입력 3종 노출: rpm / I_rms / R_ph(선택).
- R_ph 미입력 시 P_cu는 추정치(estimated=True) — GUI에 ⚠ 표기 권장.

## 알려진 한계 / 후속
1. P_cu 기하 추정은 엔드와인딩 미포함(박형 모터에서 오차 큼) →
   **R_ph 실측/Motor-CAD값 입력 권장**(상저항, 해당 온도).
2. 자석 와류손·기계손 미포함(효율 약간 과대) — QDD는 슬롯고조파가 커서
   자석 와류손 추후 검토 가치 있음.
3. γ≈199° 오프셋은 내부 권선 패턴 vs aedt 상 명칭 차이 —
   aedt boundaries['coils'](Ph_A1~C6, 극성 포함)로 정확 매핑하는
   확장 가능(object_id↔코일 인덱스 매칭 필요).
4. n_steps는 전기 1주기 기준 — 철손 고조파는 n/2차까지 반영(36 권장).
