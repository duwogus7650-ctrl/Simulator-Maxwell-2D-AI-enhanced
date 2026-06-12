# motoropt — Maxwell형 2D FEM 솔버 + AI 최적설계 (개발 중)

Ansys Maxwell 2D 트랜션트 해석을 Python으로 재현하고, 그 위에
강화학습 기반 모터 최적설계를 얹는 프로젝트입니다.

**워크플로 (목표):** `.aedt` 업로드 → 목표특성(min/max/target) 지정
→ RL이 목표에 도달하는 최적 설계변수/형상 제공 → `.aedt` 재생성 출력

## 현재 상태

| 단계 | 내용 | 상태 |
|---|---|---|
| P1 | AEDT 파서 · 파라메트릭 형상 · 적합 메시 | ✅ 완료 |
| P2 | 비선형 정자기 솔버 (무부하 단일 포지션) | ✅ 1차 검증 통과 |
| P3 | 코깅토크 · 무부하 EMF (슬라이딩 밴드) | 🔶 EMF 통과 / 코깅 정밀화 중 |
| P4 | 부하 해석 (토크 · 리플) | ✅ 평균토크 +2.3% 통과 |
| P5 | DOE + 서로게이트 NN | ✅ 주목적 R²>0.99 |
| P6 | RL(SAC)+액티브러닝 최적화, GA 벤치마크 | ✅ 자석 −21~26% (FEM 검증) |
| P7 | PyQt6 GUI 통합 · aedt 내보내기 | ✅ 5탭 데스크톱 앱 |

### P7 — PyQt6 데스크톱 앱 (gui/app.py)
- 5탭: ①Model(aedt 로드·변수표 편집·형상 미리보기) ②Objective(D-S 스펙 편집)
  ③Solve(무부하/부하 해석, QThread 비차단, |B| 필드맵) ④Optimize(액티브러닝 1라운드 버튼,
  SAC 정책 개선, FEM 검증 후보 테이블) ⑤Result(기준 vs 최적 비교표·오버레이·aedt 내보내기)
- aedt 내보내기: 최적 변수셋을 원본에 주입(VariableProp 치환) — 재파싱 라운드트립 검증,
  종속 수식(theta_one 등) 자동 전파, Maxwell에서 즉시 해석 가능
- 실행: `run_gui.bat` (또는 `venv\Scripts\python gui/app.py 모델.aedt`)
  — 시스템 Python에는 shapely/triangle이 없어 venv 필수. 예외는 다이얼로그로
  표시되며 앱은 유지됨(PyQt6 abort 방지 excepthook 적용)

### P6 — 최적화 (시나리오: 토크 ≥848.7 + EMF 6.17±5% + 자석 최소화)
- Derringer-Suich 만족도 D, 액티브러닝 4라운드(DE 최적화→FEM 검증→데이터 보강→재학습)
- 라운드1에서 서로게이트 악용 적발(예측 D 0.98 → FEM 0): 코너 외삽 → 보강 후 진실 영역 수렴
- **최고 검증 설계: 자석 −20.8%(339→269mm²), T 866.2 mNm, EMF 6.24 V, D=0.741**
- SAC(20k 스텝): 탐색 best D 0.841(DE 0.847의 99%), 정책 롤아웃은 임의 시작→24스텝 D≈0.79
  — 단발 최적화는 DE 우세, SAC 정책은 대화형 설계개선(GUI) 용도로 채택 예정
- SAC 후보 FEM 검증: 자석 −26.1%, EMF 6.173, T 853.0 — 제약 전부 충족

### P5 — DOE 121설계 + MLP 서로게이트
- 설계변수 5: a_m, T_m, T_m2/T_m, W_t, MagnetR (LHS) · 설계당 ~23초(EMF 6포지션+부하 10포지션)
- 유효 95 / 메시폭주 등 실패 26 — 설계별 자식 프로세스 격리로 OOM 면역 러너
- **T_avg R²=0.988(0.4%) · EMF R²=0.994(0.3%) · magnet_area R²=1.0** → P6 최적화 목적함수 확보
- ripple/B_tooth는 경량평가 노이즈로 학습 불가 → P6에서 FEM 검증 루프 담당(액티브러닝)으로 처리

### P4 확장 v2 검증 — 손실·효율 (PDF Rev1 p.7 대조, 2026-06-12)
- 400W(4500rpm/4.9A): **T_avg 874.4 mNm(+2.9%) · P_fe 11.8 W(Maxwell 15.8, −25%)
  · η 94.2%(자석손 2.2W 가산; Maxwell 93.1)** — scripts/run_400w_loss.py
- 200W(L_stk 15×0.93, 3.7A): T_avg 497.5(+3.6%) · P_fe 8.8(−21%) · η 93.8(92.4)
- 코깅(최적화 형상): 가상일 6.31 mNm vs Maxwell 5.95 (+6.0%) — 구 4.93은 최적화 전 기준
- P_fe는 ±25% 신뢰구간(고조파 Bertotti vs Maxwell 시간영역 차이),
  동손은 R_ph 실측 입력 필수(기하 추정 ~50% 과소) — docs/p4_loss_patch_notes.md 참조

### P4 검증 (부하: 4.9 Arms · β=0 · 적층계수 0.97 적용)
- **평균토크(가상일): 868.2 mNm** (Maxwell 848.7, **+2.3%**) — 통과
- 토크리플 pk2pk 27.9 mNm/3.2% (Maxwell ~24/2.8%), 전기 6차 펄스 지배 — 물리 일치
- 적층계수(Lamination 0.97) 유효 BH 변환 적용 → EMF도 6.10~6.25 V로 재정렬(목표 6.17 정중앙)
- 전류각 캘리브레이션: 전기위상 스캔 → MTPA(β=0) 자동 탐색 (δ*≈290°e)
- Arkkio는 +4~7% 양의 바이어스 확인 → 토크 공식 추정자는 가상일법 채택

### P3 검증 (무부하, 슬라이딩 밴드 메시)
- **EMF @1000rpm: RMS 6.20~6.36 V** (Maxwell 6.17, +0.5~3%) · 피크 8.59 V (목표 8.37~8.51) — **통과**
- 코깅(144차 폴딩): **가상일 4.93 / Arkkio 5.12 mNm pk2pk (Maxwell 5.35, −7.9%/−4.3%)**
  — 형상 감사에서 치 시작점 50µm 돌출 버그 발견·수정 후 −57%→−7.9%로 회복.
  잔여 편차는 크라운 스플라인 근사(원호) 기인 추정, Maxwell 코깅 CSV와 파형 대조 예정
- 슬라이딩 밴드: 로터/스테이터 분리 메시 + 결정론적 밴드 재연결 → 위치 간 메시 노이즈 제거
  (풀 리메시 방식 pk2pk 38.9 → 슬라이딩 6.0 → 폴딩 2.3 mNm)

### P2 검증 (400W 18s/16p SPM, 협동로봇용)
- 무부하 Az 범위: **±0.0050 Wb/m** (Maxwell ±0.0048, 오차 ≈4%)
- Newton-Raphson 7회 수렴 (잔차 < 1e-4), 해석 시간 ~1초 @ 34k 요소
- 치 자속밀도 1.5~1.9 T — Maxwell 필드 오버레이와 일치 경향

## 구조

```
motoropt/
  expressions.py   # Maxwell 변수 수식 평가기 (단위 mm/deg/rpm, AST 화이트리스트)
  aedt_parser.py   # .aedt 블록트리 파서 → 변수/재질 BH/작도/권선/모션/솔브/메시
  geometry.py      # 파라메트릭 형상 (자석 spline/eccentric_arc 스타일 자동 지원)
  meshing.py       # 평면배치 + Triangle 기반 영역 적합 메시
  materials.py     # ν(B²) 비선형 곡선, PM 선형 리코일
  solver_ms.py     # 2D 비선형 정자기 FEM (Az, 1차 삼각요소, NR + 라인서치)
  plotting.py      # 형상/메시/자장 플롯 (Maxwell 색상 체계)
scripts/
  run_p1_geometry.py   # 파싱→형상→메시 데모
  run_p2_noload.py     # 무부하 해석 데모
```

## 사용

```bash
pip install -r requirements.txt
python scripts/run_p1_geometry.py 모델.aedt -o out/
python scripts/run_p2_noload.py  모델.aedt -o out/
```

## 형상 충실도 감사 (P5 후 수행)
- 변수 33개: aedt 원시 수식 그대로 재해석 — theta_two=asin(...), H_t=(D_so/2-T_Yoke)-(D_si/2+d_1+d_2) 등 전수 일치
- Maxwell 폴리라인 기록 좌표와 직접 대조: 슈 9점, 자석 내호/측벽/스플라인 3점 — 좌표 일치
- 알려진 근사: ①스플라인 내부 곡선→3점 통과 원호 ②필렛→상부 모서리 베지어 ③원호 현 샘플링(µm 스케일)
- 발견·수정: 치 사각형 XStart가 D_si/2−0.05mm로 공극 돌출 → D_si/2로 정정. 효과: EMF 6.200→6.164(목표 6.17), T_avg +2.0%→+1.6%, 코깅 2.27→4.93mNm
- 주의: DOE 95설계는 수정 전 형상 기준(균일 편향 ~0.4%) — P6 액티브러닝에서 수정 형상으로 보강 예정

## 핵심 설계 노트
- 같은 반경을 공유하는 인터페이스(자석↔로터, 슈 외호↔코일링)는
  **통일 각도 그리드**로 생성해야 슬리버 없는 적합 메시가 나온다.
- 선분 노딩은 `shapely.union_all(lines, grid_size=1e-6)` 고정 정밀도로.
- Triangle 품질각은 q25 (q30은 예각 입력에서 비종결 위험).
- 자석 모델: 데마그 곡선에서 Br=1.2528 T, μ_rec=1.03 선형 리코일.
- `.aedt` 설계 파일은 저장소에 포함하지 않음 (.gitignore).
