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
| P4 | 부하 해석 (토크 · 리플) | ⏳ |
| P5 | DOE + 서로게이트 NN | ⏳ |
| P6 | RL(SAC) 최적화 + GA/베이지안 벤치마크 | ⏳ |
| P7 | PyQt6 GUI 통합 · aedt 내보내기 | ⏳ |

### P3 검증 (무부하, 슬라이딩 밴드 메시)
- **EMF @1000rpm: RMS 6.20~6.36 V** (Maxwell 6.17, +0.5~3%) · 피크 8.59 V (목표 8.37~8.51) — **통과**
- 코깅(144차 폴딩): 가상일 2.27 / Arkkio 2.57 mNm pk2pk (Maxwell 보고값 5.35)
  — 두 독립 추정자 일치. 잔여 편차는 자석 크라운 스플라인/필렛 근사 기인으로 추정,
  크라운 곡선 민감도 +7%/미세변형 확인. Maxwell 코깅 곡선 원본과 파형 대조 예정
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

## 핵심 설계 노트
- 같은 반경을 공유하는 인터페이스(자석↔로터, 슈 외호↔코일링)는
  **통일 각도 그리드**로 생성해야 슬리버 없는 적합 메시가 나온다.
- 선분 노딩은 `shapely.union_all(lines, grid_size=1e-6)` 고정 정밀도로.
- Triangle 품질각은 q25 (q30은 예각 입력에서 비종결 위험).
- 자석 모델: 데마그 곡선에서 Br=1.2528 T, μ_rec=1.03 선형 리코일.
- `.aedt` 설계 파일은 저장소에 포함하지 않음 (.gitignore).
