# 400W/200W 모델 손실·효율 검증 (PDF Rev1 기준, 2026-06-12)

- [x] 400W.aedt 파싱 확인 (design=4. 400W_BasicModel_Load_Optimized, 37변수)
- [x] 코깅 확인: 가상일 6.31 / Arkkio 6.63 mNm vs Maxwell 5.95 (+6.0%/+11.4%)
      ※ 이전 4.93은 최적화 전 형상/기준(Maxwell 5.35) — 직접 비교 불가
- [x] γ 캘리브레이션: γ*=190.9° (구 δ*≈290°와 등가, ini_pos 컨벤션 차이)
- [x] 400W 부하 스윕: T_avg 874.4(+2.9%) | 리플 4.05%(Arkkio) | P_fe 11.8(−25%) | η 94.2%
- [x] 200W 부하 스윕: T_avg 497.5(+3.6%) | P_fe 8.8(−21%) | η 93.8%
- [x] README + docs/p4_loss_patch_notes.md 반영 (weld 패치 회귀 항목 종결)

# GUI aedt 열기 크래시 수정 (2026-06-12)

- [x] 재현: 시스템 Python 3.14(PyQt6 있음, shapely/triangle 없음)으로 GUI 실행 시
      아무 aedt나 열면 ModuleNotFoundError → PyQt6 abort → 앱 종료
- [x] gui/app.py: open_aedt/refresh_geometry 예외 가드 + 전역 excepthook
      (에러 다이얼로그 표시, 앱 유지, venv 안내 포함)
- [x] expressions.py: '(수식) 단위' 패턴 지원 (OuterType SyntaxError 해소)
- [x] run_gui.bat 추가 (venv Python으로 실행)
- [x] 검증: 시스템 Python에서 400W 열기 → 다이얼로그+앱 유지 /
      venv에서 400W·QDD-20 정상 로드, X12·OuterType은 미지원 안내(theta_one 없음)

## 리뷰
- 패치노트의 미결 항목("400W 코깅 4.93 재현 확인")을 PDF 기준 대조로 종결.
  weld 패치로 인한 회귀 없음 — 메시 91k 정상, NR 8~9회 수렴.
- 토크 +3% 내외, 효율 +1.1~1.4%p 과대(자석 와류손·기계손 미모델 + P_fe 과소).
- P_fe 계통 편차 −21~−25% (QDD는 +23%): 고조파 Bertotti vs Maxwell
  시간영역 후처리 차이 추정 — 모델 신뢰구간 ±25%로 문서화.
- 동손 기하 추정 R_ph 79 vs 실제 158 mΩ — 엔드와인딩 미포함, 실측 입력 필수.
- 산출물: scripts/run_400w_loss.py, run_200w_loss.py,
  p400/p200_loss_results.json, p400_cog_regression.npz
