# 전면 재검토 + 견고성 하드닝 (2026-06-20)

코드베이스 전체를 4개 도메인(솔버·최적화·AEDT입출력·GUI) 병렬 리뷰 →
발견 문제 일괄 수정. 커밋 44c640c(서로게이트), 7eb8d91(하드닝 13파일).

- [x] 서로게이트: docstring이 주장만 하던 5-fold CV를 실제 계산. 측정으로
      ripple_pct/B_tooth/cogging_pp가 CV 음수=노이즈 확정 → reliable 플래그·
      노이즈 목표 경고. 구조 (64,64)→(16,16)+L2. 구버전 번들 하위호환.
- [x] 솔버 fail-loud: NR 미수렴 플래그·Arkkio 빈영역·코일 0면적 경고,
      디리클레 절대→상대공차(대형모터 경계누락), 슬라이딩밴드 잔차 진단.
- [x] 최적화: DOE 병렬 time_budget·except 분리, sweep 재질 자동검출,
      d_target 0분모 가드, SAC 언더플로 클램프.
- [x] AEDT: 다중설계 무차별 치환→첫설계 스코핑+라운드트립 검증, 보자력
      SI프리픽스 정확화, 파서 크래시 가드, parse_aedt design_name 옵션.
- [x] GUI: closeEvent 크래시방지·동시솔브 차단·모델 deepcopy·excepthook
      파일로깅·target 경계 엄격검증·Y/Δ currentData 판정.
- [x] 회귀 하네스 cp949 이모지 크래시 수정.

## 리뷰
- **검증: 내전형 7종 회귀 0.000% 편차** (Jun-13 베이스라인과 전 지표 동일) —
  모든 물리 수정이 FEM 수치 경로를 안 건드림을 입증.
- 독립 code-reviewer가 잡은 HIGH(mu_rec 최소제곱 변경)는 회귀 0%로 중립 확인.
- OMC executor(opus)×4 병렬 구현 → code-reviewer 독립 승인 패스 활용.
- 보류(정직): 다중설계 GUI 선택 UI(요청 대기), 코깅 밴드 연결 수학 재작성
  (Maxwell 실측 검증 선행 필요 — 이번엔 진단 경고만), 400W_10P12S/14P12S
  파싱실패(기존 문제 ['K','theta_x1'] 미정의, 범위 밖).

---

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
