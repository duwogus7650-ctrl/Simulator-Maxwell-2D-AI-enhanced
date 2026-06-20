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

---

# 야간 자율 세션 (2026-06-21 새벽) — 파서 수정·오프라인화·검증·배포

요청: 전체 검증 → 필요한 수정 자동진행 → 오프라인 Python 실행화 + 매뉴얼 →
최종 검증 → 폴더/GitHub 업로드. 핵심 우려: "모터마다 결과가 달라야 함".

- [x] theta_x1/K 파싱 실패 근본원인 = **변수명 대소문자 불일치**(D_SO vs D_so,
      T_yoke vs T_Yoke). Maxwell은 case-insensitive인데 파서는 case-sensitive.
      → expressions.py 해석을 대소문자 무시로(원본 casing 보존), 충돌은 fail-loud.
- [x] "모든 모터 동일결과" 우려 검증: 입력(11개 design_name·형상 전부 고유) +
      출력(회귀 9 PASS 전부 distinct, T 0.085~8.52 N·m, 100배 스프레드). 구조적
      불가능 확인. GUI는 이미 surrogate/DOE를 design_name·전류별 파일 분리.
- [x] 회귀: PASS 9 / FAIL 0 / 스킵 2(외전형 — 솔버 범위 밖). 이전 7/2/2 → 9/0/2.
- [x] 오프라인화: run_cli.py(헤드리스, 네트워크 의존성 0) + MANUAL.md(한글).
      run_cli 400W = T 0.8576/η95% 로 baseline 일치 검증.
- [x] 최종 검증: 전 모듈 컴파일 OK, 11/11 파싱, origin/main 동기(fec32d2).
- [x] GitHub 푸시 완료, MANUAL/run_cli 원격 반영 확인.
- 비고: theta_x1은 AEDT 직접 분석으로 해결 → Maxwell 영상(2.6GB) 분석 불필요.

## Review
회귀 실패 2건은 솔버/물리 버그가 아니라 파서가 Maxwell 시맨틱(대소문자 무시)을
안 따른 것. 한 줄 고치니 두 모터 모두 정상 파싱·해석. 사용자 핵심 우려는 입출력
양쪽에서 정량 반증함. 남은 관찰: 14P12S는 제네릭 스모크 전류에서 리플 776%로 거칠게
나옴(설계 실행 아님·정상). 실제 설계값은 GUI/CLI에서 모델별 전류·권선으로 산출.
