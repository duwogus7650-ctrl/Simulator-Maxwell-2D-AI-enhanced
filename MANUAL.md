# Maxwell 2D AI-Enhanced 모터 시뮬레이터 — Python 실행 매뉴얼

Ansys Maxwell `.aedt` 파일을 읽어 **2D 자기정해석(magnetostatic) FEM**으로 무부하/부하
해석을 수행하고, 토크·역기전력·철손·효율·코깅을 계산하며, DOE·서로게이트·유전
알고리즘(DE)·강화학습(SAC)으로 형상을 최적화하는 데스크톱 프로그램입니다.

> 이 프로그램은 **런타임에 네트워크를 전혀 쓰지 않습니다.** 한 번 설치만 끝내면
> 완전 오프라인에서 동작합니다. (설치 시 pip 다운로드에만 인터넷이 필요)

---

## 1. 사전 준비

| 항목 | 요구 |
|---|---|
| OS | Windows 10/11 (개발·검증 환경). Linux/macOS도 CLI는 동작 |
| Python | **3.12** 권장 (3.10~3.12 호환) |
| 디스크 | venv 포함 약 2~3 GB |
| 디스플레이 | GUI에만 필요. **CLI(run_cli.py)는 디스플레이 불필요** |

설치 여부 확인:
```bash
python --version        # Python 3.12.x
```

---

## 2. 설치 (최초 1회, 인터넷 필요)

프로젝트 폴더에서:

```bash
# 1) 가상환경 생성
py -3.12 -m venv venv            # Windows
# python3.12 -m venv venv        # Linux/macOS

# 2) 의존성 설치
venv\Scripts\python -m pip install --upgrade pip          # Windows
venv\Scripts\python -m pip install -r requirements.txt
# (Linux/macOS는 venv/bin/python 사용)
```

설치되는 패키지: numpy, scipy, shapely(≥2.0), triangle, matplotlib, PyQt6,
scikit-learn, joblib, torch.
`torch`는 SAC(강화학습) 정책에만 쓰입니다. **없어도 해석·DE 최적화는 정상 동작**합니다.

> `venv/` 폴더는 용량이 커서 GitHub에 올라가지 않습니다(.gitignore). 새 PC에서는
> 위 절차로 venv를 다시 만들면 됩니다.

---

## 3. 실행 (설치 후에는 오프라인)

### 3-1. GUI 실행

```bash
venv\Scripts\python gui\app.py                 # Windows
venv/bin/python gui/app.py                      # Linux/macOS
# 특정 파일을 바로 열기:
venv\Scripts\python gui\app.py "C:\Users\user\Desktop\aedt파일\400W.aedt"
```
Windows에서는 더블클릭용 `run_gui.bat`도 있습니다.

### 3-2. CLI(헤드리스) 실행 — 디스플레이 없이

GUI/Qt 없이 순수 Python으로 한 파일 또는 폴더를 해석합니다. 서버·오프라인·자동
검증에 적합합니다.

```bash
# 파일 하나
venv\Scripts\python run_cli.py "C:\Users\user\Desktop\aedt파일\400W.aedt"

# 폴더 전체(*.aedt)
venv\Scripts\python run_cli.py "C:\Users\user\Desktop\aedt파일"

# 옵션
venv\Scripts\python run_cli.py <대상> --current 27.1 --rpm 4500 --steps 12 --json out.json
```

| 옵션 | 의미 | 기본값 |
|---|---|---|
| `--current` | 상전류[A] | 파일의 `I_rms`, 없으면 1A |
| `--rpm` | 회전수[rpm] | 파일의 `BaseRPM`, 없으면 1000 |
| `--steps` | 부하 스윕 스텝 수 | 6 |
| `--json` | 결과를 JSON으로 저장 | (저장 안 함) |

출력 예:
```
✅ 400W.aedt [4. 400W_BasicModel_Load_Optimized]  16P18S  γ*=190.7°  T=0.8576 N·m  리플 1.57%  η 95.0%  |B|max 2.42T  | 80.8s
...
고유 결과(서로 다른 T·η 조합): 9 / 9 ✓ 모두 다름
```

---

## 4. 입력 파일(.aedt)

- 예제 모터들은 바탕화면 `aedt파일/` 폴더에 있습니다(400W, 750W/1200W, InwheelMotor,
  KRO80, SH_Reducer 등). DXF 도면은 `dxf 파일/` 폴더.
- 하나의 `.aedt`에 여러 design이 들어 있으면 **첫 번째 design**만 해석하고 나머지는
  경고로 알립니다.
- **외전형(outer-rotor, D_ro > D_so)** 모델은 현재 2D 내전형 솔버 범위 밖이라
  자동으로 건너뜁니다(명시적 보고).

---

## 5. 결과 지표 의미

| 지표 | 의미 |
|---|---|
| `T_avg_Nm` | 평균 토크 [N·m] |
| `ripple_pct` | 토크 리플 = (Tmax−Tmin)/Tavg × 100 [%] |
| `P_fe_W` | 철손(코어 로스) [W] |
| `efficiency` | 효율 (0~1) |
| `gamma_deg` | 최대토크 전류위상각 γ* [전기각°] |
| `Bmax` | 최대 자속밀도 [T] |
| `NR` | 비선형 Newton-Raphson 반복 횟수 |

> CLI/회귀의 기본 실행은 **제네릭 스모크 전류·권선**을 씁니다. NoLoad/변형 모델은
> 이 조건에서 토크가 작고 리플이 크게 나올 수 있습니다(정상). **실제 설계값**은
> GUI에서 모델별 전류·권선·Y/Δ를 제대로 설정하거나 CLI `--current`로 지정해
> 얻으세요.

---

## 6. 모터마다 결과가 다르게 나오는 이유 (중요)

각 `.aedt`는 자신의 `design_name`과 형상 변수(외경, 극·슬롯 수, 자석 두께 등)로
**독립적으로** 파싱→형상→메시→솔브됩니다. 또한 DOE 데이터셋과 서로게이트 모델은
`surrogate_{design_name}_{전류}.joblib` 처럼 **모델명·운전점별로 파일이 분리**되어
서로 다른 모터/전류의 결과가 절대 섞이지 않습니다([gui/app.py](gui/app.py)의
`_dataset_paths`). 따라서 "모든 모터가 똑같은 결과로 나오는" 일은 구조적으로
발생하지 않습니다. (예제 11개 모델은 design_name·형상이 전부 고유함을 확인했습니다.)

---

## 7. 회귀 테스트(전 모델 일괄 검증)

```bash
venv\Scripts\python scripts\run_regression.py                      # 기본: 바탕화면 aedt파일 폴더
venv\Scripts\python scripts\run_regression.py "경로\to\aedt폴더"
```
결과는 `regression_results.json`에 저장되고, 콘솔에 `PASS/FAIL/스킵` 요약이 뜹니다.

---

## 8. 완전 오프라인(air-gapped) 설치

인터넷이 전혀 없는 PC라면, 인터넷 되는 PC에서 휠을 미리 받아 옮깁니다:

```bash
# (인터넷 PC) 휠 다운로드
pip download -r requirements.txt -d wheelhouse

# wheelhouse 폴더를 대상 PC로 복사 후, (오프라인 PC)에서:
py -3.12 -m venv venv
venv\Scripts\python -m pip install --no-index --find-links wheelhouse -r requirements.txt
```
설치 후에는 §3 그대로 오프라인 실행됩니다.

---

## 9. 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| `venv 가상환경이 없습니다` | §2로 venv 생성 후 재실행 |
| `ModuleNotFoundError: shapely/triangle` | 시스템 Python으로 실행함. 반드시 `venv\Scripts\python` 사용 |
| 콘솔에 한글/기호가 깨짐 | Windows cp949 콘솔 문제. 프로그램은 stdout을 UTF-8로 재설정하지만, 안 되면 `chcp 65001` 후 실행 |
| `torch` 설치 실패 | SAC만 영향. requirements에서 torch를 빼고 설치해도 해석·DE 최적화는 동작 |
| 특정 모델이 `skip(외전형)` | 외전형은 현재 솔버 범위 밖(의도된 동작) |
| `변수명 대소문자 충돌` 에러 | 한 .aedt에 대소문자만 다른 동일 변수가 둘 있음 → 원본 수정 필요 |
| 다중 design 경고 | 한 파일에 여러 design 존재 → 첫 design만 해석(의도된 동작) |

---

## 10. 폴더 구조 요약

```
FEM 프로그램/
├─ gui/app.py            # PyQt6 GUI (메인)
├─ run_cli.py            # 오프라인 헤드리스 러너
├─ run_gui.bat           # Windows GUI 실행 배치
├─ motoropt/             # 해석·최적화 코어
│  ├─ aedt_parser.py     # .aedt 파싱
│  ├─ expressions.py     # Maxwell 변수 수식 평가(대소문자 무시)
│  ├─ geometry.py        # 형상 생성
│  ├─ sliding.py         # 슬라이딩밴드 메시
│  ├─ solver_ms.py       # 2D 자기정해석 솔버
│  ├─ sweep_loss.py      # 부하 스윕·손실·응답
│  ├─ surrogate.py       # 서로게이트(5-fold CV)
│  ├─ doe.py / rl_opt.py # DOE / 강화학습 최적화
│  └─ ...
├─ scripts/run_regression.py   # 전 모델 회귀
├─ requirements.txt
└─ MANUAL.md             # (이 문서)
```
