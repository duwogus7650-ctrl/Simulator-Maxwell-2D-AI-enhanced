"""motoropt 데스크톱 앱 (PyQt6) — Maxwell형 2D FEM 해석 + AI 최적설계.

탭: Model(aedt 로드·변수·형상) / Objective(만족도 스펙) /
    Solve(무부하·부하 해석) / Optimize(액티브러닝·SAC) / Result(비교·aedt 출력)
무거운 연산은 QThread 워커로 분리(UI 비차단).
실행:  python gui/app.py  [선택: 모델.aedt]
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


def _ensure_venv():
    """해석 패키지(shapely/triangle)가 없는 인터프리터로 실행되면
    프로젝트 venv의 Python으로 재실행한다 (triangle은 시스템 Python
    3.14용 wheel이 없어 venv 필수)."""
    import importlib.util
    if importlib.util.find_spec("shapely") and importlib.util.find_spec("triangle"):
        return
    vpy = os.path.join(_ROOT, "venv", "Scripts", "python.exe")
    if os.path.exists(vpy) and \
            os.path.normcase(vpy) != os.path.normcase(sys.executable):
        import subprocess
        print(f"[gui] 해석 패키지가 없는 Python — venv로 재실행: {vpy}",
              file=sys.stderr)
        sys.exit(subprocess.call([vpy] + sys.argv))


_ensure_venv()

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.font_manager as _fm
for _f in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",):
    try:
        _fm.fontManager.addfont(_f)
    except Exception:
        pass
_avail = {f.name for f in _fm.fontManager.ttflist}
for _name in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
    if _name in _avail:                      # Windows는 맑은 고딕
        matplotlib.rcParams["font.family"] = _name
        break
matplotlib.rcParams["axes.unicode_minus"] = False
# 다크 엔지니어링 테마 — 모든 Figure/Axes가 rcParams를 상속하므로
# 그리기 코드를 건드리지 않고 캔버스 배경·축·눈금을 어둡게 통일한다.
matplotlib.rcParams.update({
    "figure.facecolor": "#111a2b", "savefig.facecolor": "#111a2b",
    "axes.facecolor": "#0c1322", "axes.edgecolor": "#2a3a59",
    "axes.labelcolor": "#cdd7e6", "text.color": "#cdd7e6",
    "xtick.color": "#8896ad", "ytick.color": "#8896ad",
    "axes.titlecolor": "#36cdd6", "grid.color": "#1d2a44",
})
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# ── 다크 엔지니어링 테마 (mini motor-cad 스타일: 네이비 + 시안) ──────────
#   BG_APP 가장 어두움 / BG_PANEL 패널 / BG_INPUT 입력칸 / 시안 강조 / 호박색 경고
DARK_QSS = """
* { font-family: "Segoe UI", "Malgun Gothic", sans-serif; font-size: 13px;
    color: #cdd7e6; }
QMainWindow, QWidget { background: #0b0f1a; }
QToolTip { background: #111a2b; color: #cdd7e6; border: 1px solid #2a3a59; }

/* 상단 헤더 바 */
#Header { background: #0d1424; border-bottom: 2px solid #15324a; }
#HeaderTitle { font-size: 19px; font-weight: 700; letter-spacing: 1px; }
#HeaderSub  { color: #6f7f99; font-size: 11px; letter-spacing: 2px; }

/* 탭 */
QTabWidget::pane { border: 1px solid #1d2a44; background: #0b0f1a; top: -1px; }
QTabBar::tab {
    background: #0d1424; color: #7e8da6; padding: 9px 22px;
    border: 1px solid #15203a; border-bottom: none;
    margin-right: 2px; font-weight: 600; }
QTabBar::tab:selected {
    background: #111c30; color: #36cdd6;
    border-top: 2px solid #36cdd6; }
QTabBar::tab:hover:!selected { color: #b9c6da; background: #101a2d; }

/* 패널 (그룹박스) — 모서리 시안 라인 느낌 */
QGroupBox {
    background: #101829; border: 1px solid #1f2d4a; border-radius: 5px;
    margin-top: 14px; padding: 10px 8px 8px 8px; font-weight: 600; }
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 10px; padding: 1px 8px; color: #36cdd6;
    background: #101829; letter-spacing: 1px; }

QLabel { background: transparent; }
QSplitter::handle { background: #15203a; }

/* 입력칸 — 모노스페이스 숫자 */
QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox, QPlainTextEdit {
    background: #0c1322; color: #e3eaf6; border: 1px solid #243450;
    border-radius: 4px; padding: 4px 6px;
    selection-background-color: #1f6fd0; }
QDoubleSpinBox, QSpinBox, QLineEdit {
    font-family: "Consolas", "Courier New", monospace; }
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {
    border: 1px solid #36cdd6; }
QComboBox::drop-down { border: none; width: 18px; }
QComboBox QAbstractItemView {
    background: #0c1322; border: 1px solid #2a3a59;
    selection-background-color: #1f6fd0; outline: none; }
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
QSpinBox::up-button, QSpinBox::down-button {
    background: #16223a; border: none; width: 16px; }
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover,
QSpinBox::up-button:hover, QSpinBox::down-button:hover { background: #20406a; }

QPlainTextEdit {
    font-family: "Consolas", "Courier New", monospace; font-size: 12px;
    background: #080d18; border: 1px solid #1d2a44; }

/* 버튼 — 기본은 어두운 시안 테두리, 강조는 objectName */
QPushButton {
    background: #16223a; color: #cfe0ee; border: 1px solid #2a456e;
    border-radius: 4px; padding: 7px 14px; font-weight: 600; }
QPushButton:hover { background: #1d2f4f; border-color: #36cdd6; }
QPushButton:pressed { background: #122036; }
QPushButton:disabled { background: #11182a; color: #54607a;
    border-color: #1c283f; }
QPushButton#primary {
    background: #1763c4; color: #ffffff; border: 1px solid #2a7be0; }
QPushButton#primary:hover { background: #1f78e0; }
QPushButton#go {
    background: #14854a; color: #ffffff; border: 1px solid #1ca85e; }
QPushButton#go:hover { background: #18a25a; }

/* 표 */
QTableWidget, QTableView {
    background: #0c1322; alternate-background-color: #0f1828;
    gridline-color: #1d2a44; border: 1px solid #1d2a44;
    selection-background-color: #1c3a63; selection-color: #ffffff; }
QHeaderView::section {
    background: #16223a; color: #9fb0c8; padding: 6px;
    border: none; border-right: 1px solid #1d2a44;
    border-bottom: 1px solid #2a3a59; font-weight: 600; }
QTableCornerButton::section { background: #16223a; border: none; }
QCheckBox { background: transparent; }
QCheckBox::indicator, QTableWidget::indicator {
    width: 16px; height: 16px; border: 1px solid #2a456e;
    border-radius: 3px; background: #0c1322; }
QCheckBox::indicator:checked, QTableWidget::indicator:checked {
    background: #1f6fd0; border-color: #2a7be0; }

/* 진행바 */
QProgressBar {
    background: #0c1322; border: 1px solid #243450; border-radius: 4px;
    text-align: center; color: #cdd7e6; height: 18px; }
QProgressBar::chunk {
    background: #36cdd6; border-radius: 3px; }

/* 스크롤바 */
QScrollBar:vertical { background: #0b0f1a; width: 11px; margin: 0; }
QScrollBar::handle:vertical { background: #2a3a59; border-radius: 5px;
    min-height: 24px; }
QScrollBar::handle:vertical:hover { background: #36cdd6; }
QScrollBar:horizontal { background: #0b0f1a; height: 11px; margin: 0; }
QScrollBar::handle:horizontal { background: #2a3a59; border-radius: 5px;
    min-width: 24px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

QStatusBar { background: #0d1424; color: #8896ad;
    border-top: 1px solid #15203a; }
QMessageBox { background: #101829; }
"""

DESIGN_VARS = ["a_m", "T_m", "T_m2", "W_t", "MagnetR"]

OBJ_UNITS = {"T_avg": "mNm", "emf_rms": "V", "magnet_area": "mm²",
             "ripple_pct": "%", "efficiency": "0~1", "cogging_pp": "mNm",
             "Pcu_W": "W", "B_tooth_st": "T", "B_yoke": "T"}

# 응답 키 → 한글 표시명 (Objective·Result 탭 공용)
RESP_KO = {"T_avg": "평균토크", "emf_rms": "역기전력", "magnet_area": "자석면적",
           "ripple_pct": "토크리플", "B_tooth": "치자속밀도", "efficiency": "효율",
           "cogging_pp": "코깅토크", "Pcu_W": "동손",
           "B_tooth_st": "치 자속", "B_yoke": "요크 자속"}
# 방향 유형: (내부값, 한글표시) — 콤보 itemData에 내부값 저장
TYPE_KO = [("larger", "최대화 ↑"), ("smaller", "최소화 ↓"), ("target", "목표치 ◎")]
TYPE_EN2KO = {en: ko for en, ko in TYPE_KO}


def resp_label(key: str) -> str:
    """응답 키 → '평균토크 [mNm]' 표시명."""
    u = OBJ_UNITS.get(key, "")
    return f"{RESP_KO.get(key, key)} [{u}]" if u else RESP_KO.get(key, key)


def diagnose_result(fem: dict, spec: dict, fem_D: float,
                    surro_D: float | None = None,
                    hard_keys: set | None = None) -> list:
    """결과 자동 진단 — 비전문가도 알 수 있게 경고 리스트 반환.

    (1) D=0 원인: 어느 목표가 만족도 0인지·왜(값 vs 한계). 하드/소프트 구분.
    (2) AI 과대예측: 서로게이트 D ≫ 실제 FEM D.
    (3) 물리 타당성: 깊은 포화·코깅 과다·Arkkio↔가상일 괴리·비현실 효율.
    솔버가 스스로 '의심스러운 결과'를 표면화해 조용한 오류를 막는다."""
    from motoropt.objective import _D_FUNCS, _hard_pass
    hard_keys = hard_keys or set()
    out = []
    hard_viol, soft_zeros = [], []
    for k, s in spec.items():
        if k not in fem or fem[k] is None:
            continue
        if s[0] == "larger":
            why = f"{fem[k]:.4g} < 하한 {s[1]:.4g}"
        elif s[0] == "smaller":
            why = f"{fem[k]:.4g} > 상한 {s[-1]:.4g}"
        else:
            why = f"{fem[k]:.4g}, 목표 {s[2]:.4g}"
        if k in hard_keys:
            if float(_hard_pass([fem[k]], s[0], *s[1:])[0]) < 0.5:
                hard_viol.append(f"{RESP_KO.get(k, k)} 위반({why})")
        elif float(_D_FUNCS[s[0]](np.array([fem[k]], float), *s[1:])[0]) < 0.02:
            soft_zeros.append(f"{RESP_KO.get(k, k)}=0점({why})")
    if fem_D < 1e-6 and hard_viol:
        out.append("⚠ 종합 D=0 원인(필수 제약 위반): " + " · ".join(hard_viol) +
                   " → 이 설계는 🔒필수 조건을 못 지킵니다. 액티브러닝을 더 "
                   "돌려(AI가 그 영역 학습) 만족하는 설계를 찾거나, 필수가 "
                   "물리적으로 무리면 'AI 권장 목표값' 버튼으로 나머지(소프트) "
                   "목표를 풀어 해를 만드세요.")
    if fem_D < 1e-6 and soft_zeros:
        out.append("⚠ 종합 D=0 원인: " + " · ".join(soft_zeros) +
                   " → 'AI 권장 목표값' 버튼으로 한계(L/U)를 현실값으로 조정하거나 "
                   "액티브러닝을 더 돌리세요. (만족도는 곱이라 한 항목만 0이어도 전체 0)")
    elif soft_zeros:
        out.append("ℹ 일부 목표 0점: " + " · ".join(soft_zeros) +
                   " (종합엔 미반영)")
    if surro_D is not None and surro_D - fem_D > 0.25:
        out.append(f"⚠ AI(서로게이트) 과대예측: 예측 D={surro_D:.3f} → 실제 "
                   f"D={fem_D:.3f}. 학습 덜 된 영역을 골랐을 수 있음 — "
                   "액티브러닝 2~3회 더 돌리면 보정됩니다.")
    flags = []
    Bt = fem.get("B_tooth")
    if Bt and Bt > 2.5:
        flags.append(f"치 자속 {Bt:.2f}T(깊은 포화)")
    if fem.get("T_avg") and fem.get("cogging_pp") is not None and fem["T_avg"]:
        rc = fem["cogging_pp"] / fem["T_avg"] * 100
        if rc > 3:
            flags.append(f"코깅이 평균토크의 {rc:.1f}%(큼)")
    if fem.get("T_avg") and fem.get("T_arkkio"):
        dv = abs(fem["T_arkkio"] - fem["T_avg"]) / fem["T_avg"] * 100
        if dv > 8:
            flags.append(f"Arkkio↔가상일 토크 {dv:.0f}% 괴리(메시 점검)")
    eff = fem.get("efficiency")
    if eff is not None and (eff > 0.99 or eff < 0.3):
        flags.append(f"효율 {eff*100:.0f}%(운전점 확인)")
    if flags:
        out.append("⚠ 물리 타당성 점검: " + " / ".join(flags) +
                   " — 절대값은 Maxwell 교차검증 권장.")
    return out


def _obj_key(text: str) -> str:
    """테이블 표시명 'T_avg [mNm]' → 응답 키 'T_avg'."""
    return text.split(" [")[0].strip()


def _error_dialog(parent, title: str, exc: BaseException):
    """예외 → 사용자 안내 다이얼로그. PyQt6는 슬롯 내 미처리 예외 시
    앱을 abort시키므로 사용자 동작 슬롯은 반드시 이걸로 감싼다."""
    if isinstance(exc, ModuleNotFoundError):
        msg = (f"필요한 패키지가 없습니다: {exc.name}\n\n"
               "venv의 Python으로 실행하세요:\n"
               "    venv\\Scripts\\python gui\\app.py\n"
               "또는 run_gui.bat 더블클릭")
    elif isinstance(exc, KeyError):
        msg = (f"이 설계는 지원하지 않는 변수 구성입니다 "
               f"(필수 변수 {exc} 없음)")
    else:
        msg = f"{type(exc).__name__}: {exc}"
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Critical)
    box.setWindowTitle(title)
    box.setText(msg)
    box.setDetailedText("".join(traceback.format_exception(exc)))
    box.exec()


# ====================================================================== 워커
class Worker(QThread):
    log = pyqtSignal(str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    geom = pyqtSignal(object)        # 실시간 형상 (설계변수 dict)

    def __init__(self, fn, *args, **kw):
        super().__init__()
        self.fn, self.args, self.kw = fn, args, kw

    def run(self):
        try:
            self.done.emit(self.fn(self.log.emit, *self.args, **self.kw))
        except Exception:
            self.failed.emit(traceback.format_exc())


# ================================================================ 메인 윈도
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("motoropt — Maxwell 2D AI-enhanced")
        self.resize(1280, 800)
        self.model = None          # 파싱된 aedt 모델
        self.style = None
        self.geo = None
        self.aedt_path = None
        self.last_solve = None     # (solver, result, 메트릭 dict)
        self.last_responses = None # 부하 스윕 응답 dict
        self.candidates = []       # 최적화 후보 [(D, x, fem)]
        self._active_round = 0     # 액티브러닝 라운드 카운터(모델별)
        self._workers = []
        self._cur_geom_emit = lambda *a, **k: None   # 잡→GUI 형상/진행 통로
        self._obj_user_edited = set()    # 사용자가 직접 바꾼 목표 키(자동충전 보존)
        self._obj_autofilling = False    # 프로그램적 표 갱신 중 플래그

        tabs = QTabWidget()
        tabs.addTab(self._tab_model(), "① Model")
        tabs.addTab(self._tab_objective(), "② Objective")
        tabs.addTab(self._tab_solve(), "③ Solve")
        tabs.addTab(self._tab_optimize(), "④ Optimize")
        tabs.addTab(self._tab_result(), "⑤ Result")
        self.tabs = tabs

        # 상단 브랜드 헤더 (mini motor-cad 스타일)
        header = QWidget(); header.setObjectName("Header")
        hl = QHBoxLayout(header); hl.setContentsMargins(16, 8, 16, 8)
        title = QLabel("MOTOR<span style='color:#36cdd6'>OPT</span>")
        title.setObjectName("HeaderTitle")
        sub = QLabel("MAXWELL 2D · AI MOTOR DESIGN")
        sub.setObjectName("HeaderSub")
        hl.addWidget(title); hl.addSpacing(12); hl.addWidget(sub); hl.addStretch()
        self.lbl_header_model = QLabel("모델 없음")
        self.lbl_header_model.setObjectName("HeaderSub")
        hl.addWidget(self.lbl_header_model)

        central = QWidget(); v = QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        v.addWidget(header); v.addWidget(tabs, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("aedt 파일을 열어 시작하세요")

    # ---------------------------------------------------------- ① Model
    def _tab_model(self):
        w = QWidget(); lay = QHBoxLayout(w)
        left = QVBoxLayout()
        btn = QPushButton("📂 .aedt 열기")
        btn.setObjectName("primary")
        btn.clicked.connect(self.open_aedt)
        left.addWidget(btn)
        self.lbl_model = QLabel("—")
        left.addWidget(self.lbl_model)
        self.tbl_vars = QTableWidget(0, 3)
        self.tbl_vars.setHorizontalHeaderLabels(["변수", "수식", "해석값"])
        self.tbl_vars.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        left.addWidget(self.tbl_vars, 1)
        btn2 = QPushButton("🔄 변수 적용 → 형상 갱신")
        btn2.clicked.connect(self.refresh_geometry)
        left.addWidget(btn2)
        lw = QWidget(); lw.setLayout(left)

        self.fig_geo = Figure(figsize=(5, 5), tight_layout=True)
        self.cv_geo = FigureCanvasQTAgg(self.fig_geo)
        sp = QSplitter(); sp.addWidget(lw); sp.addWidget(self.cv_geo)
        sp.setSizes([520, 720])
        lay.addWidget(sp)
        return w

    def open_aedt(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Maxwell 프로젝트", "", "AEDT (*.aedt)")
        if not path:
            return
        try:
            self._load_aedt(path)
        except Exception as e:
            self.statusBar().showMessage(f"로드 실패: {os.path.basename(path)}")
            _error_dialog(self, "aedt 열기 실패", e)

    def _reset_model_state(self):
        """모델 전환 시 이전 모델의 해석·최적화 결과를 모두 비운다 —
        다른 aedt를 열었는데 직전 모델 결과가 남는 것 방지."""
        self.candidates = []
        self._active_round = 0
        if hasattr(self, "btn_active"):
            self.btn_active.setText(
                "▶ 액티브러닝 1라운드 (DE→FEM 검증→재학습)")
        for tbl in (getattr(self, "tbl_cand", None),
                    getattr(self, "tbl_res", None)):
            if tbl is not None:
                tbl.setRowCount(0)
        for fig, cv in ((getattr(self, "fig_field", None),
                         getattr(self, "cv_field", None)),
                        (getattr(self, "fig_res", None),
                         getattr(self, "cv_res", None))):
            if fig is not None and cv is not None:
                fig.clear(); cv.draw()
        for log in (getattr(self, "log_solve", None),
                    getattr(self, "log_opt", None)):
            if log is not None:
                log.clear()

    def _load_aedt(self, path):
        from motoropt.aedt_parser import parse_aedt, detect_magnet_style
        self.model = parse_aedt(path)
        self._reset_model_state()           # 직전 모델 결과 비우기
        self.style = detect_magnet_style(self.model)
        self.aedt_path = path
        self.lbl_model.setText(
            f"<b>{self.model['design_name']}</b> · 자석={self.style} · "
            f"파트 {len(self.model['parts'])} · "
            f"코일 {len(self.model['boundaries']['coils'])}")
        self.lbl_header_model.setText(
            f"▣ {self.model['design_name']}")          # 헤더에 현재 모델명
        v, raw = self.model["variables"], self.model["variables_raw"]
        keys = list(raw)
        self.tbl_vars.setRowCount(len(keys))
        for i, k in enumerate(keys):
            self.tbl_vars.setItem(i, 0, QTableWidgetItem(k))
            it = QTableWidgetItem(raw[k])
            if k not in DESIGN_VARS:
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.tbl_vars.setItem(i, 1, it)
            val = v.get(k)
            disp = f"{val:.6g}" if val is not None else "—"
            it2 = QTableWidgetItem(disp)
            it2.setFlags(it2.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.tbl_vars.setItem(i, 2, it2)
        if v.get("BaseRPM"):
            self.sp_rpm.setValue(float(v["BaseRPM"]))
        if v.get("I_rms"):
            self.sp_irms.setValue(float(v["I_rms"]))
        if v.get("Zc"):
            self.sp_turns.setValue(float(v["Zc"]))   # aedt 턴수로 초기화
        try:                                          # aedt 적층계수로 초기화
            from motoropt.aedt_parser import detect_material_names
            steel, _ = detect_material_names(self.model)
            self.sp_stack.setValue(float(
                self.model["materials"][steel].get("stacking_factor", 1.0)))
        except Exception:
            self.sp_stack.setValue(1.0)
        self._obj_user_edited.clear()          # 새 모델 → 목표 편집표시 초기화
        if not self._autofill_spec_from_dataset():
            self._reset_spec_defaults()        # 메타 없으면 기본값 복원
        self._refresh_geometry()
        self.statusBar().showMessage(f"{os.path.basename(path)} 로드 완료")

    def _apply_turns(self):
        """턴수(Zc) 입력을 모델 변수·변수표에 일괄 반영.

        Zc는 독립 변수이고 다른 수식이 참조하지 않으므로 안전하게 덮어쓴다.
        variables_raw(=DOE·액티브러닝의 vary가 재해석), variables(=부하스윕),
        변수표(=current_raw가 읽는 무부하·부하 해석)를 모두 동기화해 모든
        해석이 같은 턴수를 쓰게 한다."""
        if self.model is None:
            return
        zc = int(round(self.sp_turns.value()))
        self.model["variables_raw"]["Zc"] = str(zc)
        self.model["variables"]["Zc"] = float(zc)
        for i in range(self.tbl_vars.rowCount()):
            if self.tbl_vars.item(i, 0).text() == "Zc":
                self.tbl_vars.item(i, 1).setText(str(zc))
                it2 = self.tbl_vars.item(i, 2)
                if it2 is not None:
                    it2.setText(f"{float(zc):.6g}")
                break

    def _apply_stack(self):
        """적층계수(점적률)를 강판 재질에 반영 — 철심 BH에만 적용(EMF 불변).

        solver_ms(무부하·부하)·sweep_loss(스윕)·doe(DOE) 모두
        model['materials'][강판]['stacking_factor']를 읽으므로, 여기서 한 번
        써 두면 전 해석이 같은 적층계수를 쓴다. NuCurve가 B_eff = ks·B +
        (1−ks)·μ0·H 로 철심 포화를 키운다."""
        if self.model is None:
            return
        from motoropt.aedt_parser import detect_material_names
        try:
            steel, _ = detect_material_names(self.model)
        except Exception:
            return
        self.model["materials"][steel]["stacking_factor"] = \
            float(self.sp_stack.value())

    def current_raw(self):
        raw = dict(self.model["variables_raw"])
        for i in range(self.tbl_vars.rowCount()):
            k = self.tbl_vars.item(i, 0).text()
            raw[k] = self.tbl_vars.item(i, 1).text()
        return raw

    def refresh_geometry(self):
        if self.model is None:
            return
        try:
            self._refresh_geometry()
        except Exception as e:
            self.statusBar().showMessage("형상 갱신 실패")
            _error_dialog(self, "형상 갱신 실패", e)

    def _refresh_geometry(self):
        from motoropt.expressions import resolve_variables
        from motoropt.geometry import build_motor
        v = resolve_variables(self.current_raw())
        self.model["variables"] = v
        self.geo = build_motor(v, self.style)
        ax = self.fig_geo.gca() if self.fig_geo.axes else \
            self.fig_geo.add_subplot()
        ax.clear()
        self._draw_geo(ax, self.geo)
        ax.set_title(f"{self.model['design_name']} — 자석 "
                     f"{sum(p.area for p, _, _ in self.geo.magnets):.1f} mm²")
        self.cv_geo.draw()

    @staticmethod
    def _draw_geo(ax, geo, color_mode=True):
        from matplotlib.patches import Polygon as MplPoly
        def fill(poly, fc):
            if poly.geom_type == "MultiPolygon":
                for g in poly.geoms:
                    fill(g, fc)
                return
            ax.add_patch(MplPoly(np.asarray(poly.exterior.coords), closed=True,
                                 facecolor=fc, edgecolor="#404040", lw=.3))
            for h in poly.interiors:
                ax.add_patch(MplPoly(np.asarray(h.coords), closed=True,
                                     facecolor="white", lw=.3,
                                     edgecolor="#404040"))
        fill(geo.stator, "#c8c8c8"); fill(geo.rotor, "#c8c8c8")
        for p, ang, pol in geo.magnets:
            fill(p, "#e02020" if pol > 0 else "#2040e0")
        for c in geo.coils:
            fill(c, "#ff8c00")
        lim = geo.region_radius * 1.05
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")

    # ------------------------------------------------------ ② Objective
    def _tab_objective(self):
        w = QWidget(); lay = QVBoxLayout(w)
        lay.addWidget(QLabel(
            "<b>목표 특성</b> — 최적화에서 동시에 만족시킬 성능 목표를 고르세요.<br>"
            "<b>방향</b>: 최대화↑(클수록 좋음)·최소화↓(작을수록 좋음)·목표치◎"
            "(특정값에 맞춤). 방향에 따라 입력칸이 자동 활성화됩니다."))
        from motoropt.objective import SPEC, SPEC_EXTRA
        rows = [(k, s, True) for k, s in SPEC.items()] + \
               [(k, s, False) for k, s in SPEC_EXTRA.items()]
        self.tbl_obj = QTableWidget(len(rows), 7)
        self.tbl_obj.setHorizontalHeaderLabels(
            ["사용", "목표 특성", "방향", "하한치 (L)", "타겟값 (T)",
             "상한치 (U)", "필수 🔒"])
        self.tbl_obj.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._obj_autofilling = True       # 구성 중 편집신호 무시
        for i, (k, spec, on) in enumerate(rows):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable |
                         Qt.ItemFlag.ItemIsEnabled)
            chk.setCheckState(Qt.CheckState.Checked if on
                              else Qt.CheckState.Unchecked)
            self.tbl_obj.setItem(i, 0, chk)
            name = QTableWidgetItem(resp_label(k))
            name.setFlags(name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            name.setData(Qt.ItemDataRole.UserRole, k)      # 내부 응답 키 보관
            self.tbl_obj.setItem(i, 1, name)
            cb = QComboBox()
            for en, ko in TYPE_KO:
                cb.addItem(ko, en)                          # 표시=한글, data=내부값
            cb.setCurrentIndex([e for e, _ in TYPE_KO].index(spec[0]))
            cb.currentIndexChanged.connect(
                lambda _t, r=i: self._on_obj_dir_changed(r))
            self.tbl_obj.setCellWidget(i, 2, cb)
            if spec[0] == "target":
                vals = {3: spec[1], 4: spec[2], 5: spec[3]}
            else:
                vals = {3: spec[1], 5: spec[2]}
            for c in (3, 4, 5):
                self.tbl_obj.setItem(
                    i, c, QTableWidgetItem(
                        f"{vals[c]:.4g}" if c in vals else ""))
            hard = QTableWidgetItem()                       # 필수(하드 제약)
            hard.setFlags(Qt.ItemFlag.ItemIsUserCheckable |
                          Qt.ItemFlag.ItemIsEnabled)
            hard.setCheckState(Qt.CheckState.Unchecked)
            hard.setToolTip(
                "체크 = 하드 제약: 이 조건(방향·경계)을 반드시 만족해야 함"
                "(어기면 그 설계는 탈락=D 0). AI 권장값 조정 시에도 고정되어 "
                "바뀌지 않고, 나머지(소프트) 목표만 조정됩니다.")
            self.tbl_obj.setItem(i, 6, hard)
            self._update_obj_row_state(i)
        self._obj_autofilling = False      # 구성 끝 — 이후 편집은 사용자 것
        self.tbl_obj.itemChanged.connect(self._on_obj_item_changed)
        lay.addWidget(self.tbl_obj)
        lay.addWidget(QLabel(
            "종합 만족도 D = (∏ dᵢ)^(1/n) — 모든 목표를 동시에 만족할수록 1에 "
            "가까움.<br>• <b>최대화</b>: 하한치=미달 기준(0점), 상한치=충분 기준"
            "(만점) • <b>최소화</b>: 하한치=충분히 작음(만점), 상한치=초과 기준"
            "(0점) • <b>목표치</b>: 하한·타겟·상한 모두 사용.<br>"
            "토크리플·치자속밀도는 노이즈가 커 학습이 안 되니 목표로 쓰지 말고 "
            "후보 FEM 검증으로만 확인하세요."))
        lay.addStretch(1)
        return w

    def _spec_from_table(self) -> dict:
        """Objective 테이블의 체크된 행 → SPEC dict."""
        spec = {}
        for i in range(self.tbl_obj.rowCount()):
            if self.tbl_obj.item(i, 0).checkState() != Qt.CheckState.Checked:
                continue
            k = self.tbl_obj.item(i, 1).data(Qt.ItemDataRole.UserRole)
            typ = self.tbl_obj.cellWidget(i, 2).currentData()   # 내부값(en)
            try:
                L = float(self.tbl_obj.item(i, 3).text())
                U = float(self.tbl_obj.item(i, 5).text())
                if typ == "target":
                    spec[k] = (typ, L, float(self.tbl_obj.item(i, 4).text()), U)
                else:
                    spec[k] = (typ, L, U)
            except (TypeError, ValueError):
                raise ValueError(f"Objective 행 '{k}'의 L/T/U 값이 잘못됨")
        if not spec:
            raise ValueError("체크된 목표 특성이 없음")
        return spec

    def _hard_keys_from_table(self) -> set:
        """필수(하드 제약)로 체크된 + 사용 중인 목표 키 집합."""
        hard = set()
        for i in range(self.tbl_obj.rowCount()):
            it6 = self.tbl_obj.item(i, 6)
            if (self.tbl_obj.item(i, 0).checkState() == Qt.CheckState.Checked
                    and it6 is not None
                    and it6.checkState() == Qt.CheckState.Checked):
                hard.add(self.tbl_obj.item(i, 1)
                         .data(Qt.ItemDataRole.UserRole))
        return hard

    def relax_to_recommended(self):
        """목표가 빡빡해 해가 안 잡힐 때 AI 권장값 제안 — 필수(🔒)는 고정하고
        소프트 목표의 한계(L/U)를 이 모델 DOE의 '필수 만족 설계' 달성범위로
        조정. 변경값을 미리 보여주고 적용 여부는 사용자가 선택(수락/거부)."""
        from PyQt6.QtWidgets import QMessageBox
        from motoropt.objective import _hard_pass
        if self.model is None:
            self.log_opt.appendPlainText("⚠ 먼저 Model 탭에서 aedt를 여세요")
            return
        dataset, _ = self._dataset_paths()
        if not os.path.exists(dataset):
            QMessageBox.warning(self, "데이터 없음",
                "이 모델 DOE 데이터가 없습니다. 먼저 'DOE 생성'을 하세요.")
            return
        try:
            spec = self._spec_from_table()
        except ValueError as e:
            QMessageBox.warning(self, "목표 없음", str(e)); return
        hard = self._hard_keys_from_table()
        oks = []
        with open(dataset, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("status") == "ok":
                    oks.append(r)
        if len(oks) < 3:
            QMessageBox.warning(self, "데이터 부족",
                f"유효 DOE 설계가 {len(oks)}개뿐입니다. DOE를 더 생성하세요.")
            return

        def feasible(r):                       # 필수 제약 모두 만족하는 설계?
            for k in hard:
                if k in spec and k in r and r[k] is not None:
                    if float(_hard_pass([r[k]], spec[k][0],
                                        *spec[k][1:])[0]) < 0.5:
                        return False
            return True
        feas = [r for r in oks if feasible(r)]
        if hard and not feas:                  # 필수 자체가 데이터서 불가능
            lines = []
            for k in hard:
                vals = [r[k] for r in oks if k in r and r[k] is not None]
                if vals:
                    lines.append(f"  · {resp_label(k)}: 데이터 달성범위 "
                                 f"{min(vals):.4g} ~ {max(vals):.4g}")
            QMessageBox.warning(self, "필수 제약 불가",
                "🔒필수 제약을 동시에 만족하는 설계가 데이터에 없습니다.\n"
                "필수가 너무 빡빡하거나 DOE 탐색범위 밖입니다 — 필수 경계를 "
                "아래 범위 안으로 완화하거나 DOE 범위를 넓히세요.\n\n"
                + "\n".join(lines))
            return
        pool = feas or oks                     # 필수 만족 설계 기준(없으면 전체)
        recs, preview = {}, []
        for i in range(self.tbl_obj.rowCount()):
            if self.tbl_obj.item(i, 0).checkState() != Qt.CheckState.Checked:
                continue
            k = self.tbl_obj.item(i, 1).data(Qt.ItemDataRole.UserRole)
            if k in hard:
                preview.append(f"  🔒 {resp_label(k)}: 고정(유지)")
                continue
            if k not in spec:
                continue
            vals = [r[k] for r in pool if k in r and r[k] is not None]
            if len(vals) < 3:
                continue
            lo, hi = float(min(vals)), float(max(vals))
            if hi - lo < abs(lo) * 1e-3 + 1e-9:
                lo, hi = lo * 0.98 - 1e-6, hi * 1.02 + 1e-6
            cur = spec[k]
            recs[k] = ((cur[0], lo, cur[2], hi) if cur[0] == "target"
                       else (cur[0], lo, hi))
            preview.append(f"  · {resp_label(k)}: "
                           f"[{cur[1]:.4g}, {cur[-1]:.4g}] → [{lo:.4g}, {hi:.4g}]")
        if not recs:
            QMessageBox.information(self, "조정할 목표 없음",
                "조정 가능한 소프트 목표가 없습니다(모두 필수이거나 데이터 부족).")
            return
        msg = ("필수(🔒)는 고정하고, 아래 소프트 목표의 한계를 이 모델 데이터의 "
               "달성 가능 범위로 조정합니다"
               + (f" (🔒필수 만족 설계 {len(feas)}개 기준)" if hard else "")
               + ".\n적용하시겠습니까?\n\n" + "\n".join(preview))
        if QMessageBox.question(self, "AI 권장 목표값", msg) != \
                QMessageBox.StandardButton.Yes:
            self.log_opt.appendPlainText("ℹ AI 권장 목표값 — 취소(변경 없음)")
            return
        for k, s in recs.items():
            self._set_obj_row(k, s)
            self._obj_user_edited.discard(k)
        self.log_opt.appendPlainText(
            "🎯 AI 권장 목표값 적용: "
            + ", ".join(resp_label(k) for k in recs)
            + " → 이제 액티브러닝을 다시 돌리세요.")

    # ---------------------------------------------------------- ③ Solve
    def _tab_solve(self):
        w = QWidget(); lay = QHBoxLayout(w)
        left = QVBoxLayout()
        b1 = QPushButton("▶ 무부하 해석 (코깅·EMF용 단일 포지션)")
        b1.clicked.connect(lambda: self.run_solve(load=False))
        b2 = QPushButton("▶ 부하 해석 (입력 전류·MTPA)")
        b2.setObjectName("primary")
        b2.clicked.connect(lambda: self.run_solve(load=True))
        left.addWidget(b1); left.addWidget(b2)

        grp = QGroupBox("부하 스윕 — 효율·토크리플·손실 (전기 1주기)")
        g = QGridLayout(grp)
        self.sp_rpm = QDoubleSpinBox(); self.sp_rpm.setRange(1, 50000)
        self.sp_rpm.setDecimals(0); self.sp_rpm.setValue(3000)
        self.sp_irms = QDoubleSpinBox(); self.sp_irms.setRange(0.01, 1000)
        self.sp_irms.setDecimals(2); self.sp_irms.setValue(5.0)
        self.sp_rph = QDoubleSpinBox(); self.sp_rph.setRange(0, 10000)
        self.sp_rph.setDecimals(1); self.sp_rph.setValue(0.0)
        self.sp_nstep = QDoubleSpinBox(); self.sp_nstep.setRange(6, 120)
        self.sp_nstep.setDecimals(0); self.sp_nstep.setValue(36)
        self.sp_dcu = QDoubleSpinBox(); self.sp_dcu.setRange(0.01, 5)
        self.sp_dcu.setDecimals(3); self.sp_dcu.setValue(0.3)
        self.sp_strands = QDoubleSpinBox(); self.sp_strands.setRange(1, 200)
        self.sp_strands.setDecimals(0); self.sp_strands.setValue(11)
        self.sp_tcu = QDoubleSpinBox(); self.sp_tcu.setRange(-40, 250)
        self.sp_tcu.setDecimals(0); self.sp_tcu.setValue(80)
        self.sp_turns = QDoubleSpinBox(); self.sp_turns.setRange(1, 500)
        self.sp_turns.setDecimals(0); self.sp_turns.setValue(15)
        self.sp_turns.setToolTip(
            "코일당 턴수 (Maxwell의 도체수 Zc). aedt 값으로 자동 설정되며, "
            "여기서 바꾸면 무부하·부하·스윕·DOE 모든 해석에 반영됩니다.")
        self.sp_stack = QDoubleSpinBox(); self.sp_stack.setRange(0.5, 1.0)
        self.sp_stack.setDecimals(3); self.sp_stack.setSingleStep(0.01)
        self.sp_stack.setValue(1.0)
        self.sp_stack.setToolTip(
            "적층계수(점적률). 강판 BH에만 적용돼 철심 포화를 키움(EMF는 거의 "
            "불변). aedt 값으로 자동 설정, 없으면 1.0. Maxwell이 0.97을 썼다면 "
            "0.97 입력. 무부하·부하·스윕·DOE 모든 해석에 반영됩니다.")
        self.cb_ibase = QComboBox()
        self.cb_ibase.addItems(["상전류 (권선)", "선간전류 · Y결선",
                                "선간전류 · Δ결선"])
        for col, (lbl, w_) in enumerate(
                [("전류 [Arms]", self.sp_irms),
                 ("전류 기준 (해석은 상전류)", self.cb_ibase),
                 ("상저항 [mΩ] (0=MLT 계산)", self.sp_rph),
                 ("스텝", self.sp_nstep)]):
            g.addWidget(QLabel(lbl), 0, col)
            g.addWidget(w_, 1, col)
        for col, (lbl, w_) in enumerate(
                [("속도 [rpm]", self.sp_rpm),
                 ("나동선 지름 [mm]", self.sp_dcu),
                 ("가닥수", self.sp_strands),
                 ("권선온도 [°C]", self.sp_tcu),
                 ("턴수 (Zc)", self.sp_turns),
                 ("적층계수", self.sp_stack)]):
            g.addWidget(QLabel(lbl), 2, col)
            g.addWidget(w_, 3, col)
        self.lbl_iconv = QLabel()
        g.addWidget(self.lbl_iconv, 4, 0, 1, 6)
        self.sp_irms.valueChanged.connect(self._update_iconv)
        self.cb_ibase.currentIndexChanged.connect(self._update_iconv)
        self.sp_turns.valueChanged.connect(self._apply_turns)
        self.sp_stack.valueChanged.connect(self._apply_stack)
        self._update_iconv()
        b3 = QPushButton("▶ 부하 스윕 실행 (γ 캘리브레이션 포함 — 수 분 소요)")
        b3.setObjectName("go")
        b3.clicked.connect(self.run_load_sweep)
        g.addWidget(b3, 5, 0, 1, 6)
        left.addWidget(grp)
        self.log_solve = QPlainTextEdit(); self.log_solve.setReadOnly(True)
        left.addWidget(self.log_solve, 1)
        lw = QWidget(); lw.setLayout(left)
        self.fig_field = Figure(figsize=(5.6, 5), tight_layout=True)
        self.cv_field = FigureCanvasQTAgg(self.fig_field)
        sp = QSplitter(); sp.addWidget(lw); sp.addWidget(self.cv_field)
        sp.setSizes([430, 820])
        lay.addWidget(sp)
        return w

    def run_solve(self, load):
        if self.geo is None:
            self.log_solve.appendPlainText("⚠ 먼저 Model 탭에서 aedt를 여세요")
            return
        model, style, raw = self.model, self.style, self.current_raw()
        irms_in, i_note = self._phase_current()
        if load and i_note:
            self.log_solve.appendPlainText(i_note)

        def job(log):
            from motoropt.expressions import resolve_variables
            from motoropt.geometry import build_motor
            from motoropt.sliding import SlidingBandMesh
            from motoropt.solver_ms import Magnetostatic2D
            from motoropt.postproc import (torque_arkkio, coenergy,
                                           build_winding_map)
            from motoropt.aedt_parser import detect_material_names
            v = resolve_variables(raw)
            geo = build_motor(v, style)
            steel, magnet = detect_material_names(model)
            _ks = model["materials"][steel].get("stacking_factor", 1.0)
            log(f"재질: 강판={steel} / 자석={magnet}"
                + (f" / 적층계수 {_ks:.3f}" if _ks < 1.0 else ""))
            log("형상/메시 생성...")
            sbm = SlidingBandMesh(geo, n_band=2880)
            s = Magnetostatic2D(sbm.merge(0.0), model["materials"],
                                steel, magnet)
            if load:
                wmap = build_winding_map(s)
                Ia = (irms_in or v["I_rms"]) * math.sqrt(2)
                Zc = int(round(v["Zc"]))
                te = math.radians(290.0)
                iph = {"A": Ia * math.sin(te),
                       "B": Ia * math.sin(te - 2 * math.pi / 3),
                       "C": Ia * math.sin(te + 2 * math.pi / 3)}
                at = {}
                for ph, sides in wmap.items():
                    for ci, d in sides:
                        at[ci] = d * Zc * iph[ph]
                s.set_coil_currents(at)
            else:
                s.set_coil_currents({})
            log("Newton-Raphson 풀이...")
            res = s.solve(tol=1e-5)
            T = torque_arkkio(s, res, sbm.r_i + .005, sbm.r_o - .005,
                              v["L_stk"])
            met = {"mode": "부하" if load else "무부하",
                   "NR": res.iterations,
                   "Az_max": float(np.abs(res.A).max()),
                   "B_max": float(res.Bmag.max()),
                   "T_mNm": T * 1e3}
            log(f"수렴 {res.iterations}회 | Az±{met['Az_max']:.4f} Wb/m | "
                f"|B|max {met['B_max']:.2f} T | T {met['T_mNm']:.1f} mNm"
                + (" (Arkkio 단일포지션·과대추정)" if load else ""))
            if load:
                log("ℹ 이 토크는 Arkkio 단일포지션(+4~7% 바이어스). Maxwell과 "
                    "비교·검증은 아래 '부하 스윕'의 T_avg(가상일 1주기 평균)을 쓰세요.")
            return s, res, met

        self._spawn(job, self.log_solve.appendPlainText, self._solve_done)

    def _update_iconv(self):
        """전류 입력/기준 변경 시 해석에 쓰일 상전류를 즉시 표시."""
        I, note = self._phase_current()
        self.lbl_iconv.setText(
            f"→ 해석 상전류: <b>{I:.2f} Arms</b>"
            + (f"  ({note})" if note else ""))

    def _phase_current(self) -> tuple:
        """전류 입력 + 기준 콤보 → (상전류 Arms, 환산 설명)."""
        I = float(self.sp_irms.value())
        mode = self.cb_ibase.currentText()
        if "Δ" in mode:
            return I / math.sqrt(3.0), \
                f"선간 {I:.2f}A (Δ) → 상전류 {I/math.sqrt(3.0):.2f}A"
        if "Y" in mode:
            return I, f"선간 {I:.2f}A (Y) = 상전류 {I:.2f}A"
        return I, ""

    def run_load_sweep(self):
        if self.geo is None:
            self.log_solve.appendPlainText("⚠ 먼저 Model 탭에서 aedt를 여세요")
            return
        model, style, raw = self.model, self.style, self.current_raw()
        rpm = float(self.sp_rpm.value())
        irms, i_note = self._phase_current()
        if i_note:
            self.log_solve.appendPlainText(i_note)
        rph = float(self.sp_rph.value()) * 1e-3 or None    # mΩ → Ω, 0=MLT 계산
        nstep = int(self.sp_nstep.value())
        d_cu = float(self.sp_dcu.value())
        strands = int(self.sp_strands.value())
        tcu = float(self.sp_tcu.value())

        def job(log):
            from motoropt.expressions import resolve_variables
            from motoropt.aedt_parser import detect_material_names
            from motoropt.sweep_loss import (sweep_load_with_fields,
                                             compute_responses,
                                             calibrate_gamma)
            v = resolve_variables(raw)
            m2 = dict(model); m2["variables"] = v
            steel, magnet = detect_material_names(m2)
            ini = math.degrees(v.get("ini_pos", 0.0))
            _ks = model["materials"][steel].get("stacking_factor", 1.0)
            log(f"재질: 강판={steel} / 자석={magnet}"
                + (f" / 적층계수 {_ks:.3f}" if _ks < 1.0 else ""))
            R_in = rph
            if R_in is None:
                from motoropt.winding import phase_resistance
                w = phase_resistance(v, d_cu_mm=d_cu, strands=strands,
                                     T_cu_C=tcu)
                R_in = w["R_ph"]
                log(f"R_ph(MLT 계산) = {R_in*1e3:.1f} mΩ "
                    f"(MLT {w['MLT_mm']:.1f} mm, 직렬 {w['n_series']:.0f}턴, "
                    f"도체 {w['turn_csa_mm2']:.3f} mm², {tcu:.0f}°C)")
                log("  ℹ 동손은 MLT 추정(검증 ±10%: 400W −3%·KRO80 +8%). 더 "
                    "정확히는 상저항 [mΩ]에 실측 R_ph를 직접 입력하세요(0=MLT 자동).")
            log("γ 캘리브레이션 (4점 프로브 × 6스텝)...")
            cal = calibrate_gamma(m2, style, rpm=rpm, I_rms=irms,
                                  n_steps=6, init_pos_deg=ini)
            log(f"γ* = {cal['gamma_max_deg']:.1f}°  "
                f"(T_max≈{cal['T_max_est']:.3f} N·m)")
            log(f"부하 스윕 {nstep}스텝 @ {rpm:.0f}rpm / {irms:.2f}Arms...")
            sw = sweep_load_with_fields(m2, style, rpm=rpm, I_rms=irms,
                                        gamma_deg=cal["gamma_max_deg"],
                                        n_steps=nstep, init_pos_deg=ini,
                                        steel_name=steel, magnet_name=magnet)
            r = compute_responses(sw, m2, R_ph_ohm=R_in)
            warn = "" if rph else " (MLT 계산)"
            log(f"── 응답 ──\n"
                f"T_avg       {r['T_avg']:.4f} N·m (가상일·정확, Maxwell 비교용)"
                f"  [Arkkio {r.get('T_avg_arkkio', r['T_avg']):.4f}]\n"
                f"ripple_pct  {r['T_ripple_pct']:.2f} %  "
                f"({r['T_ripple_pp']*1e3:.1f} mNm pp, Arkkio)\n"
                f"P_fe        {r['P_fe']:.2f} W "
                f"(히 {r['P_fe_stator']['P_hyst']:.1f} / "
                f"와 {r['P_fe_stator']['P_eddy']:.1f} / "
                f"과잉 {r['P_fe_stator']['P_excess']:.1f})\n"
                f"P_cu        {r['P_cu']:.2f} W "
                f"(R_ph {r['R_ph']*1e3:.1f} mΩ{warn})\n"
                f"efficiency  {r['efficiency']*100:.2f} % "
                f"(자석 와류손·기계손 미포함)")
            return r

        self._spawn(job, self.log_solve.appendPlainText, self._sweep_done)

    def _sweep_done(self, r):
        self.last_responses = r

    def _solve_done(self, out):
        s, res, met = out
        self.last_solve = out
        self.fig_field.clear()
        ax = self.fig_field.add_subplot()
        import matplotlib.tri as mtri
        V = s.V * 1e3
        tri = mtri.Triangulation(V[:, 0], V[:, 1], s.T)
        tp = ax.tripcolor(tri, facecolors=res.Bmag, cmap="jet",
                          vmin=0, vmax=2.2)
        ax.tricontour(tri, res.A,
                      levels=np.linspace(res.A.min(), res.A.max(), 25),
                      colors="k", linewidths=.3)
        self.fig_field.colorbar(tp, ax=ax, shrink=.8, label="|B| [T]")
        ax.set_aspect("equal")
        tnote = " (Arkkio·과대)" if met["mode"] == "부하" else ""
        ax.set_title(f"{met['mode']} — T={met['T_mNm']:.1f} mNm{tnote}, "
                     f"|B|max {met['B_max']:.2f} T")
        self.cv_field.draw()

    # ------------------------------------------------------- ④ Optimize
    def _tab_optimize(self):
        w = QWidget(); lay = QHBoxLayout(w)
        left = QVBoxLayout()
        grp = QGroupBox("DOE 생성 — 서로게이트 학습 데이터 (모델별 1회)")
        g = QGridLayout(grp)
        self.sp_ndoe = QDoubleSpinBox(); self.sp_ndoe.setRange(10, 300)
        self.sp_ndoe.setDecimals(0); self.sp_ndoe.setValue(60)
        g.addWidget(QLabel("설계 수 (권장 60+, 설계당 ~20초)"), 0, 0)
        g.addWidget(self.sp_ndoe, 1, 0)
        self.btn_doe = QPushButton("▶ DOE 생성 (전류는 Solve 탭 상전류 사용)")
        self.btn_doe.setObjectName("go")
        self.btn_doe.clicked.connect(self.run_doe_build)
        g.addWidget(self.btn_doe, 2, 0)
        left.addWidget(grp)
        self.btn_active = QPushButton("▶ 액티브러닝 1라운드 (DE→FEM 검증→재학습)")
        self.btn_active.setObjectName("primary")
        self.btn_active.clicked.connect(self.run_active_round)
        self.btn_sac = QPushButton("▶ SAC 정책으로 현재 설계 개선 (24스텝)")
        self.btn_sac.clicked.connect(self.run_sac_improve)
        left.addWidget(self.btn_active); left.addWidget(self.btn_sac)
        self.btn_relax = QPushButton(
            "🎯 목표가 안 잡힐 때: AI 권장 목표값으로 조정")
        self.btn_relax.setToolTip(
            "필수(🔒)는 고정하고, 나머지(소프트) 목표의 한계(L/U)를 이 모델 "
            "DOE 데이터의 달성 가능 범위로 바꿔 D>0 해가 존재하게 만듭니다. "
            "적용 전 변경값을 보여주고, 적용 여부는 직접 선택합니다.")
        self.btn_relax.clicked.connect(self.relax_to_recommended)
        left.addWidget(self.btn_relax)
        self.pb_opt = QProgressBar(); self.pb_opt.setRange(0, 100)
        self.pb_opt.setValue(0); self.pb_opt.setFormat("%p% — 대기")
        left.addWidget(self.pb_opt)
        self.log_opt = QPlainTextEdit(); self.log_opt.setReadOnly(True)
        left.addWidget(self.log_opt, 1)
        lw = QWidget(); lw.setLayout(left)
        right = QVBoxLayout()
        right.addWidget(QLabel("<b>FEM 검증된 후보</b>"))
        self.tbl_cand = QTableWidget(0, 8)
        self.tbl_cand.setHorizontalHeaderLabels(
            ["D", "T[mNm]", "EMF[V]", "자석[mm²]",
             "a_m", "T_m", "W_t", "MagnetR"])
        self.tbl_cand.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        right.addWidget(self.tbl_cand, 1)
        right.addWidget(QLabel("<b>탐색 중 형상 (실시간)</b> — 회색=기준 / 적색=현재 최적"))
        self.fig_opt = Figure(figsize=(4.2, 4.2), tight_layout=True)
        self.cv_opt = FigureCanvasQTAgg(self.fig_opt)
        right.addWidget(self.cv_opt, 2)
        rw = QWidget(); rw.setLayout(right)
        sp = QSplitter(); sp.addWidget(lw); sp.addWidget(rw)
        sp.setSizes([460, 790])
        lay.addWidget(sp)
        return w

    def _on_opt_update(self, payload):
        """탐색 진행 표시 — payload dict {x, info, frac}.
        frac → 진행바, x(설계변수) → 실시간 형상(회색=기준, 적색=현재)."""
        frac = payload.get("frac")
        if frac is not None:
            self.pb_opt.setValue(int(max(0, min(1, frac)) * 100))
            self.pb_opt.setFormat((payload.get("info") or "") + "  %p%")
        x = payload.get("x")
        if x is None or self.model is None:
            return
        info = payload.get("info", "")
        try:
            from motoropt.geometry import build_motor
            from motoropt.doe import vary
            geo = build_motor(vary(self.model, x), self.style)
        except Exception:
            return
        self.fig_opt.clear()
        ax = self.fig_opt.add_subplot()
        if self.geo is not None:                         # 기준 형상(회색 윤곽)
            for poly in (self.geo.stator, self.geo.rotor):
                self._outline(ax, poly, "#b0b0b0")
            for p, _, _ in self.geo.magnets:
                self._outline(ax, p, "#b0b0b0")
        for poly in (geo.stator, geo.rotor):             # 현재 설계(적색)
            self._outline(ax, poly, "#d02020")
        for p, _, _ in geo.magnets:
            self._outline(ax, p, "#d02020")
        for c in geo.coils:
            self._outline(ax, c, "#e08020")
        lim = geo.region_radius * 1.05
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_title(info or "탐색 중…", fontsize=9)
        self.cv_opt.draw()

    @staticmethod
    def _outline(ax, poly, color):
        from matplotlib.patches import Polygon as MplPoly
        if poly.geom_type == "MultiPolygon":
            for gmt in poly.geoms:
                MainWindow._outline(ax, gmt, color)
            return
        ax.add_patch(MplPoly(np.asarray(poly.exterior.coords), closed=True,
                             fill=False, edgecolor=color, lw=0.7))

    def run_doe_build(self):
        if self.aedt_path is None:
            self.log_opt.appendPlainText("⚠ 먼저 Model 탭에서 aedt를 여세요")
            return
        model, style = self.model, self.style
        n = int(self.sp_ndoe.value())
        irms, i_note = self._phase_current()
        if i_note:
            self.log_opt.appendPlainText(i_note)
        dataset, _ = self._dataset_paths()
        meta_path = dataset[:-6] + ".meta.json"          # .jsonl → .meta.json
        # efficiency·cogging이 목표에 있으면 DOE도 해당 평가 추가(느림)
        try:
            _spec = self._spec_from_table()
            want_eff = "efficiency" in _spec
            want_cog = "cogging_pp" in _spec
            want_cmin = "Pcu_W" in _spec
        except ValueError:
            want_eff = want_cog = want_cmin = False
        rpm = float(self.sp_rpm.value())
        d_cu = float(self.sp_dcu.value())
        strands = int(self.sp_strands.value())
        tcu = float(self.sp_tcu.value())
        rph = float(self.sp_rph.value()) * 1e-3 or None

        def job(log):
            from scipy.stats import qmc
            from motoropt.doe import (bounds_for_model, baseline_design,
                                      calibrate_delta, evaluate_design)
            from motoropt.aedt_parser import detect_material_names
            if irms <= 0:
                log("⚠ 상전류가 0 — Solve 탭에서 정격 상전류 입력 후 실행")
                return None
            v = model["variables"]
            steel, mag = detect_material_names(model)
            bounds = bounds_for_model(v)
            log("설계변수 범위: " + ", ".join(
                f"{k} {lo:g}~{hi:g}" for k, (lo, hi) in bounds.items()))
            log(f"δ 캘리브레이션 (4점, {irms:.2f} Arms)...")
            delta = calibrate_delta(model, style, I_rms=irms,
                                    steel_name=steel, magnet_name=mag)
            log(f"δ* = {delta:.1f}°e")
            json.dump({"I_rms": irms, "delta_e_deg": delta, "bounds": bounds,
                       "design": model.get("design_name")},
                      open(meta_path, "w", encoding="utf-8"))

            keys = list(bounds)
            lo = np.array([bounds[k][0] for k in keys])
            hi = np.array([bounds[k][1] for k in keys])
            X = qmc.LatinHypercube(d=len(keys), seed=7).random(n) * (hi - lo) + lo
            designs = [baseline_design(v)] + \
                      [dict(zip(keys, map(float, row))) for row in X]
            done = set()
            if os.path.exists(dataset):                  # 이어돌리기
                with open(dataset, encoding="utf-8") as f:
                    for line in f:
                        try:
                            done.add(tuple(round(val, 6) for val in
                                           json.loads(line)["x"].values()))
                        except Exception:
                            pass
            designs = [d for d in designs
                       if tuple(round(val, 6) for val in d.values())
                       not in done]
            per = (66 if want_eff else 20) + (24 if want_cog else 0)
            if want_eff:
                log("ℹ 효율 포함 평가 — 설계당 ~66초(부하스윕). DE 탐색이 "
                    "효율도 직접 최적화하게 됩니다.")
            if want_cog:
                log("ℹ 코깅 포함 평가 — 설계당 +~24초(무부하 1주기 가상일·FFT "
                    "저차 추출, 밴드노이즈 제거). a_m에 강하게 의존해 학습 가능.")
            log(f"평가할 설계 {len(designs)}개 (기존 {len(done)}개 스킵) — "
                f"예상 {len(designs)*per//60}분")
            t0 = time.time()
            n_ok = n_fail = 0
            N = len(designs)
            emit = self._cur_geom_emit
            with open(dataset, "a", encoding="utf-8") as f:
                for i, x in enumerate(designs):
                    r = evaluate_design(model, style, x, I_rms=irms,
                                        delta_e_deg=delta, steel_name=steel,
                                        magnet_name=mag,
                                        with_efficiency=want_eff,
                                        with_cogging=want_cog,
                                        with_current_min=want_cmin, rpm=rpm,
                                        d_cu_mm=d_cu, strands=strands,
                                        T_cu_C=tcu, R_ph_ohm=rph)
                    f.write(json.dumps(r) + "\n"); f.flush()
                    if r["status"] == "ok":
                        n_ok += 1
                    else:
                        n_fail += 1
                    el = time.time() - t0
                    eta = el / (i + 1) * (N - i - 1)
                    log(f"{i+1}/{N} {r['status'][:36]} | "
                        f"경과 {el/60:.1f}분 남음 {eta/60:.1f}분")
                    payload = {"frac": (i + 1) / N,
                               "info": f"DOE {i+1}/{N}  남음 {eta/60:.1f}분"}
                    if i % 2 == 0:           # 탐색 중인 형상도 가끔 표시
                        payload["x"] = x
                    emit(payload)
            log(f"✅ DOE 완료: 유효 {n_ok} / 실패 {n_fail} → "
                f"{os.path.basename(dataset)}\n이제 액티브러닝 1라운드를 "
                "실행하세요.")
            return n_ok                     # 새로 평가된 설계 수

        self._spawn(job, self.log_opt.appendPlainText, self._doe_done,
                    busy_btns=[self.btn_doe, self.btn_active, self.btn_sac],
                    geom_slot=self._on_opt_update, notify="DOE 생성 완료")

    def _doe_done(self, n_new):
        # 새 설계가 실제로 추가됐을 때만 목표값 자동충전 — 0개(이어돌리기·중복
        # 스킵)면 사용자가 직접 설정한 상/하한치를 덮어쓰지 않는다.
        if not n_new:
            return
        if self._autofill_spec_from_dataset():
            self.log_opt.appendPlainText(
                "ℹ Objective 목표값을 이 모델의 기준 설계 성능으로 갱신했습니다 "
                "(Objective 탭에서 확인·수정 가능)")

    def _load_meta(self):
        """DOE 메타(bounds·δ·전류) — 없으면 400W 레거시 기본값."""
        dataset, _ = self._dataset_paths()
        meta_path = dataset[:-6] + ".meta.json"
        if os.path.exists(meta_path):
            mt = json.load(open(meta_path, encoding="utf-8"))
            return ({k: tuple(b) for k, b in mt["bounds"].items()},
                    float(mt["delta_e_deg"]), float(mt["I_rms"]))
        from motoropt.doe import BOUNDS, DELTA_E_DEG
        return (dict(BOUNDS), DELTA_E_DEG,
                float(self.model["variables"].get("I_rms", 0.0) or 0.0))

    def _update_obj_row_state(self, i: int):
        """방향에 따라 타겟값(T) 칸 활성/비활성 — 최대화·최소화는 L·U만 사용,
        목표치는 L·T·U 모두 사용. 비활성 칸은 회색 처리."""
        cb = self.tbl_obj.cellWidget(i, 2)
        it = self.tbl_obj.item(i, 4)
        if cb is None or it is None:
            return
        from PyQt6.QtGui import QColor
        if cb.currentData() == "target":                 # 목표치 → 타겟값 활성
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsEditable
                        | Qt.ItemFlag.ItemIsEnabled)
            it.setBackground(QColor("#12233f"))          # 활성 = 푸른 입력색
            it.setForeground(QColor("#e3eaf6"))
            it.setToolTip("목표로 맞출 값")
        else:                                            # 최대화/최소화 → 비활성
            it.setText("")
            it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable
                        & ~Qt.ItemFlag.ItemIsEnabled)
            it.setBackground(QColor("#0a0f18"))          # 비활성 = 어둡게
            it.setToolTip("최대화·최소화에서는 사용 안 함")

    def _on_obj_item_changed(self, item):
        """사용자가 L/T/U 셀을 직접 고치면 그 목표를 '편집됨'으로 표시 —
        이후 자동충전이 덮어쓰지 않는다."""
        if self._obj_autofilling or item.column() not in (3, 4, 5):
            return
        nm = self.tbl_obj.item(item.row(), 1)
        if nm is not None and nm.data(Qt.ItemDataRole.UserRole):
            self._obj_user_edited.add(nm.data(Qt.ItemDataRole.UserRole))

    def _on_obj_dir_changed(self, r):
        self._update_obj_row_state(r)
        if self._obj_autofilling:
            return
        nm = self.tbl_obj.item(r, 1)               # 방향 변경도 사용자 편집
        if nm is not None and nm.data(Qt.ItemDataRole.UserRole):
            self._obj_user_edited.add(nm.data(Qt.ItemDataRole.UserRole))

    def _set_obj_row(self, key: str, spec: tuple):
        if key in self._obj_user_edited:           # 사용자 설정 보존
            return
        for i in range(self.tbl_obj.rowCount()):
            if self.tbl_obj.item(i, 1).data(Qt.ItemDataRole.UserRole) != key:
                continue
            self._obj_autofilling = True           # 프로그램적 갱신(편집표시 안 함)
            try:
                cb = self.tbl_obj.cellWidget(i, 2)
                cb.setCurrentIndex([e for e, _ in TYPE_KO].index(spec[0]))
                vals = ({3: spec[1], 4: spec[2], 5: spec[3]}
                        if spec[0] == "target" else {3: spec[1], 5: spec[2]})
                for c in (3, 4, 5):
                    self.tbl_obj.setItem(i, c, QTableWidgetItem(
                        f"{vals[c]:.4g}" if c in vals else ""))
                self._update_obj_row_state(i)
            finally:
                self._obj_autofilling = False
            return

    def _reset_spec_defaults(self):
        from motoropt.objective import SPEC, SPEC_EXTRA
        for k, s in {**SPEC, **SPEC_EXTRA}.items():
            self._set_obj_row(k, s)

    def _autofill_spec_from_dataset(self) -> bool:
        """DOE 기준 설계(첫 샘플) 성능 → Objective 테이블 L/T/U 자동 설정.

        메타파일이 있는(=GUI DOE로 만든) 모델만 — 400W의 수동 튜닝값은 유지.
        T_avg: 기준 유지(L=기준, U=+3%) / emf_rms: 기준±5% /
        magnet_area: 절감(L=-26%, U=기준)."""
        if self.model is None:
            return False
        dataset, _ = self._dataset_paths()
        meta_path = dataset[:-6] + ".meta.json"
        if not (os.path.exists(meta_path) and os.path.exists(dataset)):
            return False
        from motoropt.doe import baseline_design
        base = {k: round(v_, 6) for k, v_
                in baseline_design(self.model["variables"]).items()}
        oks, row = [], None
        with open(dataset, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("status") != "ok":
                    continue
                oks.append(r)
                if {k: round(v_, 6) for k, v_ in r["x"].items()} == base:
                    row = r
        if not oks:
            return False
        row = row or oks[0]                        # 기준 없으면 첫 유효 행

        # 모든 목표 상하한을 이 모델의 DOE 실제 범위(최소~최대)로 설정한다.
        # 이래야 만족도가 0~1로 고르게 분포해 DE가 경사를 따라가고, 어떤
        # aedt를 넣어도 D가 0으로 붕괴하지 않는다(기준값 기반은 기준이 이미
        # 최적이면 '기준 넘어라'가 불가능→D=0 됨 — KRO80 사례).
        def rng(key):
            vals = [r[key] for r in oks if key in r and r[key] is not None]
            if len(vals) < 3:
                return None
            lo, hi = float(min(vals)), float(max(vals))
            if hi - lo < abs(lo) * 1e-3 + 1e-9:    # 거의 동일하면 ±2% 여유
                lo, hi = lo * 0.98 - 1e-6, hi * 1.02 + 1e-6
            return lo, hi

        if (r0 := rng("T_avg")):
            self._set_obj_row("T_avg", ("larger", r0[0], r0[1]))
        if (r0 := rng("cogging_pp")):
            self._set_obj_row("cogging_pp", ("smaller", r0[0], r0[1]))
        return True

    def run_active_round(self):
        if self.aedt_path is None:
            self.log_opt.appendPlainText("⚠ 먼저 Model 탭에서 aedt를 여세요")
            return
        model, style = self.model, self.style
        try:
            spec = self._spec_from_table()
        except ValueError as e:
            self.log_opt.appendPlainText(f"⚠ {e}")
            return
        hard = self._hard_keys_from_table()        # 필수(하드) 제약 키
        self._active_round += 1
        rnd = self._active_round
        self.btn_active.setText(
            f"▶ 액티브러닝 {rnd}라운드 (DE→FEM 검증→재학습)")
        dataset, surro = self._dataset_paths()
        bounds, delta, irms = self._load_meta()
        # 효율이 목표에 있으면 부하 스윕까지 평가 — Solve 탭 운전조건 사용
        want_eff = "efficiency" in spec
        want_cog = "cogging_pp" in spec
        want_cmin = "Pcu_W" in spec
        rpm = float(self.sp_rpm.value())
        d_cu = float(self.sp_dcu.value())
        strands = int(self.sp_strands.value())
        tcu = float(self.sp_tcu.value())
        rph = float(self.sp_rph.value()) * 1e-3 or None     # mΩ→Ω, 0=MLT

        def job(log):
            from scipy.optimize import differential_evolution
            from motoropt.doe import evaluate_design
            from motoropt.surrogate import (load_dataset, train_surrogate,
                                            save, X_KEYS)
            from motoropt.objective import (SurrogateObjective,
                                            desirability_from_dict)
            if not os.path.exists(dataset):
                log(f"⚠ 이 모델용 DOE 데이터셋이 없습니다: "
                    f"{os.path.basename(dataset)}\n"
                    "서로게이트 최적화는 모델별 DOE(P5, scripts/run_p5_doe.py)가 "
                    "선행되어야 합니다 — 현재 400W 모델만 데이터 보유.")
                return None
            log(f"━━━━━━━━━━ 액티브러닝 {rnd}라운드 ━━━━━━━━━━")
            log(f"서로게이트 재학습... ({os.path.basename(dataset)})")
            X, Y, ykeys = load_dataset(dataset)
            if len(X) < 20:
                log(f"⚠ 유효 샘플 {len(X)}개 — 너무 적습니다. "
                    "DOE 생성으로 60개 이상 확보 권장.")
                return None
            skipped = [k for k in spec if k not in ykeys]
            if skipped:
                log(f"ℹ DE 탐색 제외(FEM 검증에서만 평가·반영): "
                    f"{', '.join(skipped)}")
            if hard:
                log(f"🔒 하드 제약(반드시 만족): "
                    f"{', '.join(resp_label(k) for k in hard)}")
            mdl, sc, met, _ = train_surrogate(X, Y, y_keys=ykeys)
            save(mdl, sc, surro, y_keys=ykeys)
            if "efficiency" in ykeys:
                log("ℹ 데이터셋에 efficiency 포함 → DE 탐색이 효율도 직접 최적화")
            t0 = time.time()
            emit = self._cur_geom_emit
            emit({"frac": 0.10, "info": "서로게이트 학습 완료 → DE 탐색"})
            obj = SurrogateObjective(surro, bounds, spec=spec, hard_keys=hard)
            log(f"DE 최적화 (샘플 {len(X)}, δ*={delta:.1f}°, "
                f"I={irms:.2f}A)...")
            maxit = 250
            st = {"g": 0}

            def cb(xk, convergence=None):            # 세대마다 진행·형상 표시
                st["g"] += 1
                gi = st["g"]
                frac = 0.10 + 0.65 * min(gi / maxit, 1.0)
                xd_i = dict(zip(X_KEYS, map(float, obj.x_of(xk))))
                emit({"x": xd_i, "frac": frac,
                      "info": f"{rnd}R · DE {gi}세대  "
                              f"D={float(obj.D(xk)[0]):.3f}"})
                # DE는 서로게이트(빠름)라 수 초에 수렴 → 형상 모핑이 안 보임.
                # 시각화를 위해 세대마다 살짝 늦춰 변형 과정을 눈으로 보이게 함.
                time.sleep(0.05)

            r = differential_evolution(lambda u: -obj.D(u)[0],
                                       [(0, 1)] * 5, seed=0,
                                       maxiter=maxit, tol=1e-8, callback=cb)
            xd = dict(zip(X_KEYS, map(float, obj.x_of(r.x))))
            eta_note = " + 효율 부하스윕(~+60초)" if want_eff else ""
            emit({"x": xd, "frac": 0.80,
                  "info": f"DE 완료 D={-r.fun:.3f} → FEM 검증 중"})
            log(f"서로게이트 D={-r.fun:.4f} → FEM 검증 중 (~30초{eta_note})...")
            fem = evaluate_design(model, style, xd, I_rms=irms or None,
                                  delta_e_deg=delta, with_efficiency=want_eff,
                                  with_cogging=want_cog,
                                  with_current_min=want_cmin,
                                  rpm=rpm, d_cu_mm=d_cu, strands=strands,
                                  T_cu_C=tcu, R_ph_ohm=rph)
            with open(dataset, "a", encoding="utf-8") as f:
                f.write(json.dumps(fem) + "\n")
            if fem["status"] != "ok":
                emit({"frac": 1.0, "info": "FEM 실패"})
                log(f"FEM 실패: {fem['status'][:40]}")
                return None
            D = desirability_from_dict(fem, spec, hard_keys=hard)   # 검증 D
            emit({"x": xd, "frac": 1.0, "info": f"{rnd}R 완료 D={D:.3f}"})
            msg = (f"FEM D={D:.4f} | T={fem['T_avg']:.1f} "
                   f"EMF={fem['emf_rms']:.3f} A={fem['magnet_area']:.1f}")
            if "efficiency" in fem:
                msg += (f" η={fem['efficiency']*100:.1f}% "
                        f"(P_fe {fem['P_fe']:.1f} P_cu {fem['P_cu']:.1f}W)")
            log(msg)
            for wmsg in diagnose_result(fem, spec, D, surro_D=-r.fun,
                                        hard_keys=hard):
                log(wmsg)                          # 자동 진단·경고
            log(f"✅ 액티브러닝 {rnd}라운드 완료 (소요 {(time.time()-t0)/60:.1f}분)")
            return (D, xd, fem)

        self._spawn(job, self.log_opt.appendPlainText, self._cand_done,
                    busy_btns=[self.btn_doe, self.btn_active, self.btn_sac],
                    geom_slot=self._on_opt_update, notify="액티브러닝 완료")

    def run_sac_improve(self):
        if self.model is None:
            self.log_opt.appendPlainText("⚠ 먼저 Model 탭에서 aedt를 여세요")
            return
        v = self.model["variables"]
        x0 = {"a_m": v["a_m"], "T_m": v["T_m"] * 1e3,
              "T_m2_ratio": v["T_m2"] / v["T_m"],
              "W_t": v["W_t"] * 1e3, "MagnetR": v["MagnetR"] * 1e3}

        dataset, surro = self._dataset_paths()
        bounds, _, _ = self._load_meta()

        def job(log):
            try:
                import torch
            except ImportError:
                log("⚠ torch 미설치 — venv에서 'pip install torch' 후 사용 가능")
                return None
            if not os.path.exists(surro):
                log("⚠ 서로게이트 없음 — 액티브러닝 1라운드를 먼저 실행하세요")
                return None
            from motoropt.objective import SurrogateObjective
            from motoropt.rl_opt import DesignEnv, SAC
            obj = SurrogateObjective(surro, bounds)
            env = DesignEnv(obj)
            env.reset()
            env.u = np.array([(x0[k] - bounds[k][0])
                              / (bounds[k][1] - bounds[k][0])
                              for k in obj.keys]).clip(0, 1)
            env.D = float(obj.D(env.u)[0])
            log(f"시작 D={env.D:.4f}")
            agent = SAC(env.dim + 1, env.dim)
            actor_pt = os.path.join(_ROOT, "sac_actor.pt")
            if os.path.exists(actor_pt):
                agent.actor.load_state_dict(torch.load(actor_pt))
                log("학습된 SAC 정책 로드")
            else:
                log("⚠ 학습된 정책(sac_actor.pt) 없음 — 무작위 초기 정책으로 "
                    "동작해 개선 효과가 없습니다. SAC 학습(P6) 후 사용 권장, "
                    "단발 최적화는 액티브러닝 버튼이 더 정확합니다.")
            emit = self._cur_geom_emit
            s = env._obs()
            for t in range(env.h):
                s, _, _ = env.step(agent.act(s, deterministic=True))
                xd_i = dict(zip(obj.keys, map(float, obj.x_of(env.u))))
                emit({"x": xd_i, "frac": (t + 1) / env.h,
                      "info": f"SAC {t+1}/{env.h}스텝  D={env.D:.3f}"})
            xd = dict(zip(obj.keys, map(float, obj.x_of(env.u))))
            log(f"✅ SAC 개선 완료: D={env.D:.4f} → {xd}")
            return None

        self._spawn(job, self.log_opt.appendPlainText, lambda *_: None,
                    busy_btns=[self.btn_doe, self.btn_active, self.btn_sac],
                    geom_slot=self._on_opt_update, notify="SAC 개선 완료")

    def _cand_done(self, out):
        if out is None:
            return
        D, xd, fem = out
        self.candidates.append(out)
        self.candidates.sort(key=lambda c: -c[0])
        self.tbl_cand.setRowCount(len(self.candidates))
        for i, (Di, xi, fi) in enumerate(self.candidates):
            vals = [f"{Di:.4f}", f"{fi['T_avg']:.1f}",
                    f"{fi['emf_rms']:.3f}", f"{fi['magnet_area']:.1f}",
                    f"{xi['a_m']:.4f}", f"{xi['T_m']:.3f}",
                    f"{xi['W_t']:.3f}", f"{xi['MagnetR']:.3f}"]
            for j, t in enumerate(vals):
                self.tbl_cand.setItem(i, j, QTableWidgetItem(t))
        best = self.candidates[0][0]
        self.log_opt.appendPlainText(
            f"📊 누적 후보 {len(self.candidates)}개 · 현재 최고 D={best:.4f} "
            f"(이 최고값이 Result·내보내기에 쓰임 — 라운드가 나빠도 유지됨)")
        self._update_result()

    # --------------------------------------------------------- ⑤ Result
    def _tab_result(self):
        w = QWidget(); lay = QHBoxLayout(w)
        left = QVBoxLayout()
        left.addWidget(QLabel("<b>기준 vs 최적 (FEM 검증값)</b>"))
        self.tbl_res = QTableWidget(0, 3)
        self.tbl_res.setHorizontalHeaderLabels(["항목", "기준", "최적"])
        self.tbl_res.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        left.addWidget(self.tbl_res, 1)
        self.lbl_res_warn = QLabel()               # 자동 진단 경고
        self.lbl_res_warn.setWordWrap(True)
        self.lbl_res_warn.setStyleSheet("color:#f0b030;")
        left.addWidget(self.lbl_res_warn)
        btn = QPushButton("💾 최적 설계 .aedt 내보내기")
        btn.setObjectName("primary")
        btn.clicked.connect(self.export_best)
        left.addWidget(btn)
        lw = QWidget(); lw.setLayout(left)
        self.fig_res = Figure(figsize=(5.2, 5), tight_layout=True)
        self.cv_res = FigureCanvasQTAgg(self.fig_res)
        sp = QSplitter(); sp.addWidget(lw); sp.addWidget(self.cv_res)
        sp.setSizes([480, 770])
        lay.addWidget(sp)
        return w

    def _baseline_fem_row(self):
        """DOE 기준 설계의 FEM 응답 행 (없으면 None) — Result 기준값 소스."""
        if self.model is None:
            return None
        dataset, _ = self._dataset_paths()
        if not os.path.exists(dataset):
            return None
        from motoropt.doe import baseline_design
        base = {k: round(v_, 6) for k, v_
                in baseline_design(self.model["variables"]).items()}
        row = None
        with open(dataset, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("status") != "ok":
                    continue
                if {k: round(v_, 6) for k, v_ in r["x"].items()} == base:
                    return r
                row = row or r                     # 기준 못 찾으면 첫 유효 행
        return row

    def _update_result(self):
        if not self.candidates or self.geo is None:
            return
        from motoropt.objective import _D_FUNCS, _hard_pass
        D, xd, fem = self.candidates[0]
        base_area = sum(p.area for p, _, _ in self.geo.magnets)
        try:
            spec = self._spec_from_table()      # 사용자가 선택한 목표 항목
            hard = self._hard_keys_from_table()
        except ValueError:
            spec, hard = {}, set()
        base_row = self._baseline_fem_row()

        def base_of(key):                        # 항목별 기준값
            if key == "magnet_area":
                return base_area
            if base_row and key in base_row:
                return base_row[key]
            return None

        rows = [("종합 만족도 D", "—", f"{D:.4f}")]
        for key, s in spec.items():
            name = resp_label(key)
            b = base_of(key)
            b_txt = f"{b:.4g}" if b is not None else "—"
            if key in fem:                       # FEM 검증된 응답
                opt = fem[key]
                pct = f", {(opt / b - 1) * 100:+.1f}%" if b else ""
                if key in hard:                  # 🔒 필수 = 통과/위반
                    ok = float(_hard_pass([opt], s[0], *s[1:])[0]) >= 0.5
                    opt_txt = (f"{opt:.4g}  (🔒필수 "
                               f"{'✓ 만족' if ok else '✗ 위반'}{pct})")
                else:
                    d = float(_D_FUNCS[s[0]](np.array([opt], float),
                                             *s[1:])[0])
                    opt_txt = f"{opt:.4g}  (만족도 {d:.2f}{pct})"
            else:                                # efficiency 등 서로게이트 외
                opt_txt = "— (Solve 탭 부하 스윕으로 평가)"
            rows.append((name, b_txt, opt_txt))
        self.tbl_res.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, t in enumerate(r):
                self.tbl_res.setItem(i, j, QTableWidgetItem(t))
        warns = diagnose_result(fem, spec, D, hard_keys=hard)   # 자동 진단
        self.lbl_res_warn.setText("\n".join(warns) if warns
                                  else "✓ 자동 점검 통과 — 명백한 이상 없음 "
                                       "(절대값 최종확인은 Maxwell 권장)")
        self.lbl_res_warn.setStyleSheet(               # 경고=호박 / 통과=초록
            "color:#f0b030;" if warns else "color:#3fcf7a;")
        # 오버레이
        from motoropt.doe import vary
        from motoropt.geometry import build_motor
        geo1 = build_motor(vary(self.model, xd), self.style)
        self.fig_res.clear()
        ax = self.fig_res.add_subplot()
        for geo, c, lw in ((self.geo, "#555555", 1.6), (geo1, "#d62728", 1.1)):
            for p, _, _ in geo.magnets:
                xs, ys = p.exterior.xy
                ax.plot(xs, ys, color=c, lw=lw)
            for ring in [geo.stator.exterior] + list(geo.stator.interiors):
                arr = np.asarray(ring.coords)
                ax.plot(arr[:, 0], arr[:, 1], color=c, lw=lw * 0.6, alpha=.8)
        lim = self.geo.region_radius * 1.02
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title("기준(회색) vs 최적(적색)")
        self.cv_res.draw()

    def export_best(self):
        if not self.candidates or self.aedt_path is None:
            return
        from motoropt.aedt_export import export_aedt, overrides_from_design
        _, xd, _ = self.candidates[0]
        dst, _ = QFileDialog.getSaveFileName(
            self, "내보내기", "Optimized.aedt", "AEDT (*.aedt)")
        if not dst:
            return
        export_aedt(self.aedt_path, dst, overrides_from_design(xd))
        self.statusBar().showMessage(f"내보내기 완료: {dst}")

    # ----------------------------------------------------------- 공용
    _DESIGN_400W = "4. 400W_BasicModel_Load_Optimized"

    def _dataset_paths(self):
        """모델별·운전전류별 DOE 데이터셋·서로게이트 경로 (루트 기준).

        실행 위치(cwd)와 무관해야 하고 ① 모델이 다르면 설계명으로,
        ② 운전점이 다르면 상전류로 파일을 분리한다 — 다른 aedt나 다른
        전류(피크/정격)의 결과가 절대 섞이지 않도록. 400W는 레거시 유지."""
        design = (self.model or {}).get("design_name", "") or "unknown"
        if design == self._DESIGN_400W:
            return (os.path.join(_ROOT, "doe_results.jsonl"),
                    os.path.join(_ROOT, "surrogate.joblib"))
        tag = re.sub(r"[^\w]+", "_", design).strip("_")
        iph = self._phase_current()[0] if hasattr(self, "sp_irms") else 0.0
        cur = f"_{iph:.1f}A" if iph > 0 else ""
        return (os.path.join(_ROOT, f"doe_{tag}{cur}.jsonl"),
                os.path.join(_ROOT, f"surrogate_{tag}{cur}.joblib"))

    def _spawn(self, job, log_slot, done_slot, busy_btns=None,
               geom_slot=None, notify=None):
        wk = Worker(job)
        wk.log.connect(log_slot)
        wk.done.connect(done_slot)
        wk.failed.connect(lambda e: log_slot("오류:\n" + e))
        if geom_slot is not None:
            wk.geom.connect(geom_slot)
        self._cur_geom_emit = wk.geom.emit       # 잡이 형상을 보낼 통로
        btns = list(busy_btns or [])
        labels = [(b, b.text()) for b in btns]
        for b in btns:
            b.setEnabled(False)
            b.setText("⏳ 실행 중…")

        def _finish(ok):
            for b, t in labels:
                b.setEnabled(True); b.setText(t)
            if notify:
                self.statusBar().showMessage(
                    ("✅ " if ok else "⚠ ") + notify, 8000)
            try:
                QApplication.beep()
            except Exception:
                pass
        wk.done.connect(lambda *_: _finish(True))
        wk.failed.connect(lambda *_: _finish(False))
        wk.finished.connect(lambda: self._workers.remove(wk))
        self._workers.append(wk)
        wk.start()


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_QSS)               # 다크 엔지니어링 테마
    # PyQt6는 미처리 예외 시 qFatal(abort) — 다이얼로그로 대체해 앱 유지
    def hook(tp, val, tb):
        traceback.print_exception(tp, val, tb)
        try:
            _error_dialog(None, "내부 오류", val)
        except Exception:
            pass
    sys.excepthook = hook
    win = MainWindow()
    if len(sys.argv) > 1 and sys.argv[1].endswith(".aedt"):
        win.open_aedt(sys.argv[1])
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
