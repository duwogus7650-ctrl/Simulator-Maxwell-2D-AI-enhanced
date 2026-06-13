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
    QPlainTextEdit, QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
    QTabWidget, QVBoxLayout, QWidget)

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
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

DESIGN_VARS = ["a_m", "T_m", "T_m2", "W_t", "MagnetR"]

OBJ_UNITS = {"T_avg": "mNm", "emf_rms": "V", "magnet_area": "mm²",
             "ripple_pct": "%", "efficiency": "0~1"}


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
        self._workers = []

        tabs = QTabWidget()
        self.setCentralWidget(tabs)
        tabs.addTab(self._tab_model(), "① Model")
        tabs.addTab(self._tab_objective(), "② Objective")
        tabs.addTab(self._tab_solve(), "③ Solve")
        tabs.addTab(self._tab_optimize(), "④ Optimize")
        tabs.addTab(self._tab_result(), "⑤ Result")
        self.tabs = tabs
        self.statusBar().showMessage("aedt 파일을 열어 시작하세요")

    # ---------------------------------------------------------- ① Model
    def _tab_model(self):
        w = QWidget(); lay = QHBoxLayout(w)
        left = QVBoxLayout()
        btn = QPushButton("📂 .aedt 열기")
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
        if not self._autofill_spec_from_dataset():
            self._reset_spec_defaults()        # 메타 없으면 기본값 복원
        self._refresh_geometry()
        self.statusBar().showMessage(f"{os.path.basename(path)} 로드 완료")

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
            "<b>목표 특성 (Derringer-Suich 만족도)</b> — "
            "유형: larger(↑)/smaller(↓)/target(목표값)"))
        from motoropt.objective import SPEC, SPEC_EXTRA
        rows = [(k, s, True) for k, s in SPEC.items()] + \
               [(k, s, False) for k, s in SPEC_EXTRA.items()]
        self.tbl_obj = QTableWidget(len(rows), 6)
        self.tbl_obj.setHorizontalHeaderLabels(
            ["사용", "응답", "유형", "L (하한)", "T (목표)", "U (상한)"])
        self.tbl_obj.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        for i, (k, spec, on) in enumerate(rows):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable |
                         Qt.ItemFlag.ItemIsEnabled)
            chk.setCheckState(Qt.CheckState.Checked if on
                              else Qt.CheckState.Unchecked)
            self.tbl_obj.setItem(i, 0, chk)
            name = QTableWidgetItem(
                f"{k} [{OBJ_UNITS[k]}]" if k in OBJ_UNITS else k)
            name.setFlags(name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.tbl_obj.setItem(i, 1, name)
            cb = QComboBox(); cb.addItems(["larger", "smaller", "target"])
            cb.setCurrentText(spec[0])
            cb.currentTextChanged.connect(
                lambda _t, r=i: self._update_obj_row_state(r))
            self.tbl_obj.setCellWidget(i, 2, cb)
            if spec[0] == "target":
                vals = {3: spec[1], 4: spec[2], 5: spec[3]}
            else:
                vals = {3: spec[1], 5: spec[2]}
            for c in (3, 4, 5):
                self.tbl_obj.setItem(
                    i, c, QTableWidgetItem(
                        f"{vals[c]:.4g}" if c in vals else ""))
            self._update_obj_row_state(i)
        lay.addWidget(self.tbl_obj)
        lay.addWidget(QLabel(
            "종합 만족도 D = (∏ dᵢ)^(1/n) — 모든 목표를 동시에 만족할수록 "
            "1에 가까움.<br>efficiency·ripple_pct는 Solve 탭 ▶부하 스윕에서 "
            "FEM으로 평가됩니다 (efficiency는 서로게이트 응답에 없어 "
            "Optimize의 DE 탐색에는 미참여)."))
        lay.addStretch(1)
        return w

    def _spec_from_table(self) -> dict:
        """Objective 테이블의 체크된 행 → SPEC dict."""
        spec = {}
        for i in range(self.tbl_obj.rowCount()):
            if self.tbl_obj.item(i, 0).checkState() != Qt.CheckState.Checked:
                continue
            k = _obj_key(self.tbl_obj.item(i, 1).text())
            typ = self.tbl_obj.cellWidget(i, 2).currentText()
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

    # ---------------------------------------------------------- ③ Solve
    def _tab_solve(self):
        w = QWidget(); lay = QHBoxLayout(w)
        left = QVBoxLayout()
        b1 = QPushButton("▶ 무부하 해석 (코깅·EMF용 단일 포지션)")
        b1.clicked.connect(lambda: self.run_solve(load=False))
        b2 = QPushButton("▶ 부하 해석 (입력 전류·MTPA)")
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
                 ("권선온도 [°C]", self.sp_tcu)]):
            g.addWidget(QLabel(lbl), 2, col)
            g.addWidget(w_, 3, col)
        self.lbl_iconv = QLabel()
        g.addWidget(self.lbl_iconv, 4, 0, 1, 4)
        self.sp_irms.valueChanged.connect(self._update_iconv)
        self.cb_ibase.currentIndexChanged.connect(self._update_iconv)
        self._update_iconv()
        b3 = QPushButton("▶ 부하 스윕 실행 (γ 캘리브레이션 포함 — 수 분 소요)")
        b3.clicked.connect(self.run_load_sweep)
        g.addWidget(b3, 5, 0, 1, 4)
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
            log(f"재질: 강판={steel} / 자석={magnet}")
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
                f"|B|max {met['B_max']:.2f} T | T {met['T_mNm']:.1f} mNm")
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
            log(f"재질: 강판={steel} / 자석={magnet}")
            R_in = rph
            if R_in is None:
                from motoropt.winding import phase_resistance
                w = phase_resistance(v, d_cu_mm=d_cu, strands=strands,
                                     T_cu_C=tcu)
                R_in = w["R_ph"]
                log(f"R_ph(MLT 계산) = {R_in*1e3:.1f} mΩ "
                    f"(MLT {w['MLT_mm']:.1f} mm, 직렬 {w['n_series']:.0f}턴, "
                    f"도체 {w['turn_csa_mm2']:.3f} mm², {tcu:.0f}°C)")
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
                f"T_avg       {r['T_avg']:.4f} N·m\n"
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
        ax.set_title(f"{met['mode']} — T={met['T_mNm']:.1f} mNm, "
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
        b0 = QPushButton("▶ DOE 생성 (전류는 Solve 탭 상전류 사용)")
        b0.clicked.connect(self.run_doe_build)
        g.addWidget(b0, 2, 0)
        left.addWidget(grp)
        b1 = QPushButton("▶ 액티브러닝 1라운드 (DE→FEM 검증→재학습)")
        b1.clicked.connect(self.run_active_round)
        b2 = QPushButton("▶ SAC 정책으로 현재 설계 개선 (24스텝)")
        b2.clicked.connect(self.run_sac_improve)
        left.addWidget(b1); left.addWidget(b2)
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
        rw = QWidget(); rw.setLayout(right)
        sp = QSplitter(); sp.addWidget(lw); sp.addWidget(rw)
        sp.setSizes([480, 770])
        lay.addWidget(sp)
        return w

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
        # efficiency가 목표에 있으면 DOE도 효율 포함 평가(느림) — Solve 운전조건
        try:
            want_eff = "efficiency" in self._spec_from_table()
        except ValueError:
            want_eff = False
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
            per = 90 if want_eff else 20
            if want_eff:
                log("ℹ 효율 포함 평가 — 설계당 ~90초(부하스윕). DE 탐색이 "
                    "효율도 직접 최적화하게 됩니다.")
            log(f"평가할 설계 {len(designs)}개 (기존 {len(done)}개 스킵) — "
                f"예상 {len(designs)*per//60}분")
            t0 = time.time()
            n_ok = n_fail = 0
            with open(dataset, "a", encoding="utf-8") as f:
                for i, x in enumerate(designs):
                    r = evaluate_design(model, style, x, I_rms=irms,
                                        delta_e_deg=delta, steel_name=steel,
                                        magnet_name=mag,
                                        with_efficiency=want_eff, rpm=rpm,
                                        d_cu_mm=d_cu, strands=strands,
                                        T_cu_C=tcu, R_ph_ohm=rph)
                    f.write(json.dumps(r) + "\n"); f.flush()
                    if r["status"] == "ok":
                        n_ok += 1
                    else:
                        n_fail += 1
                    el = time.time() - t0
                    eta = el / (i + 1) * (len(designs) - i - 1)
                    log(f"{i+1}/{len(designs)} {r['status'][:36]} | "
                        f"경과 {el/60:.1f}분 남음 {eta/60:.1f}분")
            log(f"✅ DOE 완료: 유효 {n_ok} / 실패 {n_fail} → "
                f"{os.path.basename(dataset)}\n이제 액티브러닝 1라운드를 "
                "실행하세요.")
            return None

        self._spawn(job, self.log_opt.appendPlainText, self._doe_done)

    def _doe_done(self, _):
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
        """유형에 따라 T(목표) 칸 활성/비활성 — larger/smaller는 L·U만 사용."""
        cb = self.tbl_obj.cellWidget(i, 2)
        it = self.tbl_obj.item(i, 4)
        if cb is None or it is None:
            return
        if cb.currentText() == "target":
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsEditable
                        | Qt.ItemFlag.ItemIsEnabled)
        else:
            it.setText("")
            it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable
                        & ~Qt.ItemFlag.ItemIsEnabled)

    def _set_obj_row(self, key: str, spec: tuple):
        for i in range(self.tbl_obj.rowCount()):
            if _obj_key(self.tbl_obj.item(i, 1).text()) != key:
                continue
            self.tbl_obj.cellWidget(i, 2).setCurrentText(spec[0])
            vals = ({3: spec[1], 4: spec[2], 5: spec[3]}
                    if spec[0] == "target" else {3: spec[1], 5: spec[2]})
            for c in (3, 4, 5):
                self.tbl_obj.setItem(i, c, QTableWidgetItem(
                    f"{vals[c]:.4g}" if c in vals else ""))
            self._update_obj_row_state(i)
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
                    row = r
                    break
                row = row or r                     # 기준 없으면 첫 유효 행
        if row is None:
            return False
        T0, E0, A0 = row["T_avg"], row["emf_rms"], row["magnet_area"]
        self._set_obj_row("T_avg", ("larger", T0, T0 * 1.03))
        self._set_obj_row("emf_rms", ("target", E0 * 0.95, E0, E0 * 1.05))
        self._set_obj_row("magnet_area", ("smaller", A0 * 0.74, A0))
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
        dataset, surro = self._dataset_paths()
        bounds, delta, irms = self._load_meta()
        # 효율이 목표에 있으면 부하 스윕까지 평가 — Solve 탭 운전조건 사용
        want_eff = "efficiency" in spec
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
            mdl, sc, met, _ = train_surrogate(X, Y, y_keys=ykeys)
            save(mdl, sc, surro, y_keys=ykeys)
            if "efficiency" in ykeys:
                log("ℹ 데이터셋에 efficiency 포함 → DE 탐색이 효율도 직접 최적화")
            obj = SurrogateObjective(surro, bounds, spec=spec)
            log(f"DE 최적화 (샘플 {len(X)}, δ*={delta:.1f}°, "
                f"I={irms:.2f}A)...")
            r = differential_evolution(lambda u: -obj.D(u)[0],
                                       [(0, 1)] * 5, seed=0,
                                       maxiter=250, tol=1e-8)
            xd = dict(zip(X_KEYS, map(float, obj.x_of(r.x))))
            eta_note = " + 효율 부하스윕(~+60초)" if want_eff else ""
            log(f"서로게이트 D={-r.fun:.4f} → FEM 검증 중 (~30초{eta_note})...")
            fem = evaluate_design(model, style, xd, I_rms=irms or None,
                                  delta_e_deg=delta, with_efficiency=want_eff,
                                  rpm=rpm, d_cu_mm=d_cu, strands=strands,
                                  T_cu_C=tcu, R_ph_ohm=rph)
            with open(dataset, "a") as f:
                f.write(json.dumps(fem) + "\n")
            if fem["status"] != "ok":
                log(f"FEM 실패: {fem['status'][:40]}")
                return None
            D = desirability_from_dict(fem, spec)   # efficiency 포함 검증 D
            msg = (f"FEM D={D:.4f} | T={fem['T_avg']:.1f} "
                   f"EMF={fem['emf_rms']:.3f} A={fem['magnet_area']:.1f}")
            if "efficiency" in fem:
                msg += (f" η={fem['efficiency']*100:.1f}% "
                        f"(P_fe {fem['P_fe']:.1f} P_cu {fem['P_cu']:.1f}W)")
            log(msg)
            return (D, xd, fem)

        self._spawn(job, self.log_opt.appendPlainText, self._cand_done)

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
            s = env._obs()
            for t in range(env.h):
                s, _, _ = env.step(agent.act(s, deterministic=True))
            xd = dict(zip(obj.keys, map(float, obj.x_of(env.u))))
            log(f"개선 후 D={env.D:.4f} → {xd}")
            return None

        self._spawn(job, self.log_opt.appendPlainText, lambda *_: None)

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
        btn = QPushButton("💾 최적 설계 .aedt 내보내기")
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
        from motoropt.objective import _D_FUNCS
        D, xd, fem = self.candidates[0]
        base_area = sum(p.area for p, _, _ in self.geo.magnets)
        try:
            spec = self._spec_from_table()      # 사용자가 선택한 목표 항목
        except ValueError:
            spec = {}
        base_row = self._baseline_fem_row()

        def base_of(key):                        # 항목별 기준값
            if key == "magnet_area":
                return base_area
            if base_row and key in base_row:
                return base_row[key]
            return None

        rows = [("종합 만족도 D", "—", f"{D:.4f}")]
        for key, s in spec.items():
            unit = OBJ_UNITS.get(key, "")
            name = f"{key} [{unit}]" if unit else key
            b = base_of(key)
            b_txt = f"{b:.4g}" if b is not None else "—"
            if key in fem:                       # FEM 검증된 응답
                opt = fem[key]
                d = float(_D_FUNCS[s[0]](np.array([opt], float), *s[1:])[0])
                pct = f", {(opt / b - 1) * 100:+.1f}%" if b else ""
                opt_txt = f"{opt:.4g}  (만족도 {d:.2f}{pct})"
            else:                                # efficiency 등 서로게이트 외
                opt_txt = "— (Solve 탭 부하 스윕으로 평가)"
            rows.append((name, b_txt, opt_txt))
        self.tbl_res.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, t in enumerate(r):
                self.tbl_res.setItem(i, j, QTableWidgetItem(t))
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

    def _spawn(self, job, log_slot, done_slot):
        wk = Worker(job)
        wk.log.connect(log_slot)
        wk.done.connect(done_slot)
        wk.failed.connect(lambda e: log_slot("오류:\n" + e))
        wk.finished.connect(lambda: self._workers.remove(wk))
        self._workers.append(wk)
        wk.start()


def main():
    app = QApplication(sys.argv)
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
