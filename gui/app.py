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
import sys
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
        matplotlib.rcParams["font.family"] = "NanumGothic"
    except Exception:
        pass
matplotlib.rcParams["axes.unicode_minus"] = False
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

DESIGN_VARS = ["a_m", "T_m", "T_m2", "W_t", "MagnetR"]


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

    def _load_aedt(self, path):
        from motoropt.aedt_parser import parse_aedt, detect_magnet_style
        self.model = parse_aedt(path)
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
        self.tbl_obj = QTableWidget(3, 5)
        self.tbl_obj.setHorizontalHeaderLabels(
            ["응답", "유형", "L (하한)", "T (목표)", "U (상한)"])
        self.tbl_obj.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        from motoropt.objective import SPEC
        for i, (k, spec) in enumerate(SPEC.items()):
            self.tbl_obj.setItem(i, 0, QTableWidgetItem(k))
            cb = QComboBox(); cb.addItems(["larger", "smaller", "target"])
            cb.setCurrentText(spec[0])
            self.tbl_obj.setCellWidget(i, 1, cb)
            nums = spec[1:]
            cols = {"larger": (2, None, 4), "smaller": (2, None, 4),
                    "target": (2, 3, 4)}
            if spec[0] == "target":
                vals = {2: nums[0], 3: nums[1], 4: nums[2]}
            else:
                vals = {2: nums[0], 4: nums[1]}
            for c in (2, 3, 4):
                self.tbl_obj.setItem(
                    i, c, QTableWidgetItem(
                        f"{vals[c]:.4g}" if c in vals else ""))
        lay.addWidget(self.tbl_obj)
        lay.addWidget(QLabel("종합 만족도 D = (∏ dᵢ)^(1/n) — 모든 목표를 "
                             "동시에 만족할수록 1에 가까움"))
        lay.addStretch(1)
        return w

    # ---------------------------------------------------------- ③ Solve
    def _tab_solve(self):
        w = QWidget(); lay = QHBoxLayout(w)
        left = QVBoxLayout()
        b1 = QPushButton("▶ 무부하 해석 (코깅·EMF용 단일 포지션)")
        b1.clicked.connect(lambda: self.run_solve(load=False))
        b2 = QPushButton("▶ 부하 해석 (정격 전류·MTPA)")
        b2.clicked.connect(lambda: self.run_solve(load=True))
        left.addWidget(b1); left.addWidget(b2)
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

        def job(log):
            from motoropt.expressions import resolve_variables
            from motoropt.geometry import build_motor
            from motoropt.sliding import SlidingBandMesh
            from motoropt.solver_ms import Magnetostatic2D
            from motoropt.postproc import (torque_arkkio, coenergy,
                                           build_winding_map)
            v = resolve_variables(raw)
            geo = build_motor(v, style)
            log("형상/메시 생성...")
            sbm = SlidingBandMesh(geo, n_band=2880)
            s = Magnetostatic2D(sbm.merge(0.0), model["materials"],
                                "20PNX1200F_20C",
                                "Arnold_Magnetics_N45UH_80C")
            if load:
                wmap = build_winding_map(s)
                Ia = v["I_rms"] * math.sqrt(2)
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

    def run_active_round(self):
        if self.aedt_path is None:
            self.log_opt.appendPlainText("⚠ 먼저 Model 탭에서 aedt를 여세요")
            return
        aedt = self.aedt_path

        def job(log):
            from scipy.optimize import differential_evolution
            from motoropt.doe import BOUNDS, _init, _eval
            from motoropt.surrogate import (load_dataset, train_surrogate,
                                            save, X_KEYS, Y_KEYS)
            from motoropt.objective import SurrogateObjective, desirability
            log("서로게이트 재학습...")
            X, Y = load_dataset("doe_results.jsonl")
            mdl, sc, met, _ = train_surrogate(X, Y)
            save(mdl, sc, "surrogate.joblib")
            obj = SurrogateObjective("surrogate.joblib", BOUNDS)
            log(f"DE 최적화 (샘플 {len(X)})...")
            r = differential_evolution(lambda u: -obj.D(u)[0],
                                       [(0, 1)] * 5, seed=0,
                                       maxiter=250, tol=1e-8)
            xd = dict(zip(X_KEYS, map(float, obj.x_of(r.x))))
            log(f"서로게이트 D={-r.fun:.4f} → FEM 검증 중 (~30초)...")
            _init(aedt)
            fem = _eval(xd)
            with open("doe_results.jsonl", "a") as f:
                f.write(json.dumps(fem) + "\n")
            if fem["status"] != "ok":
                log(f"FEM 실패: {fem['status'][:40]}")
                return None
            Yv = np.array([[fem[k] for k in Y_KEYS]])
            D = float(desirability(Yv)[0])
            log(f"FEM D={D:.4f} | T={fem['T_avg']:.1f} "
                f"EMF={fem['emf_rms']:.3f} A={fem['magnet_area']:.1f}")
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

        def job(log):
            import torch
            from motoropt.doe import BOUNDS
            from motoropt.objective import SurrogateObjective
            from motoropt.rl_opt import DesignEnv, SAC
            obj = SurrogateObjective("surrogate.joblib", BOUNDS)
            env = DesignEnv(obj)
            env.reset()
            env.u = np.array([(x0[k] - BOUNDS[k][0])
                              / (BOUNDS[k][1] - BOUNDS[k][0])
                              for k in obj.keys]).clip(0, 1)
            env.D = float(obj.D(env.u)[0])
            log(f"시작 D={env.D:.4f}")
            agent = SAC(env.dim + 1, env.dim)
            if os.path.exists("sac_actor.pt"):
                agent.actor.load_state_dict(torch.load("sac_actor.pt"))
                log("학습된 SAC 정책 로드")
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

    def _update_result(self):
        if not self.candidates or self.geo is None:
            return
        D, xd, fem = self.candidates[0]
        base_area = sum(p.area for p, _, _ in self.geo.magnets)
        rows = [("종합 만족도 D", "0 (면적=상한)", f"{D:.4f}"),
                ("평균토크 [mNm]", "862.0", f"{fem['T_avg']:.1f}"),
                ("EMF RMS [V]", "6.164", f"{fem['emf_rms']:.3f}"),
                ("자석 면적 [mm²]", f"{base_area:.1f}",
                 f"{fem['magnet_area']:.1f} "
                 f"({(fem['magnet_area']/base_area-1)*100:+.1f}%)")]
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
        ax.set_xlim(0, 36); ax.set_ylim(14, 40)
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
