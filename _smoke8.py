import sys, os, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"   # 사용자 화면에 창 띄우지 않음
ROOT = r"c:\Users\user\Desktop\Maxwell-2D-AI-enhanced-main"
sys.path.insert(0, ROOT)
import gui.app as A
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
out = open(os.path.join(ROOT, "_smoke8.log"), "w", encoding="utf-8")
def P(*a): print(*a, file=out, flush=True)
app = QApplication(sys.argv)
app.setQuitOnLastWindowClosed(False)
win = A.MainWindow(); win.show()
win.open_aedt(r"C:\Users\user\Desktop\aedt파일\SH_Reducer_25_32.aedt")
win.sp_irms.setValue(5.0)
win.sp_ndoe.setValue(25)
win.run_doe_build()
t0 = time.time()
def step1():
    if win._workers and time.time()-t0 < 1500:
        QTimer.singleShot(5000, step1); return
    P("--- DOE 로그 (마지막 6줄) ---")
    P("\n".join(win.log_opt.toPlainText().splitlines()[-6:]))
    win.log_opt.clear()
    win.run_active_round()
    QTimer.singleShot(3000, step2)
def step2():
    if win._workers and time.time()-t0 < 1800:
        QTimer.singleShot(3000, step2); return
    P("--- 액티브러닝 로그 ---"); P(win.log_opt.toPlainText())
    P("후보 수:", win.tbl_cand.rowCount())
    app.quit()
QTimer.singleShot(3000, step1)
app.exec()
P("EXIT", round((time.time()-t0)/60, 1), "min"); out.close()
