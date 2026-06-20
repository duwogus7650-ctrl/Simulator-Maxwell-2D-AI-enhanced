"""P6 액티브러닝: DE 최적화 → FEM 검증 → 데이터 보강 → 재학습 (반복).

OFFLINE 재현 스크립트 — Ansys Maxwell(.aedt)가 설치된 환경에서 P6 액티브러닝을
배치로 재현하기 위한 용도. GUI(gui/app.py)와 별개로 단독 실행한다.
사용: python scripts/run_p6_active.py [경로\\to\\model.aedt]
"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
from scipy.optimize import differential_evolution
from motoropt.doe import BOUNDS, _init, _eval
from motoropt.surrogate import load_dataset, train_surrogate, save, X_KEYS
from motoropt.objective import (SurrogateObjective, desirability, SPEC)

DATA = "doe_results.jsonl"
# .aedt 경로: 인자로 받거나(없으면) 프로젝트 상대 기본값.
AEDT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "400W.aedt")
_init(AEDT)
Y_KEYS = None
HARD = set()   # 하드(필수) 제약 키 — 필요 시 채움(예: {"B_tooth_st"})

def fem(xdict):
    r = _eval(xdict)
    return r

def D_of_row(r):
    Y = np.array([[r[k] for k in Y_KEYS]])
    return float(desirability(Y, spec=SPEC, y_keys=Y_KEYS, hard_keys=HARD)[0])

log = []
for rnd in range(1, 5):
    X, Y, Y_KEYS = load_dataset(DATA)
    model, scale, met, _ = train_surrogate(X, Y, y_keys=Y_KEYS)
    rel = [k for k in Y_KEYS if met[k]["reliable"]]
    save(model, scale, "surrogate.joblib", y_keys=Y_KEYS, reliable_keys=rel)
    obj = SurrogateObjective("surrogate.joblib", BOUNDS,
                             spec=SPEC, hard_keys=HARD)
    res = differential_evolution(lambda u: -obj.D(u)[0], [(0, 1)] * 5,
                                 seed=rnd, maxiter=300, tol=1e-8)
    u = res.x
    xd = dict(zip(X_KEYS, map(float, obj.x_of(u))))
    Ypred = obj.predict(u)[0]
    t0 = time.time()
    r = fem(xd)
    ok = r["status"] == "ok"
    Dtrue = D_of_row(r) if ok else 0.0
    with open(DATA, "a", encoding="utf-8") as f:
        f.write(json.dumps(r) + "\n")
    print(f"[라운드 {rnd}] 서로게이트 D={-res.fun:.4f} → FEM D={Dtrue:.4f} "
          f"({'ok' if ok else r['status'][:25]}) {time.time()-t0:.0f}s", flush=True)
    if ok:
        print(f"  예측 T={Ypred[0]:.1f} EMF={Ypred[1]:.3f} A={Ypred[4]:.1f} | "
              f"FEM T={r['T_avg']:.1f} EMF={r['emf_rms']:.3f} A={r['magnet_area']:.1f}",
              flush=True)
        log.append({"round": rnd, "x": xd, "D_pred": -res.fun,
                    "D_true": Dtrue, "fem": {k: r[k] for k in Y_KEYS}})
json.dump(log, open("p6_active_log.json", "w"), indent=1)
print("ACTIVE DONE")
