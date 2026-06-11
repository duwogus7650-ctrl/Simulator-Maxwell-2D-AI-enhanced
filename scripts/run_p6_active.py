"""P6 액티브러닝: DE 최적화 → FEM 검증 → 데이터 보강 → 재학습 (반복)."""
import sys, json, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
from scipy.optimize import differential_evolution
from motoropt.doe import BOUNDS, _init, _eval
from motoropt.surrogate import load_dataset, train_surrogate, save, X_KEYS, Y_KEYS
from motoropt.objective import SurrogateObjective, desirability

DATA = "doe_results.jsonl"
_init("/mnt/user-data/uploads/400W.aedt")

def fem(xdict):
    r = _eval(xdict)
    return r

def D_of_row(r):
    Y = np.array([[r[k] for k in Y_KEYS]])
    return float(desirability(Y)[0])

log = []
for rnd in range(1, 5):
    X, Y = load_dataset(DATA)
    model, scale, met, _ = train_surrogate(X, Y)
    save(model, scale, "surrogate.joblib")
    obj = SurrogateObjective("surrogate.joblib", BOUNDS)
    res = differential_evolution(lambda u: -obj.D(u)[0], [(0, 1)] * 5,
                                 seed=rnd, maxiter=300, tol=1e-8)
    u = res.x
    xd = dict(zip(X_KEYS, map(float, obj.x_of(u))))
    Ypred = obj.predict(u)[0]
    t0 = time.time()
    r = fem(xd)
    ok = r["status"] == "ok"
    Dtrue = D_of_row(r) if ok else 0.0
    with open(DATA, "a") as f:
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
