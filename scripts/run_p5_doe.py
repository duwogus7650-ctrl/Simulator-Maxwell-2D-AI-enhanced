"""P5 DOE 러너 — 설계별 자식 프로세스 격리(OOM/행 방어) + 이어돌리기."""
import sys, os, json, time, resource, warnings
import multiprocessing as mp
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
from scipy.stats import qmc
from motoropt.doe import BOUNDS, _init, _eval

OUT = "doe_results.jsonl"
N = 120

def design_list():
    keys = list(BOUNDS)
    lo = np.array([BOUNDS[k][0] for k in keys])
    hi = np.array([BOUNDS[k][1] for k in keys])
    X = qmc.LatinHypercube(d=len(keys), seed=7).random(N) * (hi - lo) + lo
    ds = [dict(zip(keys, map(float, r))) for r in X]
    ds.insert(0, {"a_m": 0.89, "T_m": 2.2, "T_m2_ratio": 2.02/2.2,
                  "W_t": 3.5, "MagnetR": 0.8})
    return ds

def key_of(x): return tuple(round(x[k], 6) for k in sorted(x))

def child(x, q):
    resource.setrlimit(resource.RLIMIT_AS, (1_800_000_000, 1_800_000_000))
    _init("/mnt/user-data/uploads/400W.aedt")
    q.put(_eval(x))

def main():
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try: done.add(key_of(json.loads(line)["x"]))
            except Exception: pass
    todo = [d for d in design_list() if key_of(d) not in done]
    print(f"남은 설계 {len(todo)}개 (완료 {len(done)} 스킵)", flush=True)
    ctx = mp.get_context("fork")
    with open(OUT, "a") as f:
        for i, x in enumerate(todo):
            q = ctx.Queue()
            p = ctx.Process(target=child, args=(x, q))
            t0 = time.time(); p.start(); p.join(timeout=150)
            if p.is_alive():
                p.terminate(); p.join()
                r = {"x": x, "status": "fail: timeout"}
            elif q.empty():
                r = {"x": x, "status": f"fail: crashed(exit {p.exitcode})"}
            else:
                r = q.get()
            f.write(json.dumps(r) + "\n"); f.flush()
            if i % 5 == 0:
                print(f"{i+1}/{len(todo)} {time.time()-t0:.0f}s {r['status'][:30]}", flush=True)
    print("DOE DONE", flush=True)

if __name__ == "__main__":
    main()
