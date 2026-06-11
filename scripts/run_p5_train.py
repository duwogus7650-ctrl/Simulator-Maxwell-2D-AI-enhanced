"""DOE 데이터셋 → 서로게이트 학습 + 검증 플롯."""
import sys, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.font_manager as fm
fm.fontManager.addfont("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
matplotlib.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from motoropt.surrogate import load_dataset, train_surrogate, save, Y_KEYS

X, Y = load_dataset("doe_results.jsonl")
print(f"유효 샘플 {len(X)}개")
model, scale, metrics, (Xte, Yte, Yp) = train_surrogate(X, Y)
for k, v in metrics.items():
    print(f"{k:12s} R²={v['R2']:.3f}  MAE={v['MAE']:.3g}  rel={v['rel%']:.1f}%")
save(model, scale, "surrogate.joblib")

units = {"T_avg": "mNm", "emf_rms": "V", "ripple_pct": "%",
         "B_tooth": "T", "magnet_area": "mm²"}
fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), dpi=120)
for j, (ax, k) in enumerate(zip(axes, Y_KEYS)):
    ax.scatter(Yte[:, j], Yp[:, j], s=18, alpha=.75)
    lim = [min(Yte[:, j].min(), Yp[:, j].min()),
           max(Yte[:, j].max(), Yp[:, j].max())]
    ax.plot(lim, lim, "k--", lw=.8)
    ax.set_xlabel(f"FEM 실측 [{units[k]}]"); ax.set_ylabel("서로게이트 예측")
    ax.set_title(f"{k} (R²={metrics[k]['R2']:.3f})")
    ax.grid(alpha=.3)
fig.suptitle("서로게이트 검증 — 테스트 20% 홀드아웃", y=1.02)
fig.tight_layout(); fig.savefig("surrogate_400W.png", bbox_inches="tight")
print("saved surrogate_400W.png / surrogate.joblib")
