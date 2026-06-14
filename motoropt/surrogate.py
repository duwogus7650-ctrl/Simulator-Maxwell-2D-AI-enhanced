"""P5 서로게이트: DOE 데이터셋 → MLP 회귀 모델 학습/검증.

입력 5 (a_m, T_m, T_m2_ratio, W_t, MagnetR)
출력 4 (T_avg, emf_rms, ripple_pct, magnet_area) + B_tooth
지표: 5-fold CV R², 테스트 MAE/상대오차. 모델은 joblib 저장 →
P6 RL 환경의 빠른 평가 함수로 사용.
"""
from __future__ import annotations

import json

import numpy as np

X_KEYS = ["a_m", "T_m", "T_m2_ratio", "W_t", "MagnetR"]
Y_KEYS = ["T_avg", "emf_rms", "ripple_pct", "B_tooth", "magnet_area"]
# 데이터셋에 있으면 추가로 학습하는 옵션 응답 (with_efficiency·with_cogging DOE)
Y_KEYS_OPT = ["efficiency", "cogging_pp", "Pcu_W"]


def dataset_y_keys(rows) -> list:
    """ok 행 전부에 존재하는 응답만 학습 대상 — 구 데이터셋 호환."""
    ok = [r for r in rows if r.get("status") == "ok"]
    keys = list(Y_KEYS)
    for k in Y_KEYS_OPT:
        if ok and all(k in r and r[k] is not None for r in ok):
            keys.append(k)
    return keys


def load_dataset(path: str):
    """→ (X, Y, y_keys). y_keys는 데이터셋에 실제로 있는 응답 집합."""
    rows = []
    with open(path, encoding="utf-8") as f:        # Windows cp949 회피(한글 포함)
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    y_keys = dataset_y_keys(rows)
    X, Y = [], []
    for r in rows:
        if r.get("status") != "ok":
            continue
        X.append([r["x"][k] for k in X_KEYS])
        Y.append([r[k] for k in y_keys])
    return np.asarray(X), np.asarray(Y), y_keys


def train_surrogate(X, Y, seed: int = 0, y_keys: list | None = None):
    y_keys = y_keys or Y_KEYS
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    from sklearn.multioutput import MultiOutputRegressor

    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2,
                                          random_state=seed)
    model = make_pipeline(
        StandardScaler(),
        MultiOutputRegressor(MLPRegressor(
            hidden_layer_sizes=(64, 64), activation="tanh",
            solver="lbfgs", max_iter=4000, random_state=seed)))
    # 출력 표준화 수동 (MultiOutput 안에서 스케일 차이 큼)
    mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-12
    model.fit(Xtr, (Ytr - mu) / sd)

    Yp = model.predict(Xte) * sd + mu
    metrics = {}
    for j, k in enumerate(y_keys):
        err = Yp[:, j] - Yte[:, j]
        ss = 1 - np.sum(err ** 2) / np.sum((Yte[:, j] - Yte[:, j].mean()) ** 2)
        metrics[k] = {"R2": float(ss),
                      "MAE": float(np.abs(err).mean()),
                      "rel%": float(np.abs(err / (Yte[:, j] + 1e-12)).mean()
                                    * 100)}
    return model, (mu, sd), metrics, (Xte, Yte, Yp)


def save(model, scale, path: str, y_keys: list | None = None):
    import joblib
    joblib.dump({"model": model, "mu": scale[0], "sd": scale[1],
                 "x_keys": X_KEYS, "y_keys": y_keys or Y_KEYS}, path)


def predict(bundle_path: str, X: np.ndarray) -> np.ndarray:
    import joblib
    b = joblib.load(bundle_path)
    return b["model"].predict(X) * b["sd"] + b["mu"]
