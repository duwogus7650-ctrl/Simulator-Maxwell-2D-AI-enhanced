"""P5 서로게이트: DOE 데이터셋 → MLP 회귀 모델 학습/검증.

입력 5 (a_m, T_m, T_m2_ratio, W_t, MagnetR)
출력 4 (T_avg, emf_rms, ripple_pct, magnet_area) + B_tooth
지표: 실제 5-fold CV R²(KFold로 폴드별 재학습), 별도 홀드아웃 MAE/상대오차.
모델은 joblib 저장 → P6 RL 환경의 빠른 평가 함수로 사용.
"""
from __future__ import annotations

import json

import numpy as np

X_KEYS = ["a_m", "T_m", "T_m2_ratio", "W_t", "MagnetR"]
Y_KEYS = ["T_avg", "emf_rms", "ripple_pct", "B_tooth", "magnet_area"]
# 데이터셋에 있으면 추가로 학습하는 옵션 응답 (with_efficiency·with_cogging DOE)
Y_KEYS_OPT = ["efficiency", "cogging_pp"]   # 동손은 목표 제외(고정전류서 상수)


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


def train_surrogate(X, Y, seed: int = 0, y_keys: list | None = None,
                    arch=(16, 16), alpha=1e-2):
    y_keys = y_keys or Y_KEYS
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    from sklearn.multioutput import MultiOutputRegressor

    def _make():
        return make_pipeline(
            StandardScaler(),
            MultiOutputRegressor(MLPRegressor(
                hidden_layer_sizes=arch, activation="tanh",
                solver="lbfgs", alpha=alpha, max_iter=4000,
                random_state=seed)))

    # --- 실제 5-fold CV R² (폴드마다 새 파이프라인·폴드별 mu,sd로 출력표준화) ---
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    cv_num = np.zeros(len(y_keys))   # Σ 폴드별 (R²·가중 없이 단순평균)
    cv_cnt = 0
    for tr_idx, te_idx in kf.split(X):
        Xtr_f, Xte_f = X[tr_idx], X[te_idx]
        Ytr_f, Yte_f = Y[tr_idx], Y[te_idx]
        mu_f, sd_f = Ytr_f.mean(0), Ytr_f.std(0) + 1e-12
        m_f = _make()
        m_f.fit(Xtr_f, (Ytr_f - mu_f) / sd_f)
        Yp_f = m_f.predict(Xte_f) * sd_f + mu_f
        for j in range(len(y_keys)):
            err = Yp_f[:, j] - Yte_f[:, j]
            denom = np.sum((Yte_f[:, j] - Yte_f[:, j].mean()) ** 2)
            cv_num[j] += 1 - np.sum(err ** 2) / (denom + 1e-12)
        cv_cnt += 1
    r2_cv = cv_num / max(cv_cnt, 1)

    # --- 홀드아웃: R2_holdout / MAE / rel% + 시각화 튜플 ---
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2,
                                          random_state=seed)
    mu_h, sd_h = Ytr.mean(0), Ytr.std(0) + 1e-12
    model_h = _make()
    model_h.fit(Xtr, (Ytr - mu_h) / sd_h)
    Yp = model_h.predict(Xte) * sd_h + mu_h
    metrics = {}
    for j, k in enumerate(y_keys):
        err = Yp[:, j] - Yte[:, j]
        ss = 1 - np.sum(err ** 2) / np.sum((Yte[:, j] - Yte[:, j].mean()) ** 2)
        metrics[k] = {"R2_cv": float(r2_cv[j]),
                      "R2_holdout": float(ss),
                      "MAE": float(np.abs(err).mean()),
                      "rel%": float(np.abs(err / (Yte[:, j] + 1e-12)).mean()
                                    * 100),
                      "reliable": bool(r2_cv[j] >= 0.5)}

    # --- 프로덕션 모델: 전체 (X, Y)로 재학습(데이터 더 많음), mu,sd도 전체 Y ---
    mu, sd = Y.mean(0), Y.std(0) + 1e-12
    model = _make()
    model.fit(X, (Y - mu) / sd)
    return model, (mu, sd), metrics, (Xte, Yte, Yp)


def save(model, scale, path: str, y_keys: list | None = None,
         reliable_keys: list | None = None):
    import joblib
    joblib.dump({"model": model, "mu": scale[0], "sd": scale[1],
                 "x_keys": X_KEYS, "y_keys": y_keys or Y_KEYS,
                 "reliable_keys": reliable_keys}, path)


def predict(bundle_path: str, X: np.ndarray) -> np.ndarray:
    import joblib
    b = joblib.load(bundle_path)
    return b["model"].predict(X) * b["sd"] + b["mu"]
