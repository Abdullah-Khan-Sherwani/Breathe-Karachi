"""
Shuffled K-fold CV sanity check — mirrors the notebook's evaluation methodology
to confirm that shuffled CV inflates R² vs a proper time-based holdout.
Not used in production. Run once, compare numbers, delete.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, make_scorer
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

from config.db import get_collection, COLLECTION_FEATURE_STORE
from src.preprocess_daily_data import FEATURE_COLS

TARGET_COLS = ["AQI_t+1", "AQI_t+2", "AQI_t+3"]


def load():
    docs = list(get_collection(COLLECTION_FEATURE_STORE).find({}, {"_id": 0}))
    df = pd.DataFrame(docs)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    feat = [c for c in FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=feat + TARGET_COLS)
    return df, feat


def per_horizon_r2(y_true, y_pred):
    return [r2_score(y_true[:, h], y_pred[:, h]) for h in range(3)]


def eval_shuffled_cv(X, y, label="LGBM (shuffled 5-fold CV, shuffle=True)"):
    """Notebook-style: shuffled folds, incorrect for time series."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_per_horizon = [[], [], []]

    for fold, (tr_idx, te_idx) in enumerate(kf.split(X)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        models = []
        for h in range(3):
            m = LGBMRegressor(n_estimators=300, learning_rate=0.05,
                              max_depth=6, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=42, verbose=-1)
            m.fit(X_tr, y_tr[:, h])
            models.append(m)

        preds = np.column_stack([m.predict(X_te) for m in models])
        for h in range(3):
            r2_per_horizon[h].append(r2_score(y_te[:, h], preds[:, h]))

    print(f"\n{label}")
    print(f"  R2_t1 = {np.mean(r2_per_horizon[0]):.3f}  (std {np.std(r2_per_horizon[0]):.3f})")
    print(f"  R2_t2 = {np.mean(r2_per_horizon[1]):.3f}  (std {np.std(r2_per_horizon[1]):.3f})")
    print(f"  R2_t3 = {np.mean(r2_per_horizon[2]):.3f}  (std {np.std(r2_per_horizon[2]):.3f})")


def eval_timeseries_cv(X, y, label="LGBM (time-series 5-fold CV, no shuffle)"):
    """Correct for time series: each test fold is strictly after its training fold."""
    kf = KFold(n_splits=5, shuffle=False)
    r2_per_horizon = [[], [], []]

    for fold, (tr_idx, te_idx) in enumerate(kf.split(X)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        models = []
        for h in range(3):
            m = LGBMRegressor(n_estimators=300, learning_rate=0.05,
                              max_depth=6, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=42, verbose=-1)
            m.fit(X_tr, y_tr[:, h])
            models.append(m)

        preds = np.column_stack([m.predict(X_te) for m in models])
        for h in range(3):
            r2_per_horizon[h].append(r2_score(y_te[:, h], preds[:, h]))

    print(f"\n{label}")
    print(f"  R2_t1 = {np.mean(r2_per_horizon[0]):.3f}  (std {np.std(r2_per_horizon[0]):.3f})")
    print(f"  R2_t2 = {np.mean(r2_per_horizon[1]):.3f}  (std {np.std(r2_per_horizon[1]):.3f})")
    print(f"  R2_t3 = {np.mean(r2_per_horizon[2]):.3f}  (std {np.std(r2_per_horizon[2]):.3f})")


def eval_holdout(X, y, label="LGBM (last 30 days holdout — our current method)"):
    """What train.py actually does."""
    X_tr, y_tr = X[:-30], y[:-30]
    X_te, y_te = X[-30:], y[-30:]

    models = []
    for h in range(3):
        m = LGBMRegressor(n_estimators=300, learning_rate=0.05,
                          max_depth=6, num_leaves=31,
                          subsample=0.8, colsample_bytree=0.8,
                          random_state=42, verbose=-1)
        m.fit(X_tr, y_tr[:, h])
        models.append(m)

    preds = np.column_stack([m.predict(X_te) for m in models])
    print(f"\n{label}")
    for h, name in enumerate(["t+1", "t+2", "t+3"]):
        print(f"  R2_{name} = {r2_score(y_te[:, h], preds[:, h]):.3f}")


if __name__ == "__main__":
    print("Loading feature store...")
    df, feat = load()
    X = df[feat].values
    y = df[TARGET_COLS].values
    print(f"  {len(df)} rows, {len(feat)} features")

    print("\n" + "=" * 60)
    print("EVALUATION METHODOLOGY COMPARISON — same model, same data")
    print("=" * 60)

    eval_shuffled_cv(X, y)
    eval_timeseries_cv(X, y)
    eval_holdout(X, y)

    print("\n" + "=" * 60)
    print("Shuffled CV inflates R² because future data trains the model.")
    print("Time-series CV and holdout reflect real forecasting difficulty.")
    print("=" * 60)
