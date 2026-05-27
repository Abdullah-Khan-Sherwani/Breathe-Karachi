"""
Walk-forward holdout analysis — evaluates LGBM across multiple 30-day
windows spread across the full dataset to show R² variance by season.
Not used in production.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor

from config.db import get_collection, COLLECTION_FEATURE_STORE
from src.preprocess_daily_data import FEATURE_COLS

TARGET_COLS = ["AQI_t+1", "AQI_t+2", "AQI_t+3"]
WINDOW = 30   # test window size in days
MIN_TRAIN = 200  # minimum training rows before first test window

LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.05, max_depth=6,
                   num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                   random_state=42, verbose=-1)


def load():
    docs = list(get_collection(COLLECTION_FEATURE_STORE).find({}, {"_id": 0}))
    df = pd.DataFrame(docs)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    feat = [c for c in FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=feat + TARGET_COLS)
    return df, feat


def run_window(X, y, tr_end, te_start, te_end):
    X_tr, y_tr = X[:tr_end], y[:tr_end]
    X_te, y_te = X[te_start:te_end], y[te_start:te_end]
    if len(X_tr) < MIN_TRAIN or len(X_te) == 0:
        return None
    models = []
    for h in range(3):
        m = LGBMRegressor(**LGBM_PARAMS)
        m.fit(X_tr, y_tr[:, h])
        models.append(m)
    preds = np.column_stack([m.predict(X_te) for m in models])
    return [r2_score(y_te[:, h], preds[:, h]) for h in range(3)]


if __name__ == "__main__":
    print("Loading feature store...")
    df, feat = load()
    X = df[feat].values
    y = df[TARGET_COLS].values
    dates = df["date"].dt.date
    n = len(df)
    print(f"  {n} rows  |  {feat.__len__()} features")
    print(f"  Date range: {dates.iloc[0]} to {dates.iloc[-1]}\n")

    # Non-overlapping 30-day windows starting after MIN_TRAIN rows
    # Stride: every ~90 days to sample different seasons
    STRIDE = 90
    starts = list(range(MIN_TRAIN, n - WINDOW, STRIDE))

    print(f"{'Window':>6}  {'Train end':>12}  {'Test start':>12}  {'Test end':>12}  "
          f"{'Train rows':>10}  {'R2_t1':>7}  {'R2_t2':>7}  {'R2_t3':>7}  {'Season'}")
    print("-" * 100)

    results = []
    for i, start in enumerate(starts):
        end = start + WINDOW
        if end > n:
            break
        r2s = run_window(X, y, start, start, end)
        if r2s is None:
            continue

        window_dates = dates.iloc[start:end]
        mid_month = window_dates.iloc[len(window_dates)//2].month
        season = {12:"Winter",1:"Winter",2:"Winter",
                  3:"Spring",4:"Spring",5:"Spring",
                  6:"Summer",7:"Summer",8:"Summer",
                  9:"Autumn",10:"Autumn",11:"Autumn"}[mid_month]

        print(f"  {i+1:>4}  {str(dates.iloc[start-1]):>12}  {str(dates.iloc[start]):>12}  "
              f"{str(dates.iloc[end-1]):>12}  {start:>10}  "
              f"{r2s[0]:>7.3f}  {r2s[1]:>7.3f}  {r2s[2]:>7.3f}  {season}")
        results.append({"season": season, "r2_t1": r2s[0], "r2_t2": r2s[1], "r2_t3": r2s[2],
                        "train_rows": start, "test_start": str(dates.iloc[start])})

    if not results:
        print("No windows evaluated.")
        sys.exit(1)

    df_r = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print(f"  Windows evaluated : {len(df_r)}")
    print(f"  R2_t1 : mean={df_r.r2_t1.mean():.3f}  std={df_r.r2_t1.std():.3f}  "
          f"min={df_r.r2_t1.min():.3f}  max={df_r.r2_t1.max():.3f}")
    print(f"  R2_t2 : mean={df_r.r2_t2.mean():.3f}  std={df_r.r2_t2.std():.3f}  "
          f"min={df_r.r2_t2.min():.3f}  max={df_r.r2_t2.max():.3f}")
    print(f"  R2_t3 : mean={df_r.r2_t3.mean():.3f}  std={df_r.r2_t3.std():.3f}  "
          f"min={df_r.r2_t3.min():.3f}  max={df_r.r2_t3.max():.3f}")

    print("\nSEASONAL BREAKDOWN (mean R2_t3 per season)")
    for season, grp in df_r.groupby("season"):
        print(f"  {season:<8}: R2_t3={grp.r2_t3.mean():.3f}  (n={len(grp)} windows)")

    print("\nFINAL 30-DAY WINDOW (our current train.py holdout)")
    last = run_window(X, y, n - WINDOW, n - WINDOW, n)
    if last:
        print(f"  Train rows={n - WINDOW}  Test={dates.iloc[n-WINDOW]} to {dates.iloc[-1]}")
        print(f"  R2_t1={last[0]:.3f}  R2_t2={last[1]:.3f}  R2_t3={last[2]:.3f}")
    print("=" * 60)
