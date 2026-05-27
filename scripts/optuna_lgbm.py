"""
Optuna hyperparameter search for the 3 per-horizon LGBM regressors.

Uses TimeSeriesSplit (5 folds) on training data only — no leakage from the
30-day holdout. Optimises the mean R2 across all 3 horizons (equal weight).
Prints the best params and test-set results using the same 30-day holdout
as train_daily.py.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from config.db import get_collection, COLLECTION_FEATURE_STORE

TARGET_COLS  = ["AQI_t+1", "AQI_t+2", "AQI_t+3"]
EXCLUDE_COLS = {"date", "processed_at", "_id",
                "AQI_trend_7d", "AQI_x_wind", "NO2_lag_3", "Humidity_lag_3",
                "surface_pressure", "surface_pressure_t1", "surface_pressure_t2", "surface_pressure_t3",
                "surface_pressure_lag_1", "surface_pressure_roll_mean_7",
                "apparent_temp", "apparent_temp_t1", "apparent_temp_t2", "apparent_temp_t3",
                "apparent_temp_lag_1", "apparent_temp_roll_mean_7",
                "wind_gusts", "wind_gusts_t1", "wind_gusts_t2", "wind_gusts_t3",
                "wind_gusts_lag_1", "wind_gusts_roll_mean_7",
                } | set(TARGET_COLS)

N_TRIALS  = 100
N_CV_FOLDS = 5
TEST_DAYS  = 30


def load_data():
    docs = list(get_collection(COLLECTION_FEATURE_STORE).find({}, {"_id": 0}))
    df = pd.DataFrame(docs)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=TARGET_COLS)
    feat = [c for c in df.columns if c not in EXCLUDE_COLS]
    split_date = df["date"].max() - pd.Timedelta(days=TEST_DAYS)
    train = df[df["date"] <= split_date]
    test  = df[df["date"] >  split_date]
    return train, test, feat


def objective(trial, X_train, y_train):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth":        trial.suggest_int("max_depth", 4, 10),
        "num_leaves":       trial.suggest_int("num_leaves", 20, 127),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples":trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": 42,
        "verbose": -1,
    }

    tscv = TimeSeriesSplit(n_splits=N_CV_FOLDS)
    scaler = StandardScaler()
    scores = []

    for train_idx, val_idx in tscv.split(X_train):
        Xtr, Xvl = X_train[train_idx], X_train[val_idx]
        Xtr_sc = scaler.fit_transform(Xtr)
        Xvl_sc = scaler.transform(Xvl)

        fold_r2 = []
        for h in range(3):
            ytr = y_train[train_idx, h]
            yvl = y_train[val_idx,   h]
            m = LGBMRegressor(**params)
            m.fit(Xtr_sc, ytr)
            pred = m.predict(Xvl_sc)
            fold_r2.append(r2_score(yvl, pred))
        scores.append(np.mean(fold_r2))

    return np.mean(scores)


def evaluate_best(params, X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    results = {}
    for h, col in enumerate(TARGET_COLS):
        m = LGBMRegressor(**params, random_state=42, verbose=-1)
        m.fit(Xtr, y_train[:, h])
        pred = m.predict(Xte)
        r2 = r2_score(y_test[:, h], pred)
        results[col] = round(r2, 4)
        print(f"  {col}: R2={r2:.4f}")
    return results


def main():
    print("Loading data ...")
    train_df, test_df, feat = load_data()
    print(f"Train: {len(train_df)} rows  |  Test: {len(test_df)} rows  |  Features: {len(feat)}")

    X_train = train_df[feat].values.astype(np.float32)
    X_test  = test_df[feat].values.astype(np.float32)
    y_train = train_df[TARGET_COLS].values.astype(np.float32)
    y_test  = test_df[TARGET_COLS].values.astype(np.float32)

    print(f"\nRunning Optuna ({N_TRIALS} trials, {N_CV_FOLDS}-fold TimeSeriesCV) ...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=N_TRIALS,
        n_jobs=1,
        show_progress_bar=True,
    )

    best = study.best_params
    print(f"\nBest CV R2 (mean across horizons): {study.best_value:.4f}")
    print("Best params:")
    for k, v in best.items():
        print(f"  {k}: {v}")

    print("\nEvaluating best params on 30-day holdout:")
    evaluate_best(best, X_train, X_test, y_train, y_test)

    # Compare against default params from train_daily.py
    default = dict(
        n_estimators=500, learning_rate=0.05, max_depth=7,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
    )
    print("\nDefault params on 30-day holdout:")
    evaluate_best(default, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
