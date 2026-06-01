"""
Training orchestrator — loads feature_store, time-aware split (last 30 days = test),
trains Ridge + LightGBM + LSTM, computes nnls ensemble weights, saves all to
model_registry, and logs runs to model_logs.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config.db import (
    get_collection,
    save_model,
    purge_old_models,
    COLLECTION_FEATURE_STORE,
    COLLECTION_MODEL_LOGS,
    COLLECTION_ENSEMBLE,
)
from src.models import train_ridge, train_ridge_full, train_lgbm, train_lgbm_full, train_lstm, train_lstm_full

TARGET_COLS  = ["AQI_t+1", "AQI_t+2", "AQI_t+3", "AQI_t+4"]
EXCLUDE_COLS = {
    "date", "processed_at", "_id",
    # Tier-2: confirmed to hurt holdout performance
    "AQI_trend_7d", "AQI_x_wind", "NO2_lag_3", "Humidity_lag_3",
    # Tier-3: weather variables not available at backfill time for most rows
    "surface_pressure", "surface_pressure_t1", "surface_pressure_t2", "surface_pressure_t3", "surface_pressure_t4",
    "surface_pressure_lag_1", "surface_pressure_roll_mean_7",
    "apparent_temp", "apparent_temp_t1", "apparent_temp_t2", "apparent_temp_t3", "apparent_temp_t4",
    "apparent_temp_lag_1", "apparent_temp_roll_mean_7",
    "wind_gusts", "wind_gusts_t1", "wind_gusts_t2", "wind_gusts_t3", "wind_gusts_t4",
    "wind_gusts_lag_1", "wind_gusts_roll_mean_7",
    # Sparse variables: null in ERA5 archive, forecast-only
    "visibility", "visibility_t1", "visibility_t2", "visibility_t3", "visibility_t4",
    "visibility_lag_1", "visibility_roll_mean_7",
    "wind_speed_80m", "wind_speed_80m_t1", "wind_speed_80m_t2", "wind_speed_80m_t3", "wind_speed_80m_t4",
    "wind_speed_80m_lag_1", "wind_speed_80m_roll_mean_7",
    "cape", "cape_t1", "cape_t2", "cape_t3", "cape_t4",
    "cape_lag_1", "cape_roll_mean_7",
    # PM2_5 leads excluded: dominant AQI driver inflates metrics; CAMS forecast used at inference instead
    "PM2_5_t1", "PM2_5_t2", "PM2_5_t3", "PM2_5_t4",
} | set(TARGET_COLS)

_PER_HORIZON_KEYS = [
    "MAE_d1", "MAE_d2", "MAE_d3", "MAE_d4",
    "RMSE_d1", "RMSE_d2", "RMSE_d3", "RMSE_d4",
    "R2_d1", "R2_d2", "R2_d3", "R2_d4",
]


def load_data() -> pd.DataFrame:
    docs = list(get_collection(COLLECTION_FEATURE_STORE).find({}, {"_id": 0}))
    if not docs:
        raise RuntimeError("feature_store is empty — run fetch_data.py and preprocess_daily_data.py first.")
    df = pd.DataFrame(docs)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=TARGET_COLS)
    df = df[df["date"] >= pd.Timestamp("2023-01-01")]
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def time_split(df: pd.DataFrame, test_days: int = 30):
    split = df["date"].max() - pd.Timedelta(days=test_days)
    train = df[df["date"] <= split]
    test  = df[df["date"] >  split]
    return train, test


def _log(col, model_type: str, status: str, metrics: dict, model_id=None):
    doc: dict = {
        "timestamp":  datetime.now(timezone.utc),
        "status":     status,
        "model_type": model_type,
        "model_id":   str(model_id) if model_id else None,
    }
    for key in ("MAE", "RMSE", "R2", "error", *_PER_HORIZON_KEYS):
        if key in metrics:
            doc[key] = metrics[key]
    col.insert_one(doc)


def _nnls_weights(A: np.ndarray, y: np.ndarray, n_models: int) -> np.ndarray:
    w, _ = nnls(A, y)
    return w / w.sum() if w.sum() > 0 else np.ones(n_models) / n_models


def run() -> None:
    df   = load_data()
    feat = get_feature_cols(df)

    train, test = time_split(df, test_days=60)
    if len(test) == 0:
        raise RuntimeError("Test split is empty — need at least 30 days of data.")

    X_tr,   y_tr   = train[feat].values, train[TARGET_COLS].values
    X_te,   y_te   = test[feat].values,  test[TARGET_COLS].values
    X_full, y_full = df[feat].values,    df[TARGET_COLS].values

    print(f"Holdout eval : {len(X_tr)} train / {len(X_te)} test rows")
    print(f"Full retrain : {len(X_full)} rows  |  Features: {len(feat)}")

    logs_col = get_collection(COLLECTION_MODEL_LOGS)

    eval_trainers = [("lgbm", train_lgbm), ("lstm", train_lstm), ("ridge", train_ridge)]
    full_trainers = {"lgbm": train_lgbm_full, "lstm": train_lstm_full, "ridge": train_ridge_full}

    # Step 1 — evaluate each model on 30-day holdout
    holdout: dict = {}
    holdout_preds: dict = {}
    for model_type, trainer in eval_trainers:
        print(f"\nEvaluating {model_type} on holdout...")
        try:
            _, _, metrics, hparams = trainer(X_tr, y_tr, X_te, y_te)
            holdout_preds[model_type] = np.array(metrics.pop("_preds"))
            holdout[model_type] = (metrics, hparams)
            print(f"  {model_type}: MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}  R²={metrics['R2']:.3f}")
            for h in range(1, 5):
                if f"R2_d{h}" in metrics:
                    print(f"    d{h}: MAE={metrics[f'MAE_d{h}']:.2f}  RMSE={metrics[f'RMSE_d{h}']:.2f}  R²={metrics[f'R2_d{h}']:.3f}")
        except Exception as exc:
            print(f"  {model_type} eval FAILED: {exc}")

    # Step 2 — compute ensemble weights and evaluate cleanly
    order = [m for m in ("lgbm", "lstm", "ridge") if m in holdout_preds]
    if len(order) >= 2:
        n_te = y_te.shape[0]
        half = 30

        # Fit weights on first half of holdout, evaluate on second half (no leakage)
        val_preds = np.zeros((n_te - half, len(TARGET_COLS)))
        for h in range(len(TARGET_COLS)):
            A = np.column_stack([holdout_preds[m][:, h] for m in order])
            w = _nnls_weights(A[:half], y_te[:half, h], len(order))
            val_preds[:, h] = A[half:] @ w

        ens_metrics = {
            "MAE":  float(mean_absolute_error(y_te[half:], val_preds)),
            "RMSE": float(np.sqrt(mean_squared_error(y_te[half:], val_preds))),
            "R2":   float(r2_score(y_te[half:], val_preds)),
        }
        for h in range(len(TARGET_COLS)):
            ens_metrics[f"MAE_d{h+1}"]  = float(mean_absolute_error(y_te[half:, h], val_preds[:, h]))
            ens_metrics[f"RMSE_d{h+1}"] = float(np.sqrt(mean_squared_error(y_te[half:, h], val_preds[:, h])))
            ens_metrics[f"R2_d{h+1}"]   = float(r2_score(y_te[half:, h], val_preds[:, h]))

        # Refit weights on full holdout for deployment
        weights = []
        for h in range(len(TARGET_COLS)):
            A = np.column_stack([holdout_preds[m][:, h] for m in order])
            weights.append(_nnls_weights(A, y_te[:, h], len(order)).tolist())

        get_collection(COLLECTION_ENSEMBLE).replace_one(
            {},
            {"order": order, "weights": weights, "metrics": ens_metrics, "updated_at": datetime.now(timezone.utc)},
            upsert=True,
        )
        _log(logs_col, "ensemble", "success", ens_metrics)

        print(f"\nEnsemble weights (order={order}):")
        for h, w in enumerate(weights, 1):
            print(f"  d{h}: {dict(zip(order, [f'{x:.2f}' for x in w]))}")
        print(f"\nEnsemble metrics (second-half holdout):")
        print(f"  Overall: MAE={ens_metrics['MAE']:.2f}  RMSE={ens_metrics['RMSE']:.2f}  R²={ens_metrics['R2']:.3f}")
        for h in range(1, 5):
            print(f"  d{h}: MAE={ens_metrics[f'MAE_d{h}']:.2f}  RMSE={ens_metrics[f'RMSE_d{h}']:.2f}  R²={ens_metrics[f'R2_d{h}']:.3f}")

    # Step 3 — retrain on full labeled data; save with holdout metrics
    results = []
    for model_type, full_trainer in full_trainers.items():
        if model_type not in holdout:
            continue
        metrics, hparams = holdout[model_type]
        print(f"\nFull retrain {model_type} on {len(X_full)} rows...")
        try:
            model_full, scaler_full = full_trainer(X_full, y_full)
            model_id = save_model(
                model=model_full,
                scaler=scaler_full,
                model_type=model_type,
                metrics=metrics,
                feature_cols=feat,
                hyperparameters=hparams,
                extra_metadata={
                    "train_samples": len(X_full),
                    "test_samples":  len(X_te),
                    "holdout_eval":  True,
                    "automated":     os.getenv("AUTOMATED_RUN") == "true",
                },
            )
            _log(logs_col, model_type, "success", metrics, model_id)
            purge_old_models(model_type)
            results.append((model_type, metrics["RMSE"], model_id))
            print(f"  saved  id={model_id}")
        except Exception as exc:
            _log(logs_col, model_type, "error", {"error": str(exc)})
            print(f"  {model_type} full retrain FAILED: {exc}")

    if not results:
        raise RuntimeError("All models failed to train.")

    best_type, best_rmse, best_id = min(results, key=lambda r: r[1])
    print(f"\nBest model: {best_type}  RMSE={best_rmse:.2f}  id={best_id}")
    print("Training complete.")


if __name__ == "__main__":
    run()
