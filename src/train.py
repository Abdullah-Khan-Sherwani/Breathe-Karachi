"""
Training orchestrator — loads feature_store, time-aware split (last 30 days = test),
trains Ridge + LightGBM + LSTM, saves the best RMSE model to model_registry,
and logs all three runs to model_logs.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config.db import (
    get_collection,
    save_model,
    COLLECTION_FEATURE_STORE,
    COLLECTION_MODEL_LOGS,
)
from src.models import train_ridge, train_lgbm, train_lstm

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
    # Tier-4: columns removed from pipeline but may exist in old MongoDB docs
    "visibility", "visibility_t1", "visibility_t2", "visibility_t3",
    "visibility_lag_1", "visibility_roll_mean_7",
    "wind_speed_80m", "wind_speed_80m_t1", "wind_speed_80m_t2", "wind_speed_80m_t3",
    "wind_speed_80m_lag_1",
    "cape", "cape_t1", "cape_t2", "cape_t3", "cape_lag_1",
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
    # Rows before 2023 have weather but sparse/unreliable AQI
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
    # Always include base metrics; include per-horizon keys when present
    for key in ("MAE", "RMSE", "R2", "error", *_PER_HORIZON_KEYS):
        if key in metrics:
            doc[key] = metrics[key]
    col.insert_one(doc)


def run() -> None:
    df   = load_data()
    feat = get_feature_cols(df)

    train, test = time_split(df)
    if len(test) == 0:
        raise RuntimeError("Test split is empty — need at least 30 days of data.")

    X_tr, y_tr = train[feat].values, train[TARGET_COLS].values
    X_te, y_te = test[feat].values,  test[TARGET_COLS].values

    print(f"Train: {len(X_tr)} rows | Test: {len(X_te)} rows | Features: {len(feat)}")

    logs_col = get_collection(COLLECTION_MODEL_LOGS)
    results  = []

    trainers = [
        ("ridge", train_ridge),
        ("lgbm",  train_lgbm),
        ("lstm",  train_lstm),
    ]

    for model_type, trainer in trainers:
        print(f"\nTraining {model_type}...")
        try:
            model, scaler, metrics, hparams = trainer(X_tr, y_tr, X_te, y_te)
            model_id = save_model(
                model=model,
                scaler=scaler,
                model_type=model_type,
                metrics=metrics,
                feature_cols=feat,
                hyperparameters=hparams,
                extra_metadata={"train_samples": len(X_tr), "test_samples": len(X_te)},
            )
            _log(logs_col, model_type, "success", metrics, model_id)
            results.append((model_type, metrics["RMSE"], model_id))
            print(f"  {model_type}: MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}  R²={metrics['R2']:.3f}")
            for h in range(1, 5):
                if f"R2_d{h}" in metrics:
                    print(f"    d{h}: MAE={metrics[f'MAE_d{h}']:.2f}  RMSE={metrics[f'RMSE_d{h}']:.2f}  R²={metrics[f'R2_d{h}']:.3f}")
        except Exception as exc:
            _log(logs_col, model_type, "error", {"error": str(exc)})
            print(f"  {model_type} FAILED: {exc}")

    if not results:
        raise RuntimeError("All models failed to train.")

    best_type, best_rmse, best_id = min(results, key=lambda r: r[1])
    print(f"\nBest model: {best_type}  RMSE={best_rmse:.2f}  id={best_id}")
    print("Training complete.")


if __name__ == "__main__":
    run()
