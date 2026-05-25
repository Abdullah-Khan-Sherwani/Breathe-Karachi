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

TARGET_COLS  = ["AQI_t+1", "AQI_t+2", "AQI_t+3"]
EXCLUDE_COLS = {"date", "processed_at", "_id"} | set(TARGET_COLS)


def load_data() -> pd.DataFrame:
    docs = list(get_collection(COLLECTION_FEATURE_STORE).find({}, {"_id": 0}))
    if not docs:
        raise RuntimeError("feature_store is empty — run fetch_data.py and preprocess_daily_data.py first.")
    df = pd.DataFrame(docs)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=TARGET_COLS)
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def time_split(df: pd.DataFrame, test_days: int = 30):
    split = df["date"].max() - pd.Timedelta(days=test_days)
    train = df[df["date"] <= split]
    test  = df[df["date"] >  split]
    return train, test


def _log(col, model_type: str, status: str, metrics: dict, model_id=None):
    col.insert_one({
        "timestamp":  datetime.now(timezone.utc),
        "status":     status,
        "model_type": model_type,
        "model_id":   str(model_id) if model_id else None,
        **metrics,
    })


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
        except Exception as exc:
            _log(logs_col, model_type, "error", {"error": str(exc)})
            print(f"  {model_type} FAILED: {exc}")

    if not results:
        raise RuntimeError("All models failed to train.")

    # Mark all but the best as inactive (save_model already handles per-type retirement,
    # but we also want a single "champion" across types)
    best_type, best_rmse, best_id = min(results, key=lambda r: r[1])
    print(f"\nBest model: {best_type}  RMSE={best_rmse:.2f}  id={best_id}")

    # Retire non-champion models so load_model('best') could find the winner if needed
    # Convention: leave each model's own active record; callers choose model_type explicitly.
    print("Training complete.")


if __name__ == "__main__":
    run()
