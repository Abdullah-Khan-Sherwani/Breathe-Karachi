"""
Inference — loads the active model from model_registry, builds the appropriate
input, generates a 3-day AQI forecast, and writes one document to predictions.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pickle
from datetime import datetime, timezone, date, timedelta

import numpy as np
import pandas as pd

from config.db import (
    get_collection,
    load_model,
    COLLECTION_FEATURE_STORE,
    COLLECTION_PREDICTIONS,
    COLLECTION_MODEL_REGISTRY,
)

SEQ_LEN = 7


def _latest_active_type() -> str:
    """Return model_type of the most recently trained active model (any type)."""
    doc = get_collection(COLLECTION_MODEL_REGISTRY).find_one(
        {"status": "active"},
        sort=[("trained_at", -1)],
    )
    if doc is None:
        raise ValueError("No active model found in model_registry.")
    return doc["model_type"]


def _load_feature_store(n_rows: int) -> pd.DataFrame:
    docs = list(
        get_collection(COLLECTION_FEATURE_STORE)
        .find({"AQI": {"$exists": True}}, {"_id": 0})
        .sort("date", -1)
        .limit(n_rows)
    )
    if not docs:
        raise RuntimeError("feature_store is empty.")
    df = pd.DataFrame(docs)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _feature_cols(metadata: dict) -> list[str]:
    return metadata["features"]


def _predict_tabular(model, scaler, feat_cols: list, df: pd.DataFrame) -> np.ndarray:
    row = df.iloc[[-1]][feat_cols].values
    row_sc = scaler.transform(row)
    return model.predict(row_sc)[0]


def _predict_lstm(model, scaler, feat_cols: list, df: pd.DataFrame) -> np.ndarray:
    if len(df) < SEQ_LEN:
        raise RuntimeError(f"Need at least {SEQ_LEN} rows in feature_store for LSTM inference.")
    seq = df.iloc[-SEQ_LEN:][feat_cols].values
    seq_sc = scaler.transform(seq)
    return model.predict(seq_sc[np.newaxis, ...], verbose=0)[0]


def run() -> None:
    model_type = _latest_active_type()
    model, scaler, metadata = load_model(model_type)
    feat_cols = _feature_cols(metadata)
    model_id  = metadata["_id"]

    n_rows = SEQ_LEN if model_type == "lstm" else 1
    df = _load_feature_store(n_rows + 10)  # extra rows as buffer

    if model_type == "lstm":
        predictions = _predict_lstm(model, scaler, feat_cols, df)
    else:
        predictions = _predict_tabular(model, scaler, feat_cols, df)

    last_date = df["date"].max().date()
    forecasts = [
        {"date": (last_date + timedelta(days=i + 1)).isoformat(),
         "predicted_AQI": float(predictions[i])}
        for i in range(3)
    ]

    doc = {
        "predicted_at": datetime.now(timezone.utc),
        "model_type":   model_type,
        "model_id":     model_id,
        "forecasts":    forecasts,
    }
    get_collection(COLLECTION_PREDICTIONS).insert_one(doc)

    for f in forecasts:
        print(f"  {f['date']}: AQI = {f['predicted_AQI']:.1f}")
    print("Forecast written to predictions.")


if __name__ == "__main__":
    run()
