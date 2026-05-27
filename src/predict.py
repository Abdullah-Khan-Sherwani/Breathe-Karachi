"""
Inference — loads the active lgbm and lstm models from model_registry, ensembles
their predictions with per-horizon weights, and writes one document to predictions.
Falls back to whichever model type is available if one is missing.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from config.db import (
    get_collection,
    load_model,
    COLLECTION_FEATURE_STORE,
    COLLECTION_PREDICTIONS,
)

SEQ_LEN = 7

# Per-horizon blend weights [lgbm, lstm] — derived from holdout analysis
_ENSEMBLE_WEIGHTS = [
    (0.60, 0.40),   # day 1: LGBM dominates
    (0.15, 0.85),   # day 2: LSTM dominates
    (0.00, 1.00),   # day 3: LSTM only
    (0.00, 1.00),   # day 4: LSTM only
]


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


def _predict_tabular(model, scaler, feat_cols: list, df: pd.DataFrame) -> np.ndarray:
    row = df.iloc[[-1]][feat_cols].values.astype(float)
    # If the forecast API failed, some lead features may be NaN — fill with column median.
    if np.isnan(row).any():
        col_medians = np.nanmedian(df[feat_cols].values, axis=0)
        nan_mask = np.isnan(row[0])
        row[0, nan_mask] = col_medians[nan_mask]
    row_sc = scaler.transform(row)
    return model.predict(row_sc)[0]


def _predict_lstm(model, scaler, feat_cols: list, df: pd.DataFrame) -> np.ndarray:
    if len(df) < SEQ_LEN:
        raise RuntimeError(f"Need at least {SEQ_LEN} rows in feature_store for LSTM inference.")
    x_sc, y_sc = scaler if isinstance(scaler, tuple) else (scaler, None)
    seq = df.iloc[-SEQ_LEN:][feat_cols].values.astype(float)
    # Fill any NaN (e.g. forecast API failure) with column median from available rows.
    if np.isnan(seq).any():
        col_medians = np.nanmedian(df[feat_cols].values, axis=0)
        nan_mask = np.isnan(seq)
        seq[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
    seq_sc = x_sc.transform(seq)
    pred_sc = model.predict(seq_sc[np.newaxis, ...], verbose=0)
    if y_sc is not None:
        return y_sc.inverse_transform(pred_sc)[0]
    return pred_sc[0]


def _try_load(model_type: str):
    """Return (model, scaler, metadata) or None if not found."""
    try:
        return load_model(model_type)
    except ValueError:
        return None


def run() -> None:
    df = _load_feature_store(SEQ_LEN + 10)

    lgbm_result = _try_load("lgbm")
    lstm_result = _try_load("lstm")

    if lgbm_result is None and lstm_result is None:
        raise ValueError("No active lgbm or lstm model found in model_registry.")

    preds_lgbm: np.ndarray | None = None
    preds_lstm: np.ndarray | None = None
    component_models: dict = {}

    if lgbm_result is not None:
        lgbm_model, lgbm_scaler, lgbm_meta = lgbm_result
        preds_lgbm = _predict_tabular(lgbm_model, lgbm_scaler, lgbm_meta["features"], df)
        component_models["lgbm"] = lgbm_meta["_id"]

    if lstm_result is not None:
        lstm_model, lstm_scaler, lstm_meta = lstm_result
        preds_lstm = _predict_lstm(lstm_model, lstm_scaler, lstm_meta["features"], df)
        component_models["lstm"] = lstm_meta["_id"]

    predictions = np.zeros(4)
    for d, (w_lgbm, w_lstm) in enumerate(_ENSEMBLE_WEIGHTS):
        if preds_lgbm is not None and preds_lstm is not None:
            predictions[d] = w_lgbm * preds_lgbm[d] + w_lstm * preds_lstm[d]
        elif preds_lgbm is not None:
            predictions[d] = preds_lgbm[d]
        else:
            predictions[d] = preds_lstm[d]

    last_date = df["date"].max().date()
    forecasts = [
        {"date": (last_date + timedelta(days=i + 1)).isoformat(),
         "predicted_AQI": float(predictions[i])}
        for i in range(4)
    ]

    doc = {
        "predicted_at":    datetime.now(timezone.utc),
        "model_type":      "ensemble",
        "component_models": component_models,
        "forecasts":       forecasts,
    }
    get_collection(COLLECTION_PREDICTIONS).insert_one(doc)

    for f in forecasts:
        print(f"  {f['date']}: AQI = {f['predicted_AQI']:.1f}")
    print("Forecast written to predictions.")


if __name__ == "__main__":
    run()
