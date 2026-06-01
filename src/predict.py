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
    COLLECTION_ENSEMBLE,
)

SEQ_LEN = 7


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

    last_date = df["date"].max().date()
    today = datetime.now(timezone.utc).date()
    staleness_days = (today - last_date).days

    print(f"[predict] Anchor date (latest feature_store row): {last_date}")
    print(f"[predict] Today (UTC): {today}  |  Staleness: {staleness_days} day(s)")
    if staleness_days > 2:
        print(f"[predict] WARNING: Feature store is {staleness_days} days stale — "
              f"predictions anchored to {last_date}, not today.")

    # Load ensemble weights from MongoDB (computed during training via nnls)
    ens_doc = get_collection(COLLECTION_ENSEMBLE).find_one({})
    if ens_doc is None:
        raise ValueError("No ensemble_config found — run train.py first to compute weights.")
    ens_order   = ens_doc["order"]                        # e.g. ["lgbm", "lstm", "ridge"]
    ens_weights = [np.array(w) for w in ens_doc["weights"]]  # list of 4 weight vectors

    _PREDICT_FN = {
        "lgbm":  _predict_tabular,
        "ridge": _predict_tabular,
        "lstm":  _predict_lstm,
    }

    component_preds: dict[str, np.ndarray] = {}
    component_models: dict = {}
    for model_type in ens_order:
        result = _try_load(model_type)
        if result is None:
            print(f"[predict] {model_type.upper()} not found — skipping.")
            continue
        model, scaler, meta = result
        print(f"[predict] Loaded {model_type.upper()}  id={meta['_id']}  trained_at={meta.get('trained_at','?')}")
        component_preds[model_type] = _PREDICT_FN[model_type](model, scaler, meta["features"], df)
        component_models[model_type] = meta["_id"]

    if not component_preds:
        raise ValueError("No models loaded — cannot predict.")

    # Blend using per-horizon nnls weights; fall back to equal weight for missing models
    predictions = np.zeros(4)
    for h in range(4):
        available = [m for m in ens_order if m in component_preds]
        w = np.array([ens_weights[h][ens_order.index(m)] for m in available])
        w = w / w.sum()
        predictions[h] = sum(w[i] * component_preds[m][h] for i, m in enumerate(available))

    print()
    print(f"[predict] Ensemble order: {ens_order}")
    print(f"  {'Date':<12}  " + "  ".join(f"{m.upper():>7}" for m in ens_order) + f"  {'Ensemble':>9}")
    forecasts = []
    for i in range(4):
        fdate = (last_date + timedelta(days=i + 1)).isoformat()
        vals  = "  ".join(f"{component_preds[m][i]:>7.1f}" if m in component_preds else f"{'n/a':>7}" for m in ens_order)
        w_str = "/".join(f"{ens_weights[i][j]:.2f}" for j in range(len(ens_order)))
        print(f"  {fdate:<12}  {vals}  {predictions[i]:>9.1f}  w=[{w_str}]")
        forecasts.append({"date": fdate, "predicted_AQI": float(predictions[i])})

    doc = {
        "predicted_at":     datetime.now(timezone.utc),
        "model_type":       "ensemble",
        "component_models": component_models,
        "anchor_date":      last_date.isoformat(),
        "forecasts":        forecasts,
    }
    get_collection(COLLECTION_PREDICTIONS).insert_one(doc)
    print()
    print("Forecast written to predictions.")


if __name__ == "__main__":
    run()
