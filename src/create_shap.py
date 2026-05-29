"""
SHAP explainability — computes mean |SHAP| values across all 4 forecast horizons
for the most recent prediction using the latest active LGBM model. Results are
persisted to MongoDB so the dashboard can load them without local files.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timezone

from config.db import (
    get_collection,
    load_model,
    COLLECTION_FEATURE_STORE,
    COLLECTION_MODEL_REGISTRY,
    COLLECTION_SHAP,
)

TOP_FEATURES = 15


def _latest_active_lgbm_type() -> str:
    doc = get_collection(COLLECTION_MODEL_REGISTRY).find_one(
        {"model_type": "lgbm", "status": "active"},
        sort=[("trained_at", -1)],
    )
    if doc is None:
        raise ValueError("No active lgbm model found in model_registry.")
    return doc["model_type"]


def _load_feature_store(feat_cols: list) -> pd.DataFrame:
    docs = list(
        get_collection(COLLECTION_FEATURE_STORE)
        .find({"AQI": {"$exists": True}}, {"_id": 0})
        .sort("date", -1)
        .limit(200)
    )
    df = pd.DataFrame(docs)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df[feat_cols].dropna()


def run() -> None:
    import shap

    model_type = _latest_active_lgbm_type()
    model, scaler, metadata = load_model(model_type)
    feat_cols = metadata["features"]

    df = _load_feature_store(feat_cols)
    if df.empty:
        raise RuntimeError("feature_store has no processable rows.")

    X = df.values.astype(float)
    X_sc = scaler.transform(X)
    X_sc_df = pd.DataFrame(X_sc, columns=feat_cols)

    # model.models is a list of 4 LGBMRegressors (one per horizon)
    horizon_shap_values = []
    for lgbm in model.models:
        explainer = shap.TreeExplainer(lgbm)
        sv = explainer.shap_values(X_sc_df)  # shape (n_rows, n_features)
        horizon_shap_values.append(np.abs(sv))

    # Average mean |SHAP| across all 4 horizons, computed on the whole background set
    mean_abs_shap = np.mean([sv.mean(axis=0) for sv in horizon_shap_values], axis=0)

    importance = pd.DataFrame({"feature": feat_cols, "importance": mean_abs_shap})
    importance = importance.sort_values("importance", ascending=False).head(TOP_FEATURES)

    exp_list = [
        {"feature": row["feature"], "importance": float(row["importance"])}
        for _, row in importance.iterrows()
    ]

    get_collection(COLLECTION_SHAP).insert_one({
        "created_at":  datetime.now(timezone.utc),
        "model_type":  model_type,
        "explanation": exp_list,
    })

    print(f"SHAP: top {TOP_FEATURES} features saved to '{COLLECTION_SHAP}'")
    for entry in exp_list[:5]:
        print(f"  {entry['feature']:<40}  {entry['importance']:.4f}")


if __name__ == "__main__":
    run()
