"""
LIME explainability — explains the most recent prediction using the latest
active model. Outputs HTML, CSV, and PNG to lime_explanations/.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from config.db import (
    get_collection,
    load_model,
    COLLECTION_FEATURE_STORE,
    COLLECTION_MODEL_REGISTRY,
    COLLECTION_LIME,
)

SEQ_LEN     = 7
OUT_DIR     = Path(__file__).parent.parent / "lime_explanations"
TOP_FEATURES = 15


def _latest_active_type() -> str:
    doc = get_collection(COLLECTION_MODEL_REGISTRY).find_one(
        {"status": "active"},
        sort=[("trained_at", -1)],
    )
    if doc is None:
        raise ValueError("No active model found.")
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


def _make_predict_fn(model, scaler, model_type: str, feat_cols: list):
    """Return a function (n_samples, n_features) → (n_samples,) for LIME."""
    def predict_tabular(X: np.ndarray) -> np.ndarray:
        X_sc = scaler.transform(X)
        preds = model.predict(X_sc)
        return preds[:, 0]  # return AQI_t+1

    def predict_lstm(X: np.ndarray) -> np.ndarray:
        results = []
        for row in X:
            # tile single row into SEQ_LEN steps so LSTM sees a valid sequence
            seq = np.tile(row, (SEQ_LEN, 1))
            seq_sc = scaler.transform(seq)
            pred = model.predict(seq_sc[np.newaxis, ...], verbose=0)[0]
            results.append(pred[0])
        return np.array(results)

    return predict_lstm if model_type == "lstm" else predict_tabular


def run() -> None:
    import lime.lime_tabular
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(exist_ok=True)

    model_type = _latest_active_type()
    model, scaler, metadata = load_model(model_type)
    feat_cols = metadata["features"]

    df = _load_feature_store(feat_cols)
    if df.empty:
        raise RuntimeError("feature_store has no processable rows.")

    train_data = df.values
    instance   = train_data[-1]

    predict_fn = _make_predict_fn(model, scaler, model_type, feat_cols)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=train_data,
        feature_names=feat_cols,
        mode="regression",
        discretize_continuous=True,
        random_state=42,
    )

    explanation = explainer.explain_instance(
        data_row=instance,
        predict_fn=predict_fn,
        num_features=TOP_FEATURES,
    )

    # HTML
    explanation.save_to_file(str(OUT_DIR / "lime_explanation.html"))

    # CSV
    exp_list = explanation.as_list()
    pd.DataFrame(exp_list, columns=["feature", "weight"]).to_csv(
        OUT_DIR / "lime_explanation.csv", index=False
    )

    # PNG bar chart
    features, weights = zip(*exp_list)
    colors = ["#e74c3c" if w < 0 else "#2ecc71" for w in weights]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(features)), weights, color=colors)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=9)
    ax.set_xlabel("LIME weight (impact on AQI_t+1)")
    ax.set_title(f"LIME Explanation — {model_type} — top {TOP_FEATURES} features")
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "lime_explanation.png", dpi=150)
    plt.close(fig)

    # Persist to MongoDB so the dashboard can load it without local files
    get_collection(COLLECTION_LIME).insert_one({
        "created_at":  datetime.now(timezone.utc),
        "model_type":  model_type,
        "explanation": [{"feature": f, "weight": w} for f, w in exp_list],
    })

    print(f"LIME artefacts saved to {OUT_DIR}")


if __name__ == "__main__":
    run()
