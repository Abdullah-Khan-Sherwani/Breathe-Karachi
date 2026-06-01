"""
SHAP explainability — computes mean |SHAP| values for all active models and their
NNLS ensemble. Results are persisted to MongoDB so the dashboard can load them.

Explainer strategy:
  LGBM  → shap.TreeExplainer     (exact, fast; one explainer per horizon)
  Ridge → shap.LinearExplainer   (exact, fast; one explainer per horizon)
  LSTM  → Expected Gradients (GradientSHAP), implemented natively with
                                  tf.GradientTape. shap's GradientExplainer/
                                  DeepExplainer rely on the removed
                                  tf.keras.backend.learning_phase and are broken
                                  on Keras 3 / TF 2.17. Expected Gradients is the
                                  same Shapley-consistent estimand GradientExplainer
                                  targets: it attributes over the real SEQ_LEN-day
                                  window against baselines sampled from the data.
  Ensemble → weighted sum of component SHAP values (valid: SHAP is linear for Σ wᵢfᵢ)
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
    COLLECTION_SHAP,
    COLLECTION_ENSEMBLE,
)

TOP_FEATURES = 15
SEQ_LEN = 7

# LSTM Expected-Gradients budget
LSTM_EG_BASELINES = 20    # reference windows sampled from the data per instance
LSTM_EG_STEPS     = 20    # interpolation steps along each baseline->input path
LSTM_EXPLAIN_N    = 10    # most-recent windows to explain


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


def _try_load(model_type: str):
    try:
        return load_model(model_type)
    except Exception:
        return None


def _shap_lgbm(model, scaler, feat_cols: list, df_vals: np.ndarray) -> np.ndarray:
    """TreeExplainer on each of the 4 per-horizon LGBMRegressors; average mean|SHAP|."""
    import shap
    X_sc = scaler.transform(df_vals)
    X_df = pd.DataFrame(X_sc, columns=feat_cols)
    per_horizon = []
    for lgbm in model.models:
        explainer = shap.TreeExplainer(lgbm)
        sv = explainer.shap_values(X_df)    # (n, n_features)
        per_horizon.append(np.abs(sv).mean(axis=0))
    return np.mean(per_horizon, axis=0)     # (n_features,)


def _shap_ridge(model, scaler, feat_cols: list, df_vals: np.ndarray) -> np.ndarray:
    """LinearExplainer on each of the 4 Ridge estimators in MultiOutputRegressor."""
    import shap
    X_sc = scaler.transform(df_vals)
    X_df = pd.DataFrame(X_sc, columns=feat_cols)
    per_horizon = []
    for ridge in model.estimators_:         # one Ridge per horizon
        explainer = shap.LinearExplainer(ridge, X_df)
        sv = explainer.shap_values(X_df)    # (n, n_features)
        per_horizon.append(np.abs(sv).mean(axis=0))
    return np.mean(per_horizon, axis=0)     # (n_features,)


def _shap_lstm(model, scaler, feat_cols: list, df_vals: np.ndarray) -> np.ndarray:
    """
    Expected Gradients (GradientSHAP) for the Keras LSTM — a Shapley-consistent
    attribution computed natively with tf.GradientTape (shap's deep explainers
    are broken on Keras 3 / TF 2.17). For each explained window we integrate the
    model gradient along straight-line paths from baselines sampled out of the
    data, average over baselines, sum the per-(timestep, feature) attributions
    over the SEQ_LEN window to a per-feature value, and average |attribution|
    across the 4 horizons. Operates in the scaled feature space.
    """
    import tensorflow as tf
    x_sc, y_sc = scaler if isinstance(scaler, tuple) else (scaler, None)

    X_sc = x_sc.transform(df_vals).astype("float32")            # (n, n_features)
    seqs = np.array(
        [X_sc[i:i + SEQ_LEN] for i in range(len(X_sc) - SEQ_LEN + 1)],
        dtype="float32",
    )                                                           # (n_seq, SEQ_LEN, F)
    if len(seqs) < 2:
        raise ValueError("Too few rows to build LSTM sequences.")

    F = len(feat_cols)
    rng = np.random.default_rng(42)
    n_base  = min(LSTM_EG_BASELINES, len(seqs))
    explain = seqs[-min(LSTM_EXPLAIN_N, len(seqs)):]
    alphas  = np.linspace(0.0, 1.0, LSTM_EG_STEPS + 1, dtype="float32")

    def _grad_h(inp: np.ndarray, h: int) -> np.ndarray:
        x = tf.convert_to_tensor(inp)
        with tf.GradientTape() as tape:
            tape.watch(x)
            out = model(x, training=False)[:, h]
        return tape.gradient(out, x).numpy()

    per_horizon = []
    for h in range(4):
        feat_attr = np.zeros(F)
        for x in explain:                                       # x: (SEQ_LEN, F)
            base = seqs[rng.choice(len(seqs), size=n_base, replace=False)]   # (B, SEQ, F)
            diff = x[None] - base                                            # (B, SEQ, F)
            # interpolate each baseline->x path: (B, STEPS+1, SEQ, F)
            interp = base[:, None] + alphas[None, :, None, None] * diff[:, None]
            grads = _grad_h(interp.reshape(-1, SEQ_LEN, F), h).reshape(
                n_base, LSTM_EG_STEPS + 1, SEQ_LEN, F)
            avg_grads = ((grads[:, :-1] + grads[:, 1:]) / 2.0).mean(axis=1)   # trapezoid -> (B, SEQ, F)
            ig = (diff * avg_grads).mean(axis=0)                             # avg baselines -> (SEQ, F)
            feat_attr += np.abs(ig.sum(axis=0))                             # sum over timesteps -> (F,)
        # Convert from scaled-output to original AQI units so the LSTM is
        # commensurate with the LGBM/Ridge SHAP in the ensemble combination.
        scale_h = float(y_sc.scale_[h]) if y_sc is not None else 1.0
        per_horizon.append((feat_attr / len(explain)) * scale_h)
    return np.mean(per_horizon, axis=0)                         # (n_features,)


_SHAP_FN = {
    "lgbm":  _shap_lgbm,
    "ridge": _shap_ridge,
    "lstm":  _shap_lstm,
}


def run() -> None:
    import shap  # noqa: F401 — ensures shap is importable before any work starts

    # ── Load ensemble weights ──────────────────────────────────────────────────
    ens_doc     = get_collection(COLLECTION_ENSEMBLE).find_one({})
    ens_order   = ens_doc["order"]   if ens_doc else []
    ens_weights = ens_doc["weights"] if ens_doc else []  # list of 4 weight vectors

    # ── Load models ────────────────────────────────────────────────────────────
    loaded: dict = {}
    for mt in ["lgbm", "ridge", "lstm"]:
        result = _try_load(mt)
        if result is not None:
            loaded[mt] = result
            print(f"Loaded {mt}")
        else:
            print(f"  {mt} not found in registry — skipping")

    if not loaded:
        raise RuntimeError("No active models found in model_registry.")

    # Use feat_cols from LGBM if available, otherwise first loaded model
    primary_type = "lgbm" if "lgbm" in loaded else next(iter(loaded))
    _, _, primary_meta = loaded[primary_type]
    feat_cols = primary_meta["features"]

    # ── Compute per-model SHAP ─────────────────────────────────────────────────
    per_model_importance: dict[str, np.ndarray] = {}
    per_model_feats:      dict[str, list]        = {}

    for mt, (model, scaler, meta) in loaded.items():
        mfeat = meta["features"]
        try:
            print(f"Computing SHAP for {mt}...")
            df_m = _load_feature_store(mfeat)
            if df_m.empty:
                print(f"  {mt}: no rows — skip")
                continue
            imp = _SHAP_FN[mt](model, scaler, mfeat, df_m.values.astype(float))
            per_model_importance[mt] = imp
            per_model_feats[mt]      = mfeat
            top_feat = mfeat[int(np.argmax(imp))]
            print(f"  {mt}: done  (top feature: {top_feat}  {imp.max():.4f})")
        except Exception as exc:
            print(f"  {mt} SHAP failed: {exc}")

    if not per_model_importance:
        raise RuntimeError("All model SHAP computations failed.")

    # ── Compute ensemble SHAP (exact for linear combinations) ─────────────────
    # Φ(Σ wᵢfᵢ) = Σ wᵢ Φ(fᵢ)  — valid because SHAP satisfies linearity
    ensemble_importance: np.ndarray | None = None
    ensemble_feat_cols  = feat_cols

    if ens_weights and len(per_model_importance) >= 2:
        # Average the per-horizon weights into a single scalar weight per model type
        n_horizons = len(ens_weights)
        avg_w: dict[str, float] = {mt: 0.0 for mt in ens_order}
        for w_vec in ens_weights:
            for i, mt in enumerate(ens_order):
                avg_w[mt] = avg_w.get(mt, 0.0) + w_vec[i] / n_horizons

        ens_vec   = np.zeros(len(feat_cols))
        total_w   = 0.0
        for mt, imp in per_model_importance.items():
            w = avg_w.get(mt, 0.0)
            if w <= 0 or per_model_feats[mt] != feat_cols:
                continue        # skip if not in ensemble or different feature set
            ens_vec  += w * imp
            total_w  += w

        if total_w > 0:
            ensemble_importance = ens_vec / total_w
            print(f"Ensemble SHAP computed (contributing models: "
                  f"{[mt for mt in per_model_importance if avg_w.get(mt, 0) > 0]}, "
                  f"total_w={total_w:.3f})")

    if ensemble_importance is None:
        # Fall back to primary model
        ensemble_importance = per_model_importance.get(primary_type,
                              next(iter(per_model_importance.values())))
        ensemble_feat_cols  = per_model_feats.get(primary_type,
                              next(iter(per_model_feats.values())))
        print(f"Ensemble SHAP unavailable — using {primary_type} as primary.")

    # ── Build top-N ranked lists ───────────────────────────────────────────────
    def _top_n(features: list, importance: np.ndarray, n: int = TOP_FEATURES) -> list:
        df_imp = pd.DataFrame({"feature": features, "importance": importance})
        df_imp = df_imp.sort_values("importance", ascending=False).head(n)
        return [{"feature": r["feature"], "importance": float(r["importance"])}
                for _, r in df_imp.iterrows()]

    ensemble_list  = _top_n(ensemble_feat_cols, ensemble_importance)
    per_model_list = {
        mt: _top_n(per_model_feats[mt], per_model_importance[mt])
        for mt in per_model_importance
    }

    # ── Persist to MongoDB ─────────────────────────────────────────────────────
    get_collection(COLLECTION_SHAP).insert_one({
        "created_at":  datetime.now(timezone.utc),
        "model_type":  "ensemble",
        "explanation": ensemble_list,   # top-N ensemble importances (dashboard reads this)
        "per_model":   per_model_list,  # per-model breakdown (for report / extended tab)
    })

    print(f"\nSHAP saved — top {TOP_FEATURES} ensemble features:")
    for entry in ensemble_list[:5]:
        print(f"  {entry['feature']:<40}  {entry['importance']:.4f}")
    print(f"\nPer-model coverage: {list(per_model_list.keys())}")


if __name__ == "__main__":
    run()
