"""
SILENT SHADOW forecast that INCLUDES PM2_5 lead features (which prod deliberately
excludes). Trains the full LGBM + LSTM + Ridge ensemble on all labeled data with
PM2_5_t1..t4 added back, predicts the next 4 days from the latest feature_store row
(PM2_5 leads come from the CAMS forecast already in the store), and writes ONE doc
to a SEPARATE collection `predictions_pm25` for side-by-side comparison with prod.

Fully isolated: prod predict.py / train.py / collections are untouched. Run it
alongside the normal pipeline (e.g. right after predict.py).
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from datetime import datetime, timezone, timedelta

import numpy as np

from config.db import get_collection
from src.train import (load_data, EXCLUDE_COLS, TARGET_COLS, time_split,
                       _perturb_lead_features, _nnls_weights)
from src.models import train_ridge, train_lgbm, train_lstm, \
    train_ridge_full, train_lgbm_full, train_lstm_full
from src.predict import _load_feature_store, _predict_tabular, _predict_lstm, SEQ_LEN

PM25 = {"PM2_5_t1", "PM2_5_t2", "PM2_5_t3", "PM2_5_t4"}
SHADOW_COLLECTION = "predictions_pm25"
ORDER = ["lgbm", "lstm", "ridge"]
NH = len(TARGET_COLS)
_EVAL = {"lgbm": train_lgbm, "lstm": train_lstm, "ridge": train_ridge}
_FULL = {"lgbm": train_lgbm_full, "lstm": train_lstm_full, "ridge": train_ridge_full}
_INFER = {"lgbm": _predict_tabular, "ridge": _predict_tabular, "lstm": _predict_lstm}


def run() -> None:
    df = load_data()
    feat = [c for c in df.columns if c not in (EXCLUDE_COLS - PM25)]   # prod set + PM2_5 leads
    _perturb_lead_features(df, feat)
    print(f"[shadow] features={len(feat)} (incl PM2_5 leads) | labeled rows={len(df)}")

    # 1) ensemble weights from a 60-day holdout (per-horizon nnls), like train.py
    tr, te = time_split(df, test_days=60)
    Xtr, ytr = tr[feat].values, tr[TARGET_COLS].values
    Xte, yte = te[feat].values, te[TARGET_COLS].values
    preds = {}
    for m in ORDER:
        _, _, met, _ = _EVAL[m](Xtr, ytr, Xte, yte)
        preds[m] = np.asarray(met.pop("_preds"))
        print(f"[shadow] holdout {m}: MAE={met['MAE']:.2f} RMSE={met['RMSE']:.2f} R2={met['R2']:.3f}")
    weights = []
    for h in range(NH):
        A = np.column_stack([preds[m][:, h] for m in ORDER])
        weights.append(_nnls_weights(A, yte[:, h], len(ORDER)))

    # 2) full retrain on every labeled row
    Xf, yf = df[feat].values, df[TARGET_COLS].values
    fitted = {}
    for m in ORDER:
        model, scaler = _FULL[m](Xf, yf)
        fitted[m] = (model, scaler)
        print(f"[shadow] full-retrained {m} on {len(Xf)} rows")

    # 3) inference from the latest feature_store row(s)
    live = _load_feature_store(SEQ_LEN + 10)
    anchor = live["date"].max().date()
    comp = {m: np.asarray(_INFER[m](fitted[m][0], fitted[m][1], feat, live)) for m in ORDER}

    norm_w = [weights[h] / weights[h].sum() for h in range(NH)]
    blended = np.array([sum(norm_w[h][i] * comp[m][h] for i, m in enumerate(ORDER))
                        for h in range(NH)])

    # Store the ensemble forecast AND each component model's forecast, so the
    # shadow can be compared model-by-model against prod.
    forecasts = [{
        "date":          (anchor + timedelta(days=k + 1)).isoformat(),
        "predicted_AQI": float(blended[k]),                       # nnls ensemble
        "components":    {m: float(comp[m][k]) for m in ORDER},   # lgbm/lstm/ridge
    } for k in range(NH)]
    print(f"[shadow] anchor={anchor}  (order={ORDER})")
    for k, f in enumerate(forecasts):
        comps = "  ".join(f"{m}={f['components'][m]:.1f}" for m in ORDER)
        print(f"  {f['date']}: ensemble={f['predicted_AQI']:.1f}  ({comps})")

    get_collection(SHADOW_COLLECTION).insert_one({
        "predicted_at":       datetime.now(timezone.utc),
        "model_type":         "ensemble_pm25_shadow",
        "anchor_date":        anchor.isoformat(),
        "features_incl_pm25": True,
        "ensemble_order":     ORDER,
        "ensemble_weights":   [w.tolist() for w in norm_w],
        "forecasts":          forecasts,
    })
    print(f"[shadow] written to `{SHADOW_COLLECTION}` (separate from prod predictions).")


if __name__ == "__main__":
    run()
