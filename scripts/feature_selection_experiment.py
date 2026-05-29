"""
Feature Selection Experiment — Karachi AQI Predictor
=====================================================
Standalone experiment script. Prints all results to stdout.
No models are saved to MongoDB, disk, or anywhere.

Usage:
    python scripts/feature_selection_experiment.py              # all 4 horizons, full run
    python scripts/feature_selection_experiment.py --horizon 1  # single horizon
    python scripts/feature_selection_experiment.py --quick      # 3-split RFECV, skip LSTM
    python scripts/feature_selection_experiment.py --horizon 1 2 --quick
"""

import sys
import argparse
import time
import warnings
from pathlib import Path

# Allow imports from the project root
sys.path.append(str(Path(__file__).parent.parent))

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_regression, RFECV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

# ---------------------------------------------------------------------------
# Re-use data-loading logic from the main training pipeline
# ---------------------------------------------------------------------------
from src.train import load_data, get_feature_cols, time_split, TARGET_COLS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOP_N       = 30          # features to show in ranking tables
LGBM_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)
LGBM_PARAMS_FAST = dict(           # lighter params for RFECV inner loops
    n_estimators=200,
    learning_rate=0.08,
    max_depth=5,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)
SEQ_LEN = 7       # LSTM lookback window, matches lstm_model.py


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _section(title: str) -> None:
    print()
    print(f"--- {title} ---")


def _print_ranked_table(feature_names, scores, label: str, top_n: int = TOP_N) -> list:
    """Print a sorted ranking table and return list of top-N feature names."""
    order   = np.argsort(scores)[::-1]
    top_idx = order[:top_n]
    print(f"\n  Rank  {'Feature':<45}  {label}")
    print(f"  {'----':<5}  {'-------':<45}  {'-----'}")
    for rank, idx in enumerate(top_idx, start=1):
        print(f"  {rank:<5}  {feature_names[idx]:<45}  {scores[idx]:.6f}")
    return [feature_names[i] for i in top_idx]


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ---------------------------------------------------------------------------
# Section 1 — Mutual Information ranking
# ---------------------------------------------------------------------------

def run_mutual_information(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    horizons: list[int],
) -> dict[int, list[str]]:
    """
    Compute mutual_info_regression for each requested horizon.
    Returns {horizon_index: [top-N feature names]}.
    """
    _banner("SECTION 1 — MUTUAL INFORMATION RANKING")
    print(
        "  Estimating mutual information between every feature and each AQI target.\n"
        "  Higher MI score = stronger statistical dependency on the target."
    )

    results: dict[int, list[str]] = {}
    for h in horizons:
        target_name = TARGET_COLS[h]
        _section(f"Horizon t+{h + 1}  ({target_name})")
        t0 = time.perf_counter()
        scores = mutual_info_regression(
            X_train, y_train[:, h], random_state=42, n_neighbors=5
        )
        elapsed = time.perf_counter() - t0
        top_feats = _print_ranked_table(feature_names, scores, "MI Score", TOP_N)
        results[h] = top_feats
        print(f"\n  Completed in {elapsed:.1f}s")

    return results


# ---------------------------------------------------------------------------
# Section 2 — LGBM built-in feature importance + Section 3 permutation importance
# ---------------------------------------------------------------------------

def run_lgbm_importances(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    horizons: list[int],
) -> tuple[dict[int, list[str]], dict[int, list[str]], dict[int, LGBMRegressor]]:
    """
    Train one LGBMRegressor per horizon on the train split.
    Section 2: built-in split-based feature_importances_.
    Section 3: permutation importance on the held-out test split.
    Returns (lgbm_top_feats, perm_top_feats, trained_models).
    """
    _banner("SECTION 2 — LGBM BUILT-IN FEATURE IMPORTANCE")
    print(
        "  Training LGBMRegressor per horizon (all features).\n"
        "  Importance = cumulative gain across all splits of each feature."
    )

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)

    lgbm_top: dict[int, list[str]] = {}
    perm_top: dict[int, list[str]] = {}
    models:   dict[int, LGBMRegressor] = {}

    for h in horizons:
        target_name = TARGET_COLS[h]
        _section(f"Horizon t+{h + 1}  ({target_name})")
        t0 = time.perf_counter()

        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(X_tr_sc, y_train[:, h])
        models[h] = model

        importances = model.feature_importances_.astype(float)
        top_feats   = _print_ranked_table(feature_names, importances, "Gain Importance", TOP_N)
        lgbm_top[h] = top_feats

        test_rmse  = _rmse(y_test[:, h], model.predict(X_te_sc))
        train_rmse = _rmse(y_train[:, h], model.predict(X_tr_sc))
        print(f"\n  Train RMSE: {train_rmse:.3f}  |  Test RMSE: {test_rmse:.3f}")
        print(f"  Elapsed: {time.perf_counter() - t0:.1f}s")

    # -----------------------------------------------------------------------
    _banner("SECTION 3 — PERMUTATION IMPORTANCE")
    print(
        "  Model-agnostic ranking: randomly shuffling each feature one at a time\n"
        "  and measuring how much test-set RMSE degrades.\n"
        "  Uses the same trained LGBM models from Section 2."
    )

    for h in horizons:
        target_name = TARGET_COLS[h]
        _section(f"Horizon t+{h + 1}  ({target_name})")
        t0 = time.perf_counter()

        scaler_h = StandardScaler()
        X_tr_sc  = scaler_h.fit_transform(X_train)
        X_te_sc  = scaler_h.transform(X_test)

        model = models[h]
        result = permutation_importance(
            model, X_te_sc, y_test[:, h],
            n_repeats=10,
            random_state=42,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        perm_means = result.importances_mean
        top_feats  = _print_ranked_table(feature_names, perm_means, "Mean RMSE Increase", TOP_N)
        perm_top[h] = top_feats
        print(f"\n  Elapsed: {time.perf_counter() - t0:.1f}s")

    return lgbm_top, perm_top, models


# ---------------------------------------------------------------------------
# Section 4 — RFECV with TimeSeriesSplit
# ---------------------------------------------------------------------------

def run_rfecv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    horizons: list[int],
    n_splits: int = 5,
) -> dict[int, list[str]]:
    """
    RFECV with TimeSeriesSplit to find the minimal feature subset that does not
    hurt RMSE vs. the full-feature model.
    Returns {horizon_index: [selected feature names]}.
    """
    _banner("SECTION 4 — RFECV  (Recursive Feature Elimination with CV)")
    print(
        f"  TimeSeriesSplit(n_splits={n_splits}) | scoring=neg_RMSE\n"
        "  Eliminates the weakest feature per iteration; keeps the count that\n"
        "  maximises cross-validated RMSE without degrading below full-feature baseline."
    )

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rfecv_top: dict[int, list[str]] = {}

    for h in horizons:
        target_name = TARGET_COLS[h]
        _section(f"Horizon t+{h + 1}  ({target_name})")
        t0 = time.perf_counter()

        estimator = LGBMRegressor(**LGBM_PARAMS_FAST)
        selector  = RFECV(
            estimator=estimator,
            step=1,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            min_features_to_select=5,
            n_jobs=-1,
        )
        selector.fit(X_tr_sc, y_train[:, h])

        selected_mask  = selector.support_
        selected_feats = [feature_names[i] for i, s in enumerate(selected_mask) if s]
        n_total        = len(feature_names)
        n_selected     = len(selected_feats)

        print(f"\n  Optimal feature count : {n_selected} / {n_total}")
        print(f"  CV RMSE (optimal)     : {-selector.cv_results_['mean_test_score'][selector.n_features_ - 1]:.4f}")
        print(f"  CV RMSE (full set)    : {-selector.cv_results_['mean_test_score'][-1]:.4f}")
        print(f"\n  Selected features ({n_selected}):")
        for feat in sorted(selected_feats):
            print(f"    {feat}")

        rfecv_top[h] = selected_feats
        print(f"\n  Elapsed: {time.perf_counter() - t0:.1f}s")

    return rfecv_top


# ---------------------------------------------------------------------------
# Section 5 — Consensus feature set
# ---------------------------------------------------------------------------

def compute_consensus(
    mi_top:    dict[int, list[str]],
    lgbm_top:  dict[int, list[str]],
    perm_top:  dict[int, list[str]],
    horizons:  list[int],
    min_methods:  int = 3,   # must be in top-30 of all 3 ranking methods
    min_horizons: int = 3,   # across at least this many of the chosen horizons
) -> list[str]:
    """
    Consensus = features that appear in top-{TOP_N} of MI + LGBM + Perm
    for at least min_horizons of the tested horizons.
    """
    _banner("SECTION 5 — CONSENSUS FEATURE SET")
    print(
        f"  A feature enters the consensus set when it appears in the top-{TOP_N}\n"
        f"  of ALL THREE ranking methods for at least {min_horizons} of "
        f"{len(horizons)} horizon(s) tested.\n"
        "  These are the safest candidates for inclusion in a leaner feature set."
    )

    horizon_votes: dict[str, int] = {}

    for h in horizons:
        in_mi   = set(mi_top.get(h, []))
        in_lgbm = set(lgbm_top.get(h, []))
        in_perm = set(perm_top.get(h, []))

        # Feature must be in ALL three ranking methods for this horizon
        triple_overlap = in_mi & in_lgbm & in_perm

        for feat in triple_overlap:
            horizon_votes[feat] = horizon_votes.get(feat, 0) + 1

    consensus = sorted(
        [feat for feat, votes in horizon_votes.items() if votes >= min_horizons],
        key=lambda f: -horizon_votes[f],
    )

    print(f"\n  Consensus set size: {len(consensus)} features")
    print(f"\n  {'Feature':<45}  {'Horizons agreeing'}")
    print(f"  {'-------':<45}  {'------------------'}")
    for feat in consensus:
        votes = horizon_votes[feat]
        print(f"  {feat:<45}  {votes} / {len(horizons)}")

    if not consensus:
        print(
            "\n  WARNING: Empty consensus set. "
            "Consider relaxing min_horizons or using union of per-method top sets."
        )

    return consensus


# ---------------------------------------------------------------------------
# Section 6 — LSTM quick-test: all features vs. consensus set
# ---------------------------------------------------------------------------

def _build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Slide a SEQ_LEN window across X, aligning each window to y[i + seq_len - 1]."""
    Xs, ys = [], []
    for i in range(len(X) - seq_len + 1):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len - 1])
    return np.array(Xs), np.array(ys)


def _build_lstm(n_features: int):
    """Replicate the same architecture used in lstm_model.py."""
    import tensorflow as tf
    from tensorflow.keras.regularizers import L2

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(SEQ_LEN, n_features)),
        tf.keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=L2(1e-3)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.LSTM(32, kernel_regularizer=L2(1e-3)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(4),
    ])
    model.compile(optimizer="adam", loss="huber")
    return model


def _train_and_eval_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label: str,
) -> dict:
    """
    Train LSTM with the same StandardScaler + sequence approach as lstm_model.py.
    Returns metrics dict with MAE, RMSE, R2 plus per-horizon breakdowns.
    No model is saved anywhere.
    """
    import tensorflow as tf
    tf.keras.backend.clear_session()

    x_sc = StandardScaler()
    y_sc = StandardScaler()

    X_tr_sc = x_sc.fit_transform(X_train)
    X_te_sc = x_sc.transform(X_test)

    y_tr_sc = y_sc.fit_transform(y_train)
    y_te_sc = y_sc.transform(y_test)

    # Build sequences across the full time-ordered dataset, then split
    X_all_sc = np.vstack([X_tr_sc, X_te_sc])
    y_all_sc = np.vstack([y_tr_sc, y_te_sc])

    X_seq, y_seq = _build_sequences(X_all_sc, y_all_sc, SEQ_LEN)

    n_train_seq = len(X_train) - SEQ_LEN + 1
    X_s_tr = X_seq[:n_train_seq]
    y_s_tr = y_seq[:n_train_seq]
    X_s_te = X_seq[n_train_seq:]
    y_s_te = y_seq[n_train_seq:]

    n_features = X_tr_sc.shape[1]
    print(f"  [{label}]  train sequences: {len(X_s_tr)}  |  test sequences: {len(X_s_te)}  |  features: {n_features}")

    model = _build_lstm(n_features)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=25, restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5
    )

    t0 = time.perf_counter()
    model.fit(
        X_s_tr, y_s_tr,
        validation_split=0.1,
        epochs=150,
        batch_size=16,
        callbacks=[early_stop, reduce_lr],
        verbose=0,
    )
    train_secs = time.perf_counter() - t0
    print(f"  [{label}]  training finished in {train_secs:.1f}s")

    if len(X_s_te) == 0:
        # Edge case: not enough data for even one test sequence
        print(f"  [{label}]  WARNING: no test sequences — using last training sequence for evaluation.")
        preds_sc = model.predict(X_s_tr[-1:], verbose=0)
        preds    = y_sc.inverse_transform(preds_sc)
        y_true   = y_sc.inverse_transform(y_s_tr[-1:])
    else:
        preds_sc = model.predict(X_s_te, verbose=0)
        preds    = y_sc.inverse_transform(preds_sc)
        y_true   = y_sc.inverse_transform(y_s_te)

    mae  = float(mean_absolute_error(y_true, preds))
    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    r2   = float(r2_score(y_true, preds))

    metrics: dict = {"MAE": mae, "RMSE": rmse, "R2": r2, "train_secs": train_secs}
    for i in range(preds.shape[1]):
        yh = y_true[:, i]
        ph = preds[:, i]
        metrics[f"MAE_d{i + 1}"]  = float(mean_absolute_error(yh, ph))
        metrics[f"RMSE_d{i + 1}"] = float(np.sqrt(mean_squared_error(yh, ph)))
        metrics[f"R2_d{i + 1}"]   = float(r2_score(yh, ph))

    # Model is intentionally discarded here — no saving
    tf.keras.backend.clear_session()

    return metrics


def _print_lstm_metrics(label: str, metrics: dict) -> None:
    print(f"\n  {label}")
    print(f"  {'':->50}")
    print(f"  Overall   MAE={metrics['MAE']:.3f}  RMSE={metrics['RMSE']:.3f}  R2={metrics['R2']:.4f}")
    for h in range(1, 5):
        k_mae  = f"MAE_d{h}"
        k_rmse = f"RMSE_d{h}"
        k_r2   = f"R2_d{h}"
        if k_mae in metrics:
            print(
                f"  t+{h}       "
                f"MAE={metrics[k_mae]:.3f}  "
                f"RMSE={metrics[k_rmse]:.3f}  "
                f"R2={metrics[k_r2]:.4f}"
            )


def run_lstm_comparison(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    consensus_feats: list[str],
) -> None:
    """Train two LSTMs (all features vs. consensus set) and print comparison."""
    _banner("SECTION 6 — LSTM COMPARISON: ALL FEATURES vs. CONSENSUS SET")

    if not consensus_feats:
        print(
            "  Skipping LSTM comparison — consensus set is empty.\n"
            "  Re-run with more horizons or relax min_horizons in compute_consensus()."
        )
        return

    n_all       = X_train.shape[1]
    n_consensus = len(consensus_feats)
    print(
        f"  All features : {n_all}\n"
        f"  Consensus    : {n_consensus}\n"
        f"  Reduction    : {n_all - n_consensus} features dropped "
        f"({(n_all - n_consensus) / n_all * 100:.1f}%)\n"
    )

    # --- Run A: all features ---
    _section("Run A — All Features")
    metrics_all = _train_and_eval_lstm(X_train, y_train, X_test, y_test, "ALL FEATURES")

    # --- Run B: consensus features only ---
    _section("Run B — Consensus Features Only")
    feat_idx      = [feature_names.index(f) for f in consensus_feats if f in feature_names]
    missing       = [f for f in consensus_feats if f not in feature_names]
    if missing:
        print(f"  WARNING: {len(missing)} consensus features not found in feature list: {missing}")

    X_tr_cs = X_train[:, feat_idx]
    X_te_cs = X_test[:, feat_idx]
    metrics_cs = _train_and_eval_lstm(X_tr_cs, y_train, X_te_cs, y_test, "CONSENSUS FEATURES")

    # --- Side-by-side comparison ---
    _section("Side-by-side Comparison")
    _print_lstm_metrics("All Features", metrics_all)
    _print_lstm_metrics("Consensus Features", metrics_cs)

    print(f"\n  Delta (Consensus - All Features):")
    print(f"  {'':->50}")
    delta_mae  = metrics_cs["MAE"]  - metrics_all["MAE"]
    delta_rmse = metrics_cs["RMSE"] - metrics_all["RMSE"]
    delta_r2   = metrics_cs["R2"]   - metrics_all["R2"]
    mae_pct    = delta_mae  / max(abs(metrics_all["MAE"]),  1e-9) * 100
    rmse_pct   = delta_rmse / max(abs(metrics_all["RMSE"]), 1e-9) * 100

    def _sign(v):
        return "+" if v >= 0 else ""

    print(f"  Overall   ΔMAE={_sign(delta_mae)}{delta_mae:.3f} ({_sign(mae_pct)}{mae_pct:.1f}%)  "
          f"ΔRMSE={_sign(delta_rmse)}{delta_rmse:.3f} ({_sign(rmse_pct)}{rmse_pct:.1f}%)  "
          f"ΔR2={_sign(delta_r2)}{delta_r2:.4f}")

    for h in range(1, 5):
        dm = metrics_cs.get(f"MAE_d{h}", float("nan"))  - metrics_all.get(f"MAE_d{h}", float("nan"))
        dr = metrics_cs.get(f"RMSE_d{h}", float("nan")) - metrics_all.get(f"RMSE_d{h}", float("nan"))
        d2 = metrics_cs.get(f"R2_d{h}", float("nan"))   - metrics_all.get(f"R2_d{h}", float("nan"))
        print(f"  t+{h}       ΔMAE={_sign(dm)}{dm:.3f}  ΔRMSE={_sign(dr)}{dr:.3f}  ΔR2={_sign(d2)}{d2:.4f}")

    verdict = "BETTER" if delta_rmse < 0 else ("SIMILAR" if abs(rmse_pct) < 2.0 else "WORSE")
    print(f"\n  Verdict: Consensus-feature LSTM is {verdict} than all-feature LSTM on test RMSE.")
    print(
        "\n  Note: LSTM results have variance from random weight initialisation.\n"
        "  Single-run comparison is indicative; repeat with fixed seeds for rigour."
    )


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature selection experiment for Karachi AQI Predictor."
    )
    parser.add_argument(
        "--horizon",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4],
        default=[1, 2, 3, 4],
        help="Which prediction horizons to analyse (1-indexed, e.g. --horizon 1 2). Default: all 4.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Quick mode: limits RFECV to 3 TimeSeriesSplit folds and skips Section 6 (LSTM)."
            " Useful for a fast sanity-check run."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args    = parse_args()
    horizons = [h - 1 for h in sorted(set(args.horizon))]  # convert to 0-indexed
    quick   = args.quick
    rfecv_splits = 3 if quick else 5

    total_start = time.perf_counter()

    _banner("KARACHI AQI PREDICTOR — FEATURE SELECTION EXPERIMENT")
    print(f"  Horizons    : {[f't+{h + 1}' for h in horizons]}")
    print(f"  Quick mode  : {quick}")
    print(f"  RFECV splits: {rfecv_splits}")
    print(f"  Top-N shown : {TOP_N}")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    _section("Loading data from MongoDB feature_store")
    t0 = time.perf_counter()
    df   = load_data()
    feat = get_feature_cols(df)
    print(f"  Rows loaded   : {len(df)}")
    print(f"  Feature count : {len(feat)}")
    print(f"  Date range    : {df['date'].min().date()}  ->  {df['date'].max().date()}")
    print(f"  Data load time: {time.perf_counter() - t0:.1f}s")

    train_df, test_df = time_split(df, test_days=30)
    print(f"  Train rows    : {len(train_df)}  (up to {train_df['date'].max().date()})")
    print(f"  Test rows     : {len(test_df)}   (from {test_df['date'].min().date()})")

    if len(test_df) == 0:
        sys.exit("ERROR: test split is empty — need at least 30 days of data.")

    feature_names = feat  # ordered list of strings
    X_train = train_df[feature_names].values.astype(np.float64)
    y_train = train_df[TARGET_COLS].values.astype(np.float64)
    X_test  = test_df[feature_names].values.astype(np.float64)
    y_test  = test_df[TARGET_COLS].values.astype(np.float64)

    # Sanity-check for NaNs
    nan_in_X = np.isnan(X_train).sum()
    nan_in_y = np.isnan(y_train).sum()
    if nan_in_X > 0 or nan_in_y > 0:
        print(f"  WARNING: NaNs in X_train={nan_in_X}, y_train={nan_in_y} — filling with 0.")
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test  = np.nan_to_num(X_test,  nan=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0)
        y_test  = np.nan_to_num(y_test,  nan=0.0)

    # ------------------------------------------------------------------
    # Run the five analysis sections
    # ------------------------------------------------------------------
    mi_top   = run_mutual_information(X_train, y_train, feature_names, horizons)
    lgbm_top, perm_top, _ = run_lgbm_importances(
        X_train, y_train, X_test, y_test, feature_names, horizons
    )
    rfecv_top = run_rfecv(X_train, y_train, feature_names, horizons, n_splits=rfecv_splits)

    # ------------------------------------------------------------------
    # Consensus set: combines MI + LGBM + permutation top-30s
    # Use min_horizons = len(horizons) when only 1 or 2 are tested,
    # otherwise require agreement across at least 3 of 4 horizons.
    # ------------------------------------------------------------------
    min_horizons = min(len(horizons), 3)
    consensus    = compute_consensus(
        mi_top, lgbm_top, perm_top, horizons, min_horizons=min_horizons
    )

    # ------------------------------------------------------------------
    # Print RFECV features alongside consensus for convenience
    # ------------------------------------------------------------------
    _banner("RFECV SELECTED FEATURES — SUMMARY ACROSS HORIZONS")
    rfecv_union = sorted(set(f for feats in rfecv_top.values() for f in feats))
    rfecv_inter = sorted(
        set.intersection(*[set(feats) for feats in rfecv_top.values()])
        if rfecv_top else set()
    )
    print(f"\n  RFECV union (any horizon)          : {len(rfecv_union)} features")
    print(f"  RFECV intersection (all horizons)  : {len(rfecv_inter)} features")
    if rfecv_inter:
        print("\n  Features selected by RFECV for ALL tested horizons:")
        for f in rfecv_inter:
            print(f"    {f}")
    print(f"\n  Consensus (MI + LGBM + Perm agree) : {len(consensus)} features")
    in_both = sorted(set(consensus) & set(rfecv_inter))
    print(f"  In consensus AND RFECV intersection: {len(in_both)} features")
    if in_both:
        print("\n  These features are the strongest candidates overall:")
        for f in in_both:
            print(f"    {f}")

    # ------------------------------------------------------------------
    # Section 6 — LSTM comparison (skipped in quick mode)
    # ------------------------------------------------------------------
    if quick:
        _banner("SECTION 6 — LSTM COMPARISON")
        print("  Skipped in --quick mode.")
    else:
        run_lstm_comparison(
            X_train, y_train, X_test, y_test, feature_names, consensus
        )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    total_elapsed = time.perf_counter() - total_start
    _banner("EXPERIMENT COMPLETE")
    print(f"  Total runtime : {total_elapsed / 60:.1f} minutes")
    print(f"  Consensus set : {len(consensus)} features")
    if consensus:
        print("\n  Consensus features to evaluate for inclusion:")
        for f in consensus:
            print(f"    {f}")
    print(
        "\n  Next steps:"
        "\n    1. Compare RFECV intersection with consensus list."
        "\n    2. Re-train production model with consensus features and compare holdout RMSE."
        "\n    3. Update EXCLUDE_COLS in src/train.py if consensus features improve metrics."
        "\n    4. Features absent from both consensus and RFECV are strong candidates for exclusion."
    )


if __name__ == "__main__":
    main()
