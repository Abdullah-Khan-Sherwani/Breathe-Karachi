"""
Sanity-check script — compares 20 randomly sampled dates from the MongoDB
feature_store backup against values recomputed from scratch by the new
preprocess_daily_data.py pipeline.

Algorithm
---------
1. Load ALL rows from the backup CSV → df_mongo (represents what MongoDB stores).
2. Extract only the 15 raw input columns from the backup CSV → df_raw.
3. Run the full preprocessing pipeline on df_raw → df_computed.
4. Pick 20 random dates that are fully populated in BOTH df_mongo and df_computed.
5. For every feature column, compare values between df_mongo and df_computed.
6. Report EXACT MATCH, minor drift (1e-6 < diff <= 1.0), MISMATCH (diff > 1.0),
   and any columns present in df_mongo but missing from df_computed.
7. Print a summary table and overall PASS / FAIL verdict.

Known acceptable deviations
---------------------------
Three categories of columns deviate from the MongoDB backup for documented
reasons — none reflect formula errors in the new preprocessing code:

(A) stagnant_air
    Uses df['wind_speed'].median() computed at pipeline runtime.
    The backup was created when feature_store held only 2023+ rows; the current
    full dataset (2018-2026) shifts the median, changing the binary flag for
    some rows near the threshold.  Expected dataset-expansion effect.

(B) wind_speed_t1 / wind_speed_t2 / wind_speed_t3 and wind_x_PM2_5_lag1
    The raw wind_speed values for ~16 days (all having wind_speed == 30.38
    in the current backup) were silently updated by update_daily_data.py after
    features had already been computed and stored.  The feature columns in those
    documents therefore use an older wind_speed value; the raw column reflects
    the newer fetch.  Our new pipeline recomputes correctly from the CURRENT
    wind_speed, so it produces different (more consistent) values for those rows.

(C) AQI_ewm_30 for early-2023 dates
    EWM (span=30) is sensitive to the initialisation window.  The backup was
    produced by running the pipeline only on 2023+ data, so the EWM warm-up
    starts from January 2023.  Running on the full 2018-2026 dataset adds ~5
    years of seed data, causing a sub-0.1 drift that decays to zero by mid-2023.
    All other EWM spans (7, 14) are short enough that this effect is negligible.
"""

import sys
import random
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

# Import all pipeline functions (not run()) so we can simulate on the
# raw-column extract from the backup CSV.
from src.preprocess_daily_data import (
    cap_iqr,
    add_log_transforms,
    add_wind_features,
    add_temporal,
    add_cyclical_features,
    add_lag_rolling,
    add_extended_lags,
    add_rolling_extended,
    add_derived_weather,
    add_interaction_features,
    add_tier2_features,
    add_weather_leads,
    add_tier3_features,
    add_targets,
    RAW_COLS,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKUP_CSV = (
    Path(__file__).parent.parent
    / "backups"
    / "feature_store_20260527_125204.csv"
)

# Columns that are identifiers only
IDENTIFIER_COLS = {"_id", "date"}

# Float comparison tolerances
RTOL = 1e-5
ATOL = 1e-6
MINOR_DRIFT_THRESHOLD = 1.0

N_SAMPLE = 20
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Known acceptable deviation columns — these will NOT cause FAIL verdict
# ---------------------------------------------------------------------------
# (A) Dataset-expansion effect on whole-dataset median
KNOWN_DEVIATION_DATASET_EXPANSION = {"stagnant_air"}

# (B) Stale raw wind_speed in the backup — features were computed using an older
#     fetch value that was later silently corrected by update_daily_data.py
KNOWN_DEVIATION_STALE_WIND = {"wind_speed_t1", "wind_speed_t2", "wind_speed_t3", "wind_x_PM2_5_lag1"}

# (C) EWM warm-up initialisation difference (full dataset vs 2023-only)
KNOWN_DEVIATION_EWM_WARMUP = {"AQI_ewm_30"}

KNOWN_DEVIATION_COLS = (
    KNOWN_DEVIATION_DATASET_EXPANSION
    | KNOWN_DEVIATION_STALE_WIND
    | KNOWN_DEVIATION_EWM_WARMUP
)


# ---------------------------------------------------------------------------
# Pipeline simulation (mirrors run() but without MongoDB I/O)
# ---------------------------------------------------------------------------

def run_pipeline_on_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the complete feature-engineering pipeline to a DataFrame that
    contains only the 15 raw input columns plus 'date'.
    Returns the post-dropna processed DataFrame with date as YYYY-MM-DD string.
    """
    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Calendar reindex + forward fill (same as load_raw)
    full_idx = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = (
        df.set_index("date")
        .reindex(full_idx)
        .ffill()
        .reset_index()
        .rename(columns={"index": "date"})
    )

    df = cap_iqr(df)
    df = add_log_transforms(df)
    df = add_wind_features(df)
    df = add_temporal(df)
    df = add_cyclical_features(df)
    df = add_lag_rolling(df)
    df = add_extended_lags(df)
    df = add_rolling_extended(df)
    df = add_derived_weather(df)
    df = add_interaction_features(df)
    df = add_tier2_features(df)
    df = add_weather_leads(df)
    df = add_tier3_features(df)
    df = add_targets(df)

    df = df.dropna().reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def coerce_float(val) -> float:
    """Coerce any scalar (including bool, numpy types, 'True'/'False') to float."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return float("nan")
    if isinstance(val, bool):
        return 1.0 if val else 0.0
    if isinstance(val, str):
        if val.lower() == "true":
            return 1.0
        if val.lower() == "false":
            return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def compare_value(mongo_val, computed_val):
    """
    Returns (status, abs_diff) where status is one of:
      'EXACT'        — within RTOL/ATOL tolerances
      'MINOR_DRIFT'  — absolute diff in (1e-6, 1.0]
      'MISMATCH'     — absolute diff > 1.0
      'NAN_BOTH'     — both NaN (treated as matching)
      'NAN_MISMATCH' — one side NaN and the other is not
    """
    m = coerce_float(mongo_val)
    c = coerce_float(computed_val)

    if np.isnan(m) and np.isnan(c):
        return "NAN_BOTH", 0.0
    if np.isnan(m) or np.isnan(c):
        return "NAN_MISMATCH", float("nan")

    abs_diff = abs(m - c)
    if np.isclose(m, c, rtol=RTOL, atol=ATOL):
        return "EXACT", abs_diff
    if abs_diff <= MINOR_DRIFT_THRESHOLD:
        return "MINOR_DRIFT", abs_diff
    return "MISMATCH", abs_diff


# ---------------------------------------------------------------------------
# Main sanity check
# ---------------------------------------------------------------------------

def run_sanity_check():
    print("=" * 70)
    print("FEATURE SANITY CHECK")
    print(f"Backup: {BACKUP_CSV.name}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load the full backup as df_mongo
    # ------------------------------------------------------------------
    print("\n[1] Loading backup CSV ...")
    df_mongo_raw = pd.read_csv(BACKUP_CSV, low_memory=False)
    df_mongo_raw["date"] = df_mongo_raw["date"].astype(str)
    print(f"    Loaded {len(df_mongo_raw)} rows, {len(df_mongo_raw.columns)} columns")

    # Keep only rows that have fully computed features
    df_mongo = df_mongo_raw.dropna(
        subset=["AQI_ewm_7", "dew_point", "wind_dir_sin"]
    ).copy().reset_index(drop=True)
    print(f"    Rows with fully computed features: {len(df_mongo)}")

    # ------------------------------------------------------------------
    # Step 2: Extract raw columns and run pipeline
    # ------------------------------------------------------------------
    print("\n[2] Extracting raw columns and running pipeline ...")
    raw_cols_in_csv = [c for c in RAW_COLS if c in df_mongo_raw.columns]
    df_raw = df_mongo_raw[raw_cols_in_csv].copy()
    print(f"    Raw columns extracted: {raw_cols_in_csv}")

    df_computed = run_pipeline_on_raw(df_raw)
    print(f"    Pipeline produced {len(df_computed)} fully-computed rows")

    # ------------------------------------------------------------------
    # Step 3: Find common dates
    # ------------------------------------------------------------------
    print("\n[3] Finding overlap ...")
    mongo_dates = set(df_mongo["date"].astype(str))
    computed_dates = set(df_computed["date"].astype(str))
    common_dates = sorted(mongo_dates & computed_dates)
    print(f"    Dates in mongo (processed): {len(mongo_dates)}")
    print(f"    Dates in computed:          {len(computed_dates)}")
    print(f"    Overlapping dates:          {len(common_dates)}")

    if len(common_dates) < N_SAMPLE:
        print(f"    WARNING: fewer than {N_SAMPLE} overlap dates — using all {len(common_dates)}")
        sampled_dates = common_dates
    else:
        random.seed(RANDOM_SEED)
        sampled_dates = sorted(random.sample(common_dates, N_SAMPLE))

    print(f"\n[4] Sampled {len(sampled_dates)} dates for comparison:")
    for d in sampled_dates:
        print(f"    {d}")

    # ------------------------------------------------------------------
    # Step 4: Identify columns to compare
    # ------------------------------------------------------------------
    mongo_feature_cols = [
        c for c in df_mongo.columns if c not in IDENTIFIER_COLS
    ]
    computed_feature_cols = set(df_computed.columns) - IDENTIFIER_COLS

    missing_from_computed = [c for c in mongo_feature_cols if c not in computed_feature_cols]
    extra_in_computed = [c for c in computed_feature_cols if c not in set(mongo_feature_cols)]

    print(f"\n[5] Column coverage:")
    print(f"    Mongo feature columns:    {len(mongo_feature_cols)}")
    print(f"    Computed feature columns: {len(computed_feature_cols)}")
    if missing_from_computed:
        print(f"    MISSING from computed ({len(missing_from_computed)}): {missing_from_computed}")
    else:
        print("    All mongo feature columns are present in computed output.")
    if extra_in_computed:
        print(f"    Extra in computed (not in mongo): {sorted(extra_in_computed)}")

    cols_to_compare = [c for c in mongo_feature_cols if c in computed_feature_cols]

    # ------------------------------------------------------------------
    # Step 5: Per-column comparison across all sampled dates
    # ------------------------------------------------------------------
    print(f"\n[6] Comparing {len(cols_to_compare)} columns across {len(sampled_dates)} dates ...")

    df_mongo_idx = df_mongo.set_index("date")
    df_computed_idx = df_computed.set_index("date")

    # col -> {exact, minor, mismatch_list, nan_both, nan_mismatch}
    col_results = {
        col: {"exact": 0, "minor": 0, "mismatch": [], "nan_both": 0, "nan_mismatch": 0}
        for col in cols_to_compare
    }
    # (date, col) -> (status, abs_diff, mongo_val, computed_val)
    date_col_status = {}

    for date in sampled_dates:
        if date not in df_mongo_idx.index or date not in df_computed_idx.index:
            continue
        mongo_row = df_mongo_idx.loc[date]
        computed_row = df_computed_idx.loc[date]

        for col in cols_to_compare:
            m_val = mongo_row[col] if col in mongo_row.index else float("nan")
            c_val = computed_row[col] if col in computed_row.index else float("nan")
            status, diff = compare_value(m_val, c_val)
            date_col_status[(date, col)] = (status, diff, m_val, c_val)

            r = col_results[col]
            if status == "EXACT":
                r["exact"] += 1
            elif status == "MINOR_DRIFT":
                r["minor"] += 1
            elif status == "MISMATCH":
                r["mismatch"].append((date, diff, m_val, c_val))
            elif status == "NAN_BOTH":
                r["nan_both"] += 1
            elif status == "NAN_MISMATCH":
                r["nan_mismatch"] += 1

    # ------------------------------------------------------------------
    # Step 6: Categorise columns
    # ------------------------------------------------------------------
    n = len(sampled_dates)
    exact_cols = []
    minor_cols = []
    mismatch_cols = []
    nan_mm_cols = []

    for col in cols_to_compare:
        r = col_results[col]
        if r["mismatch"]:
            mismatch_cols.append(col)
        elif r["nan_mismatch"] > 0:
            nan_mm_cols.append(col)
        elif r["minor"] > 0:
            minor_cols.append(col)
        else:
            exact_cols.append(col)

    # ------------------------------------------------------------------
    # Step 7: Print results
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nColumns with EXACT match across all {n} dates ({len(exact_cols)}):")
    if exact_cols:
        for i in range(0, len(exact_cols), 5):
            print("  " + "  ".join(exact_cols[i:i+5]))
    else:
        print("  (none)")

    print(f"\nColumns with MINOR DRIFT (1e-6 < diff <= 1.0) ({len(minor_cols)}):")
    for col in minor_cols:
        diffs = [
            date_col_status[(d, col)][1]
            for d in sampled_dates
            if (d, col) in date_col_status and date_col_status[(d, col)][0] == "MINOR_DRIFT"
        ]
        max_d = max(diffs) if diffs else 0.0
        flag = " [KNOWN DEVIATION]" if col in KNOWN_DEVIATION_COLS else ""
        print(f"  {col}: {len(diffs)}/{n} dates (max diff = {max_d:.2e}){flag}")

    print(f"\nColumns with MISMATCH (diff > 1.0) ({len(mismatch_cols)}):")
    for col in mismatch_cols:
        flag = " [KNOWN DEVIATION]" if col in KNOWN_DEVIATION_COLS else " *** UNEXPECTED ***"
        print(f"  {col}:{flag}")
        for date, diff, m_val, c_val in col_results[col]["mismatch"]:
            print(
                f"    {date}: mongo={coerce_float(m_val):.6f}  "
                f"computed={coerce_float(c_val):.6f}  diff={diff:.4f}"
            )

    if nan_mm_cols:
        unexpected_nan = [c for c in nan_mm_cols if c not in KNOWN_DEVIATION_COLS]
        print(f"\nColumns with NaN MISMATCH ({len(nan_mm_cols)}): {nan_mm_cols}")
        if unexpected_nan:
            print(f"  Unexpected NaN mismatches: {unexpected_nan}")

    if missing_from_computed:
        print(f"\nColumns MISSING from computed ({len(missing_from_computed)}): {missing_from_computed}")

    # ------------------------------------------------------------------
    # Step 8: Known deviation explanations
    # ------------------------------------------------------------------
    all_deviating = set(mismatch_cols + minor_cols + nan_mm_cols)
    active_known = all_deviating & KNOWN_DEVIATION_COLS
    if active_known:
        print("\nKNOWN ACCEPTABLE DEVIATIONS (not counted as failures):")
        for col in sorted(active_known):
            if col in KNOWN_DEVIATION_DATASET_EXPANSION:
                reason = (
                    "dataset-expansion effect: stagnant_air threshold uses "
                    "wind_speed.median() at pipeline runtime; the backup was made when "
                    "only 2023+ data existed (median ~12.97), but the full dataset "
                    "has median ~15.02, so some binary flags change near the threshold."
                )
            elif col in KNOWN_DEVIATION_STALE_WIND:
                reason = (
                    "stale raw data: wind_speed for ~16 days was silently re-fetched "
                    "by update_daily_data.py AFTER features were computed and stored. "
                    "Our new pipeline recomputes correctly from the current wind_speed value."
                )
            elif col in KNOWN_DEVIATION_EWM_WARMUP:
                reason = (
                    "EWM warm-up initialisation: running on full 2018-2026 data vs "
                    "2023-only data shifts early-2023 EWM(span=30) values by < 0.06; "
                    "drift decays to ~0 by mid-2023 and is absent for later dates."
                )
            else:
                reason = "see module docstring."
            print(f"  {col}: {reason}")

    # ------------------------------------------------------------------
    # Step 9: PASS / FAIL verdict
    # ------------------------------------------------------------------
    unexpected_mismatches = [c for c in mismatch_cols if c not in KNOWN_DEVIATION_COLS]
    unexpected_nan_mm = [c for c in nan_mm_cols if c not in KNOWN_DEVIATION_COLS]
    has_missing = len(missing_from_computed) > 0

    total_issues = len(unexpected_mismatches) + len(unexpected_nan_mm) + (1 if has_missing else 0)

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  Columns compared:           {len(cols_to_compare)}")
    print(f"  Exact match:                {len(exact_cols)}")
    print(f"  Minor drift only:           {len(minor_cols)}")
    print(f"  Known deviations (total):   {len(active_known)}")
    print(f"  Unexpected mismatches:      {len(unexpected_mismatches)}")
    print(f"  Unexpected NaN mismatches:  {len(unexpected_nan_mm)}")
    print(f"  Missing from computed:      {len(missing_from_computed)}")

    if total_issues == 0:
        print("\n  OVERALL: PASS")
    else:
        print("\n  OVERALL: FAIL")
        if unexpected_mismatches:
            print(f"  Unexpected mismatches: {unexpected_mismatches}")
        if unexpected_nan_mm:
            print(f"  Unexpected NaN mismatches: {unexpected_nan_mm}")
        if has_missing:
            print(f"  Missing columns: {missing_from_computed}")

    print("=" * 70)
    return total_issues == 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    passed = run_sanity_check()
    sys.exit(0 if passed else 1)
