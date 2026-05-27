"""
Sanity check for new variables added in the feature engineering expansion.
Runs directly against MongoDB — no backup CSV required.

Checks:
  1. Row count is intact
  2. New raw columns are present and not all-NaN
  3. Engineered features derived from new vars exist
  4. Lead features (t+1/t+2/t+3) are populated for recent rows
  5. Value ranges are physically plausible
  6. No corruption of existing core columns
  7. Targets are still present
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from config.db import get_collection, COLLECTION_FEATURE_STORE

# ---------------------------------------------------------------------------
# Expected columns and plausible value ranges
# ---------------------------------------------------------------------------

NEW_RAW_COLS = {
    "BLH":          (0,    5000),   # boundary layer height in metres
    "cloud_cover":  (0,    100),    # percentage
    "shortwave_rad":(0,    1000),   # W/m2
    "uv_index":     (0,    20),     # index (from CAMS AQ archive, not ERA5)
    "aod":          (0,    5),      # dimensionless
    "dust":         (0,    5000),   # ug/m3
}

NEW_ENGINEERED_COLS = [
    "vpd", "log_aod",
    "BLH_lag_1", "cloud_cover_lag_1", "shortwave_rad_lag_1",
    "uv_index_lag_1", "aod_lag_1", "dust_lag_1",
    "BLH_roll_mean_7", "aod_roll_mean_7", "dust_roll_mean_7",
]

NEW_LEAD_COLS = (
    [f"BLH_t{d}"             for d in [1, 2, 3]]
    + [f"cloud_cover_t{d}"   for d in [1, 2, 3]]
    + [f"shortwave_rad_t{d}" for d in [1, 2, 3]]
    + [f"PM2_5_t{d}"         for d in [1, 2, 3]]
    + [f"aod_t{d}"           for d in [1, 2, 3]]
    + [f"dust_t{d}"          for d in [1, 2, 3]]
    + [f"uv_index_t{d}"      for d in [1, 2, 3]]
)

# Columns always present (weather from 2018) vs only from ~Jan-2023 (AQ data).
# Checked against different denominators to avoid false failures from the
# pre-existing dataset structure (not corruption).
# Raw weather cols present for all rows (2018+):
WEATHER_CORE_COLS = ["Temperature", "Humidity", "Precipitation"]
# Computed cols that only exist in processed AQ rows (2023+):
AQ_CORE_COLS = ["AQI", "PM2_5", "PM10", "NO2", "SO2", "CO", "O3",
                "AQI_lag_1", "AQI_roll_mean_7", "AQI_ewm_7",
                "dew_point", "wind_dir_sin", "wind_dir_cos"]

TARGET_COLS = ["AQI_t+1", "AQI_t+2", "AQI_t+3"]

PASS_MARK = "\033[92mPASS\033[0m"
FAIL_MARK = "\033[91mFAIL\033[0m"


def run() -> bool:
    print("=" * 65)
    print("SANITY CHECK — NEW FEATURE ENGINEERING")
    print("=" * 65)

    col = get_collection(COLLECTION_FEATURE_STORE)
    docs = list(col.find({}, {"_id": 0}))
    if not docs:
        print("FAIL: feature_store is empty.")
        return False

    df = pd.DataFrame(docs)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    total_rows = len(df)
    print(f"\nLoaded {total_rows} rows  ({df['date'].min().date()} to {df['date'].max().date()})")

    failures = []

    # ------------------------------------------------------------------
    # 1. Row count — just flag if suspiciously low
    # ------------------------------------------------------------------
    print(f"\n[1] Row count: {total_rows}", end="  ")
    if total_rows < 100:
        print(FAIL_MARK)
        failures.append("row count < 100")
    else:
        print(PASS_MARK)

    # ------------------------------------------------------------------
    # 2. New raw columns — present and not all-NaN
    # ------------------------------------------------------------------
    print("\n[2] New raw columns:")
    for col_name, (lo, hi) in NEW_RAW_COLS.items():
        present = col_name in df.columns
        if not present:
            print(f"    {col_name:<22} MISSING  {FAIL_MARK}")
            failures.append(f"{col_name} missing")
            continue
        non_null = df[col_name].notna().sum()
        pct = 100 * non_null / total_rows
        in_range = df[col_name].dropna().between(lo, hi).all()
        vmin = df[col_name].min()
        vmax = df[col_name].max()
        status = PASS_MARK if non_null > 0 and in_range else FAIL_MARK
        if non_null == 0 or not in_range:
            failures.append(f"{col_name} range/coverage issue")
        print(f"    {col_name:<22} {non_null:>5}/{total_rows} non-null ({pct:.1f}%)  "
              f"range [{vmin:.2f}, {vmax:.2f}]  {status}")

    # ------------------------------------------------------------------
    # 3. Engineered features from new vars
    # ------------------------------------------------------------------
    print("\n[3] Engineered features from new vars:")
    for col_name in NEW_ENGINEERED_COLS:
        present = col_name in df.columns
        non_null = df[col_name].notna().sum() if present else 0
        pct = 100 * non_null / total_rows if present else 0
        if not present or non_null == 0:
            status = FAIL_MARK
            failures.append(f"{col_name} missing or all-NaN")
        else:
            status = PASS_MARK
        label = col_name if present else f"{col_name} (MISSING)"
        print(f"    {label:<28} {non_null:>5}/{total_rows} non-null ({pct:.1f}%)  {status}")

    # ------------------------------------------------------------------
    # 4. Lead features — checked against AQ-era rows only (leads are
    # computed from AQ/weather features and only persist for processed rows)
    # Exclude last 3 rows which intentionally have NaN leads.
    # ------------------------------------------------------------------
    df_aq_pre = df[df["AQI"].notna()].iloc[:-3] if df["AQI"].notna().sum() > 3 else df[df["AQI"].notna()]
    n_aq_pre = len(df_aq_pre)
    print(f"\n[4] Lead features (AQ processed rows, excl. last 3: {n_aq_pre}):")
    for col_name in NEW_LEAD_COLS:
        present = col_name in df.columns
        if not present:
            print(f"    {col_name:<22} MISSING  {FAIL_MARK}")
            failures.append(f"{col_name} missing")
            continue
        non_null_hist = df_aq_pre[col_name].notna().sum()
        pct = 100 * non_null_hist / n_aq_pre if n_aq_pre > 0 else 0
        # Expect >90% populated for processed AQ rows
        status = PASS_MARK if pct > 90 else FAIL_MARK
        if pct <= 90:
            failures.append(f"{col_name} lead coverage {pct:.1f}% < 90%")
        print(f"    {col_name:<22} {non_null_hist:>5}/{n_aq_pre} ({pct:.1f}%)  {status}")

    # ------------------------------------------------------------------
    # 5. Core existing columns untouched
    # Weather cols checked against all rows; AQ cols checked against the
    # subset that has AQ data (~2023 onwards) so threshold is meaningful.
    # ------------------------------------------------------------------
    df_aq = df[df["AQI"].notna()]
    n_aq = len(df_aq)
    print(f"\n[5] Core columns integrity  (AQ rows: {n_aq}/{total_rows}):")
    for col_name in WEATHER_CORE_COLS:
        present = col_name in df.columns
        non_null = df[col_name].notna().sum() if present else 0
        pct = 100 * non_null / total_rows if present else 0
        status = PASS_MARK if present and pct > 80 else FAIL_MARK
        if not present or pct <= 80:
            failures.append(f"weather core col {col_name} degraded ({pct:.1f}%)")
        print(f"    {col_name:<28} {non_null:>5}/{total_rows} ({pct:.1f}%)  {status}")
    for col_name in AQ_CORE_COLS:
        present = col_name in df.columns
        non_null = df_aq[col_name].notna().sum() if present else 0
        pct = 100 * non_null / n_aq if (present and n_aq > 0) else 0
        status = PASS_MARK if present and pct > 80 else FAIL_MARK
        if not present or pct <= 80:
            failures.append(f"AQ core col {col_name} degraded ({pct:.1f}% of AQ rows)")
        print(f"    {col_name:<28} {non_null:>5}/{n_aq} AQ rows ({pct:.1f}%)  {status}")

    # ------------------------------------------------------------------
    # 6. Targets
    # ------------------------------------------------------------------
    print("\n[6] Target columns:")
    df_with_targets = df.dropna(subset=TARGET_COLS)
    pct = 100 * len(df_with_targets) / total_rows
    status = PASS_MARK if len(df_with_targets) > 50 else FAIL_MARK
    if len(df_with_targets) <= 50:
        failures.append("too few rows with all 3 targets")
    print(f"    Rows with all 3 targets: {len(df_with_targets)}/{total_rows} ({pct:.1f}%)  {status}")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("VERDICT")
    print("=" * 65)
    if failures:
        print(f"  FAIL  ({len(failures)} issue(s)):")
        for f in failures:
            print(f"    - {f}")
    else:
        print("  PASS  — all checks clean.")
    print("=" * 65)
    return len(failures) == 0


if __name__ == "__main__":
    passed = run()
    sys.exit(0 if passed else 1)
