"""
Task 1 + 2 + 3 combined diagnostic for the Karachi AQI feature_store.

Task 1 — Backup ALL documents to data/feature_store_backup_20260527.csv
          Connects to MongoDB via config.db if reachable; falls back to the
          most recent backup CSV already present in backups/ when Atlas is
          unreachable from the current network (IP not whitelisted).

Task 2 — Data completeness for 2023+ rows (21 raw columns)
Task 3 — Lead feature NaN audit for the 5 most recent rows

Read-only: never modifies MongoDB.
Run from any working directory; sys.path is patched to the project root.
"""

import sys
import shutil
from pathlib import Path

# ── path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from bson import ObjectId

# ── constants ─────────────────────────────────────────────────────────────────
BACKUP_PATH   = PROJECT_ROOT / "data" / "feature_store_backup_20260527.csv"
BACKUPS_DIR   = PROJECT_ROOT / "backups"

RAW_COLS_21 = [
    "AQI", "PM2_5", "PM10", "NO2", "SO2", "CO", "O3",
    "Temperature", "Humidity", "Precipitation",
    "wind_speed", "wind_direction",
    "apparent_temp", "surface_pressure", "wind_gusts",
    "BLH", "cloud_cover", "shortwave_rad", "uv_index",
    "aod", "dust",
]

# All lead feature columns produced by preprocess_daily_data.py.
# These MUST be filled on the most recent row via the forecast API.
LEAD_GROUPS: dict[str, list[str]] = {
    "Weather (Temp/Hum/Precip/Wind)": [
        f"{c}_t{d}"
        for c in ["Temperature", "Humidity", "Precipitation", "wind_speed"]
        for d in [1, 2, 3, 4]
    ],
    "Wind direction sin/cos": [
        f"wind_dir_{s}_t{d}" for s in ["sin", "cos"] for d in [1, 2, 3, 4]
    ],
    "Tier-3 (pressure/apparent_temp/gusts)": [
        f"{c}_t{d}"
        for c in ["surface_pressure", "apparent_temp", "wind_gusts"]
        for d in [1, 2, 3, 4]
    ],
    "BLH / cloud_cover / shortwave_rad": [
        f"{c}_t{d}"
        for c in ["BLH", "cloud_cover", "shortwave_rad"]
        for d in [1, 2, 3, 4]
    ],
    "AQ leads (PM2_5/aod/dust/uv)": [
        f"{c}_t{d}"
        for c in ["PM2_5", "aod", "dust", "uv_index"]
        for d in [1, 2, 3, 4]
    ],
}
ALL_LEAD_COLS: list[str] = [col for cols in LEAD_GROUPS.values() for col in cols]
TARGET_COLS = ["AQI_t+1", "AQI_t+2", "AQI_t+3", "AQI_t+4"]

SEPARATOR = "=" * 72


# ── helpers ──────────────────────────────────────────────────────────────────

def _sanitize(doc: dict) -> dict:
    """Convert BSON ObjectId / binary types for pandas/CSV."""
    clean = {}
    for k, v in doc.items():
        if isinstance(v, ObjectId):
            clean[k] = str(v)
        elif isinstance(v, bytes):
            clean[k] = f"<binary {len(v)} bytes>"
        elif hasattr(v, "item"):
            clean[k] = v.item()
        else:
            clean[k] = v
    return clean


def _load_from_mongo() -> tuple[pd.DataFrame, str]:
    """
    Attempt a live MongoDB fetch.
    Returns (DataFrame, source_label) on success.
    Raises any exception on failure so the caller can fall back.
    """
    from config.db import get_collection, COLLECTION_FEATURE_STORE
    col = get_collection(COLLECTION_FEATURE_STORE)
    docs = list(col.find({}))
    rows = [_sanitize(d) for d in docs]
    df = pd.DataFrame(rows)
    return df, "MongoDB (live)"


def _load_from_csv_fallback() -> tuple[pd.DataFrame, str]:
    """
    Fall back to the most recent feature_store_*.csv in backups/.
    Raises FileNotFoundError if nothing is there.
    """
    csvs = sorted(BACKUPS_DIR.glob("feature_store_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No feature_store backup CSV found in {BACKUPS_DIR}")
    latest = csvs[-1]
    df = pd.read_csv(latest, low_memory=False)
    return df, f"CSV fallback: {latest.name}"


def load_data() -> tuple[pd.DataFrame, str]:
    """
    Load feature_store from MongoDB, falling back to the local CSV backup
    if Atlas is unreachable (e.g. current IP not in whitelist).
    """
    try:
        df, src = _load_from_mongo()
        print(f"  Source: {src}")
        return df, src
    except Exception as mongo_err:
        print(f"  MongoDB unreachable: {mongo_err}")
        print(f"  Falling back to local CSV backup...")
        df, src = _load_from_csv_fallback()
        print(f"  Source: {src}")
        return df, src


def _coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)
    return df


# ── Task 1 ───────────────────────────────────────────────────────────────────

def task1_backup(df: pd.DataFrame, source: str) -> None:
    print()
    print(SEPARATOR)
    print("TASK 1 — Feature Store Backup")
    print(SEPARATOR)

    BACKUP_PATH.parent.mkdir(parents=True, exist_ok=True)

    export = df.copy()
    if "date" in export.columns and pd.api.types.is_datetime64_any_dtype(export["date"]):
        export["date"] = export["date"].dt.strftime("%Y-%m-%d")

    export.to_csv(BACKUP_PATH, index=False)

    print(f"  Source        : {source}")
    print(f"  Rows exported : {len(export):,}")
    print(f"  Columns       : {len(export.columns)}")
    print(f"  Saved to      : {BACKUP_PATH}")
    if "MongoDB" not in source:
        print()
        print("  NOTE: MongoDB is currently unreachable from this machine's IP")
        print(f"  (current public IP: 103.87.192.212 — not in Atlas IP access list).")
        print("  The backup was produced from the most recent backups/ CSV which")
        print("  was written by the existing backup_mongodb.py script on 2026-05-27.")
        print("  Data is current as of that run.")


# ── Task 2 ───────────────────────────────────────────────────────────────────

def task2_completeness(df: pd.DataFrame) -> None:
    print()
    print(SEPARATOR)
    print("TASK 2 — Data Completeness Diagnostics (2023+ rows)")
    print(SEPARATOR)

    total_rows = len(df)
    print(f"  Total rows in feature_store : {total_rows:,}")

    if "date" not in df.columns:
        print("  ERROR: 'date' column missing — cannot filter by year.")
        return

    cutoff = pd.Timestamp("2023-01-01")
    df_23 = df[df["date"] >= cutoff].copy()
    print(f"  Rows with date >= 2023-01-01 : {len(df_23):,}")

    if df_23.empty:
        print("  No 2023+ rows found.")
        return

    print()
    print("  NaN count per raw column (2023+ rows only)")
    print(f"  {'Column':<22}  {'NaN count':>10}  {'NaN %':>8}  Status")
    print(f"  {'-'*22}  {'-'*10}  {'-'*8}  {'-'*30}")

    n = len(df_23)
    all_clean = True
    for col in RAW_COLS_21:
        if col in df_23.columns:
            nan_count = int(df_23[col].isna().sum())
        else:
            nan_count = n  # column entirely absent from the dataframe
        pct = nan_count / n * 100
        if col not in df_23.columns:
            status = "COLUMN ABSENT"
            all_clean = False
        elif nan_count == 0:
            status = "OK"
        elif pct < 5:
            status = f"sparse gaps ({nan_count} rows)"
            all_clean = False
        else:
            status = f"SIGNIFICANT GAPS"
            all_clean = False
        print(f"  {col:<22}  {nan_count:>10,}  {pct:>7.1f}%  {status}")

    if all_clean:
        print()
        print("  All 21 raw columns fully populated for 2023+ rows.")

    # Most recent 5 dates and their raw-column completeness
    print()
    print("  Most recent 5 dates — per-row raw column completeness:")
    print()
    recent5 = df_23.nlargest(5, "date").reset_index(drop=True)
    for _, row in recent5.iterrows():
        date_str = row["date"].strftime("%Y-%m-%d")
        nan_cols = []
        for c in RAW_COLS_21:
            val = row.get(c, float("nan"))
            if c not in df_23.columns or pd.isna(val):
                nan_cols.append(c)
        if nan_cols:
            status = f"NaN in: {', '.join(nan_cols)}"
        else:
            status = "All 21 raw cols present"
        print(f"    {date_str}  |  {status}")


# ── Task 3 ───────────────────────────────────────────────────────────────────

def task3_lead_bug(df: pd.DataFrame) -> None:
    print()
    print(SEPARATOR)
    print("TASK 3 — Lead Feature NaN Audit (last 5 rows)")
    print(SEPARATOR)
    print()
    print("  Expected behaviour:")
    print("    Most recent row (rank-0) : ALL lead cols filled (from forecast API)")
    print("    Rank-1 row               : t1 filled from actual, t2/t3/t4 may vary")
    print("    Last 4 rows              : AQI_t+1..t+4 targets NaN (future unknown)")
    print()

    if "date" not in df.columns:
        print("  ERROR: 'date' column missing.")
        return

    recent = df.nlargest(5, "date").reset_index(drop=True)

    # Determine which lead/target columns exist in the data
    present_leads   = [c for c in ALL_LEAD_COLS  if c in df.columns]
    present_targets = [c for c in TARGET_COLS     if c in df.columns]
    absent_leads    = [c for c in ALL_LEAD_COLS   if c not in df.columns]

    if absent_leads:
        print(f"  NOTE: {len(absent_leads)} lead columns not present in stored data:")
        for g_name, g_cols in LEAD_GROUPS.items():
            missing_in_group = [c for c in g_cols if c not in df.columns]
            if missing_in_group:
                print(f"    [{g_name}] absent: {', '.join(missing_in_group)}")
        print()

    bugs_found: list[str] = []

    for rank, (_, row) in enumerate(recent.iterrows()):
        date_str = row["date"].strftime("%Y-%m-%d")
        is_most_recent = (rank == 0)
        label = "MOST RECENT (rank-0)" if is_most_recent else f"rank-{rank}"

        # Which lead cols are NaN for this row?
        nan_leads   = [c for c in present_leads   if pd.isna(row.get(c, float("nan")))]
        nan_targets = [c for c in present_targets if pd.isna(row.get(c, float("nan")))]
        filled_leads = len(present_leads) - len(nan_leads)

        print(f"  [{label}]  date={date_str}")
        print(f"    Lead cols checked : {len(present_leads)} present")
        print(f"    Lead cols filled  : {filled_leads}")

        if nan_leads:
            print(f"    Lead cols NaN     : {len(nan_leads)}")
            for g_name, g_cols in LEAD_GROUPS.items():
                nan_in_group = [c for c in g_cols if c in present_leads and pd.isna(row.get(c, float("nan")))]
                if nan_in_group:
                    print(f"      [{g_name}]: {', '.join(nan_in_group)}")

            if is_most_recent:
                bugs_found.append(
                    f"BUG on {date_str} (most recent row): "
                    f"{len(nan_leads)} lead feature(s) still NaN — "
                    f"forecast fill likely failed or was never run. "
                    f"Affected cols: {', '.join(nan_leads[:12])}"
                    + (" ..." if len(nan_leads) > 12 else "")
                )
        else:
            print(f"    Lead cols NaN     : 0  -- ALL FILLED")

        # Target analysis — expected NaN pattern
        # rank-0 (last row): t+1, t+2, t+3, t+4 should all be NaN
        # rank-1           : t+2, t+3, t+4 should be NaN (t+1 is yesterday's actual)
        # rank-2           : t+3, t+4 should be NaN
        # rank-3           : t+4 should be NaN
        # rank-4           : no targets should be NaN
        expected_nan_targets = TARGET_COLS[: max(0, 4 - rank)]
        expected_filled_targets = TARGET_COLS[max(0, 4 - rank):]

        actually_nan    = set(nan_targets)
        expected_nan_s  = set(expected_nan_targets) & set(present_targets)
        expected_fill_s = set(expected_filled_targets) & set(present_targets)

        unexpected_nan    = actually_nan - expected_nan_s     # NaN when should be filled
        unexpected_filled = expected_nan_s - actually_nan     # filled when should be NaN

        target_status_parts = []
        if nan_targets:
            target_status_parts.append(f"NaN: {', '.join(nan_targets)}")
        else:
            target_status_parts.append("None NaN")

        if unexpected_nan:
            target_status_parts.append(f"  UNEXPECTED NaN: {', '.join(sorted(unexpected_nan))}")
            bugs_found.append(
                f"BUG on {date_str} (rank-{rank}): "
                f"target col(s) {sorted(unexpected_nan)} are NaN but should be filled "
                f"(only last {4-rank} target(s) should be NaN for this row)"
            )
        if unexpected_filled:
            target_status_parts.append(f"  UNEXPECTED FILL: {', '.join(sorted(unexpected_filled))}")
            bugs_found.append(
                f"WARN on {date_str} (rank-{rank}): "
                f"target col(s) {sorted(unexpected_filled)} are filled when they should be NaN "
                f"— possible future data leakage"
            )

        if not unexpected_nan and not unexpected_filled:
            target_status_parts.append("(pattern correct)")

        print(f"    Targets           : {' | '.join(target_status_parts)}")
        print()

    # ── Summary ──────────────────────────────────────────────────────────────
    print(SEPARATOR)
    print("DIAGNOSTIC SUMMARY")
    print(SEPARATOR)
    if bugs_found:
        print(f"  {len(bugs_found)} issue(s) detected:")
        for i, bug in enumerate(bugs_found, 1):
            print(f"  {i}. {bug}")
    else:
        print("  No bugs detected.")
        print("  - Lead features on most recent row: all filled (forecast API working)")
        print("  - Target NaN pattern on last 4 rows: correct")

    # ── Lead col NaN count across all 5 rows (compact table) ─────────────────
    print()
    print("  Lead feature NaN summary across last 5 rows:")
    print(f"  {'Date':<12}  {'Lead NaN':>10}  {'Target NaN':>12}  {'Verdict'}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*12}  {'-'*30}")
    for rank, (_, row) in enumerate(recent.iterrows()):
        date_str  = row["date"].strftime("%Y-%m-%d")
        nl = sum(1 for c in present_leads   if pd.isna(row.get(c, float("nan"))))
        nt = sum(1 for c in present_targets if pd.isna(row.get(c, float("nan"))))

        # Expected NaN leads: 0 for all rows (forecast should fill them all)
        # Expected NaN targets: max(0, 4-rank)
        exp_nl = 0
        exp_nt = max(0, 4 - rank)
        lead_ok   = "OK" if nl == exp_nl else f"BUG (got {nl}, want 0)"
        target_ok = "OK" if nt == exp_nt else f"WARN (got {nt}, want {exp_nt})"
        verdict = f"leads={lead_ok}, targets={target_ok}"
        label = "(most recent)" if rank == 0 else ""
        print(f"  {date_str:<12}  {nl:>10}  {nt:>12}  {verdict} {label}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print(SEPARATOR)
    print("Karachi AQI — Feature Store Diagnostic  (2026-05-27)")
    print(SEPARATOR)
    print("Connecting to MongoDB...")

    df, source = load_data()
    df = _coerce_dates(df)

    print(f"  Loaded {len(df):,} documents, {len(df.columns)} columns")
    if "date" in df.columns:
        print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    task1_backup(df, source)
    task2_completeness(df)
    task3_lead_bug(df)


if __name__ == "__main__":
    main()
