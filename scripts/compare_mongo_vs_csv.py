"""
Phase 1: Deep comparison of MongoDB feature_store vs local_raw_dataset.csv
Pulls MongoDB rows with date >= 2023-01-01, aligns on common dates,
and compares per-column values.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from config.db import get_collection, COLLECTION_FEATURE_STORE

# Columns that should be essentially identical (not IQR-capped)
NON_CAPPED = ["AQI", "PM2_5", "CO", "wind_direction", "apparent_temp", "surface_pressure", "wind_gusts"]
# Columns that are IQR-capped in MongoDB
IQR_CAPPED = ["PM10", "SO2", "NO2", "O3", "Temperature", "Humidity", "Precipitation", "wind_speed"]
# All 15 raw columns shared between MongoDB and local CSV
SHARED_COLS = NON_CAPPED + IQR_CAPPED

# MongoDB uses "PM2_5", CSV may have variations — will handle aliasing below

def run():
    print("=" * 60)
    print("PHASE 1: MongoDB vs local_raw_dataset.csv Comparison")
    print("=" * 60)

    # ── Load MongoDB rows >= 2023-01-01 ──────────────────────────────
    print("\n[1/4] Loading MongoDB rows with date >= 2023-01-01 ...")
    col = get_collection(COLLECTION_FEATURE_STORE)
    docs = list(col.find({"date": {"$gte": "2023-01-01"}}, {"_id": 0}))
    mongo_df = pd.DataFrame(docs)
    print(f"  Rows from MongoDB: {len(mongo_df)}")
    print(f"  Date range: {mongo_df['date'].min()} to {mongo_df['date'].max()}")
    print(f"  Columns available: {sorted(mongo_df.columns.tolist())}")

    # Keep only the 15 raw columns present in MongoDB
    mongo_raw_cols = ["date"] + [c for c in SHARED_COLS if c in mongo_df.columns]
    mongo_df = mongo_df[mongo_raw_cols].copy()
    mongo_df["date"] = pd.to_datetime(mongo_df["date"])

    # ── Load local CSV ────────────────────────────────────────────────
    print("\n[2/4] Loading data/local_raw_dataset.csv ...")
    csv_path = Path(__file__).parent.parent / "data" / "local_raw_dataset.csv"
    csv_df = pd.read_csv(csv_path)
    print(f"  Rows in CSV: {len(csv_df)}")
    print(f"  Date range: {csv_df['date'].min()} to {csv_df['date'].max()}")
    print(f"  Columns in CSV: {sorted(csv_df.columns.tolist())}")
    csv_df["date"] = pd.to_datetime(csv_df["date"])

    # ── Align on common dates ─────────────────────────────────────────
    print("\n[3/4] Aligning on common dates ...")
    mongo_dates = set(mongo_df["date"].dt.date)
    csv_dates = set(csv_df["date"].dt.date)

    # Dates in MongoDB range (2023-01-01 to 2026-05-24) that are missing from MongoDB
    csv_in_range = {d for d in csv_dates if pd.Timestamp("2023-01-01").date() <= d <= pd.Timestamp("2026-05-24").date()}
    missing_in_mongo = sorted(csv_in_range - mongo_dates)
    common_dates = sorted(mongo_dates & csv_dates)

    print(f"  Dates in MongoDB (>= 2023-01-01): {len(mongo_dates)}")
    print(f"  Dates in CSV (in 2023-01-01 to 2026-05-24 range): {len(csv_in_range)}")
    print(f"  Common dates: {len(common_dates)}")
    print(f"  Dates in CSV range but missing from MongoDB: {len(missing_in_mongo)}")
    if missing_in_mongo:
        print(f"  Missing dates: {missing_in_mongo[:20]}")
        if len(missing_in_mongo) > 20:
            print(f"  ... and {len(missing_in_mongo)-20} more")

    # Filter both to common dates only
    common_set = set(pd.Timestamp(d) for d in common_dates)
    m = mongo_df[mongo_df["date"].isin(common_set)].set_index("date").sort_index()
    c = csv_df[csv_df["date"].isin(common_set)].set_index("date").sort_index()

    # ── Per-column comparison ─────────────────────────────────────────
    print("\n[4/4] Per-column comparison ...")
    shared_in_both = [col for col in SHARED_COLS if col in m.columns and col in c.columns]
    print(f"  Shared columns in both: {shared_in_both}\n")

    print(f"  {'Column':<20} {'Type':<12} {'Mean Diff':>10} {'Max AbsDiff':>12} {'Rows>5':>8} {'Rows>50':>8}")
    print("  " + "-" * 78)

    anomalous_rows = {}
    for col in shared_in_both:
        diff = (m[col] - c[col]).abs()
        mean_diff = diff.mean()
        max_diff = diff.max()
        rows_gt5 = (diff > 5).sum()
        rows_gt50 = (diff > 50).sum()
        ctype = "non-capped" if col in NON_CAPPED else "IQR-capped"
        print(f"  {col:<20} {ctype:<12} {mean_diff:>10.4f} {max_diff:>12.4f} {rows_gt5:>8} {rows_gt50:>8}")

        # Flag anomalous rows
        threshold = 5 if col in NON_CAPPED else 50
        bad = diff[diff > threshold]
        if len(bad) > 0:
            for date_idx, val in bad.items():
                key = str(date_idx.date())
                if key not in anomalous_rows:
                    anomalous_rows[key] = []
                anomalous_rows[key].append(f"{col}(diff={val:.2f})")

    # ── Anomalous rows summary ────────────────────────────────────────
    print(f"\n  Anomalous rows (diff > 5 for non-capped, diff > 50 for IQR-capped):")
    if anomalous_rows:
        print(f"  Total anomalous dates: {len(anomalous_rows)}")
        for date_str, issues in sorted(anomalous_rows.items())[:30]:
            print(f"    {date_str}: {', '.join(issues)}")
        if len(anomalous_rows) > 30:
            print(f"  ... and {len(anomalous_rows)-30} more")
    else:
        print("  None found — data looks clean!")

    # ── Check for columns in CSV but not in MongoDB ───────────────────
    print("\n  Columns in CSV but not in MongoDB feature_store rows:")
    csv_only_cols = [c for c in csv_df.columns if c != "date" and c not in m.columns]
    print(f"  {csv_only_cols}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total rows compared:         {len(common_dates)}")
    print(f"  Missing from MongoDB (in range): {len(missing_in_mongo)}")
    non_capped_issues = sum(1 for col in NON_CAPPED if col in shared_in_both and (m[col]-c[col]).abs().max() > 5)
    print(f"  Non-capped columns with diff>5:  {non_capped_issues}")
    iqr_corruption = sum(1 for col in IQR_CAPPED if col in shared_in_both and (m[col]-c[col]).abs().max() > 200)
    print(f"  IQR-capped columns with diff>200 (possible corruption): {iqr_corruption}")
    print(f"  Anomalous dates total:           {len(anomalous_rows)}")

    if non_capped_issues == 0 and iqr_corruption == 0:
        print("\n  PHASE 1 RESULT: PASS — No data corruption detected.")
        print("  IQR-capping diffs are expected. OK to proceed with Phase 2.")
    else:
        print("\n  PHASE 1 RESULT: REVIEW REQUIRED")
    print("=" * 60)

if __name__ == "__main__":
    run()
