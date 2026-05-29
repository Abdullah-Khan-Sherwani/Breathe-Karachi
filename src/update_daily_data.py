"""
Incremental update — fetches every day missing from feature_store up to
yesterday and upserts it. Also resyncs the last RESYNC_DAYS days to pick up
CAMS reanalysis revisions (Open-Meteo retroactively refines recent values).

Runs hourly via GitHub Actions.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from datetime import date, datetime, timezone, timedelta
from config.db import get_collection, COLLECTION_FEATURE_STORE
from src.fetch_data import fetch_day, to_daily_record, upsert

PKT = timezone(timedelta(hours=5))

# Re-fetch the last N days on every run so that CAMS reanalysis revisions
# (which typically settle within 48–72 h) are picked up automatically.
# upsert() uses $set on raw observed fields only — no lead or target columns
# are touched, so this is safe to run unconditionally.
RESYNC_DAYS = 3


def latest_stored_date() -> date:
    """Return the most recent date already in feature_store."""
    col = get_collection(COLLECTION_FEATURE_STORE)
    doc = col.find_one({"date": {"$exists": True}}, sort=[("date", -1)])
    if doc is None:
        from src.fetch_data import BACKFILL_START
        return BACKFILL_START - timedelta(days=1)
    return date.fromisoformat(doc["date"])


def run() -> None:
    today = datetime.now(PKT).date()
    last  = latest_stored_date()

    # ── 1. Forward-fill: insert genuinely missing days ─────────────────────────
    missing = pd.date_range(last + timedelta(days=1), today - timedelta(days=1))
    if not missing.empty:
        for d in missing:
            day = d.date().isoformat()
            df  = fetch_day(day)
            if df is None or df.empty:
                print(f"  skip {day} (no data returned)")
                continue
            upsert(to_daily_record(day, df))
            print(f"  inserted {day}")
        print(f"Inserted {len(missing)} new day(s).")
    else:
        print("No new days to insert.")

    # ── 2. Resync: re-fetch last RESYNC_DAYS to catch CAMS revisions ───────────
    # Only resyncs days that already exist in feature_store (upsert=True in
    # upsert() means it would insert too, but these rows already exist from step 1).
    resync_start = today - timedelta(days=RESYNC_DAYS)
    resync_end   = today - timedelta(days=1)          # never fetch today (incomplete)
    resync_dates = pd.date_range(resync_start, resync_end)

    resynced = 0
    for d in resync_dates:
        day = d.date().isoformat()
        df  = fetch_day(day)
        if df is None or df.empty:
            print(f"  resync skip {day} (no data returned)")
            continue
        upsert(to_daily_record(day, df))
        resynced += 1

    print(f"Resynced {resynced} day(s) ({resync_start} to {resync_end}).")


if __name__ == "__main__":
    run()
