"""
Incremental update — fetches every day missing from feature_store up to
yesterday and upserts it. Runs hourly via GitHub Actions.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from datetime import date, timedelta
from config.db import get_collection, COLLECTION_FEATURE_STORE
from src.fetch_data import fetch_day, to_daily_record, upsert


def latest_stored_date() -> date:
    """Return the most recent date already in feature_store."""
    col = get_collection(COLLECTION_FEATURE_STORE)
    doc = col.find_one({"date": {"$exists": True}}, sort=[("date", -1)])
    if doc is None:
        from src.fetch_data import BACKFILL_START
        return BACKFILL_START - timedelta(days=1)
    return date.fromisoformat(doc["date"])


def run() -> None:
    last  = latest_stored_date()
    today = date.today()
    missing = pd.date_range(last + timedelta(days=1), today - timedelta(days=1))

    if missing.empty:
        print("feature_store is up to date.")
        return

    for d in missing:
        day = d.date().isoformat()
        df  = fetch_day(day)
        if df is None or df.empty:
            print(f"  skip {day} (no data)")
            continue
        upsert(to_daily_record(day, df))
        print(f"  upserted {day}")

    print(f"Updated {len(missing)} day(s).")


if __name__ == "__main__":
    run()
