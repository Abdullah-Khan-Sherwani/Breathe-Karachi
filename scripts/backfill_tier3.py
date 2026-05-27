"""
Backfill apparent_temperature, surface_pressure, wind_gusts_10m
for all historical dates already in feature_store.
Run once before reprocessing.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import requests
import pandas as pd
import numpy as np
from datetime import date
from config.db import get_collection, COLLECTION_FEATURE_STORE

LAT, LON, TIMEZONE = 24.8607, 67.0011, "Asia/Karachi"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
NEW_PARAMS   = "apparent_temperature,surface_pressure,wind_gusts_10m"

COL_MAP = {
    "apparent_temperature": "apparent_temp",
    "surface_pressure":     "surface_pressure",
    "wind_gusts_10m":       "wind_gusts",
}


def fetch_chunk(start: str, end: str) -> pd.DataFrame | None:
    try:
        resp = requests.get(
            ARCHIVE_URL,
            params={
                "latitude": LAT, "longitude": LON, "timezone": TIMEZONE,
                "start_date": start, "end_date": end,
                "hourly": NEW_PARAMS,
            },
            timeout=30,
        ).json()
        if "hourly" not in resp:
            print(f"  API error {start}–{end}: {resp.get('reason', resp)}")
            return None
        df = pd.DataFrame(resp["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        return df
    except Exception as e:
        print(f"  fetch error {start}–{end}: {e}")
        return None


def run() -> None:
    col = get_collection(COLLECTION_FEATURE_STORE)

    # Get date range from existing docs
    docs = list(col.find({}, {"_id": 0, "date": 1}))
    dates = sorted(d["date"] for d in docs if "date" in d)
    if not dates:
        print("feature_store is empty.")
        return

    start_date = dates[0]
    end_date   = dates[-1]
    print(f"Backfilling {start_date} to {end_date} ...")

    # Fetch in yearly chunks
    start = date.fromisoformat(start_date)
    end   = date.fromisoformat(end_date)
    updated = 0

    current = start
    while current <= end:
        chunk_end = min(date(current.year, 12, 31), end)
        print(f"  {current} to {chunk_end} ...", end=" ", flush=True)
        df = fetch_chunk(current.isoformat(), chunk_end.isoformat())
        if df is None:
            current = date(current.year + 1, 1, 1)
            continue

        df["date"] = df["time"].dt.date
        for day, grp in df.groupby("date"):
            row = grp.drop(columns=["time", "date"]).mean()
            record = {COL_MAP[c]: float(row[c]) for c in COL_MAP if c in row and not np.isnan(row[c])}
            if record:
                col.update_one({"date": day.isoformat()}, {"$set": record})
                updated += 1

        print(f"{(chunk_end - current).days + 1} days")
        current = date(current.year + 1, 1, 1)

    print(f"Done. Updated {updated} documents.")


if __name__ == "__main__":
    run()
