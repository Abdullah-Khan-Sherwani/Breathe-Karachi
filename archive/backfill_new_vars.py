"""
Backfill new weather and air-quality variables for all dates already in
feature_store.  Run once before reprocessing with preprocess_daily_data.py.

New weather vars  : BLH, cloud_cover, shortwave_rad
New AQ vars       : aod, dust, uv_index
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

ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
AQ_URL       = "https://air-quality-api.open-meteo.com/v1/air-quality"

WEATHER_PARAMS = "boundary_layer_height,cloud_cover,shortwave_radiation"
AQ_PARAMS = "aerosol_optical_depth,dust,uv_index"

WEATHER_COL_MAP = {
    "boundary_layer_height": "BLH",
    "cloud_cover":           "cloud_cover",
    "shortwave_radiation":   "shortwave_rad",
}
AQ_COL_MAP = {
    "aerosol_optical_depth": "aod",
    "dust":                  "dust",
    "uv_index":              "uv_index",
}


def _fetch_chunk(url: str, params_str: str, start: str, end: str) -> pd.DataFrame | None:
    try:
        resp = requests.get(
            url,
            params={
                "latitude": LAT, "longitude": LON, "timezone": TIMEZONE,
                "start_date": start, "end_date": end,
                "hourly": params_str,
            },
            timeout=60,
        ).json()
        if "hourly" not in resp:
            print(f"    API error ({url.split('/')[4]}): {resp.get('reason', str(resp)[:120])}")
            return None
        df = pd.DataFrame(resp["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        return df
    except Exception as e:
        print(f"    fetch error {start}–{end}: {e}")
        return None


def _daily_means(df: pd.DataFrame, col_map: dict) -> dict[str, dict]:
    """Return {date_iso: {mapped_col: value}} daily averages."""
    df = df.copy()
    df["date"] = df["time"].dt.date
    out = {}
    for day, grp in df.groupby("date"):
        row = grp.drop(columns=["time", "date"]).mean()
        record = {}
        for raw_col, mapped in col_map.items():
            if raw_col in row and not np.isnan(row[raw_col]):
                record[mapped] = float(row[raw_col])
        if record:
            out[day.isoformat()] = record
    return out


def run() -> None:
    col = get_collection(COLLECTION_FEATURE_STORE)

    docs = list(col.find({}, {"_id": 0, "date": 1}))
    dates = sorted(d["date"] for d in docs if "date" in d)
    if not dates:
        print("feature_store is empty — nothing to backfill.")
        return

    start_d = date.fromisoformat(dates[0])
    end_d   = date.fromisoformat(dates[-1])
    print(f"Backfilling new variables: {start_d} to {end_d}")
    print(f"  Weather : {WEATHER_PARAMS}")
    print(f"  AQ      : {AQ_PARAMS}\n")

    total_updated = 0
    current = start_d

    while current <= end_d:
        chunk_end = min(date(current.year, 12, 31), end_d)
        s, e = current.isoformat(), chunk_end.isoformat()
        print(f"  Chunk {s} to {e} ...", end=" ", flush=True)

        wthr_df = _fetch_chunk(ARCHIVE_URL, WEATHER_PARAMS, s, e)
        aq_df   = _fetch_chunk(AQ_URL,      AQ_PARAMS,      s, e)

        wthr_daily = _daily_means(wthr_df, WEATHER_COL_MAP) if wthr_df is not None else {}
        aq_daily   = _daily_means(aq_df,   AQ_COL_MAP)      if aq_df   is not None else {}

        all_dates = sorted(set(wthr_daily) | set(aq_daily))
        chunk_updated = 0
        for day in all_dates:
            record = {**wthr_daily.get(day, {}), **aq_daily.get(day, {})}
            if record:
                col.update_one({"date": day}, {"$set": record})
                chunk_updated += 1

        print(f"{chunk_updated} days updated")
        total_updated += chunk_updated
        current = date(current.year + 1, 1, 1)

    print(f"\nDone. {total_updated} documents updated with new variables.")


if __name__ == "__main__":
    run()
