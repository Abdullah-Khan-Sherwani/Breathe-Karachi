"""
One-time historical backfill — fetches daily AQI + weather from Open-Meteo
for every day from BACKFILL_START to today and upserts into feature_store.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import requests
import pandas as pd
from datetime import date, timedelta
from tqdm import tqdm
from config.db import get_collection, COLLECTION_FEATURE_STORE

LAT, LON, TIMEZONE = 24.8607, 67.0011, "Asia/Karachi"
BACKFILL_START = date(2023, 1, 1)

AIR_URL     = "https://air-quality-api.open-meteo.com/v1/air-quality"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

AIR_PARAMS     = "us_aqi,pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone"
WEATHER_PARAMS = "temperature_2m,relative_humidity_2m,precipitation"

COL_MAP = {
    "us_aqi": "AQI", "pm2_5": "PM2_5", "pm10": "PM10",
    "nitrogen_dioxide": "NO2", "sulphur_dioxide": "SO2",
    "carbon_monoxide": "CO", "ozone": "O3",
    "temperature_2m": "Temperature", "relative_humidity_2m": "Humidity",
    "precipitation": "Precipitation",
}


def fetch_day(day: str) -> pd.DataFrame | None:
    """Fetch hourly air quality + weather for one date. Returns merged DataFrame or None."""
    base = {"latitude": LAT, "longitude": LON, "timezone": TIMEZONE,
            "start_date": day, "end_date": day}
    try:
        air  = requests.get(AIR_URL,     params={**base, "hourly": AIR_PARAMS},     timeout=15).json()["hourly"]
        wthr = requests.get(ARCHIVE_URL, params={**base, "hourly": WEATHER_PARAMS}, timeout=15).json()["hourly"]
    except Exception:
        return None

    df = pd.merge(pd.DataFrame(air), pd.DataFrame(wthr), on="time")
    df["time"] = pd.to_datetime(df["time"])
    return df


def to_daily_record(day: str, df: pd.DataFrame) -> dict:
    """Average hourly DataFrame to a single dict keyed by standardised column names."""
    row = df.drop(columns=["time"]).mean().rename(COL_MAP)
    return {"date": day, **row.to_dict()}


def upsert(record: dict) -> None:
    col = get_collection(COLLECTION_FEATURE_STORE)
    col.update_one({"date": record["date"]}, {"$set": record}, upsert=True)


def backfill(start: date = BACKFILL_START, end: date = None) -> None:
    end = end or date.today() - timedelta(days=1)
    days = pd.date_range(start, end)
    for d in tqdm(days, desc="Backfilling"):
        day = d.date().isoformat()
        df  = fetch_day(day)
        if df is None or df.empty:
            continue
        upsert(to_daily_record(day, df))
    print(f"Backfill complete: {start} to {end}")


if __name__ == "__main__":
    backfill()
