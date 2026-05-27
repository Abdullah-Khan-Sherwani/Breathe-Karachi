import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from pymongo import ASCENDING
from config.db import get_collection, COLLECTION_HOURLY

LAT, LON, TIMEZONE = 24.8607, 67.0011, "Asia/Karachi"
AIR_URL     = "https://air-quality-api.open-meteo.com/v1/air-quality"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

AIR_PARAMS     = "us_aqi,pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone"
WEATHER_PARAMS = "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,apparent_temperature,surface_pressure,wind_gusts_10m"

COL_MAP = {
    "us_aqi": "AQI", "pm2_5": "PM2_5", "pm10": "PM10",
    "nitrogen_dioxide": "NO2", "sulphur_dioxide": "SO2",
    "carbon_monoxide": "CO", "ozone": "O3",
    "temperature_2m": "Temperature", "relative_humidity_2m": "Humidity",
    "precipitation": "Precipitation",
    "wind_speed_10m": "wind_speed", "wind_direction_10m": "wind_direction",
    "apparent_temperature": "apparent_temp",
    "surface_pressure": "surface_pressure",
    "wind_gusts_10m": "wind_gusts",
}


def _ensure_index(col):
    try:
        col.create_index([("time", ASCENDING)], unique=True)
    except Exception:
        pass


def _latest_stored_time(col) -> datetime:
    doc = col.find_one({"time": {"$exists": True}}, sort=[("time", -1)])
    if doc is None:
        return datetime.now(timezone.utc) - timedelta(days=7)
    dt = datetime.fromisoformat(doc["time"])
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _fetch_hourly(start_date: str, end_date: str) -> pd.DataFrame | None:
    base = {"latitude": LAT, "longitude": LON, "timezone": TIMEZONE,
            "start_date": start_date, "end_date": end_date}
    try:
        air_resp = requests.get(
            AIR_URL,
            params={**base, "hourly": AIR_PARAMS},
            timeout=20,
        )
        air_resp.raise_for_status()
        air = air_resp.json()["hourly"]
    except Exception as e:
        print(f"  air-quality API error ({start_date} to {end_date}): {e}")
        return None

    try:
        wthr_resp = requests.get(
            FORECAST_URL,
            params={**base, "hourly": WEATHER_PARAMS},
            timeout=20,
        )
        wthr_resp.raise_for_status()
        wthr = wthr_resp.json()["hourly"]
    except Exception as e:
        print(f"  forecast API error ({start_date} to {end_date}): {e}")
        return None

    df = pd.merge(pd.DataFrame(air), pd.DataFrame(wthr), on="time")
    df["time"] = pd.to_datetime(df["time"])
    return df


def run() -> None:
    col = get_collection(COLLECTION_HOURLY)
    _ensure_index(col)

    latest = _latest_stored_time(col)
    now_utc = datetime.now(timezone.utc)

    start_date = latest.date().isoformat()
    end_date   = now_utc.date().isoformat()

    df = _fetch_hourly(start_date, end_date)
    if df is None or df.empty:
        print("No hourly data returned.")
        return

    cutoff = latest + timedelta(hours=1)
    cutoff_naive = cutoff.replace(tzinfo=None)
    df = df[df["time"] >= cutoff_naive]

    if df.empty:
        print("hourly_feature_store is up to date.")
        return

    fetched_at = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    docs_inserted = 0

    for _, row in df.iterrows():
        time_str = row["time"].isoformat()
        doc = {
            "time": time_str,
            "date": row["time"].date().isoformat(),
            "hour": int(row["time"].hour),
            "fetched_at": fetched_at,
        }
        for api_col, mapped_col in COL_MAP.items():
            val = row.get(api_col)
            if pd.notna(val):
                doc[mapped_col] = float(val)
            else:
                doc[mapped_col] = None

        col.update_one({"time": time_str}, {"$set": doc}, upsert=True)
        docs_inserted += 1

    if docs_inserted:
        date_from = df["time"].min().date().isoformat()
        date_to   = df["time"].max().date().isoformat()
        print(f"Updated {docs_inserted} hourly rows ({date_from} to {date_to}).")
    else:
        print("hourly_feature_store is up to date.")


if __name__ == "__main__":
    run()
