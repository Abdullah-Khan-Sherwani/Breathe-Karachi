"""
Build hourly AQI feature dataset — local CSV, no MongoDB.

Fetches hourly Open-Meteo data from START_DATE to yesterday,
engineers features, and computes 3-day-ahead daily-mean targets
WITHOUT any autoregressive rollout.

Targets (each is the mean of a future 24-hour window):
  AQI_day1_mean  = mean(AQI[t+1h  .. t+24h])
  AQI_day2_mean  = mean(AQI[t+25h .. t+48h])
  AQI_day3_mean  = mean(AQI[t+49h .. t+72h])

To add future weather forecast features later, set:
  USE_WEATHER_FORECAST_FEATURES = True
and implement fetch_forecast() below — then re-run this script.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import requests
import numpy as np
import pandas as pd
from datetime import date, timedelta

# ── Toggles ───────────────────────────────────────────────────────────────────
USE_WEATHER_FORECAST_FEATURES = False   # flip True when forecast data is ready

# ── API config ────────────────────────────────────────────────────────────────
LAT, LON, TIMEZONE = 24.8607, 67.0011, "Asia/Karachi"
START_DATE  = date(2022, 8, 5)           # earliest date with Karachi AQI data
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "hourly_features.csv"

AIR_URL     = "https://air-quality-api.open-meteo.com/v1/air-quality"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

AIR_PARAMS     = "us_aqi,pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone"
WEATHER_PARAMS = "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m"

COL_MAP = {
    "us_aqi": "AQI", "pm2_5": "PM2_5", "pm10": "PM10",
    "nitrogen_dioxide": "NO2", "sulphur_dioxide": "SO2",
    "carbon_monoxide": "CO", "ozone": "O3",
    "temperature_2m": "Temperature", "relative_humidity_2m": "Humidity",
    "precipitation": "Precipitation",
    "wind_speed_10m": "wind_speed", "wind_direction_10m": "wind_direction",
}

IQR_COLS = ["PM10", "SO2", "NO2", "O3", "Temperature", "Humidity", "Precipitation", "wind_speed"]

# ── Feature / target column lists ─────────────────────────────────────────────
FEATURE_COLS = [
    # Current-hour pollutants
    "AQI", "PM10", "NO2", "SO2", "O3",
    "log_PM2_5", "log_CO",
    # Current-hour weather
    "Temperature", "Humidity", "Precipitation",
    "wind_speed", "wind_sin", "wind_cos",
    # AQI hourly lags
    "AQI_lag_1h", "AQI_lag_2h", "AQI_lag_3h", "AQI_lag_6h",
    "AQI_lag_12h", "AQI_lag_24h", "AQI_lag_48h", "AQI_lag_168h",
    # AQI rolling windows (all shifted — no leakage)
    "AQI_roll_mean_3h", "AQI_roll_mean_6h",
    "AQI_roll_mean_12h", "AQI_roll_mean_24h",
    "AQI_roll_std_24h", "AQI_roll_min_24h", "AQI_roll_max_24h",
    # Pollutant lags
    "log_PM2_5_lag_1h", "log_PM2_5_lag_24h",
    "PM10_lag_1h", "PM10_lag_24h",
    # Temporal
    "hour_sin", "hour_cos",
    "month", "is_rush_hour",
    # Season dummies (Autumn = dropped reference)
    "season_Spring", "season_Summer", "season_Winter",
    # Weekday dummies (Monday=0 = dropped reference)
    "weekday_1", "weekday_2", "weekday_3", "weekday_4", "weekday_5", "weekday_6",
]

# Placeholder — implement and append to FEATURE_COLS when toggled on
FORECAST_FEATURE_COLS = [
    "temp_d1",       "temp_d2",       "temp_d3",
    "humidity_d1",   "humidity_d2",   "humidity_d3",
    "precip_d1",     "precip_d2",     "precip_d3",
    "wind_speed_d1", "wind_speed_d2", "wind_speed_d3",
]

TARGET_COLS = ["AQI_day1_mean", "AQI_day2_mean", "AQI_day3_mean"]


# ── Fetch ──────────────────────────────────────────────────────────────────────
def fetch_chunk(start: str, end: str) -> pd.DataFrame | None:
    """Fetch hourly air quality + weather for a date range in one API call pair."""
    base = {"latitude": LAT, "longitude": LON, "timezone": TIMEZONE,
            "start_date": start, "end_date": end}
    try:
        air_resp  = requests.get(AIR_URL,     params={**base, "hourly": AIR_PARAMS},     timeout=60)
        wthr_resp = requests.get(ARCHIVE_URL, params={**base, "hourly": WEATHER_PARAMS}, timeout=60)
        air  = air_resp.json()
        wthr = wthr_resp.json()
    except Exception as e:
        print(f"    fetch error {start}–{end}: {e}")
        return None

    if "hourly" not in air or "hourly" not in wthr:
        print(f"    missing hourly key for {start}–{end}")
        return None

    df = pd.merge(
        pd.DataFrame(air["hourly"]),
        pd.DataFrame(wthr["hourly"]),
        on="time"
    )
    df["time"] = pd.to_datetime(df["time"])
    return df.rename(columns=COL_MAP)


def fetch_all(start: date, end: date) -> pd.DataFrame:
    """Fetch data in yearly chunks and concatenate."""
    chunks = []
    current = start
    while current <= end:
        chunk_end = min(date(current.year, 12, 31), end)
        print(f"  {current} to {chunk_end} ...", end=" ", flush=True)
        df = fetch_chunk(current.isoformat(), chunk_end.isoformat())
        if df is not None and not df.empty:
            chunks.append(df)
            print(f"{len(df)} rows")
        else:
            print("skipped")
        current = date(current.year + 1, 1, 1)

    df = pd.concat(chunks, ignore_index=True)
    df = df.sort_values("time").reset_index(drop=True)

    # Reindex to complete hourly calendar — fills any gaps
    full_range = pd.date_range(df["time"].min(), df["time"].max(), freq="h")
    df = df.set_index("time").reindex(full_range).rename_axis("time").reset_index()
    df = df.ffill()

    # Drop hours where AQI is still null (before data availability)
    df = df.dropna(subset=["AQI"]).reset_index(drop=True)
    return df


# ── Feature engineering ────────────────────────────────────────────────────────
def cap_iqr(df: pd.DataFrame) -> pd.DataFrame:
    for col in IQR_COLS:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return df


def add_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    df["log_PM2_5"] = np.log1p(df["PM2_5"])
    df["log_CO"]    = np.log1p(df["CO"])
    return df.drop(columns=["PM2_5", "CO"])


def add_wind(df: pd.DataFrame) -> pd.DataFrame:
    df["wind_sin"] = np.sin(np.deg2rad(df["wind_direction"]))
    df["wind_cos"] = np.cos(np.deg2rad(df["wind_direction"]))
    return df.drop(columns=["wind_direction"])


def add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    hour    = df["time"].dt.hour
    month   = df["time"].dt.month
    weekday = df["time"].dt.weekday

    df["hour_sin"]     = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"]     = np.cos(2 * np.pi * hour / 24)
    df["month"]        = month
    df["is_rush_hour"] = hour.isin([7, 8, 9, 17, 18, 19]).astype(int)

    season_map = {12: "Winter", 1: "Winter", 2: "Winter",
                   3: "Spring",  4: "Spring",  5: "Spring",
                   6: "Summer",  7: "Summer",  8: "Summer",
                   9: "Autumn", 10: "Autumn", 11: "Autumn"}
    season = month.map(season_map)

    # Drop any stale dummies from previous runs before recreating
    stale = [c for c in df.columns if c.startswith("season_") or c.startswith("weekday_")]
    df = df.drop(columns=stale, errors="ignore")

    season_dummies  = pd.get_dummies(season,  prefix="season",  drop_first=True).astype(int)
    weekday_dummies = pd.get_dummies(weekday, prefix="weekday", drop_first=True).astype(int)
    return pd.concat([df, season_dummies, weekday_dummies], axis=1)


def add_lags_rolling(df: pd.DataFrame) -> pd.DataFrame:
    aqi = df["AQI"]

    for h in [1, 2, 3, 6, 12, 24, 48, 168]:
        df[f"AQI_lag_{h}h"] = aqi.shift(h)

    aqi_s1 = aqi.shift(1)   # shifted series — no current-hour leakage
    for w in [3, 6, 12, 24]:
        df[f"AQI_roll_mean_{w}h"] = aqi_s1.rolling(w).mean()
    df["AQI_roll_std_24h"] = aqi_s1.rolling(24).std()
    df["AQI_roll_min_24h"] = aqi_s1.rolling(24).min()
    df["AQI_roll_max_24h"] = aqi_s1.rolling(24).max()

    df["log_PM2_5_lag_1h"]  = df["log_PM2_5"].shift(1)
    df["log_PM2_5_lag_24h"] = df["log_PM2_5"].shift(24)
    df["PM10_lag_1h"]  = df["PM10"].shift(1)
    df["PM10_lag_24h"] = df["PM10"].shift(24)

    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute forward-looking 24h window means without any Python loop.

    Proof of reversed-series rolling trick:
      future_mean(s, offset=k, window=w)[i] = mean(s[i+k], ..., s[i+k+w-1])

    Verified with unit examples before use.
    """
    def future_mean(series: pd.Series, offset: int, window: int) -> pd.Series:
        rev    = series.iloc[::-1].reset_index(drop=True)
        rolled = rev.rolling(window, min_periods=window).mean().shift(offset)
        return rolled.iloc[::-1].reset_index(drop=True)

    df["AQI_day1_mean"] = future_mean(df["AQI"], offset=1,  window=24).values
    df["AQI_day2_mean"] = future_mean(df["AQI"], offset=25, window=24).values
    df["AQI_day3_mean"] = future_mean(df["AQI"], offset=49, window=24).values
    return df


# ── Optional: forecast features (stub) ────────────────────────────────────────
def add_forecast_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder — add actual future weather features here when ready.
    These must come from Open-Meteo forecast API at inference time,
    and from a stored forecast archive (not actual values) at training time.
    """
    raise NotImplementedError(
        "Forecast features not yet implemented. "
        "Set USE_WEATHER_FORECAST_FEATURES = False to skip."
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def build() -> None:
    end_date = date.today() - timedelta(days=1)

    print(f"Fetching hourly data: {START_DATE} to {end_date}")
    df = fetch_all(START_DATE, end_date)
    print(f"  Raw rows after gap-fill + AQI filter: {len(df)}")

    print("Engineering features ...")
    df = cap_iqr(df)
    df = add_log_transforms(df)
    df = add_wind(df)
    df = add_temporal(df)
    df = add_lags_rolling(df)
    df = add_targets(df)

    if USE_WEATHER_FORECAST_FEATURES:
        df = add_forecast_features(df)

    all_cols = ["time"] + FEATURE_COLS + TARGET_COLS
    if USE_WEATHER_FORECAST_FEATURES:
        all_cols += FORECAST_FEATURE_COLS

    # Keep only valid rows (lag warmup + target window fully populated)
    df = df.dropna(subset=FEATURE_COLS + TARGET_COLS).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[[c for c in all_cols if c in df.columns]].to_csv(OUTPUT_PATH, index=False)

    print(f"\nDataset saved: {OUTPUT_PATH}")
    print(f"  Rows            : {len(df)}")
    print(f"  Features        : {len(FEATURE_COLS)}" +
          (f" + {len(FORECAST_FEATURE_COLS)} forecast" if USE_WEATHER_FORECAST_FEATURES else ""))
    print(f"  Date range      : {df['time'].iloc[0]}  to  {df['time'].iloc[-1]}")
    print(f"  AQI_day3 mean   : {df['AQI_day3_mean'].mean():.1f}  "
          f"std={df['AQI_day3_mean'].std():.1f}")


if __name__ == "__main__":
    build()
