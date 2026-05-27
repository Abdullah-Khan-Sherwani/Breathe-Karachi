"""
Feature engineering — loads all raw daily rows from feature_store,
computes engineered features, and upserts processed documents back.
Runs hourly after update_daily_data.py.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from config.db import get_collection, COLLECTION_FEATURE_STORE

IQR_COLS = ["PM10", "SO2", "NO2", "O3", "Temperature", "Humidity", "Precipitation"]


def load_raw() -> pd.DataFrame:
    docs = list(get_collection(COLLECTION_FEATURE_STORE).find({}, {"_id": 0}))
    df   = pd.DataFrame(docs)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


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
    return df


def add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    df["month"]   = df["date"].dt.month
    df["season"]  = df["date"].dt.month.map(
        {12: "Winter", 1: "Winter", 2: "Winter",
          3: "Spring",  4: "Spring",  5: "Spring",
          6: "Summer",  7: "Summer",  8: "Summer",
          9: "Autumn", 10: "Autumn", 11: "Autumn"}
    )
    df["weekday"] = df["date"].dt.weekday

    season_dummies  = pd.get_dummies(df["season"],  prefix="season",  drop_first=True)
    weekday_dummies = pd.get_dummies(df["weekday"], prefix="weekday", drop_first=True)
    df = pd.concat([df.drop(columns=["season", "weekday"]), season_dummies, weekday_dummies], axis=1)
    return df


def add_lag_rolling(df: pd.DataFrame) -> pd.DataFrame:
    df["AQI_lag_1"]       = df["AQI"].shift(1)
    df["AQI_lag_2"]       = df["AQI"].shift(2)
    df["AQI_roll_mean_3"] = df["AQI"].shift(1).rolling(3).mean()
    df["AQI_roll_std_3"]  = df["AQI"].shift(1).rolling(3).std()
    df["AQI_diff"]        = df["AQI"].shift(1).diff()
    return df


def add_tier2_features(df: pd.DataFrame) -> pd.DataFrame:
    # 7-day linear slope of AQI (uses only past values — shift(1) before rolling)
    aqi_lagged = df["AQI"].shift(1)
    df["AQI_trend_7d"] = aqi_lagged.rolling(7).apply(
        lambda arr: np.polyfit(np.arange(7), arr, 1)[0], raw=True
    )

    # Interaction: current-day AQI × wind_speed (both already used as raw features)
    if "wind_speed" in df.columns:
        df["AQI_x_wind"] = df["AQI"] * df["wind_speed"]

    # Lag-3 features motivated by cross-correlation peaks (NO2, Humidity peak at lag 3)
    df["NO2_lag_3"]      = df["NO2"].shift(3)
    df["Humidity_lag_3"] = df["Humidity"].shift(3)

    return df


def add_tier3_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering for new Open-Meteo variables (surface_pressure, apparent_temp, wind_gusts)."""
    tier3_vars = {
        "surface_pressure": "sp",
        "apparent_temp":    "at",
        "wind_gusts":       "wg",
    }
    for col, _ in tier3_vars.items():
        if col not in df.columns:
            continue
        # Forecast windows (actual future values used at training; forecast API at inference)
        df[f"{col}_t1"] = df[col].shift(-1)
        df[f"{col}_t2"] = df[col].shift(-2)
        df[f"{col}_t3"] = df[col].shift(-3)
        # Lagged and rolling context
        df[f"{col}_lag_1"]      = df[col].shift(1)
        df[f"{col}_roll_mean_7"] = df[col].shift(1).rolling(7).mean()
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df["AQI_t+1"] = df["AQI"].shift(-1)
    df["AQI_t+2"] = df["AQI"].shift(-2)
    df["AQI_t+3"] = df["AQI"].shift(-3)
    return df


def run() -> None:
    df = load_raw()
    df = df.ffill()
    df = cap_iqr(df)
    df = add_log_transforms(df)
    df = add_temporal(df)
    df = add_lag_rolling(df)
    df = add_tier2_features(df)
    df = add_tier3_features(df)
    df = add_targets(df)
    df = df.dropna().reset_index(drop=True)

    col = get_collection(COLLECTION_FEATURE_STORE)
    for record in df.to_dict("records"):
        record["date"] = record["date"].date().isoformat()
        col.update_one({"date": record["date"]}, {"$set": record}, upsert=True)

    print(f"Preprocessed and upserted {len(df)} rows.")


if __name__ == "__main__":
    run()
