"""
Feature engineering — loads all raw daily rows from feature_store,
computes engineered features, and upserts processed documents back.
Runs hourly after update_daily_data.py.

Produces 120 columns matching the feature_store schema:
  - 15 raw inputs (AQI, pollutants, weather, tier-3 weather)
  - 100 engineered features
  - 3 targets (AQI_t+1, AQI_t+2, AQI_t+3)
  - date identifier
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import requests
from config.db import get_collection, COLLECTION_FEATURE_STORE

# IQR capping applied to these columns (includes wind_speed).
# Quantiles are computed on ALL loaded rows so the thresholds are stable
# across incremental pipeline runs.
IQR_COLS = [
    "PM10", "SO2", "NO2", "O3",
    "Temperature", "Humidity", "Precipitation", "wind_speed",
]

RAW_COLS = [
    "date",
    "AQI", "PM2_5", "PM10", "NO2", "SO2", "CO", "O3",
    "Temperature", "Humidity", "Precipitation",
    "wind_speed", "wind_direction",
    "apparent_temp", "surface_pressure", "wind_gusts",
]

# Columns produced by forward-shifting — NaN for the last 3 rows by design.
# These are filled from the forecast API before upserting, so they should not
# trigger dropna() on the main pipeline.
_LEAD_COLS = (
    [f"{c}_t{d}" for c in ["Temperature", "Humidity", "Precipitation", "wind_speed"] for d in [1, 2, 3]]
    + [f"wind_dir_{s}_t{d}" for s in ["sin", "cos"] for d in [1, 2, 3]]
    + [f"{c}_t{d}" for c in ["surface_pressure", "apparent_temp", "wind_gusts"] for d in [1, 2, 3]]
)
_TARGET_COLS = ["AQI_t+1", "AQI_t+2", "AQI_t+3"]

_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_FORECAST_HOURLY = (
    "temperature_2m,relative_humidity_2m,precipitation,"
    "wind_speed_10m,wind_direction_10m,"
    "apparent_temperature,surface_pressure,wind_gusts_10m"
)
_FORECAST_COL_MAP = {
    "temperature_2m": "Temperature",
    "relative_humidity_2m": "Humidity",
    "precipitation": "Precipitation",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_direction",
    "apparent_temperature": "apparent_temp",
    "surface_pressure": "surface_pressure",
    "wind_gusts_10m": "wind_gusts",
}
LAT, LON, TIMEZONE = 24.8607, 67.0011, "Asia/Karachi"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw() -> pd.DataFrame:
    """
    Load all documents from feature_store, keep only RAW_COLS, reindex to a
    complete daily calendar and forward-fill any gaps.
    Returns a DataFrame sorted by date with a clean integer index.
    """
    docs = list(get_collection(COLLECTION_FEATURE_STORE).find({}, {"_id": 0}))
    df = pd.DataFrame(docs)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Keep only the raw input columns (drop any previously computed features)
    cols_present = [c for c in RAW_COLS if c in df.columns]
    df = df[cols_present].copy()

    # Reindex to a complete calendar and forward-fill gaps
    full_idx = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = (
        df.set_index("date")
        .reindex(full_idx)
        .ffill()
        .reset_index()
        .rename(columns={"index": "date"})
    )
    return df


# ---------------------------------------------------------------------------
# IQR capping
# ---------------------------------------------------------------------------

def cap_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip outliers in IQR_COLS using the 1.5*IQR rule.
    Bounds are computed on ALL rows in df so they are consistent across runs.
    """
    for col in IQR_COLS:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return df


# ---------------------------------------------------------------------------
# Log transforms
# ---------------------------------------------------------------------------

def add_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Add log1p transforms; original columns are kept unchanged."""
    df["log_PM2_5"] = np.log1p(df["PM2_5"])
    df["log_CO"] = np.log1p(df["CO"])
    return df


# ---------------------------------------------------------------------------
# Wind features
# ---------------------------------------------------------------------------

def add_wind_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode wind_direction as sine and cosine components.
    wind_sin / wind_cos are kept as separate columns alongside the
    wind_dir_sin / wind_dir_cos added later in add_derived_weather —
    both pairs contain identical values, matching the schema in MongoDB.
    wind_direction is NOT dropped.
    """
    if "wind_direction" in df.columns:
        wd_rad = np.deg2rad(df["wind_direction"])
        df["wind_sin"] = np.sin(wd_rad)
        df["wind_cos"] = np.cos(wd_rad)
    return df


# ---------------------------------------------------------------------------
# Temporal / calendar features
# ---------------------------------------------------------------------------

def add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add month (numeric), season dummies (drop_first=True → Autumn is baseline),
    and weekday dummies (drop_first=True → weekday_0/Monday is baseline).
    """
    df["month"] = df["date"].dt.month
    df["season"] = df["date"].dt.month.map({
        12: "Winter",  1: "Winter",  2: "Winter",
         3: "Spring",  4: "Spring",  5: "Spring",
         6: "Summer",  7: "Summer",  8: "Summer",
         9: "Autumn", 10: "Autumn", 11: "Autumn",
    })
    df["weekday"] = df["date"].dt.weekday

    season_dummies = pd.get_dummies(df["season"], prefix="season", drop_first=True)
    weekday_dummies = pd.get_dummies(df["weekday"], prefix="weekday", drop_first=True)

    df = pd.concat(
        [df.drop(columns=["season", "weekday"]), season_dummies, weekday_dummies],
        axis=1,
    )
    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sine/cosine encodings for month, day-of-year, and day-of-week."""
    month = df["date"].dt.month
    doy = df["date"].dt.day_of_year
    dow = df["date"].dt.dayofweek

    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    df["weekday_sin"] = np.sin(2 * np.pi * dow / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * dow / 7)
    return df


# ---------------------------------------------------------------------------
# Lag / rolling features (shorter windows, tier-1)
# ---------------------------------------------------------------------------

def add_lag_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Short-window lag and rolling statistics for AQI and supporting variables.
    All rolling windows are applied to AQI.shift(1) so no future data leaks in.
    """
    aqi_s = df["AQI"].shift(1)

    df["AQI_lag_1"] = aqi_s
    df["AQI_lag_2"] = df["AQI"].shift(2)

    df["AQI_roll_mean_3"] = aqi_s.rolling(3).mean()
    df["AQI_roll_std_3"] = aqi_s.rolling(3).std()
    df["AQI_roll_min_3"] = aqi_s.rolling(3).min()
    df["AQI_roll_max_3"] = aqi_s.rolling(3).max()

    df["AQI_diff"] = aqi_s.diff()

    # Rolling stats on other columns (shift(1) for leak-free computation)
    df["Temperature_roll_mean_7"] = df["Temperature"].shift(1).rolling(7).mean()
    df["Humidity_roll_mean_7"] = df["Humidity"].shift(1).rolling(7).mean()
    df["log_PM2_5_lag_1"] = df["log_PM2_5"].shift(1)
    df["PM10_lag_1"] = df["PM10"].shift(1)
    df["wind_speed_lag_1"] = df["wind_speed"].shift(1)

    return df


# ---------------------------------------------------------------------------
# Extended lag features (longer windows, tier-2 lags)
# ---------------------------------------------------------------------------

def add_extended_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Longer AQI lags and pollutant lags.
    PM2_5_lag_1 must be computed here so add_derived_weather can use it
    for the wind_x_PM2_5_lag1 interaction term.
    """
    df["AQI_lag_3"] = df["AQI"].shift(3)
    df["AQI_lag_7"] = df["AQI"].shift(7)
    df["AQI_lag_14"] = df["AQI"].shift(14)

    df["PM2_5_lag_1"] = df["PM2_5"].shift(1)
    df["PM2_5_lag_2"] = df["PM2_5"].shift(2)
    df["PM2_5_lag_7"] = df["PM2_5"].shift(7)
    df["CO_lag_1"] = df["CO"].shift(1)
    df["NO2_lag_1"] = df["NO2"].shift(1)

    return df


# ---------------------------------------------------------------------------
# Extended rolling features (medium-window rolling stats)
# ---------------------------------------------------------------------------

def add_rolling_extended(df: pd.DataFrame) -> pd.DataFrame:
    """
    7-day and 14-day rolling statistics for AQI and PM2_5.
    All windows are applied to shift(1) series for leak-free computation.
    AQI_diff_2 is the second difference of the lagged AQI series.
    """
    aqi_s = df["AQI"].shift(1)
    pm25_s = df["PM2_5"].shift(1)

    df["AQI_roll_mean_7"] = aqi_s.rolling(7).mean()
    df["AQI_roll_std_7"] = aqi_s.rolling(7).std()
    df["AQI_roll_max_7"] = aqi_s.rolling(7).max()
    df["AQI_roll_min_7"] = aqi_s.rolling(7).min()

    df["AQI_roll_mean_14"] = aqi_s.rolling(14).mean()
    df["AQI_roll_std_14"] = aqi_s.rolling(14).std()
    df["AQI_roll_max_14"] = aqi_s.rolling(14).max()

    df["AQI_ewm_7"] = aqi_s.ewm(span=7, adjust=False).mean()
    df["AQI_ewm_14"] = aqi_s.ewm(span=14, adjust=False).mean()
    df["AQI_ewm_30"] = aqi_s.ewm(span=30, adjust=False).mean()

    df["PM2_5_roll_mean_7"] = pm25_s.rolling(7).mean()
    df["PM2_5_ewm_7"] = pm25_s.ewm(span=7, adjust=False).mean()

    df["AQI_diff_2"] = aqi_s.diff().diff()

    return df


# ---------------------------------------------------------------------------
# Derived weather / meteorological features
# ---------------------------------------------------------------------------

def add_derived_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Physics-inspired and threshold-based weather features.

    MUST be called after add_extended_lags so that PM2_5_lag_1 exists.

    stagnant_air uses df['wind_speed'].median() — the whole-dataset median
    at pipeline runtime.  This value shifts slightly as new data is added,
    which is expected behaviour.
    """
    df["dew_point"] = df["Temperature"] - ((100 - df["Humidity"]) / 5.0)
    df["temp_inversion"] = ((df["Temperature"] < 20) & (df["Humidity"] < 50)).astype(float)
    df["AQI_high_flag"] = (df["AQI"] > 150).astype(float)

    if "wind_speed" in df.columns and "wind_direction" in df.columns:
        wd_rad = np.deg2rad(df["wind_direction"])
        df["wind_dir_sin"] = np.sin(wd_rad)
        df["wind_dir_cos"] = np.cos(wd_rad)
        df["stagnant_air"] = (df["wind_speed"] < df["wind_speed"].median()).astype(float)

        if "PM2_5_lag_1" in df.columns:
            df["wind_x_PM2_5_lag1"] = df["wind_speed"] * df["PM2_5_lag_1"]

    return df


# ---------------------------------------------------------------------------
# Interaction features
# ---------------------------------------------------------------------------

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multiplicative interaction terms between pollutant and weather variables.
    AQI_x_month_sin requires month_sin from add_cyclical_features.
    """
    df["PM2_5_x_Humidity"] = df["PM2_5"] * df["Humidity"]
    df["CO_x_Temperature"] = df["CO"] * df["Temperature"]

    if "month_sin" in df.columns:
        df["AQI_x_month_sin"] = df["AQI"] * df["month_sin"]
    else:
        df["AQI_x_month_sin"] = df["AQI"]

    return df


# ---------------------------------------------------------------------------
# Tier-2 features (trend slope, cross-correlations)
# ---------------------------------------------------------------------------

def add_tier2_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    7-day linear slope of AQI, AQI × wind_speed interaction,
    and lag-3 features for NO2 and Humidity (cross-correlation peaks).
    """
    aqi_lagged = df["AQI"].shift(1)
    df["AQI_trend_7d"] = aqi_lagged.rolling(7).apply(
        lambda arr: np.polyfit(np.arange(7), arr, 1)[0], raw=True
    )

    if "wind_speed" in df.columns:
        df["AQI_x_wind"] = df["AQI"] * df["wind_speed"]

    df["NO2_lag_3"] = df["NO2"].shift(3)
    df["Humidity_lag_3"] = df["Humidity"].shift(3)

    return df


# ---------------------------------------------------------------------------
# Weather lead features
# ---------------------------------------------------------------------------

def add_weather_leads(df: pd.DataFrame) -> pd.DataFrame:
    """
    Future weather values (t+1, t+2, t+3) for temperature, humidity,
    precipitation, wind speed, and wind direction (as sin/cos components).
    These represent actual future observations used in training; at inference
    time they are supplied from a weather forecast API.
    """
    for col in ["Temperature", "Humidity", "Precipitation", "wind_speed"]:
        if col in df.columns:
            for lag in [1, 2, 3]:
                df[f"{col}_t{lag}"] = df[col].shift(-lag)

    if "wind_direction" in df.columns:
        for lag in [1, 2, 3]:
            wd_lead = df["wind_direction"].shift(-lag)
            wd_rad = np.deg2rad(wd_lead)
            df[f"wind_dir_sin_t{lag}"] = np.sin(wd_rad)
            df[f"wind_dir_cos_t{lag}"] = np.cos(wd_rad)

    return df


# ---------------------------------------------------------------------------
# Tier-3 features (surface_pressure, apparent_temp, wind_gusts)
# ---------------------------------------------------------------------------

def add_tier3_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lead windows (t+1/t+2/t+3), lag-1, and 7-day rolling mean for the three
    Open-Meteo tier-3 variables: surface_pressure, apparent_temp, wind_gusts.
    """
    tier3_vars = ["surface_pressure", "apparent_temp", "wind_gusts"]
    for col in tier3_vars:
        if col not in df.columns:
            continue
        df[f"{col}_t1"] = df[col].shift(-1)
        df[f"{col}_t2"] = df[col].shift(-2)
        df[f"{col}_t3"] = df[col].shift(-3)
        df[f"{col}_lag_1"] = df[col].shift(1)
        df[f"{col}_roll_mean_7"] = df[col].shift(1).rolling(7).mean()
    return df


# ---------------------------------------------------------------------------
# Target variables
# ---------------------------------------------------------------------------

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Next-1/2/3-day AQI targets for supervised learning."""
    df["AQI_t+1"] = df["AQI"].shift(-1)
    df["AQI_t+2"] = df["AQI"].shift(-2)
    df["AQI_t+3"] = df["AQI"].shift(-3)
    return df


# ---------------------------------------------------------------------------
# Forecast-fill for last 3 rows
# ---------------------------------------------------------------------------

def _fetch_daily_forecast() -> dict[str, dict]:
    """
    Fetch next 4 days of hourly weather from Open-Meteo forecast API and
    average to daily. Returns {date_str: {col: value}} for forecast dates.
    """
    try:
        resp = requests.get(
            _FORECAST_URL,
            params={
                "latitude": LAT, "longitude": LON, "timezone": TIMEZONE,
                "hourly": _FORECAST_HOURLY,
                "forecast_days": 4,
            },
            timeout=15,
        ).json()
        hourly = resp.get("hourly", {})
        df = pd.DataFrame(hourly)
        df["time"] = pd.to_datetime(df["time"])
        df["date"] = df["time"].dt.date
        df = df.drop(columns=["time"])
        df = df.rename(columns=_FORECAST_COL_MAP)
        daily = df.groupby("date").mean()
        return {d.isoformat(): row.to_dict() for d, row in daily.iterrows()}
    except Exception as e:
        print(f"  forecast fetch failed: {e}")
        return {}


def fill_lead_features_with_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """
    For the last 3 rows (which have NaN in lead/target columns from shift(-N)),
    fill lead weather features using Open-Meteo forecast data.
    Targets (AQI_t+1/+2/+3) remain NaN — they are unknowable today.
    """
    lead_nan_mask = df[_LEAD_COLS[0]].isna() if _LEAD_COLS[0] in df.columns else pd.Series(False, index=df.index)
    if not lead_nan_mask.any():
        return df

    forecast = _fetch_daily_forecast()
    if not forecast:
        return df

    df = df.copy()
    for idx in df.index[lead_nan_mask]:
        anchor = df.at[idx, "date"]
        anchor_date = anchor.date() if hasattr(anchor, "date") else pd.Timestamp(anchor).date()
        for offset in [1, 2, 3]:
            fdate = (pd.Timestamp(anchor_date) + pd.Timedelta(days=offset)).date().isoformat()
            if fdate not in forecast:
                continue
            fw = forecast[fdate]
            for col in ["Temperature", "Humidity", "Precipitation", "wind_speed"]:
                if col in fw:
                    df.at[idx, f"{col}_t{offset}"] = fw[col]
            if "wind_direction" in fw:
                rad = np.deg2rad(fw["wind_direction"])
                df.at[idx, f"wind_dir_sin_t{offset}"] = np.sin(rad)
                df.at[idx, f"wind_dir_cos_t{offset}"] = np.cos(rad)
            for col in ["surface_pressure", "apparent_temp", "wind_gusts"]:
                if col in fw:
                    df.at[idx, f"{col}_t{offset}"] = fw[col]
    return df


# ---------------------------------------------------------------------------
# MongoDB upsert helpers
# ---------------------------------------------------------------------------

def _to_python(value):
    """
    Convert numpy scalars and pandas booleans to native Python types
    for MongoDB BSON compatibility.
    """
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return float(value)
    if isinstance(value, bool):
        return float(value)
    return value


def _sanitise_record(record: dict) -> dict:
    out = {}
    for k, v in record.items():
        converted = _to_python(v)
        # Don't store NaN in MongoDB — omit the field instead so $exists checks
        # still return False for genuinely missing values (e.g. targets on last 3 rows).
        if isinstance(converted, float) and np.isnan(converted):
            continue
        out[k] = converted
    return out


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run() -> None:
    # 1. Load raw data (calendar-complete, forward-filled)
    df = load_raw()

    # 2. Outlier capping
    df = cap_iqr(df)

    # 3. Log transforms (keeps originals)
    df = add_log_transforms(df)

    # 4. Wind sin/cos (adds wind_sin, wind_cos; does NOT drop wind_direction)
    df = add_wind_features(df)

    # 5. Temporal dummies (month, season_*, weekday_*)
    df = add_temporal(df)

    # 6. Cyclical encodings (month_sin/cos, doy_sin/cos, weekday_sin/cos)
    df = add_cyclical_features(df)

    # 7. Short-window lag/rolling (AQI_lag_1/2, roll_mean/std/min/max_3, diff;
    #    Temperature_roll_mean_7, Humidity_roll_mean_7, log_PM2_5_lag_1,
    #    PM10_lag_1, wind_speed_lag_1)
    df = add_lag_rolling(df)

    # 8. Extended lags (AQI_lag_3/7/14, PM2_5_lag_1/2/7, CO_lag_1, NO2_lag_1)
    #    PM2_5_lag_1 must exist before add_derived_weather
    df = add_extended_lags(df)

    # 9. Extended rolling (AQI roll 7/14, ewm 7/14/30, PM2_5 roll, AQI_diff_2)
    df = add_rolling_extended(df)

    # 10. Derived weather (dew_point, temp_inversion, AQI_high_flag, stagnant_air,
    #     wind_dir_sin, wind_dir_cos, wind_x_PM2_5_lag1)
    #     MUST follow add_extended_lags so PM2_5_lag_1 exists
    df = add_derived_weather(df)

    # 11. Interaction features (PM2_5_x_Humidity, CO_x_Temperature, AQI_x_month_sin)
    df = add_interaction_features(df)

    # 12. Tier-2 (AQI_trend_7d, AQI_x_wind, NO2_lag_3, Humidity_lag_3)
    df = add_tier2_features(df)

    # 13. Weather leads (Temperature/Humidity/Precipitation/wind_speed _t1/t2/t3,
    #     wind_dir_sin/cos_t1/t2/t3)
    df = add_weather_leads(df)

    # 14. Tier-3 leads/lags (surface_pressure, apparent_temp, wind_gusts)
    df = add_tier3_features(df)

    # 15. Targets
    df = add_targets(df)

    # For the last 3 rows, lead features are NaN because no future observations
    # exist yet. Fill them from the Open-Meteo forecast API so inference is
    # always anchored to the most recent day (targets remain NaN intentionally).
    df = fill_lead_features_with_forecast(df)

    # Drop rows where core (non-lead, non-target) features are NaN.
    # Lead cols and target cols are intentionally excluded from this check:
    # lead cols are now forecast-filled; target cols are NaN for the last 3
    # rows by design (we don't know future AQI) and that's fine for inference.
    _skip = set(_LEAD_COLS + _TARGET_COLS + ["date", "processed_at"])
    core_cols = [c for c in df.columns if c not in _skip]
    df = df.dropna(subset=core_cols).reset_index(drop=True)

    # Upsert all processed rows to MongoDB, keyed on date string
    col = get_collection(COLLECTION_FEATURE_STORE)
    upserted = 0
    for record in df.to_dict("records"):
        record["date"] = record["date"].date().isoformat()
        record = _sanitise_record(record)
        col.update_one({"date": record["date"]}, {"$set": record}, upsert=True)
        upserted += 1

    print(f"Preprocessed and upserted {upserted} rows.")


if __name__ == "__main__":
    run()
