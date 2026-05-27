"""
Autoregressive LGBM with weather forecast features.

Each stage feeds its prediction forward as an input to the next:
  Stage 1: base_features + weather_d1                   -> AQI_day1_mean
  Stage 2: base_features + weather_d2 + pred_day1       -> AQI_day2_mean
  Stage 3: base_features + weather_d3 + pred_day2       -> AQI_day3_mean

weather_d1/d2/d3 = actual future 24h window means used as a training proxy.
At inference time replace with Open-Meteo 3-day forecast API output.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor

from scripts.build_hourly_dataset import FEATURE_COLS, TARGET_COLS

DATA_PATH = Path(__file__).parent.parent / "data" / "hourly_features.csv"

N_TEST   = 720
N_BUFFER = 72

LGBM_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

# Actual future window means — proxy for 3-day weather forecast at training time.
# Precipitation excluded: near-zero variance in Karachi (~200mm/yr, 2-3 monsoon
# events) and poor forecast reliability during monsoon make it trivial/noisy.
FORECAST_COLS = [
    "temp_d1",     "temp_d2",     "temp_d3",
    "humidity_d1", "humidity_d2", "humidity_d3",
    "wind_d1",     "wind_d2",     "wind_d3",
]


def future_mean(series: pd.Series, offset: int, window: int) -> pd.Series:
    rev    = series.iloc[::-1].reset_index(drop=True)
    rolled = rev.rolling(window, min_periods=window).mean().shift(offset)
    return rolled.iloc[::-1].reset_index(drop=True)


def add_forecast_weather(df: pd.DataFrame) -> pd.DataFrame:
    for src, prefix in [
        ("Temperature", "temp"),
        ("Humidity",    "humidity"),
        ("wind_speed",  "wind"),
    ]:
        df[f"{prefix}_d1"] = future_mean(df[src], offset=1,  window=24).values
        df[f"{prefix}_d2"] = future_mean(df[src], offset=25, window=24).values
        df[f"{prefix}_d3"] = future_mean(df[src], offset=49, window=24).values
    return df


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "R2":   round(float(r2_score(y_true, y_pred)), 4),
        "MAE":  round(float(mean_absolute_error(y_true, y_pred)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
    }


def print_results(label: str, m1: dict, m2: dict, m3: dict) -> None:
    print(f"\n{label}")
    print(f"  {'Horizon':<12}  {'R2':>7}  {'MAE':>7}  {'RMSE':>7}")
    print("  " + "-" * 38)
    for name, m in [("day1 (t+24h)", m1), ("day2 (t+48h)", m2), ("day3 (t+72h)", m3)]:
        print(f"  {name:<12}  {m['R2']:>7.3f}  {m['MAE']:>7.1f}  {m['RMSE']:>7.1f}")


def load_and_prepare():
    df = pd.read_csv(DATA_PATH, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    feat = [c for c in FEATURE_COLS if c in df.columns]
    df = add_forecast_weather(df)

    all_needed = feat + FORECAST_COLS + TARGET_COLS
    df = df.dropna(subset=all_needed).reset_index(drop=True)
    n = len(df)

    train_df = df.iloc[: n - N_TEST - N_BUFFER]
    test_df  = df.iloc[n - N_TEST :]

    print(f"Dataset       : {n} rows  |  {len(feat)} base + 12 forecast weather features")
    print(f"Train rows    : {len(train_df)}  "
          f"({train_df['time'].iloc[0].date()} to {train_df['time'].iloc[-1].date()})")
    print(f"Test rows     : {len(test_df)}   "
          f"({test_df['time'].iloc[0].date()} to {test_df['time'].iloc[-1].date()})")

    return train_df, test_df, feat


def train_autoregressive(train_df, test_df, feat):
    print("\nTraining autoregressive LGBM ...")

    scaler = StandardScaler()
    X_base_tr = scaler.fit_transform(train_df[feat].values.astype(np.float32))
    X_base_te = scaler.transform(test_df[feat].values.astype(np.float32))

    y_train = train_df[TARGET_COLS].values.astype(np.float32)
    y_test  = test_df[TARGET_COLS].values.astype(np.float32)

    def wx(df: pd.DataFrame, day: int) -> np.ndarray:
        cols = [f"temp_d{day}", f"humidity_d{day}", f"precip_d{day}", f"wind_d{day}"]
        return df[cols].values.astype(np.float32)

    wx1_tr, wx1_te = wx(train_df, 1), wx(test_df, 1)
    wx2_tr, wx2_te = wx(train_df, 2), wx(test_df, 2)
    wx3_tr, wx3_te = wx(train_df, 3), wx(test_df, 3)

    # Stage 1
    X1_tr = np.hstack([X_base_tr, wx1_tr])
    X1_te = np.hstack([X_base_te, wx1_te])
    m1 = LGBMRegressor(**LGBM_PARAMS)
    m1.fit(X1_tr, y_train[:, 0])
    p1_tr = m1.predict(X1_tr).reshape(-1, 1)
    p1_te = m1.predict(X1_te).reshape(-1, 1)
    print("  day1 done")

    # Stage 2: previous predicted day1 + day2 weather
    X2_tr = np.hstack([X_base_tr, wx2_tr, p1_tr])
    X2_te = np.hstack([X_base_te, wx2_te, p1_te])
    m2 = LGBMRegressor(**LGBM_PARAMS)
    m2.fit(X2_tr, y_train[:, 1])
    p2_tr = m2.predict(X2_tr).reshape(-1, 1)
    p2_te = m2.predict(X2_te).reshape(-1, 1)
    print("  day2 done")

    # Stage 3: previous predicted day2 + day3 weather
    X3_tr = np.hstack([X_base_tr, wx3_tr, p2_tr])
    X3_te = np.hstack([X_base_te, wx3_te, p2_te])
    m3 = LGBMRegressor(**LGBM_PARAMS)
    m3.fit(X3_tr, y_train[:, 2])
    p3_te = m3.predict(X3_te)
    print("  day3 done")

    res1 = metrics(y_test[:, 0], p1_te.ravel())
    res2 = metrics(y_test[:, 1], p2_te.ravel())
    res3 = metrics(y_test[:, 2], p3_te)
    print_results("Autoregressive LGBM + weather forecast features (actual proxy)", res1, res2, res3)

    print("\n" + "=" * 70)
    print("COMPARISON (same 30-day holdout)")
    print(f"  {'Model':<42}  {'R2_d1':>7}  {'R2_d2':>7}  {'R2_d3':>7}")
    print("  " + "-" * 65)
    print(f"  {'Daily LGBM (baseline)':<42}  {0.773:>7.3f}  {0.284:>7.3f}  {0.120:>7.3f}")
    print(f"  {'Hourly LGBM (no forecast feats)':<42}  {0.875:>7.3f}  {0.335:>7.3f}  {-0.072:>7.3f}")
    print(f"  {'Hourly LGBM autoregressive (this run)':<42}  {res1['R2']:>7.3f}  {res2['R2']:>7.3f}  {res3['R2']:>7.3f}")
    print("=" * 70)
    print("Note: weather_dN = actual future 24h window mean (training proxy).")
    print("      At inference, use Open-Meteo 3-day forecast API values.")

    return (m1, m2, m3), scaler


if __name__ == "__main__":
    train_df, test_df, feat = load_and_prepare()
    train_autoregressive(train_df, test_df, feat)
