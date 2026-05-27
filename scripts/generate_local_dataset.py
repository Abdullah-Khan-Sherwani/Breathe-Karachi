"""
Generate a complete local raw dataset (2023-01-01 → today-1) from Open-Meteo APIs,
save to data/local_raw_dataset.csv, and compare column-by-column with the MongoDB backup.

Run this when MongoDB is unreachable or to verify data integrity independently.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import requests
import numpy as np
from datetime import date, timedelta

LAT, LON, TIMEZONE = 24.8607, 67.0011, "Asia/Karachi"

AIR_URL     = "https://air-quality-api.open-meteo.com/v1/air-quality"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

AIR_PARAMS     = (
    "us_aqi,pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone,"
    "aerosol_optical_depth,dust,uv_index"
)
WEATHER_PARAMS = (
    "temperature_2m,relative_humidity_2m,precipitation,"
    "wind_speed_10m,wind_direction_10m,"
    "apparent_temperature,surface_pressure,wind_gusts_10m,"
    "boundary_layer_height,cloud_cover,shortwave_radiation"
)

COL_MAP = {
    "us_aqi": "AQI", "pm2_5": "PM2_5", "pm10": "PM10",
    "nitrogen_dioxide": "NO2", "sulphur_dioxide": "SO2",
    "carbon_monoxide": "CO", "ozone": "O3",
    "aerosol_optical_depth": "aod", "dust": "dust", "uv_index": "uv_index",
    "temperature_2m": "Temperature", "relative_humidity_2m": "Humidity",
    "precipitation": "Precipitation",
    "wind_speed_10m": "wind_speed", "wind_direction_10m": "wind_direction",
    "apparent_temperature": "apparent_temp",
    "surface_pressure": "surface_pressure",
    "wind_gusts_10m": "wind_gusts",
    "boundary_layer_height": "BLH",
    "cloud_cover": "cloud_cover",
    "shortwave_radiation": "shortwave_rad",
}

RAW_COL_ORDER = [
    "date",
    "AQI", "PM2_5", "PM10", "NO2", "SO2", "CO", "O3",
    "Temperature", "Humidity", "Precipitation",
    "wind_speed", "wind_direction",
    "apparent_temp", "surface_pressure", "wind_gusts",
    "BLH", "cloud_cover", "shortwave_rad",
    "uv_index", "aod", "dust",
]


def fetch_range_to_df(start: str, end: str) -> pd.DataFrame | None:
    base = {"latitude": LAT, "longitude": LON, "timezone": TIMEZONE,
            "start_date": start, "end_date": end}
    try:
        air  = requests.get(AIR_URL,     params={**base, "hourly": AIR_PARAMS},     timeout=30).json()["hourly"]
        wthr = requests.get(ARCHIVE_URL, params={**base, "hourly": WEATHER_PARAMS}, timeout=30).json()["hourly"]
    except Exception as e:
        print(f"  fetch error {start}–{end}: {e}")
        return None

    df = pd.merge(pd.DataFrame(air), pd.DataFrame(wthr), on="time")
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date.astype(str)
    df = df.drop(columns=["time"])

    # Daily average
    daily = df.groupby("date").mean().reset_index()
    daily = daily.rename(columns=COL_MAP)
    return daily


def generate_dataset(start: date, end: date) -> pd.DataFrame:
    """Fetch all 21 raw columns for the given date range in yearly chunks."""
    all_chunks = []
    current = start
    while current <= end:
        chunk_end = min(date(current.year, 12, 31), end)
        print(f"  Fetching {current} to {chunk_end}...", end=" ", flush=True)
        df = fetch_range_to_df(current.isoformat(), chunk_end.isoformat())
        if df is not None and not df.empty:
            all_chunks.append(df)
            print(f"{len(df)} rows")
        else:
            print("FAILED")
        current = date(current.year + 1, 1, 1)

    if not all_chunks:
        raise RuntimeError("No data fetched from Open-Meteo.")

    result = pd.concat(all_chunks, ignore_index=True)
    result = result.sort_values("date").reset_index(drop=True)

    # Reorder columns
    ordered_cols = [c for c in RAW_COL_ORDER if c in result.columns]
    remaining = [c for c in result.columns if c not in ordered_cols]
    result = result[ordered_cols + remaining]
    return result


def compare_with_backup(local_df: pd.DataFrame, backup_path: str) -> None:
    """Compare local generated dataset against the MongoDB backup CSV."""
    if not Path(backup_path).exists():
        print(f"\nBackup not found: {backup_path} — skipping comparison")
        return

    backup = pd.read_csv(backup_path)
    backup["date"] = backup["date"].astype(str)

    # Align on common dates
    common_dates = set(local_df["date"]) & set(backup["date"])
    print(f"\n=== Comparison Summary ===")
    print(f"  Local dataset rows:  {len(local_df)} (dates {local_df['date'].min()} to {local_df['date'].max()})")
    print(f"  Backup rows:         {len(backup)} (dates {backup['date'].min()} to {backup['date'].max()})")
    print(f"  Common dates:        {len(common_dates)}")

    # Dates only in local
    only_local = set(local_df["date"]) - set(backup["date"])
    only_backup = set(backup["date"]) - set(local_df["date"])
    if only_local:
        print(f"  Dates ONLY in local: {sorted(only_local)[:5]}{'...' if len(only_local)>5 else ''}")
    if only_backup:
        print(f"  Dates only in backup (pre-2023): {len(only_backup)} rows (expected)")

    # Columns comparison
    new_cols = [c for c in local_df.columns if c not in backup.columns]
    missing_cols = [c for c in backup.columns if c not in local_df.columns and c not in ("_id", "processed_at")]
    if new_cols:
        print(f"\n  NEW columns in local (not in backup): {new_cols}")
    if missing_cols:
        print(f"  MISSING in local (in backup but not local): {missing_cols}")

    # Numeric column alignment for common dates and shared columns
    shared_num_cols = [
        c for c in local_df.columns
        if c in backup.columns and c != "date"
        and pd.api.types.is_numeric_dtype(local_df[c])
    ]

    local_aligned  = local_df[local_df["date"].isin(common_dates)].set_index("date").sort_index()
    backup_aligned = backup[backup["date"].isin(common_dates)].set_index("date").sort_index()

    print(f"\n  Numeric column alignment ({len(shared_num_cols)} shared columns):")
    mismatches = []
    for col in shared_num_cols:
        a = local_aligned[col].dropna()
        b = backup_aligned[col].dropna()
        common_idx = a.index.intersection(b.index)
        if len(common_idx) == 0:
            continue
        diff = (a.loc[common_idx] - b.loc[common_idx]).abs()
        mean_diff = diff.mean()
        max_diff = diff.max()
        if mean_diff > 1.0:
            mismatches.append((col, mean_diff, max_diff))

    if mismatches:
        print(f"  Columns with mean absolute diff > 1.0 (possible schema change):")
        for col, mean_d, max_d in sorted(mismatches, key=lambda x: -x[1])[:10]:
            print(f"    {col:40s}  mean_diff={mean_d:.2f}  max_diff={max_d:.2f}")
    else:
        print(f"  All {len(shared_num_cols)} shared numeric columns match within tolerance.")

    print()


def run():
    start = date(2023, 1, 1)
    end   = date.today() - timedelta(days=1)  # yesterday (Open-Meteo archive lag)

    print(f"Generating local raw dataset: {start} to {end}")
    local_df = generate_dataset(start, end)

    out_path = Path(__file__).parent.parent / "data" / "local_raw_dataset.csv"
    out_path.parent.mkdir(exist_ok=True)
    local_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(local_df)} rows, {len(local_df.columns)} columns)")

    # Show null summary for new variables
    new_vars = ["BLH", "cloud_cover", "shortwave_rad", "uv_index", "aod", "dust"]
    print("\nNew variable coverage:")
    for v in new_vars:
        if v in local_df.columns:
            pct = local_df[v].notna().mean() * 100
            print(f"  {v}: {pct:.1f}% non-null")

    # Compare with backup
    backup_path = Path(__file__).parent.parent / "data" / "feature_store_backup_20260527.csv"
    compare_with_backup(local_df, str(backup_path))


if __name__ == "__main__":
    run()
