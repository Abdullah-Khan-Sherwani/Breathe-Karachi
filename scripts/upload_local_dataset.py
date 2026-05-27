"""
Upload the locally generated raw dataset to MongoDB feature_store.

Run this AFTER MongoDB Atlas IP access is restored and after
scripts/generate_local_dataset.py has been run.

This script:
  1. Reads data/local_raw_dataset.csv (22-column raw dataset 2023-2026)
  2. Upserts all rows into feature_store (adds BLH, cloud_cover, shortwave_rad,
     uv_index, aod, dust to every existing row, and inserts any missing rows)
  3. Prints a summary of what was upserted

After this script, run:
  python src/preprocess_daily_data.py   (regenerates all engineered features)
  python src/train.py                   (retrain models)
  python src/predict.py                 (generate fresh 4-day forecast)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from config.db import get_collection, COLLECTION_FEATURE_STORE


def _to_python(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return float(value)
    return value


def _sanitise(record: dict) -> dict:
    out = {}
    for k, v in record.items():
        converted = _to_python(v)
        if isinstance(converted, float) and np.isnan(converted):
            continue
        out[k] = converted
    return out


def run():
    csv_path = Path(__file__).parent.parent / "data" / "local_raw_dataset.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Run scripts/generate_local_dataset.py first: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded local_raw_dataset.csv: {len(df)} rows, {len(df.columns)} columns")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    col = get_collection(COLLECTION_FEATURE_STORE)
    upserted = 0
    for record in df.to_dict("records"):
        record = _sanitise(record)
        col.update_one({"date": record["date"]}, {"$set": record}, upsert=True)
        upserted += 1

    print(f"\nUpserted {upserted} rows to feature_store.")
    print("Next steps:")
    print("  python src/preprocess_daily_data.py")
    print("  python src/train.py")
    print("  python src/predict.py")


if __name__ == "__main__":
    run()
