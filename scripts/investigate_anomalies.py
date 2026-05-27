"""
Investigate the flagged anomalous dates in non-capped columns.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from config.db import get_collection, COLLECTION_FEATURE_STORE

def run():
    # The 3 non-capped columns with diff > 5
    # AQI: 2024-12-29 (diff=17.21)
    # CO: 2024-09-05 (diff=47.25), 2024-12-29 (diff=45.29)
    # wind_direction: 2024-12-29 (diff=87.54) — and max > 50

    dates_of_interest = ["2024-09-05", "2024-12-29"]

    col = get_collection(COLLECTION_FEATURE_STORE)

    csv_path = Path(__file__).parent.parent / "data" / "local_raw_dataset.csv"
    csv_df = pd.read_csv(csv_path)
    csv_df["date"] = pd.to_datetime(csv_df["date"]).dt.date.astype(str)

    print("Investigating non-capped column anomalies:")
    print("=" * 70)

    for d in dates_of_interest:
        mongo_doc = col.find_one({"date": d}, {"_id": 0})
        csv_row = csv_df[csv_df["date"] == d]

        print(f"\nDate: {d}")
        print(f"  {'Column':<20} {'MongoDB':>15} {'CSV':>15} {'Diff':>10}")
        print(f"  {'-'*60}")

        cols_to_check = ["AQI", "PM2_5", "CO", "wind_direction", "wind_speed",
                         "apparent_temp", "surface_pressure", "wind_gusts"]
        for c in cols_to_check:
            m_val = mongo_doc.get(c, "N/A") if mongo_doc else "N/A"
            if not csv_row.empty and c in csv_row.columns:
                csv_val = csv_row[c].values[0]
            else:
                csv_val = "N/A"
            try:
                diff = abs(float(m_val) - float(csv_val))
                print(f"  {c:<20} {float(m_val):>15.4f} {float(csv_val):>15.4f} {diff:>10.4f}")
            except:
                print(f"  {c:<20} {str(m_val):>15} {str(csv_val):>15}")

    # Check if CO diff is because MongoDB value is IQR-capped for CO
    print("\n\nNote: CO is NOT in IQR_COLS — differences could indicate:")
    print("  1. Different API fetch time (daily average vs different hour range)")
    print("  2. Data correction/update in Open-Meteo API since original fetch")
    print("\nLet's check CO distribution for context:")

    all_mongo = list(col.find(
        {"date": {"$gte": "2024-09-01", "$lte": "2024-09-10"}},
        {"_id": 0, "date": 1, "CO": 1}
    ))
    csv_sept = csv_df[(csv_df["date"] >= "2024-09-01") & (csv_df["date"] <= "2024-09-10")]

    print(f"\n  {'Date':<12} {'MongoDB CO':>12} {'CSV CO':>12} {'Diff':>10}")
    print(f"  {'-'*50}")
    csv_dict = dict(zip(csv_sept["date"], csv_sept["CO"]))
    for doc in sorted(all_mongo, key=lambda x: x["date"]):
        d = doc["date"]
        m_co = doc.get("CO", "N/A")
        csv_co = csv_dict.get(d, "N/A")
        try:
            diff = abs(float(m_co) - float(csv_co))
            print(f"  {d:<12} {float(m_co):>12.2f} {float(csv_co):>12.2f} {diff:>10.2f}")
        except:
            print(f"  {d:<12} {str(m_co):>12} {str(csv_co):>12}")

    print("\n\nChecking wind_direction for 2024-12-29 context:")
    all_mongo_dec = list(col.find(
        {"date": {"$gte": "2024-12-25", "$lte": "2025-01-03"}},
        {"_id": 0, "date": 1, "wind_direction": 1, "AQI": 1, "CO": 1}
    ))
    csv_dec = csv_df[(csv_df["date"] >= "2024-12-25") & (csv_df["date"] <= "2025-01-03")]
    csv_dec_dict = {row["date"]: row for _, row in csv_dec.iterrows()}

    print(f"\n  {'Date':<12} {'Mongo wind_dir':>14} {'CSV wind_dir':>14} {'Mongo AQI':>10} {'CSV AQI':>10} {'Mongo CO':>10} {'CSV CO':>10}")
    print(f"  {'-'*90}")
    for doc in sorted(all_mongo_dec, key=lambda x: x["date"]):
        d = doc["date"]
        csv_row_d = csv_dec_dict.get(d, {})
        m_wd = doc.get("wind_direction", "N/A")
        csv_wd = csv_row_d.get("wind_direction", "N/A") if hasattr(csv_row_d, 'get') else "N/A"
        m_aqi = doc.get("AQI", "N/A")
        csv_aqi = csv_row_d.get("AQI", "N/A") if hasattr(csv_row_d, 'get') else "N/A"
        m_co = doc.get("CO", "N/A")
        csv_co = csv_row_d.get("CO", "N/A") if hasattr(csv_row_d, 'get') else "N/A"
        try:
            print(f"  {d:<12} {float(m_wd):>14.2f} {float(csv_wd):>14.2f} {float(m_aqi):>10.2f} {float(csv_aqi):>10.2f} {float(m_co):>10.2f} {float(csv_co):>10.2f}")
        except:
            print(f"  {d:<12} {str(m_wd):>14} {str(csv_wd):>14} {str(m_aqi):>10} {str(csv_aqi):>10} {str(m_co):>10} {str(csv_co):>10}")

if __name__ == "__main__":
    run()
