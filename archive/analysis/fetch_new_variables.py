import matplotlib
matplotlib.use('Agg')
import sys; sys.path.append('.')
import requests, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from datetime import date

LAT, LON, TZ = 24.8607, 67.0011, "Asia/Karachi"
START, END = "2023-01-01", "2024-12-31"

# Step 1: Try all candidate new weather variables from archive API
CANDIDATE_WEATHER = [
    "surface_pressure",
    "cloud_cover",
    "vapour_pressure_deficit",
    "et0_fao_evapotranspiration",
    "uv_index",
    "sunshine_duration",
    "wind_gusts_10m",
    "shortwave_radiation",
    "dew_point_2m",
    "apparent_temperature",
    "boundary_layer_height",  # may not exist — try it
    "soil_temperature_0_to_7cm",
]

# Fetch in batches to avoid URL length issues
def fetch_archive(variables, start, end):
    resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={"latitude": LAT, "longitude": LON, "timezone": TZ,
                "start_date": start, "end_date": end,
                "hourly": ",".join(variables)},
        timeout=90
    )
    if resp.status_code != 200:
        print(f"  Archive error {resp.status_code}: {resp.text[:200]}")
        return None
    data = resp.json()
    if "hourly" not in data:
        print(f"  No hourly in response")
        return None
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df

# Fetch in two batches
batch1 = CANDIDATE_WEATHER[:7]
batch2 = CANDIDATE_WEATHER[7:]
print(f"Fetching archive batch 1: {batch1}")
df_w1 = fetch_archive(batch1, START, END)
print(f"  Got vars: {list(df_w1.columns) if df_w1 is not None else 'FAILED'}")
print(f"Fetching archive batch 2: {batch2}")
df_w2 = fetch_archive(batch2, START, END)
print(f"  Got vars: {list(df_w2.columns) if df_w2 is not None else 'FAILED'}")

# Merge batches
if df_w1 is not None and df_w2 is not None:
    df_new = df_w1.merge(df_w2, on="time", how="outer")
elif df_w1 is not None:
    df_new = df_w1
else:
    df_new = df_w2

# Step 2: Try air quality new variables
CANDIDATE_AQ = ["boundary_layer_height", "dust", "european_aqi", "alder_pollen",
                 "ammonia", "nitrogen_monoxide", "pm10_wildfires"]
print(f"\nFetching air quality new vars: {CANDIDATE_AQ}")
resp_aq = requests.get(
    "https://air-quality-api.open-meteo.com/v1/air-quality",
    params={"latitude": LAT, "longitude": LON, "timezone": TZ,
            "start_date": START, "end_date": END,
            "hourly": ",".join(CANDIDATE_AQ)},
    timeout=90
)
print(f"  AQ status: {resp_aq.status_code}")
df_aq_new = None
if resp_aq.status_code == 200:
    aq_data = resp_aq.json()
    available_aq = list(aq_data.get("hourly", {}).keys())
    print(f"  Available AQ vars: {available_aq}")
    df_aq_new = pd.DataFrame(aq_data["hourly"])
    if "time" in df_aq_new.columns:
        df_aq_new["time"] = pd.to_datetime(df_aq_new["time"])

# Step 3: Aggregate to daily means
print("\nAggregating to daily means...")
def to_daily(df_hourly, time_col="time"):
    df_hourly = df_hourly.copy()
    df_hourly["date"] = df_hourly[time_col].dt.date
    numeric_cols = df_hourly.select_dtypes(include=np.number).columns.tolist()
    return df_hourly.groupby("date")[numeric_cols].mean().reset_index()

daily_new = to_daily(df_new) if df_new is not None else pd.DataFrame()
daily_aq  = to_daily(df_aq_new) if df_aq_new is not None else pd.DataFrame()

# Merge all new daily data
if not daily_new.empty and not daily_aq.empty:
    daily_all = daily_new.merge(daily_aq, on="date", how="outer")
elif not daily_new.empty:
    daily_all = daily_new
elif not daily_aq.empty:
    daily_all = daily_aq
else:
    print("ERROR: No new variable data fetched!")
    daily_all = pd.DataFrame()

# Step 4: Load existing AQI from hourly CSV (daily average)
df_h = pd.read_csv("data/hourly_features.csv", parse_dates=["time"])
df_h["date"] = df_h["time"].dt.date
daily_aqi = df_h.groupby("date").agg(AQI=("AQI","mean"), wind_speed=("wind_speed","mean"),
                                       Temperature=("Temperature","mean")).reset_index()
daily_aqi["date"] = pd.to_datetime(daily_aqi["date"])
# Create AQI_t3 (AQI 3 days in the future)
daily_aqi["AQI_t3"] = daily_aqi["AQI"].shift(-3)

if not daily_all.empty:
    daily_all["date"] = pd.to_datetime(daily_all["date"])
    merged = daily_aqi.merge(daily_all, on="date", how="inner")
    merged = merged.dropna(subset=["AQI","AQI_t3"])

    print("\n=== CORRELATION WITH AQI (lag 0) and AQI_t+3 ===")
    print(f"{'Variable':<35}  {'r with AQI':>12}  {'r with AQI_t+3':>15}  {'Worth Adding?':>14}")
    print("-" * 80)

    rows = []
    new_vars = [c for c in daily_all.columns if c != "date"]
    for var in new_vars:
        if var not in merged.columns: continue
        data = merged[[var, "AQI", "AQI_t3"]].dropna()
        if len(data) < 30: continue
        r0  = data[var].corr(data["AQI"])
        r3  = data[var].corr(data["AQI_t3"])
        worth = "YES" if abs(r3) > 0.15 else "no"
        print(f"  {var:<35}  {r0:>12.4f}  {r3:>15.4f}  {worth:>14}")
        rows.append({"variable": var, "correlation_day0": r0, "correlation_day3": r3,
                     "worth_adding": worth == "YES", "available": True})

    result_df = pd.DataFrame(rows).sort_values("correlation_day3", key=abs, ascending=False)
    result_df.to_csv("analysis/tables/new_variables_correlation.csv", index=False)

    # Highlight top variables for day3
    top_d3 = result_df[result_df["worth_adding"]]
    print(f"\nVariables with |r| > 0.15 with day3 AQI: {len(top_d3)}")
    if not top_d3.empty:
        print(top_d3[["variable","correlation_day3"]].to_string(index=False))

    # Step 5: Boundary layer height deep-dive (if available)
    blh_cols = [c for c in merged.columns if "boundary" in c.lower() or "blh" in c.lower()]
    if blh_cols:
        blh_col = blh_cols[0]
        print(f"\n=== BOUNDARY LAYER HEIGHT ANALYSIS ({blh_col}) ===")
        merged_blh = merged[[blh_col, "AQI", "AQI_t3"]].dropna()
        high_aqi = merged_blh[merged_blh["AQI"] > 150][blh_col]
        low_aqi  = merged_blh[merged_blh["AQI"] < 50][blh_col]
        print(f"High AQI days (>150) — mean BLH: {high_aqi.mean():.1f}m, std: {high_aqi.std():.1f}m  (n={len(high_aqi)})")
        print(f"Low AQI days (<50)   — mean BLH: {low_aqi.mean():.1f}m, std: {low_aqi.std():.1f}m  (n={len(low_aqi)})")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].scatter(merged_blh[blh_col], merged_blh["AQI"], alpha=0.4, s=20)
        axes[0].set_xlabel("Boundary Layer Height (m)"); axes[0].set_ylabel("AQI")
        axes[0].set_title(f"BLH vs AQI (r={merged_blh[blh_col].corr(merged_blh['AQI']):.3f})")
        axes[0].grid(alpha=0.3)
        axes[1].scatter(merged_blh[blh_col], merged_blh["AQI_t3"], alpha=0.4, s=20, color='tomato')
        axes[1].set_xlabel("Boundary Layer Height (m)"); axes[1].set_ylabel("AQI 3 days ahead")
        axes[1].set_title(f"BLH vs AQI_t+3 (r={merged_blh[blh_col].corr(merged_blh['AQI_t3']):.3f})")
        axes[1].grid(alpha=0.3)
        plt.suptitle("Boundary Layer Height Analysis — Karachi", fontsize=12)
        plt.tight_layout()
        plt.savefig("analysis/plots/boundary_layer_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved boundary_layer_analysis.png")
    else:
        print("\nBoundary layer height not available from Open-Meteo for Karachi")

    # Plot top new variables correlation chart
    if len(result_df) > 0:
        fig, ax = plt.subplots(figsize=(12, max(5, len(result_df)*0.5)))
        colors = ['green' if abs(r) > 0.15 else 'steelblue' for r in result_df["correlation_day3"]]
        ax.barh(result_df["variable"], result_df["correlation_day3"], color=colors, alpha=0.8)
        ax.axvline(0.15, color='red', linestyle='--', lw=1.5, label='|r|=0.15 threshold')
        ax.axvline(-0.15, color='red', linestyle='--', lw=1.5)
        ax.axvline(0, color='black', lw=0.5)
        ax.set_xlabel("Pearson r with AQI_t+3"); ax.set_title("New Variable Correlation with Day3 AQI")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("analysis/plots/new_variables_correlation.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved new_variables_correlation.png")
else:
    print("No data to analyze — all fetches failed")
    pd.DataFrame(columns=["variable","correlation_day0","correlation_day3","worth_adding","available"]
                ).to_csv("analysis/tables/new_variables_correlation.csv", index=False)
