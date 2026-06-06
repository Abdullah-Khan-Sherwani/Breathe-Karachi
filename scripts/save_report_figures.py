"""
Generate and save EDA figures for the project report.
Reads from MongoDB feature_store (current data) and saves PNGs to report_figures/.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf

from config.db import get_collection, COLLECTION_FEATURE_STORE

OUT = Path(__file__).parent.parent / "report_figures"
OUT.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "axes.titlesize": 11, "axes.labelsize": 10})

BANDS = [
    (0,   50,  "Good",           "#00e400"),
    (51,  100, "Moderate",       "#d4d400"),
    (101, 150, "Unhealthy (SG)", "#ff7e00"),
    (151, 200, "Unhealthy",      "#ff0000"),
    (201, 300, "Very Unhealthy", "#8f3f97"),
]

# ── Load ─────────────────────────────────────────────────────────────────────
print("Loading feature_store …")
docs = list(get_collection(COLLECTION_FEATURE_STORE).find({}, {"_id": 0}))
df = pd.DataFrame(docs)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
dfa = df[df["AQI"].notna()].copy()
dfa["month"]  = dfa["date"].dt.month
SEASON = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
          6:"Summer",7:"Summer",8:"Summer",9:"Autumn",10:"Autumn",11:"Autumn"}
dfa["season"] = dfa["month"].map(SEASON)
aqi = dfa["AQI"]
print(f"  {len(dfa):,} days with AQI  ({dfa['date'].min().date()} -> {dfa['date'].max().date()})")

# ── Fig 1: AQI time series with 30-day rolling mean ──────────────────────────
print("Fig 1: time series …")
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.plot(dfa["date"], dfa["AQI"], color="#5b9bd5", lw=0.7, alpha=0.55, label="Daily AQI")
ax.plot(dfa["date"], dfa["AQI"].rolling(30, min_periods=1).mean(),
        color="#e8743b", lw=2, label="30-day rolling mean")
for lo, hi, lab, col in BANDS:
    ax.axhspan(lo, min(hi, dfa["AQI"].max() + 5), color=col, alpha=0.07)
ax.set_title("Karachi Daily US AQI (2022–2026)")
ax.set_ylabel("US AQI"); ax.set_xlabel("")
ax.legend(fontsize=9, loc="upper right")
plt.tight_layout()
fig.savefig(OUT / "fig_timeseries.png", bbox_inches="tight")
plt.close()

# ── Fig 2: Mean AQI by month (bar) ───────────────────────────────────────────
print("Fig 2: monthly mean …")
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
monthly = dfa.groupby("month")["AQI"].agg(["mean","std"]).reindex(range(1,13))
fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(range(1,13), monthly["mean"], color="#4c72b0",
              yerr=monthly["std"], capsize=3, error_kw={"elinewidth":0.8})
ax.set_xticks(range(1,13)); ax.set_xticklabels(MONTH_NAMES)
ax.set_ylabel("Mean US AQI"); ax.set_title("Mean AQI by Calendar Month (±1 SD)")
ax.axhline(100, color="#ff7e00", ls="--", lw=1, label="Unhealthy (SG) threshold")
ax.axhline(150, color="#ff0000", ls="--", lw=1, label="Unhealthy threshold")
ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(OUT / "fig_monthly_aqi.png", bbox_inches="tight")
plt.close()

# ── Fig 3: AQI distribution by season (violin) ───────────────────────────────
print("Fig 3: seasonal violin …")
season_order = ["Winter","Spring","Summer","Autumn"]
palette = {"Winter":"#5b9bd5","Spring":"#70ad47","Summer":"#ed7d31","Autumn":"#ffc000"}
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.violinplot(data=dfa, x="season", y="AQI", order=season_order,
               palette=palette, inner="box", ax=ax, cut=0)
ax.axhline(100, color="#ff7e00", ls="--", lw=1.2, label="Unhealthy (SG) — 100")
ax.axhline(150, color="#ff0000", ls="--", lw=1.2, label="Unhealthy — 150")
ax.set_title("AQI Distribution by Season"); ax.set_xlabel(""); ax.set_ylabel("US AQI")
ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(OUT / "fig_seasonal_violin.png", bbox_inches="tight")
plt.close()

# ── Fig 4: Correlation of observed variables with AQI ────────────────────────
print("Fig 4: correlation bar …")
obs_cols = ["PM2_5","PM10","NO2","SO2","CO","O3",
            "Temperature","Humidity","wind_speed","BLH",
            "aod","dust","Precipitation","shortwave_rad","cloud_cover"]
obs_cols = [c for c in obs_cols if c in dfa.columns]
corr = dfa[obs_cols + ["AQI"]].corr()["AQI"].drop("AQI").sort_values()
colors = ["#c0504d" if v > 0 else "#4c72b0" for v in corr]
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.barh(corr.index, corr.values, color=colors)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("Pearson r with same-day AQI")
ax.set_title("Feature Correlation with AQI (Observed Variables)")
plt.tight_layout()
fig.savefig(OUT / "fig_corr_aqi.png", bbox_inches="tight")
plt.close()

# ── Fig 5: ACF / PACF ────────────────────────────────────────────────────────
print("Fig 5: ACF/PACF …")
s = dfa.set_index("date")["AQI"].asfreq("D").interpolate("time")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
plot_acf(s.dropna(),  lags=21, ax=ax1); ax1.set_title("ACF — Daily AQI")
plot_pacf(s.dropna(), lags=21, ax=ax2, method="ywm"); ax2.set_title("PACF — Daily AQI")
plt.tight_layout()
fig.savefig(OUT / "fig_acf_pacf.png", bbox_inches="tight")
plt.close()

a = acf(s.dropna(), nlags=4)
print("  ACF by horizon:", {h: round(a[h], 3) for h in [1,2,3,4]})

print(f"\nAll figures saved to  {OUT}/")
