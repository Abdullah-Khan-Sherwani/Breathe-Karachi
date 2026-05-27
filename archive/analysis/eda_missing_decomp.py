import matplotlib
matplotlib.use('Agg')
import sys; sys.path.append('.')
from dotenv import load_dotenv; load_dotenv()
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from config.db import get_collection, COLLECTION_FEATURE_STORE

docs = list(get_collection(COLLECTION_FEATURE_STORE).find({}, {'_id':0}))
df = pd.DataFrame(docs)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').drop_duplicates('date').reset_index(drop=True)

# Missing data analysis
missing_rows = []
for col in df.columns:
    if col in ['date','processed_at','_id']: continue
    pct = df[col].isna().mean() * 100
    if pct > 0:
        missing_dates = df[df[col].isna()]['date']
        missing_rows.append({
            'feature': col, 'pct_missing': round(pct, 2),
            'n_missing': df[col].isna().sum(),
            'first_missing': missing_dates.min() if len(missing_dates)>0 else None,
            'last_missing': missing_dates.max() if len(missing_dates)>0 else None
        })
miss_df = pd.DataFrame(missing_rows).sort_values('pct_missing', ascending=False)
print("=== MISSING DATA ===")
print(miss_df.head(20).to_string(index=False))
miss_df.to_csv('analysis/tables/missing_data.csv', index=False)

# Seasonal decomposition of AQI
aqi_series = df.set_index('date')['AQI'].dropna()
# Ensure no major gaps — use only last 5 years for cleaner decomposition
aqi_recent = aqi_series[aqi_series.index >= '2020-01-01']
# Resample to ensure daily frequency
aqi_resampled = aqi_recent.resample('D').mean().interpolate()
try:
    result = seasonal_decompose(aqi_resampled, model='additive', period=365, extrapolate_trend='freq')
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    result.observed.plot(ax=axes[0], color='steelblue'); axes[0].set_title('Observed'); axes[0].set_ylabel('AQI')
    result.trend.plot(ax=axes[1], color='orange'); axes[1].set_title('Trend'); axes[1].set_ylabel('AQI')
    result.seasonal.plot(ax=axes[2], color='green'); axes[2].set_title('Seasonal'); axes[2].set_ylabel('AQI')
    result.resid.plot(ax=axes[3], color='red'); axes[3].set_title('Residual'); axes[3].set_ylabel('AQI')
    for ax in axes: ax.grid(alpha=0.3)
    plt.suptitle('AQI Seasonal Decomposition (2020-2026)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('analysis/plots/eda_seasonal_decomp.png', dpi=150, bbox_inches='tight'); plt.close()
    print("Saved seasonal decomposition plot")
except Exception as e:
    print(f"Decomp error: {e}")
print("Done.")
