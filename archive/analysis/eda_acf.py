import matplotlib
matplotlib.use('Agg')
import sys; sys.path.append('.')
from dotenv import load_dotenv; load_dotenv()
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf

# --- Daily ACF ---
from config.db import get_collection, COLLECTION_FEATURE_STORE
docs = list(get_collection(COLLECTION_FEATURE_STORE).find({'AQI': {'$exists': True}}, {'_id':0,'date':1,'AQI':1}))
df_d = pd.DataFrame(docs)
df_d['date'] = pd.to_datetime(df_d['date'])
df_d = df_d.sort_values('date').drop_duplicates('date').reset_index(drop=True)
aqi_daily = df_d['AQI'].dropna()

nlags = 30
acf_daily, confint_d = acf(aqi_daily, nlags=nlags, alpha=0.05)
pacf_daily, confint_pd = pacf(aqi_daily, nlags=nlags, alpha=0.05)
sig_thresh_d = 1.96 / np.sqrt(len(aqi_daily))

print("=== DAILY ACF VALUES ===")
for lag in range(1, nlags+1):
    sig = "*" if abs(acf_daily[lag]) > sig_thresh_d else ""
    print(f"  lag {lag:2d}: ACF={acf_daily[lag]:.4f} {sig}")

first_insig = next((i for i in range(1, nlags+1) if abs(acf_daily[i]) < sig_thresh_d), None)
print(f"\nSignificance threshold: {sig_thresh_d:.4f}")
print(f"First insignificant lag: {first_insig}")
print(f"ACF at lag 2 (48h): {acf_daily[2]:.4f}")
print(f"ACF at lag 3 (72h): {acf_daily[3]:.4f}")

# --- Hourly ACF ---
df_h = pd.read_csv('data/hourly_features.csv', parse_dates=['time'])
aqi_hourly = df_h['AQI'].dropna()
acf_hourly = acf(aqi_hourly, nlags=168, fft=True)
sig_thresh_h = 1.96 / np.sqrt(len(aqi_hourly))

print("\n=== HOURLY ACF AT KEY LAGS ===")
for lag in [1,2,3,6,12,24,48,72,168]:
    sig = "*" if abs(acf_hourly[lag]) > sig_thresh_h else ""
    print(f"  lag {lag:4d}h: ACF={acf_hourly[lag]:.4f} {sig}")

# --- Plot daily ACF/PACF ---
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
lags_d = np.arange(nlags+1)
axes[0].bar(lags_d, acf_daily, color='steelblue', alpha=0.7, label='ACF')
axes[0].axhline(sig_thresh_d, color='red', linestyle='--', label=f'95% CI (±{sig_thresh_d:.3f})')
axes[0].axhline(-sig_thresh_d, color='red', linestyle='--')
axes[0].axvline(2, color='orange', linestyle=':', lw=2, label='lag 2 (48h)')
axes[0].axvline(3, color='green', linestyle=':', lw=2, label='lag 3 (72h)')
axes[0].set_title('Daily AQI Autocorrelation Function (ACF)')
axes[0].set_xlabel('Lag (days)'); axes[0].set_ylabel('ACF'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].bar(lags_d, pacf_daily, color='tomato', alpha=0.7)
axes[1].axhline(sig_thresh_d, color='red', linestyle='--')
axes[1].axhline(-sig_thresh_d, color='red', linestyle='--')
axes[1].set_title('Daily AQI Partial Autocorrelation Function (PACF)')
axes[1].set_xlabel('Lag (days)'); axes[1].set_ylabel('PACF'); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('analysis/plots/acf_daily.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Plot hourly ACF ---
fig, ax = plt.subplots(figsize=(16, 5))
lags_h = np.arange(169)
ax.plot(lags_h, acf_hourly, color='steelblue', lw=1.5)
ax.axhline(sig_thresh_h, color='red', linestyle='--', alpha=0.7)
ax.axhline(-sig_thresh_h, color='red', linestyle='--', alpha=0.7)
for lag, label in [(24,'24h'),(48,'48h'),(72,'72h'),(168,'168h')]:
    ax.axvline(lag, color='orange', linestyle=':', lw=1.5, label=f'lag {label}')
ax.set_title('Hourly AQI ACF (up to 7 days)'); ax.set_xlabel('Lag (hours)'); ax.set_ylabel('ACF')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('analysis/plots/acf_hourly.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Save CSV ---
acf_df = pd.DataFrame({'lag_days': lags_d, 'acf_daily': acf_daily,
                       'pacf_daily': pacf_daily, 'sig_threshold': sig_thresh_d})
acf_df.to_csv('analysis/tables/acf_values.csv', index=False)
print("\nSaved: analysis/plots/acf_daily.png, analysis/plots/acf_hourly.png, analysis/tables/acf_values.csv")
