import matplotlib
matplotlib.use('Agg')
import sys; sys.path.append('.')
from dotenv import load_dotenv; load_dotenv()
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from config.db import get_collection, COLLECTION_FEATURE_STORE

docs = list(get_collection(COLLECTION_FEATURE_STORE).find({}, {'_id':0}))
df = pd.DataFrame(docs)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').drop_duplicates('date').reset_index(drop=True)

TARGET_COLS = ['AQI_t+1', 'AQI_t+2', 'AQI_t+3']
FEAT_WEATHER = ['PM10','NO2','SO2','O3','log_PM2_5','log_CO','Temperature','Humidity','Precipitation','wind_speed']

# Cross-correlation of features with FUTURE AQI (targets)
print("=== CROSS-CORRELATION WITH TARGETS ===")
rows = []
for feat in FEAT_WEATHER:
    if feat not in df.columns: continue
    for lag in range(0, 8):
        col_lagged = df[feat].shift(lag)  # feature lagged by 'lag' days (looking back)
        for tgt in TARGET_COLS:
            if tgt in df.columns:
                r = col_lagged.corr(df[tgt])
                rows.append({'feature':feat, 'lag':lag, 'target':tgt, 'r':r})
xcorr_df = pd.DataFrame(rows)
xcorr_df.to_csv('analysis/tables/xcorr_feature_lag.csv', index=False)

# Print correlations with day3 at lag 0
day3_corr = xcorr_df[(xcorr_df['target']=='AQI_t+3') & (xcorr_df['lag']==0)].sort_values('r', key=abs, ascending=False)
print("Correlation with AQI_t+3 (lag=0):")
print(day3_corr[['feature','r']].to_string(index=False))

# Heatmap: features vs targets at various lags
pivot = xcorr_df[xcorr_df['target']=='AQI_t+3'].pivot(index='feature', columns='lag', values='r')
fig, ax = plt.subplots(figsize=(14, 7))
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax, cbar_kws={'label':'Pearson r'})
ax.set_title('Cross-Correlation: Feature (at lag d) vs AQI_t+3')
ax.set_xlabel('Lag (days, 0=same day)'); ax.set_ylabel('Feature')
plt.tight_layout()
plt.savefig('analysis/plots/eda_xcorr_heatmap.png', dpi=150, bbox_inches='tight'); plt.close()

# Feature-feature correlation heatmap (LGBM 39 features)
FEAT_BASE = ['AQI','PM10','NO2','SO2','O3','log_PM2_5','log_CO','Temperature','Humidity',
             'Precipitation','wind_speed','wind_sin','wind_cos','wind_speed_lag_1',
             'AQI_lag_1','AQI_lag_2','AQI_lag_3','AQI_lag_7','AQI_roll_mean_3',
             'AQI_roll_std_3','AQI_roll_mean_7','AQI_roll_std_7','AQI_roll_min_3',
             'AQI_roll_max_3','AQI_diff','log_PM2_5_lag_1','PM10_lag_1',
             'Temperature_roll_mean_7','Humidity_roll_mean_7','month',
             'season_Spring','season_Summer','season_Winter',
             'weekday_1','weekday_2','weekday_3','weekday_4','weekday_5','weekday_6']

corr_cols = [c for c in FEAT_BASE + TARGET_COLS if c in df.columns]
corr_matrix = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(22, 18))
sns.heatmap(corr_matrix, cmap='RdBu_r', center=0, ax=ax, square=True,
            xticklabels=True, yticklabels=True, cbar_kws={'shrink':0.5},
            linewidths=0.1)
ax.set_title('Feature-Feature Correlation Matrix (LGBM 39 features + targets)', fontsize=12)
plt.xticks(fontsize=7, rotation=90); plt.yticks(fontsize=7)
plt.tight_layout()
plt.savefig('analysis/plots/eda_feature_corr_heatmap.png', dpi=150, bbox_inches='tight'); plt.close()
print("Saved cross-correlation and feature heatmap plots")
