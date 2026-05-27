import matplotlib
matplotlib.use('Agg')
import sys; sys.path.append('.')
from dotenv import load_dotenv; load_dotenv()
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from config.db import get_collection, COLLECTION_FEATURE_STORE

# Daily data
docs = list(get_collection(COLLECTION_FEATURE_STORE).find({}, {'_id':0}))
df = pd.DataFrame(docs)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').drop_duplicates('date').reset_index(drop=True)
SEASON_MAP = {12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',
              6:'Summer',7:'Summer',8:'Summer',9:'Autumn',10:'Autumn',11:'Autumn'}
df['season'] = df['date'].dt.month.map(SEASON_MAP)

# Seasonal statistics
rows = []
for season in ['Spring','Summer','Autumn','Winter']:
    s = df[df['season']==season]['AQI'].dropna()
    rolling_std = df[df['season']==season]['AQI'].rolling(7).std().dropna()
    rows.append({'season':season,'mean_aqi':s.mean(),'std_aqi':s.std(),
                 'pct_above_100':(s>100).mean()*100,'pct_above_150':(s>150).mean()*100,
                 'skewness':s.skew(),'predictability_proxy_std':rolling_std.mean(),
                 'n_days':len(s)})
stats_df = pd.DataFrame(rows)
print("=== SEASONAL STATISTICS ===")
print(stats_df.to_string(index=False))
stats_df.to_csv('analysis/tables/seasonal_stats.csv', index=False)

# Summer wind speed analysis
print("\n=== SUMMER vs OTHER WIND SPEED ===")
if 'wind_speed' in df.columns:
    summer_wind = df[df['season']=='Summer']['wind_speed'].dropna()
    other_wind  = df[df['season']!='Summer']['wind_speed'].dropna()
    t_stat, p_val = stats.ttest_ind(summer_wind, other_wind)
    print(f"Summer wind_speed: mean={summer_wind.mean():.2f}, std={summer_wind.std():.2f}")
    print(f"Other seasons:     mean={other_wind.mean():.2f}, std={other_wind.std():.2f}")
    print(f"t-test p-value: {p_val:.4f}")

# Violin plot AQI by season
fig, ax = plt.subplots(figsize=(10, 6))
SEASON_ORDER = ['Spring','Summer','Autumn','Winter']
SEASON_PAL = {'Spring':'#4CAF50','Summer':'#F44336','Autumn':'#FF9800','Winter':'#2196F3'}
df_violin = df[df['season'].isin(SEASON_ORDER)][['season','AQI']].dropna()
sns.violinplot(data=df_violin, x='season', y='AQI', order=SEASON_ORDER, palette=SEASON_PAL, ax=ax)
ax.axhline(100, color='orange', linestyle='--', lw=1.5, label='Unhealthy threshold')
ax.axhline(150, color='red', linestyle='--', lw=1.5, label='Very Unhealthy threshold')
ax.set_title('AQI Distribution by Season'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('analysis/plots/eda_aqi_by_season.png', dpi=150, bbox_inches='tight'); plt.close()

# AQI by hour of day from hourly CSV
df_h = pd.read_csv('data/hourly_features.csv', parse_dates=['time'])
df_h['hour'] = df_h['time'].dt.hour
fig, ax = plt.subplots(figsize=(14, 6))
df_h.boxplot(column='AQI', by='hour', ax=ax, grid=False, flierprops=dict(markersize=2))
ax.set_title('AQI by Hour of Day (diurnal pattern)')
ax.set_xlabel('Hour (0=midnight)'); ax.set_ylabel('AQI')
plt.suptitle('')
# Rush hours highlighted
for rush_h in [7,8,9,17,18,19]:
    ax.axvline(rush_h+0.5, color='red', alpha=0.3, lw=2)
plt.tight_layout()
plt.savefig('analysis/plots/eda_aqi_by_hour.png', dpi=150, bbox_inches='tight'); plt.close()

# Rush hour t-test
rush_aqi = df_h[df_h['hour'].isin([7,8,9,17,18,19])]['AQI'].dropna()
other_aqi = df_h[~df_h['hour'].isin([7,8,9,17,18,19])]['AQI'].dropna()
t_stat_rh, p_rh = stats.ttest_ind(rush_aqi, other_aqi)
print(f"\n=== RUSH HOUR ANALYSIS ===")
print(f"Rush hours AQI:    mean={rush_aqi.mean():.2f}, std={rush_aqi.std():.2f}")
print(f"Non-rush AQI:      mean={other_aqi.mean():.2f}, std={other_aqi.std():.2f}")
print(f"t-test p-value: {p_rh:.4f}")
print("\nSaved seasonal, hourly plots")
