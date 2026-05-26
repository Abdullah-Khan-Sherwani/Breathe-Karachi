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
SEASON_MAP = {12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',
              6:'Summer',7:'Summer',8:'Summer',9:'Autumn',10:'Autumn',11:'Autumn'}
df['season'] = df['date'].dt.month.map(SEASON_MAP)

# 1. AQI + pollutant distributions by season
SEASON_COLORS = {'Winter':'#2196F3','Spring':'#4CAF50','Summer':'#F44336','Autumn':'#FF9800'}
pollutants = ['AQI','PM10','NO2','SO2','O3','log_PM2_5','log_CO']
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for i, col in enumerate(pollutants):
    ax = axes[i//4][i%4]
    if col not in df.columns: continue
    for season, color in SEASON_COLORS.items():
        data = df.loc[df['season']==season, col].dropna()
        if len(data) > 5:
            data.plot.kde(ax=ax, color=color, label=season, lw=2)
    ax.set_title(col); ax.legend(fontsize=7); ax.grid(alpha=0.3)
axes[1][3].set_visible(False)
plt.suptitle('Pollutant Distributions by Season', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('analysis/plots/eda_distributions.png', dpi=150, bbox_inches='tight'); plt.close()

# 2. Full AQI time series
fig, ax = plt.subplots(figsize=(20, 6))
ax.plot(df['date'], df['AQI'], color='steelblue', lw=0.8, alpha=0.8)
ax.axhline(50, color='green', linestyle='--', lw=1, alpha=0.6, label='Good (50)')
ax.axhline(100, color='gold', linestyle='--', lw=1, alpha=0.6, label='Moderate (100)')
ax.axhline(150, color='orange', linestyle='--', lw=1, alpha=0.6, label='Unhealthy (150)')
ax.axhline(200, color='red', linestyle='--', lw=1, alpha=0.6, label='Very Unhealthy (200)')
# shade summers
for year in range(df['date'].dt.year.min(), df['date'].dt.year.max()+1):
    ax.axvspan(pd.Timestamp(f'{year}-06-01'), pd.Timestamp(f'{year}-08-31'), alpha=0.07, color='red')
ax.set_title('Karachi AQI 2018–2026 (red shading = Summer Jun-Aug)'); ax.legend(fontsize=8)
ax.set_xlabel('Date'); ax.set_ylabel('AQI'); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('analysis/plots/eda_timeseries.png', dpi=150, bbox_inches='tight'); plt.close()

# 3. AQI by month boxplot
df['month'] = df['date'].dt.month
fig, ax = plt.subplots(figsize=(14, 6))
df.boxplot(column='AQI', by='month', ax=ax, grid=False)
ax.set_title('AQI by Month'); ax.set_xlabel('Month'); ax.set_ylabel('AQI')
plt.suptitle('')
plt.tight_layout()
plt.savefig('analysis/plots/eda_aqi_by_month.png', dpi=150, bbox_inches='tight'); plt.close()

# 4. AQI by year boxplot
df['year'] = df['date'].dt.year
fig, ax = plt.subplots(figsize=(14, 6))
df.boxplot(column='AQI', by='year', ax=ax, grid=False)
ax.set_title('AQI by Year'); ax.set_xlabel('Year'); ax.set_ylabel('AQI')
plt.suptitle('')
plt.tight_layout()
plt.savefig('analysis/plots/eda_aqi_by_year.png', dpi=150, bbox_inches='tight'); plt.close()

print("Saved distributions, timeseries, by_month, by_year plots")
