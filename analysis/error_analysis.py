import matplotlib
matplotlib.use('Agg')
import sys; sys.path.append('.')
from dotenv import load_dotenv; load_dotenv()
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import acf
from sklearn.metrics import r2_score
from config.db import get_collection, COLLECTION_FEATURE_STORE, load_model

model_lgbm, scaler_lgbm, metadata = load_model('lgbm')
feat = metadata['features']
TARGET_COLS = ['AQI_t+1', 'AQI_t+2', 'AQI_t+3']
docs = list(get_collection(COLLECTION_FEATURE_STORE).find({}, {'_id': 0}))
df = pd.DataFrame(docs)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').drop_duplicates('date').reset_index(drop=True)
df_clean = df.dropna(subset=feat + TARGET_COLS).reset_index(drop=True)
split_date = df_clean['date'].max() - pd.Timedelta(days=30)
train_df = df_clean[df_clean['date'] <= split_date]
test_df  = df_clean[df_clean['date'] >  split_date].reset_index(drop=True)
X_train = scaler_lgbm.transform(train_df[feat].values)
X_test  = scaler_lgbm.transform(test_df[feat].values)
y_train = train_df[TARGET_COLS].values
y_test  = test_df[TARGET_COLS].values
preds_lgbm = model_lgbm.predict(X_test)

SEASON_MAP = {12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',
              6:'Summer',7:'Summer',8:'Summer',9:'Autumn',10:'Autumn',11:'Autumn'}
test_df['season'] = test_df['date'].dt.month.map(SEASON_MAP)
test_df['month']  = test_df['date'].dt.month

HORIZON_NAMES = ['day1 (t+1)', 'day2 (t+2)', 'day3 (t+3)']

# --- 1. Scatter plots: predicted vs actual ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
SEASON_COLORS = {'Spring':'#4CAF50','Summer':'#F44336','Autumn':'#FF9800','Winter':'#2196F3'}
for h, ax in enumerate(axes):
    actual = y_test[:, h]; pred = preds_lgbm[:, h]
    for season, color in SEASON_COLORS.items():
        mask = test_df['season'] == season
        ax.scatter(actual[mask], pred[mask], alpha=0.7, color=color, label=season, s=50)
    lo, hi = min(actual.min(), pred.min()), max(actual.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, label='Perfect')
    r2 = r2_score(actual, pred)
    ax.set_title(f'{HORIZON_NAMES[h]}\nR²={r2:.4f}')
    ax.set_xlabel('Actual AQI'); ax.set_ylabel('Predicted AQI')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
plt.suptitle('Daily LGBM: Predicted vs Actual by Horizon + Season', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('analysis/plots/error_scatter_all.png', dpi=150, bbox_inches='tight'); plt.close()

# --- 2. Residuals + Gaussian test ---
print("=== RESIDUAL DISTRIBUTION ===")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for h, ax in enumerate(axes):
    residuals = y_test[:, h] - preds_lgbm[:, h]
    ax.hist(residuals, bins=20, density=True, alpha=0.6, color='steelblue', label='Residuals')
    mu, std = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, std), 'r-', lw=2, label=f'Normal(μ={mu:.1f},σ={std:.1f})')
    shapiro_stat, shapiro_p = stats.shapiro(residuals[:50])  # Shapiro-Wilk needs ≤5000
    ax.set_title(f'{HORIZON_NAMES[h]}\nShapiro-Wilk p={shapiro_p:.4f}')
    ax.set_xlabel('Residual'); ax.set_ylabel('Density'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print(f"  {HORIZON_NAMES[h]}: mean={mu:.2f}, std={std:.2f}, Shapiro p={shapiro_p:.4f}")
    bias = "OVER-predicts" if mu < 0 else "UNDER-predicts" if mu > 0 else "unbiased"
    print(f"    Bias: {bias} (mean residual = {mu:.2f})")
plt.suptitle('Residual Distributions', fontsize=12); plt.tight_layout()
plt.savefig('analysis/plots/error_residuals.png', dpi=150, bbox_inches='tight'); plt.close()

# --- 3. Error by season and month ---
for h in range(3):
    test_df[f'abs_err_d{h+1}'] = np.abs(y_test[:, h] - preds_lgbm[:, h])

season_err = test_df.groupby('season')[[f'abs_err_d{h+1}' for h in range(3)]].mean().reset_index()
month_err  = test_df.groupby('month')[[f'abs_err_d{h+1}'  for h in range(3)]].mean().reset_index()
season_err.to_csv('analysis/tables/error_by_season.csv', index=False)
month_err.to_csv('analysis/tables/error_by_month.csv', index=False)

print("\n=== ERROR BY SEASON ===")
print(season_err.to_string(index=False))
print("\n=== ERROR BY MONTH ===")
print(month_err.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
x = np.arange(len(season_err)); w = 0.25
for i, h in enumerate(range(3)):
    axes[0].bar(x + i*w, season_err[f'abs_err_d{h+1}'], w, label=f'day{h+1}', alpha=0.8)
axes[0].set_xticks(x + w); axes[0].set_xticklabels(season_err['season'], rotation=15)
axes[0].set_title('MAE by Season'); axes[0].set_ylabel('MAE'); axes[0].legend(); axes[0].grid(alpha=0.3)
x_m = np.arange(len(month_err))
for i, h in enumerate(range(3)):
    axes[1].bar(x_m + i*w, month_err[f'abs_err_d{h+1}'], w, label=f'day{h+1}', alpha=0.8)
axes[1].set_xticks(x_m + w); axes[1].set_xticklabels(month_err['month'])
axes[1].set_title('MAE by Month'); axes[1].set_ylabel('MAE'); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('analysis/plots/error_by_season_month.png', dpi=150, bbox_inches='tight'); plt.close()

# --- 4. Error vs AQI level ---
bins = [0, 50, 100, 150, 999]; labels = ['Good(0-50)','Moderate(51-100)','Unhealthy(101-150)','VeryUnhealthy(151+)']
test_df['aqi_bin'] = pd.cut(test_df['AQI'], bins=bins, labels=labels, right=True)
aqi_level_err = test_df.groupby('aqi_bin')[[f'abs_err_d{h+1}' for h in range(3)]].mean().reset_index()
print("\n=== ERROR BY AQI LEVEL ===")
print(aqi_level_err.to_string(index=False))
aqi_level_err.to_csv('analysis/tables/error_by_aqi_level.csv', index=False)

fig, ax = plt.subplots(figsize=(10, 6))
x_aqi = np.arange(len(aqi_level_err))
for i, h in enumerate(range(3)):
    ax.bar(x_aqi + i*0.25, aqi_level_err[f'abs_err_d{h+1}'], 0.25, label=f'day{h+1}', alpha=0.8)
ax.set_xticks(x_aqi + 0.25); ax.set_xticklabels(aqi_level_err['aqi_bin'], rotation=15)
ax.set_title('MAE by AQI Level'); ax.set_ylabel('MAE'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('analysis/plots/error_by_aqi_level.png', dpi=150, bbox_inches='tight'); plt.close()

# --- 5. Residual ACF ---
print("\n=== RESIDUAL AUTOCORRELATION ===")
for h in range(3):
    resid = y_test[:, h] - preds_lgbm[:, h]
    if len(resid) > 5:
        acf_resid = acf(resid, nlags=min(7, len(resid)//2), fft=False)
        print(f"  {HORIZON_NAMES[h]}: ACF lags 1-7: {[f'{v:.3f}' for v in acf_resid[1:8]]}")
        if abs(acf_resid[1]) > 0.3:
            print(f"    *** SERIAL CORRELATION DETECTED — systematic missing feature ***")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for h, ax in enumerate(axes):
    resid = y_test[:, h] - preds_lgbm[:, h]
    if len(resid) > 3:
        acf_r = acf(resid, nlags=min(7, len(resid)//2-1), fft=False)
        ax.bar(range(len(acf_r)), acf_r, color='steelblue', alpha=0.7)
        thresh = 1.96 / np.sqrt(len(resid))
        ax.axhline(thresh, color='red', linestyle='--', lw=1)
        ax.axhline(-thresh, color='red', linestyle='--', lw=1)
        ax.set_title(f'Residual ACF — {HORIZON_NAMES[h]}')
        ax.set_xlabel('Lag (days)'); ax.set_ylabel('ACF'); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('analysis/plots/error_residual_acf.png', dpi=150, bbox_inches='tight'); plt.close()

# --- 6. Worst predicted days ---
test_df['abs_err_d3'] = np.abs(y_test[:, 2] - preds_lgbm[:, 2])
worst10 = test_df.nlargest(10, 'abs_err_d3')[['date','AQI','season','abs_err_d3'] +
          (['wind_speed'] if 'wind_speed' in test_df.columns else []) +
          (['Precipitation'] if 'Precipitation' in test_df.columns else [])]
worst10['predicted_d3'] = preds_lgbm[test_df.nlargest(10,'abs_err_d3').index, 2]
print("\n=== TOP 10 WORST DAY3 PREDICTIONS ===")
print(worst10.to_string(index=False))
worst10.to_csv('analysis/tables/worst_predicted_days.csv', index=False)
print("\nSaved all error analysis plots and tables.")
