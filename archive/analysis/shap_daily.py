import matplotlib
matplotlib.use('Agg')
import sys; sys.path.append('.')
from dotenv import load_dotenv; load_dotenv()
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import shap

from config.db import get_collection, COLLECTION_FEATURE_STORE, load_model

# Load model + data
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
test_df  = df_clean[df_clean['date'] >  split_date]
X_train = scaler_lgbm.transform(train_df[feat].values)
X_test  = scaler_lgbm.transform(test_df[feat].values)
y_test  = test_df[TARGET_COLS].values

horizon_names = ['day1 (t+1)', 'day2 (t+2)', 'day3 (t+3)']
all_shap = []

for h, name in enumerate(horizon_names):
    estimator = model_lgbm.estimators_[h]
    explainer = shap.TreeExplainer(estimator)
    sv = explainer.shap_values(X_test)  # shape (n_test, n_feat)
    all_shap.append(sv)

    mean_abs = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1]

    print(f"\n=== SHAP TOP 15 FEATURES — {name} ===")
    for rank, idx in enumerate(top_idx[:15], 1):
        print(f"  {rank:2d}. {feat[idx]:<30}  mean|SHAP|={mean_abs[idx]:.4f}")

    # Beeswarm
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv, X_test, feature_names=feat, show=False, max_display=15)
    plt.title(f'SHAP Beeswarm — {name}', fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(f'analysis/plots/shap_daily_lgbm_day{h+1}_beeswarm.png', dpi=150, bbox_inches='tight')
    plt.close('all')

    # Bar
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv, X_test, feature_names=feat, plot_type='bar', show=False, max_display=15)
    plt.title(f'SHAP Bar — {name}', fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(f'analysis/plots/shap_daily_lgbm_day{h+1}_bar.png', dpi=150, bbox_inches='tight')
    plt.close('all')

    # Save CSV
    top_df = pd.DataFrame({'feature': [feat[i] for i in top_idx],
                           'mean_abs_shap': mean_abs[top_idx],
                           'rank': range(1, len(feat)+1)})
    top_df.to_csv(f'analysis/tables/shap_daily_lgbm_day{h+1}.csv', index=False)

# === Cross-horizon comparison ===
print("\n=== CROSS-HORIZON SHAP RANKING COMPARISON ===")
print(f"{'Feature':<30}  {'Day1 rank':>10}  {'Day2 rank':>10}  {'Day3 rank':>10}  {'Day3 mean|SHAP|':>15}")

rankings = {}
for h in range(3):
    mean_abs_h = np.abs(all_shap[h]).mean(axis=0)
    ranked = {feat[idx]: rank+1 for rank, idx in enumerate(np.argsort(mean_abs_h)[::-1])}
    rankings[h] = ranked

# Show top 20 by day3
mean_abs_d3 = np.abs(all_shap[2]).mean(axis=0)
top_d3_idx = np.argsort(mean_abs_d3)[::-1][:20]

comp_rows = []
for idx in top_d3_idx:
    f = feat[idx]
    r1, r2, r3 = rankings[0][f], rankings[1][f], rankings[2][f]
    print(f"  {f:<30}  {r1:>10}  {r2:>10}  {r3:>10}  {mean_abs_d3[idx]:>15.4f}")
    comp_rows.append({'feature':f,'rank_d1':r1,'rank_d2':r2,'rank_d3':r3,'mean_abs_shap_d3':mean_abs_d3[idx]})

comp_df = pd.DataFrame(comp_rows)
comp_df.to_csv('analysis/tables/shap_feature_ranking_comparison.csv', index=False)

# Features in top-10 d1 but NOT top-10 d3
d1_top10 = set([feat[i] for i in np.argsort(np.abs(all_shap[0]).mean(axis=0))[::-1][:10]])
d3_top10 = set([feat[i] for i in np.argsort(np.abs(all_shap[2]).mean(axis=0))[::-1][:10]])
print("\nIn top-10 for DAY1 but NOT day3 (autoregressive-only features):")
print("  ", d1_top10 - d3_top10)
print("\nIn top-10 for DAY3 but NOT day1 (long-range signal features):")
print("  ", d3_top10 - d1_top10)

# Day1 SHAP fraction: AQI_lag_1 vs all others
mean_abs_d1 = np.abs(all_shap[0]).mean(axis=0)
total_shap_d1 = mean_abs_d1.sum()
if 'AQI_lag_1' in feat:
    aqi_lag1_idx = feat.index('AQI_lag_1')
    aqi_lag1_shap = mean_abs_d1[aqi_lag1_idx]
    print(f"\nDay1 SHAP fraction — AQI_lag_1: {aqi_lag1_shap/total_shap_d1*100:.1f}%")
    print(f"Day1 SHAP fraction — all others: {(total_shap_d1-aqi_lag1_shap)/total_shap_d1*100:.1f}%")

print("\nSaved all SHAP plots and tables.")
