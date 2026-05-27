import matplotlib
matplotlib.use('Agg')
import sys; sys.path.append('.')
from dotenv import load_dotenv; load_dotenv()
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
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
test_df = df_clean[df_clean['date'] > split_date]
X_test  = scaler_lgbm.transform(test_df[feat].values)
y_test  = test_df[TARGET_COLS].values

print("Computing permutation importance for day3 model...")
result = permutation_importance(model_lgbm.estimators_[2], X_test, y_test[:,2],
                                 n_repeats=10, random_state=42, n_jobs=-1)
perm_mean = result.importances_mean
perm_std  = result.importances_std
top_idx   = np.argsort(perm_mean)[::-1]

print("\n=== PERMUTATION IMPORTANCE — Day3 ===")
perm_rows = []
for rank, idx in enumerate(top_idx[:20], 1):
    print(f"  {rank:2d}. {feat[idx]:<30}  mean_decrease_R2={perm_mean[idx]:.4f} +/- {perm_std[idx]:.4f}")
    perm_rows.append({'rank':rank,'feature':feat[idx],'perm_importance':perm_mean[idx],'perm_std':perm_std[idx]})

perm_df = pd.DataFrame(perm_rows)
perm_df.to_csv('analysis/tables/permutation_importance_day3.csv', index=False)

# Compare with SHAP
shap_df = pd.read_csv('analysis/tables/shap_daily_lgbm_day3.csv')
comp = perm_df.merge(shap_df[['feature','rank']].rename(columns={'rank':'shap_rank'}), on='feature', how='left')
comp['perm_rank'] = comp.index + 1
print("\n=== SHAP vs PERMUTATION IMPORTANCE COMPARISON (Day3) ===")
print(comp[['feature','perm_rank','shap_rank']].head(15).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(len(perm_rows[:15])), [r['perm_importance'] for r in perm_rows[:15]],
        xerr=[r['perm_std'] for r in perm_rows[:15]], color='steelblue', alpha=0.8, capsize=4)
ax.set_yticks(range(len(perm_rows[:15]))); ax.set_yticklabels([r['feature'] for r in perm_rows[:15]])
ax.invert_yaxis(); ax.set_xlabel('Mean Decrease in R2'); ax.set_title('Permutation Importance — Day3 LGBM')
ax.grid(alpha=0.3); plt.tight_layout()
plt.savefig('analysis/plots/permutation_importance_day3.png', dpi=150, bbox_inches='tight'); plt.close()
print("Saved permutation importance plot and CSV.")
