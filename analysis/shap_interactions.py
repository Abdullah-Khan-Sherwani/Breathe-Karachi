import matplotlib
matplotlib.use('Agg')
import sys; sys.path.append('.')
from dotenv import load_dotenv; load_dotenv()
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import shap
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
X_test = scaler_lgbm.transform(test_df[feat].values)

# Day3 interactions (limit to 50 rows for speed)
estimator_d3 = model_lgbm.estimators_[2]
explainer = shap.TreeExplainer(estimator_d3)
# Try interaction values on subset
try:
    sv_int = explainer.shap_interaction_values(X_test[:50])  # shape (50, n_feat, n_feat)
    # Mean absolute interaction matrix
    mean_int = np.abs(sv_int).mean(axis=0)
    np.fill_diagonal(mean_int, 0)  # zero out self-interactions

    # Top 5 pairs
    pairs = []
    for i in range(len(feat)):
        for j in range(i+1, len(feat)):
            pairs.append((feat[i], feat[j], mean_int[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)

    print("\n=== TOP 10 SHAP INTERACTION PAIRS (Day3) ===")
    int_rows = []
    for f1, f2, val in pairs[:10]:
        print(f"  {f1:<25} x {f2:<25}  mean|interaction|={val:.4f}")
        int_rows.append({'feature_1':f1,'feature_2':f2,'mean_abs_interaction':val})
    pd.DataFrame(int_rows).to_csv('analysis/tables/shap_interactions_day3.csv', index=False)

    # Heatmap of top features
    top_feat_int = list(set([p[0] for p in pairs[:15]] + [p[1] for p in pairs[:15]]))
    idx_map = {f: feat.index(f) for f in top_feat_int}
    int_sub = mean_int[np.ix_([idx_map[f] for f in top_feat_int],[idx_map[f] for f in top_feat_int])]
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(int_sub, xticklabels=top_feat_int, yticklabels=top_feat_int,
                cmap='YlOrRd', annot=True, fmt='.3f', ax=ax, cbar_kws={'label':'mean|SHAP interaction|'})
    ax.set_title('SHAP Interaction Values — Day3 Model (top interacting features)')
    plt.xticks(rotation=45, ha='right', fontsize=8); plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig('analysis/plots/shap_daily_lgbm_day3_interactions.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print("Saved interaction heatmap")
except Exception as e:
    print(f"Interaction values failed: {e}")
    print("Falling back to standard SHAP bar plot for day3")
