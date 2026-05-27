import matplotlib
matplotlib.use('Agg')
import sys; sys.path.append('.')
from dotenv import load_dotenv; load_dotenv()
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import shap
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scripts.build_hourly_dataset import FEATURE_COLS as HOURLY_FEAT, TARGET_COLS as HOURLY_TGT

df_h = pd.read_csv('data/hourly_features.csv', parse_dates=['time'])
feat_h = [c for c in HOURLY_FEAT if c in df_h.columns]
df_h_clean = df_h.dropna(subset=feat_h + HOURLY_TGT).reset_index(drop=True)

N_TEST, N_BUFFER = 720, 72
n = len(df_h_clean)
train_h = df_h_clean.iloc[:n - N_TEST - N_BUFFER]
test_h  = df_h_clean.iloc[n - N_TEST:]

scaler_h = StandardScaler()
X_tr_h = scaler_h.fit_transform(train_h[feat_h].values.astype(np.float32))
X_te_h = scaler_h.transform(test_h[feat_h].values.astype(np.float32))
y_tr_h = train_h[HOURLY_TGT].values.astype(np.float32)
y_te_h = test_h[HOURLY_TGT].values.astype(np.float32)

LGBM_PARAMS = dict(n_estimators=300,learning_rate=0.05,max_depth=7,num_leaves=63,
                   subsample=0.8,colsample_bytree=0.8,random_state=42,verbose=-1)
hourly_models = []
for h in range(3):
    m = LGBMRegressor(**LGBM_PARAMS)
    m.fit(X_tr_h, y_tr_h[:,h])
    pred = m.predict(X_te_h)
    r2 = r2_score(y_te_h[:,h], pred)
    print(f"Hourly LGBM {HOURLY_TGT[h]}: R²={r2:.4f}")
    hourly_models.append(m)

# SHAP for day1 and day3
X_te_shap = X_te_h[:300]  # limit for speed
for h_idx in [0, 2]:
    explainer_h = shap.TreeExplainer(hourly_models[h_idx])
    sv_h = explainer_h.shap_values(X_te_shap)
    mean_abs_h = np.abs(sv_h).mean(axis=0)
    top_idx_h = np.argsort(mean_abs_h)[::-1]

    print(f"\n=== HOURLY SHAP TOP 15 — {HOURLY_TGT[h_idx]} ===")
    for rank, idx in enumerate(top_idx_h[:15], 1):
        print(f"  {rank:2d}. {feat_h[idx]:<30}  mean|SHAP|={mean_abs_h[idx]:.4f}")

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv_h, X_te_shap, feature_names=feat_h, show=False, max_display=15)
    plt.title(f'SHAP Beeswarm (Hourly LGBM) — {HOURLY_TGT[h_idx]}', fontsize=11, pad=20)
    plt.tight_layout()
    plt.savefig(f'analysis/plots/shap_hourly_lgbm_day{h_idx+1}_beeswarm.png', dpi=150, bbox_inches='tight')
    plt.close('all')

    top_h_df = pd.DataFrame({'feature':[feat_h[i] for i in top_idx_h],
                              'mean_abs_shap':mean_abs_h[top_idx_h],'rank':range(1,len(feat_h)+1)})
    top_h_df.to_csv(f'analysis/tables/shap_hourly_lgbm_day{h_idx+1}.csv', index=False)

print("Saved hourly SHAP plots and tables.")
