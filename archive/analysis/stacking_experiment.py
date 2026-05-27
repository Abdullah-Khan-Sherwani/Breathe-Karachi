import matplotlib
matplotlib.use('Agg')
import sys; sys.path.append('.')
from dotenv import load_dotenv; load_dotenv()
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
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
X_train_raw = scaler_lgbm.transform(train_df[feat].values)
X_test_raw  = scaler_lgbm.transform(test_df[feat].values)
y_train = train_df[TARGET_COLS].values
y_test  = test_df[TARGET_COLS].values

LGBM_PARAMS = dict(n_estimators=300,learning_rate=0.05,max_depth=6,num_leaves=31,
                   subsample=0.8,colsample_bytree=0.8,random_state=42,verbose=-1)
tscv = TimeSeriesSplit(n_splits=5)

print("=== STACKING EXPERIMENT ===")
print(f"Train: {len(X_train_raw)} | Test: {len(X_test_raw)}")

# ── Baseline (no stacking) ──
m_d1_b = LGBMRegressor(**LGBM_PARAMS); m_d1_b.fit(X_train_raw, y_train[:,0])
m_d2_b = LGBMRegressor(**LGBM_PARAMS); m_d2_b.fit(X_train_raw, y_train[:,1])
m_d3_b = LGBMRegressor(**LGBM_PARAMS); m_d3_b.fit(X_train_raw, y_train[:,2])
baseline_d1 = r2_score(y_test[:,0], m_d1_b.predict(X_test_raw))
baseline_d2 = r2_score(y_test[:,1], m_d2_b.predict(X_test_raw))
baseline_d3 = r2_score(y_test[:,2], m_d3_b.predict(X_test_raw))
print(f"\nBaseline (no stacking): d1={baseline_d1:.4f}  d2={baseline_d2:.4f}  d3={baseline_d3:.4f}")

# ── OOF Day1 ──
oof_d1 = np.zeros(len(X_train_raw))
for tr_idx, val_idx in tscv.split(X_train_raw):
    m = LGBMRegressor(**LGBM_PARAMS)
    m.fit(X_train_raw[tr_idx], y_train[tr_idx, 0])
    oof_d1[val_idx] = m.predict(X_train_raw[val_idx])
m_d1_full = LGBMRegressor(**LGBM_PARAMS); m_d1_full.fit(X_train_raw, y_train[:,0])
test_d1_pred = m_d1_full.predict(X_test_raw)

# ── Stacked Day2 ──
X_tr_s2 = np.column_stack([X_train_raw, oof_d1])
X_te_s2 = np.column_stack([X_test_raw,  test_d1_pred])
m_d2_s = LGBMRegressor(**LGBM_PARAMS); m_d2_s.fit(X_tr_s2, y_train[:,1])
stacked_d2 = r2_score(y_test[:,1], m_d2_s.predict(X_te_s2))
print(f"Stacked day2 (+ OOF d1 pred): d2={stacked_d2:.4f}  Delta={stacked_d2-baseline_d2:+.4f}")

# ── OOF Day2 (from stacked model) ──
oof_d2 = np.zeros(len(X_train_raw))
for tr_idx, val_idx in tscv.split(X_train_raw):
    X_fold_tr_s2 = np.column_stack([X_train_raw[tr_idx], oof_d1[tr_idx]])
    X_fold_val_s2 = np.column_stack([X_train_raw[val_idx], oof_d1[val_idx]])
    m = LGBMRegressor(**LGBM_PARAMS)
    m.fit(X_fold_tr_s2, y_train[tr_idx, 1])
    oof_d2[val_idx] = m.predict(X_fold_val_s2)
test_d2_pred = m_d2_s.predict(X_te_s2)

# ── Stacked Day3 (+ d1 + d2 predictions) ──
X_tr_s3 = np.column_stack([X_train_raw, oof_d1, oof_d2])
X_te_s3 = np.column_stack([X_test_raw,  test_d1_pred, test_d2_pred])
m_d3_s = LGBMRegressor(**LGBM_PARAMS); m_d3_s.fit(X_tr_s3, y_train[:,2])
stacked_d3 = r2_score(y_test[:,2], m_d3_s.predict(X_te_s3))
print(f"Stacked day3 (+ OOF d1+d2 preds): d3={stacked_d3:.4f}  Delta={stacked_d3-baseline_d3:+.4f}")

print(f"""
============================================================
STACKING RESULTS SUMMARY
------------------------------------------------------------
              Baseline      Stacked       Delta
Day1:         {baseline_d1:.4f}        (no stacking)    n/a
Day2:         {baseline_d2:.4f}        {stacked_d2:.4f}        {stacked_d2-baseline_d2:+.4f}
Day3:         {baseline_d3:.4f}        {stacked_d3:.4f}        {stacked_d3-baseline_d3:+.4f}
============================================================
""")

# Save CSV
results = pd.DataFrame([
    {'horizon':'day1','baseline_r2':baseline_d1,'stacked_r2':baseline_d1,'delta':0.0,'method':'no stacking'},
    {'horizon':'day2','baseline_r2':baseline_d2,'stacked_r2':stacked_d2,'delta':stacked_d2-baseline_d2,'method':'+ OOF day1 pred'},
    {'horizon':'day3','baseline_r2':baseline_d3,'stacked_r2':stacked_d3,'delta':stacked_d3-baseline_d3,'method':'+ OOF day1+day2 preds'},
])
results.to_csv('analysis/tables/stacking_results.csv', index=False)

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for col_idx, h in enumerate([1, 2]):
    ax = axes[col_idx]
    horizon_name = f'day{h+1}'
    actual = y_test[:, h]
    pred_base = m_d2_b.predict(X_test_raw) if h==1 else m_d3_b.predict(X_test_raw)
    pred_stacked = m_d2_s.predict(X_te_s2) if h==1 else m_d3_s.predict(X_te_s3)
    x_range = range(len(actual))
    ax.plot(x_range, actual, 'k-', lw=1.5, label='Actual', alpha=0.8)
    ax.plot(x_range, pred_base, 'b--', lw=1.2, label=f'Baseline R²={r2_score(actual,pred_base):.3f}', alpha=0.7)
    ax.plot(x_range, pred_stacked, 'r-', lw=1.2, label=f'Stacked R²={r2_score(actual,pred_stacked):.3f}', alpha=0.7)
    ax.set_title(f'{horizon_name}: Baseline vs Stacked'); ax.set_xlabel('Test index'); ax.set_ylabel('AQI')
    ax.legend(); ax.grid(alpha=0.3)
plt.suptitle('Prediction Stacking Comparison (30-day holdout)', fontsize=12)
plt.tight_layout()
plt.savefig('analysis/plots/stacking_comparison.png', dpi=150, bbox_inches='tight'); plt.close()
print("Saved stacking results and comparison plot.")
