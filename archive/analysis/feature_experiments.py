import matplotlib
matplotlib.use('Agg')
import sys; sys.path.append('.')
from dotenv import load_dotenv; load_dotenv()
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy.stats import linregress
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from config.db import get_collection, COLLECTION_FEATURE_STORE

# ── Load data ──────────────────────────────────────────────────────────────────
docs = list(get_collection(COLLECTION_FEATURE_STORE).find({}, {'_id': 0}))
df = pd.DataFrame(docs)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').drop_duplicates('date').reset_index(drop=True)

print("Available columns:", sorted(df.columns.tolist()))

# ── Base feature set (production 39 features) ─────────────────────────────────
FEAT_BASE = [
    'AQI','PM10','NO2','SO2','O3','log_PM2_5','log_CO',
    'Temperature','Humidity','Precipitation','wind_speed','wind_sin','wind_cos',
    'wind_speed_lag_1','AQI_lag_1','AQI_lag_2','AQI_lag_3','AQI_lag_7',
    'AQI_roll_mean_3','AQI_roll_std_3','AQI_roll_mean_7','AQI_roll_std_7',
    'AQI_roll_min_3','AQI_roll_max_3','AQI_diff','log_PM2_5_lag_1','PM10_lag_1',
    'Temperature_roll_mean_7','Humidity_roll_mean_7','month',
    'season_Spring','season_Summer','season_Winter',
    'weekday_1','weekday_2','weekday_3','weekday_4','weekday_5','weekday_6'
]
TARGET_COLS = ['AQI_t+1', 'AQI_t+2', 'AQI_t+3']

LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=31,
                   subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)

# ── Training helper ────────────────────────────────────────────────────────────
def run_experiment(df_in, extra_feats, label):
    feat_exp = FEAT_BASE + [f for f in extra_feats if f not in FEAT_BASE]
    available = [f for f in feat_exp if f in df_in.columns]
    missing = [f for f in feat_exp if f not in df_in.columns]
    if missing:
        print(f'  WARNING [{label}]: missing {missing} — skipping')
        return None

    df_exp = df_in.dropna(subset=available + TARGET_COLS).reset_index(drop=True)
    split_date = df_exp['date'].max() - pd.Timedelta(days=30)
    tr = df_exp[df_exp['date'] <= split_date]
    te = df_exp[df_exp['date'] >  split_date]
    if len(te) == 0:
        print(f'  [{label}] empty test — skipping'); return None

    sc = StandardScaler()
    X_tr = sc.fit_transform(tr[available].values)
    X_te = sc.transform(te[available].values)

    r2s = []
    for h in range(3):
        m = LGBMRegressor(**LGBM_PARAMS)
        m.fit(X_tr, tr[TARGET_COLS[h]].values)
        r2s.append(round(r2_score(te[TARGET_COLS[h]].values, m.predict(X_te)), 4))

    print(f'  {label:<40}  d1={r2s[0]:.4f}  d2={r2s[1]:.4f}  d3={r2s[2]:.4f}')
    return r2s

# ── Baseline ──────────────────────────────────────────────────────────────────
print("\n=== FEATURE ENGINEERING EXPERIMENTS ===")
print(f"  {'Experiment':<40}  {'R2_d1':>7}  {'R2_d2':>7}  {'R2_d3':>7}")
print("  " + "-"*65)
baseline = run_experiment(df, [], 'baseline (39 feat)')

# ── Compute new features not already in DB ────────────────────────────────────

# AQI_lag_14 (2-week lag) — already in DB
if 'AQI_lag_14' not in df.columns:
    df['AQI_lag_14'] = df['AQI'].shift(14)

# NO2_lag_1 — already in DB
if 'NO2_lag_1' not in df.columns:
    df['NO2_lag_1'] = df['NO2'].shift(1)

# O3_lag_1
if 'O3_lag_1' not in df.columns:
    df['O3_lag_1'] = df['O3'].shift(1)

# AQI x wind_speed interaction
if 'AQI_x_wind' not in df.columns:
    df['AQI_x_wind'] = df['AQI'] * df['wind_speed']

# AQI x Humidity interaction — use DB version if available
if 'AQI_x_Humidity' not in df.columns and 'PM2_5_x_Humidity' not in df.columns:
    df['AQI_x_Humidity'] = df['AQI'] * df['Humidity']

# 7-day trend slope (linear regression slope of last 7 AQI values, shifted by 1 to avoid leakage)
if 'AQI_trend_7d' not in df.columns:
    slopes = np.full(len(df), np.nan)
    aqi_vals = df['AQI'].shift(1).values  # shift 1 to prevent leakage
    for i in range(7, len(df)):
        window = aqi_vals[i-7:i]
        if not np.isnan(window).any():
            slopes[i] = linregress(np.arange(7), window).slope
    df['AQI_trend_7d'] = slopes

# AQI_ewm_7 — already in DB
if 'AQI_ewm_7' not in df.columns:
    df['AQI_ewm_7'] = df['AQI'].shift(1).ewm(span=7, adjust=False).mean()

# AQI_ewm_14 — already in DB
if 'AQI_ewm_14' not in df.columns:
    df['AQI_ewm_14'] = df['AQI'].shift(1).ewm(span=14, adjust=False).mean()

# is_high_pollution_event (AQI > 150 in last 3 days) — check DB version AQI_high_flag
if 'is_high_event_3d' not in df.columns:
    df['is_high_event_3d'] = (df['AQI'].shift(1).rolling(3).max() > 150).astype(float)

# Day-of-year cyclic — already in DB as doy_sin/doy_cos
if 'doy_sin' not in df.columns:
    df['doy'] = df['date'].dt.dayofyear
    df['doy_sin'] = np.sin(2 * np.pi * df['doy'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['doy'] / 365)

# AQI_roll_min_7 — already in DB
if 'AQI_roll_min_7' not in df.columns:
    df['AQI_roll_min_7'] = df['AQI'].shift(1).rolling(7).min()

# AQI_roll_max_7 — already in DB
if 'AQI_roll_max_7' not in df.columns:
    df['AQI_roll_max_7'] = df['AQI'].shift(1).rolling(7).max()

# wind x PM2_5 lag interaction — already in DB as wind_x_PM2_5_lag1
if 'wind_x_PM2_5_lag1' not in df.columns and 'log_PM2_5_lag_1' in df.columns:
    df['wind_x_PM2_5_lag1'] = df['wind_speed'] * df['log_PM2_5_lag_1']

# Humidity x PM2_5 interaction (lag version) — already in DB as PM2_5_x_Humidity
if 'Humidity_x_PM2_5_lag' not in df.columns and 'log_PM2_5_lag_1' in df.columns:
    df['Humidity_x_PM2_5_lag'] = df['Humidity'] * df['log_PM2_5_lag_1']

# ── Individual experiments ────────────────────────────────────────────────────
experiments = [
    (['AQI_lag_14'],            'add AQI_lag_14 (2-week lag)'),
    (['NO2_lag_1'],             'add NO2_lag_1'),
    (['O3_lag_1'],              'add O3_lag_1'),
    (['AQI_x_wind'],            'add AQI x wind_speed'),
    (['AQI_x_Humidity'],        'add AQI x Humidity'),
    (['AQI_trend_7d'],          'add AQI_trend_7d (7d slope)'),
    (['AQI_ewm_7'],             'add AQI_ewm_7'),
    (['AQI_ewm_14'],            'add AQI_ewm_14'),
    (['is_high_event_3d'],      'add is_high_event_3d flag'),
    (['doy_sin','doy_cos'],     'add doy_sin+doy_cos (cyclic)'),
    (['AQI_roll_min_7','AQI_roll_max_7'], 'add AQI_roll_min/max_7d'),
    (['wind_x_PM2_5_lag1'],     'add wind x PM2_5_lag1'),
    (['Humidity_x_PM2_5_lag'],  'add Humidity x PM2_5_lag'),
    # Check if DB-resident advanced features already help
    (['AQI_ewm_30'],            'add AQI_ewm_30 (DB)'),
    (['AQI_lag_14','AQI_ewm_7','AQI_trend_7d'], 'combined: lag14+ewm7+slope'),
]

results = []
for extra_feats, label in experiments:
    r2s = run_experiment(df, extra_feats, label)
    if r2s is not None and baseline is not None:
        results.append({
            'experiment': label,
            'features_added': str(extra_feats),
            'r2_d1': r2s[0], 'r2_d2': r2s[1], 'r2_d3': r2s[2],
            'delta_d1': round(r2s[0] - baseline[0], 4),
            'delta_d2': round(r2s[1] - baseline[1], 4),
            'delta_d3': round(r2s[2] - baseline[2], 4),
        })

# ── Best combination experiment ───────────────────────────────────────────────
if results and baseline:
    winners_d3 = [r for r in results if r['delta_d3'] > 0.01]
    if winners_d3:
        best_feats = []
        for r in sorted(winners_d3, key=lambda x: x['delta_d3'], reverse=True)[:5]:
            best_feats.extend(eval(r['features_added']))
        best_feats = list(dict.fromkeys(best_feats))  # deduplicate
        print(f"\n  Best combination feats: {best_feats}")
        combo_r2 = run_experiment(df, best_feats, f'BEST COMBINATION ({len(best_feats)} new feats)')
        if combo_r2:
            results.append({'experiment':'best_combination','features_added':str(best_feats),
                            'r2_d1':combo_r2[0],'r2_d2':combo_r2[1],'r2_d3':combo_r2[2],
                            'delta_d1':round(combo_r2[0]-baseline[0],4),
                            'delta_d2':round(combo_r2[1]-baseline[1],4),
                            'delta_d3':round(combo_r2[2]-baseline[2],4)})
    else:
        print("\n  No experiments improved day3 by >0.01")

# ── Season-specific model (Summer vs global) ──────────────────────────────────
print("\n=== SEASON-SPECIFIC MODEL EXPERIMENT ===")
SEASON_MAP = {12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',
              6:'Summer',7:'Summer',8:'Summer',9:'Autumn',10:'Autumn',11:'Autumn'}
df['season_label'] = df['date'].dt.month.map(SEASON_MAP)

df_feat = df.dropna(subset=FEAT_BASE + TARGET_COLS).reset_index(drop=True)
split_date = df_feat['date'].max() - pd.Timedelta(days=30)

# Cross-val within summer data (since test window may not overlap with summer)
summer_df = df_feat[df_feat['season_label'] == 'Summer'].reset_index(drop=True)
if len(summer_df) > 50:
    tscv = TimeSeriesSplit(n_splits=3)
    summer_r2s_cv = [[], [], []]
    for tr_idx, val_idx in tscv.split(summer_df):
        sc_s = StandardScaler()
        X_tr_s = sc_s.fit_transform(summer_df.iloc[tr_idx][FEAT_BASE].values)
        X_val_s = sc_s.transform(summer_df.iloc[val_idx][FEAT_BASE].values)
        for h in range(3):
            m = LGBMRegressor(**LGBM_PARAMS)
            m.fit(X_tr_s, summer_df.iloc[tr_idx][TARGET_COLS[h]].values)
            pred = m.predict(X_val_s)
            r2_val = r2_score(summer_df.iloc[val_idx][TARGET_COLS[h]].values, pred)
            summer_r2s_cv[h].append(r2_val)
    for h in range(3):
        avg = np.mean(summer_r2s_cv[h])
        print(f"  Summer-only CV R2_d{h+1}: {avg:.4f}  (global baseline: {baseline[h]:.4f})")
else:
    print(f"  Not enough summer data for CV (n={len(summer_df)})")

# ── Save results ──────────────────────────────────────────────────────────────
if results:
    res_df = pd.DataFrame(results)
    res_df.to_csv('analysis/tables/feature_engineering_results.csv', index=False)
    print("\n=== SUMMARY: Features improving day3 R2 ===")
    improvers = res_df[res_df['delta_d3'] > 0].sort_values('delta_d3', ascending=False)
    print(improvers[['experiment','r2_d3','delta_d3']].to_string(index=False))

    # Bar chart of day3 delta
    fig, ax = plt.subplots(figsize=(12, max(5, len(res_df)*0.4)))
    colors = ['green' if d > 0.01 else ('orange' if d > 0 else 'steelblue') for d in res_df['delta_d3']]
    ax.barh(res_df['experiment'], res_df['delta_d3'], color=colors, alpha=0.8)
    ax.axvline(0, color='black', lw=1)
    ax.axvline(0.05, color='red', linestyle='--', lw=1.5, label='Material threshold (+0.05)')
    ax.set_xlabel('Delta R2 (day3 vs baseline)'); ax.set_title('Feature Engineering: Day3 R2 Improvement')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('analysis/plots/feature_engineering_delta.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved feature_engineering_delta.png")

    # Also plot all 3 horizons
    fig, axes = plt.subplots(1, 3, figsize=(18, max(5, len(res_df)*0.4)))
    for h_idx, (ax, h_name) in enumerate(zip(axes, ['day1','day2','day3'])):
        vals = res_df[f'delta_d{h_idx+1}']
        colors = ['green' if d > 0.01 else ('orange' if d > 0 else '#aaa') for d in vals]
        ax.barh(res_df['experiment'], vals, color=colors, alpha=0.8)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_xlabel(f'Delta R2 {h_name}'); ax.set_title(f'{h_name}: R2 delta')
        ax.grid(alpha=0.3)
    plt.suptitle('Feature Engineering Experiments - All Horizons', fontsize=12)
    plt.tight_layout()
    plt.savefig('analysis/plots/feature_engineering_all_horizons.png', dpi=150, bbox_inches='tight')
    plt.close()

print("\nDone. All results saved to analysis/tables/feature_engineering_results.csv")
