import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor

from config.db import get_collection, COLLECTION_FEATURE_STORE

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_COLS  = ["AQI_t+1", "AQI_t+2", "AQI_t+3"]

# Strictly excluding only leakage features and targets.
EXCLUDE_COLS = {"date", "processed_at", "_id"} | set(TARGET_COLS)

# Parameters for the combinatorial search (lowered estimators for speed)
CV_RF_PARAMS = dict(
    n_estimators=50,       # Kept relatively low so hundreds of CV fits don't take forever
    max_depth=7,
    min_samples_split=5,
    random_state=42,
    n_jobs=1               # Set to 1 because RFECV will use n_jobs=-1 to parallelize the search
)

# Parameters for the final holdout evaluation (higher estimators for accuracy)
FINAL_RF_PARAMS = dict(
    n_estimators=300,
    max_depth=7,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1              # Use all cores for the final fits
)

# ── Data Loading & Prep (Read-Only) ───────────────────────────────────────────
def load_data() -> pd.DataFrame:
    docs = list(get_collection(COLLECTION_FEATURE_STORE).find({}, {"_id": 0}))
    if not docs:
        raise RuntimeError("feature_store is empty.")
    df = pd.DataFrame(docs)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=TARGET_COLS)
    return df

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS]

def time_split(df: pd.DataFrame, test_days: int = 30):
    split_date = df["date"].max() - pd.Timedelta(days=test_days)
    train = df[df["date"] <= split_date]
    test  = df[df["date"] >  split_date]
    return train, test

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "R2":   round(float(r2_score(y_true, y_pred)), 4),
        "MAE":  round(float(mean_absolute_error(y_true, y_pred)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
    }

# ── Combinatorial Search (RFECV) ──────────────────────────────────────────────
def run_rfecv_combinations(X_train, y_train, feat_names):
    print("\n[Step 4] Initializing Time-Series Cross-Validation...")
    tscv = TimeSeriesSplit(n_splits=3)
    global_support = np.zeros(len(feat_names), dtype=bool)
    
    print("[Step 5] Beginning Recursive Feature Elimination with Cross-Validation (RFECV)...")
    print("         (Note: Random Forest CV can take some time depending on your CPU core count)\n")
    
    for h, target in enumerate(TARGET_COLS):
        print(f"  --> Exploring combinations for {target}...")
        
        estimator = RandomForestRegressor(**CV_RF_PARAMS)
        
        # RFECV drops 2 features at a time (step=2) to balance speed and granularity.
        selector = RFECV(
            estimator=estimator,
            step=2, 
            cv=tscv,
            scoring="r2",
            min_features_to_select=10,
            n_jobs=-1 # Use all available CPU cores for the CV loops
        )
        
        print(f"      [Target {target}]: Fitting combinations...")
        selector.fit(X_train, y_train[:, h])
        
        support = selector.get_support()
        optimal_num = support.sum()
        max_cv_score = selector.cv_results_['mean_test_score'].max()
        
        print(f"      [Target {target}]: Found optimal combination! Kept {optimal_num} features. (Peak CV R2: {max_cv_score:.3f})")
        
        # Logical OR: If a feature is part of the optimal combo for ANY horizon, we keep it
        global_support = global_support | support

    selected_features = [feat_names[i] for i in range(len(feat_names)) if global_support[i]]
    print(f"\n[Step 6] Consolidating optimal features...")
    print(f"         Total unique features kept across all horizons: {len(selected_features)}")
    
    return selected_features, global_support

def evaluate_subset(X_tr, X_te, y_train, y_test):
    """Evaluates the features on the 30-day holdout using the robust FINAL_RF_PARAMS."""
    preds = []
    for h in range(y_train.shape[1]):
        m = RandomForestRegressor(**FINAL_RF_PARAMS)
        m.fit(X_tr, y_train[:, h])
        preds.append(m.predict(X_te))
        
    preds = np.column_stack(preds)
    
    m1 = compute_metrics(y_test[:, 0], preds[:, 0])
    m2 = compute_metrics(y_test[:, 1], preds[:, 1])
    m3 = compute_metrics(y_test[:, 2], preds[:, 2])
    
    avg_r2 = np.mean([m1['R2'], m2['R2'], m3['R2']])
    return avg_r2, m1, m2, m3

# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("RANDOM FOREST RFECV COMBINATORIAL SELECTION")
    print("=" * 60)
    
    print("\n[Step 1] Loading data from database...")
    df = load_data()
    feat_names = get_feature_cols(df)
    
    print(f"         Total rows fetched: {len(df)}")
    print(f"         Initial features to evaluate: {len(feat_names)}")
    
    print("\n[Step 2] Applying Time-Aware Split (30-day holdout)...")
    train_df, test_df = time_split(df)
    
    X_train_raw = train_df[feat_names].values.astype(np.float32)
    X_test_raw  = test_df[feat_names].values.astype(np.float32)
    y_train = train_df[TARGET_COLS].values.astype(np.float32)
    y_test  = test_df[TARGET_COLS].values.astype(np.float32)

    print("\n[Step 3] Imputing NaNs and Scaling features...")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(imputer.fit_transform(X_train_raw))
    X_test  = scaler.transform(imputer.transform(X_test_raw))

    # 1. Run the combinatorial search
    best_feats, best_mask = run_rfecv_combinations(X_train, y_train, feat_names)
    
    # 2. Evaluate the optimal subset vs All Features
    print("\n[Step 7] Evaluating Final Subsets on Holdout Test Set...")
    print("         Training models on All Features vs RFECV Optimal Features...")
    
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON 30-DAY HOLDOUT (RANDOM FOREST)")
    print("=" * 60)
    print(f"  {'Feature Set':<20} | {'Avg R2':>8} || {'d1 R2':>7} | {'d2 R2':>7} | {'d3 R2':>7}")
    print("  " + "-" * 60)
    
    # Evaluate All Features
    r2_all, m1a, m2a, m3a = evaluate_subset(X_train, X_test, y_train, y_test)
    print(f"  {'All Features':<20} | {r2_all:>8.3f} || {m1a['R2']:>7.3f} | {m2a['R2']:>7.3f} | {m3a['R2']:>7.3f}")
    
    # Evaluate RFECV Optimal Features
    X_tr_sel = X_train[:, best_mask]
    X_te_sel = X_test[:, best_mask]
    r2_sel, m1s, m2s, m3s = evaluate_subset(X_tr_sel, X_te_sel, y_train, y_test)
    print(f"  {'RFECV Optimal':<20} | {r2_sel:>8.3f} || {m1s['R2']:>7.3f} | {m2s['R2']:>7.3f} | {m3s['R2']:>7.3f}")
    
    print("\n[Step 8] Pipeline Complete. Optimized Feature List Below:")
    print("  " + ", ".join(best_feats))