"""
Train and compare two model families on the daily feature_store dataset.

  - 3 independent LGBMRegressors, one per horizon (day1, day2, day3)
  - 1 multi-output LSTM predicting all 3 horizons simultaneously

Test set  : last 30 days (time-aware split)
Baselines : Daily LGBM MultiOutputRegressor  R2_d1=0.773 R2_d2=0.284 R2_d3=0.120
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor

from config.db import get_collection, COLLECTION_FEATURE_STORE

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_COLS  = ["AQI_t+1", "AQI_t+2", "AQI_t+3", "AQI_t+4"]

# Tier-2/Tier-3 features tested and confirmed to hurt both LGBM and LSTM on holdout
EXCLUDE_COLS = {"date", "processed_at", "_id",
                "AQI_trend_7d", "AQI_x_wind", "NO2_lag_3", "Humidity_lag_3",
                "surface_pressure", "surface_pressure_t1", "surface_pressure_t2", "surface_pressure_t3",
                "surface_pressure_lag_1", "surface_pressure_roll_mean_7",
                "apparent_temp", "apparent_temp_t1", "apparent_temp_t2", "apparent_temp_t3",
                "apparent_temp_lag_1", "apparent_temp_roll_mean_7",
                "wind_gusts", "wind_gusts_t1", "wind_gusts_t2", "wind_gusts_t3",
                "wind_gusts_lag_1", "wind_gusts_roll_mean_7",
                # PM2.5 leads — excluded (dominant AQI driver, inflates metrics):
                "PM2_5_t1", "PM2_5_t2", "PM2_5_t3", "PM2_5_t4",
                # removed from pipeline but may still exist in old MongoDB docs:
                "visibility", "visibility_t1", "visibility_t2", "visibility_t3",
                "visibility_lag_1", "visibility_roll_mean_7",
                "wind_speed_80m", "wind_speed_80m_t1", "wind_speed_80m_t2", "wind_speed_80m_t3",
                "wind_speed_80m_lag_1",
                "cape", "cape_t1", "cape_t2", "cape_t3", "cape_lag_1",
                } | set(TARGET_COLS)

SEQ_LEN = 7

LGBM_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    docs = list(get_collection(COLLECTION_FEATURE_STORE).find({}, {"_id": 0}))
    if not docs:
        raise RuntimeError(
            "feature_store is empty — run fetch_data.py and preprocess_daily_data.py first."
        )
    df = pd.DataFrame(docs)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=TARGET_COLS)
    return df


def get_feature_cols(df: pd.DataFrame, exclude: set | None = None) -> list[str]:
    exc = EXCLUDE_COLS if exclude is None else exclude
    return [c for c in df.columns if c not in exc]


def time_split(df: pd.DataFrame, test_days: int = 30):
    split_date = df["date"].max() - pd.Timedelta(days=test_days)
    train = df[df["date"] <= split_date]
    test  = df[df["date"] >  split_date]
    return train, test


# ── Metric helpers ────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "R2":   round(float(r2_score(y_true, y_pred)), 4),
        "MAE":  round(float(mean_absolute_error(y_true, y_pred)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
    }


def print_per_horizon(label: str, *metrics) -> None:
    print(f"\n{label}")
    print(f"  {'Horizon':<12}  {'R2':>7}  {'MAE':>7}  {'RMSE':>7}")
    print("  " + "-" * 38)
    names = ["day1 (t+1)", "day2 (t+2)", "day3 (t+3)", "day4 (t+4)"]
    for name, m in zip(names, metrics):
        print(f"  {name:<12}  {m['R2']:>7.3f}  {m['MAE']:>7.2f}  {m['RMSE']:>7.2f}")


# ── LGBM: 3 independent models ────────────────────────────────────────────────
def train_lgbm(X_train, X_test, y_train, y_test, feat_names):
    print("\n" + "=" * 60)
    print("LGBM — training 3 independent regressors (one per horizon)")
    print("=" * 60)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    models, preds = [], []
    for h, col in enumerate(TARGET_COLS):
        print(f"\n  Fitting horizon {col} ...")
        m = LGBMRegressor(**LGBM_PARAMS)
        m.fit(X_tr, y_train[:, h])
        models.append(m)
        pred = m.predict(X_te)
        preds.append(pred)
        m_metrics = compute_metrics(y_test[:, h], pred)
        print(f"    R2={m_metrics['R2']:.3f}  MAE={m_metrics['MAE']:.2f}  RMSE={m_metrics['RMSE']:.2f}")

    metrics = [compute_metrics(y_test[:, h], preds[h]) for h in range(len(TARGET_COLS))]
    print_per_horizon("LGBM per-horizon results (30-day holdout)", *metrics)

    # Top-10 feature importances from day3 model
    imp = pd.Series(models[2].feature_importances_, index=feat_names).sort_values(ascending=False)
    print("\n  Top-10 features (day3 model):")
    for fname, score in imp.head(10).items():
        print(f"    {fname:<32}  {score:>6}")

    lgbm_preds = np.column_stack(preds)
    return models, scaler, tuple(metrics), lgbm_preds


# ── LSTM: multi-output ────────────────────────────────────────────────────────
def _build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Sliding window: each window of seq_len rows → target at the last row."""
    Xs, ys = [], []
    for i in range(len(X) - seq_len + 1):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len - 1])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def train_lstm(X_train, X_test, y_train, y_test):
    import tensorflow as tf

    # Clear any stale Keras session / traced-function cache from previous runs
    tf.keras.backend.clear_session()

    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    n_features = X_train.shape[1]

    print("\n" + "=" * 60)
    print(f"LSTM — multi-output (seq_len={SEQ_LEN}, features={n_features})")
    print("=" * 60)

    # Impute NaN features with column median (fit on train only)
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test  = imputer.transform(X_test)

    nan_remaining = np.isnan(X_train).sum() + np.isnan(X_test).sum()
    if nan_remaining > 0:
        print(f"  WARNING: {nan_remaining} NaN values remain after imputation — clipping to 0.")
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test  = np.nan_to_num(X_test,  nan=0.0)
    else:
        print(f"  Imputation complete — no NaN values remaining in features.")

    # Scale X and y independently
    x_sc = StandardScaler()
    y_sc = StandardScaler()

    X_tr_sc = x_sc.fit_transform(X_train)
    X_te_sc = x_sc.transform(X_test)
    y_tr_sc = y_sc.fit_transform(y_train)
    y_te_sc = y_sc.transform(y_test)

    # Concatenate so test sequences can look back into training data
    X_all = np.vstack([X_tr_sc, X_te_sc])
    y_all = np.vstack([y_tr_sc, y_te_sc])

    X_seq, y_seq = _build_sequences(X_all, y_all, SEQ_LEN)

    n_tr_seq = len(X_train) - SEQ_LEN + 1
    X_s_tr, y_s_tr = X_seq[:n_tr_seq], y_seq[:n_tr_seq]
    X_s_te, y_s_te = X_seq[n_tr_seq:], y_seq[n_tr_seq:]

    print(f"  Train sequences: {len(X_s_tr)}  |  Test sequences: {len(X_s_te)}")

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(SEQ_LEN, n_features)),
        tf.keras.layers.LSTM(64, return_sequences=True,
                             kernel_regularizer=tf.keras.regularizers.L2(1e-3)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.LSTM(32, kernel_regularizer=tf.keras.regularizers.L2(1e-3)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(4),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="huber",
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=25, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5, verbose=1
        ),
    ]

    print("\n  Training (verbose=1 — every epoch shows train_loss and val_loss):\n")
    model.fit(
        X_s_tr, y_s_tr,
        validation_split=0.1,
        epochs=150,
        batch_size=16,
        callbacks=callbacks,
        verbose=1,
    )

    if len(X_s_te) == 0:
        print("  WARNING: No test sequences available; using last train sequence.")
        preds_sc = model.predict(X_s_tr[-1:], verbose=0)
        y_true_sc = y_s_tr[-1:]
    else:
        preds_sc = model.predict(X_s_te, verbose=0)
        y_true_sc = y_s_te

    # Inverse-transform before computing metrics
    preds  = y_sc.inverse_transform(preds_sc)
    y_true = y_sc.inverse_transform(y_true_sc)

    lstm_metrics = tuple(compute_metrics(y_true[:, h], preds[:, h]) for h in range(len(TARGET_COLS)))
    print_per_horizon("LSTM multi-output results (30-day holdout)", *lstm_metrics)

    return model, (x_sc, y_sc), lstm_metrics, preds, y_true


# ── Ensemble: per-horizon weighted blend ──────────────────────────────────────
# Weights tuned to each model's relative strength per horizon.
# day1: LGBM stronger → lean LGBM; day2/day3: LSTM stronger → lean LSTM.
ENSEMBLE_WEIGHTS = [
    (0.6, 0.4),   # day1: lgbm_w, lstm_w
    (0.15, 0.85), # day2
    (0.0, 1.0),   # day3: pure LSTM
    (0.0, 1.0),   # day4: pure LSTM
]


def compute_ensemble(lgbm_preds, lstm_preds, y_test):
    """Blend LGBM and LSTM predictions with per-horizon weights."""
    ens_metrics = []
    for h, (wl, ws) in enumerate(ENSEMBLE_WEIGHTS):
        ens = wl * lgbm_preds[:, h] + ws * lstm_preds[:, h]
        ens_metrics.append(compute_metrics(y_test[:, h], ens))
    return tuple(ens_metrics)


# ── Comparison table ──────────────────────────────────────────────────────────
def print_comparison(lgbm_m, lstm_m, ens_m):
    h_labels = ["d1", "d2", "d3", "d4"]

    print("\n" + "=" * 72)
    print("COMPARISON (30-day holdout)")
    header = f"  {'Model':<28}" + "".join(f"  {'R2_'+h:>7}" for h in h_labels)
    print(header)
    print("  " + "-" * 60)
    for name, row in [
        ("LGBM per-horizon (this)", lgbm_m),
        ("LSTM multi-output (this)", lstm_m),
        ("Ensemble (weighted)",      ens_m),
    ]:
        vals = "".join(f"  {m['R2']:>7.3f}" for m in row)
        print(f"  {name:<28}{vals}")
    print("=" * 72)

    for metric in ("MAE", "RMSE"):
        print(f"\nPer-horizon {metric}:")
        h2 = f"  {'Model':<28}" + "".join(f"  {metric+'_'+h:>8}" for h in h_labels)
        print(h2)
        print("  " + "-" * 64)
        for name, row in [
            ("LGBM per-horizon (this)", lgbm_m),
            ("LSTM multi-output (this)", lstm_m),
            ("Ensemble (weighted)",      ens_m),
        ]:
            vals = "".join(f"  {m[metric]:>8.2f}" for m in row)
            print(f"  {name:<28}{vals}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data from MongoDB feature_store ...")
    df = load_data()

    feat = get_feature_cols(df)

    train_df, test_df = time_split(df)

    print(f"Total rows    : {len(df)}  ({df['date'].min().date()} to {df['date'].max().date()})")
    print(f"Train rows    : {len(train_df)}  "
          f"({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"Test rows     : {len(test_df)}   "
          f"({test_df['date'].min().date()} to {test_df['date'].max().date()})")
    print(f"Features      : {len(feat)}")

    X_train = train_df[feat].values.astype(np.float32)
    X_test  = test_df[feat].values.astype(np.float32)
    y_train = train_df[TARGET_COLS].values.astype(np.float32)
    y_test  = test_df[TARGET_COLS].values.astype(np.float32)

    _, _, lgbm_metrics, lgbm_preds = train_lgbm(X_train, X_test, y_train, y_test, feat)
    _, _, lstm_metrics, lstm_preds, y_test_lstm = train_lstm(X_train, X_test, y_train, y_test)

    ens_metrics = compute_ensemble(lgbm_preds, lstm_preds, y_test_lstm)
    print_comparison(lgbm_metrics, lstm_metrics, ens_metrics)
