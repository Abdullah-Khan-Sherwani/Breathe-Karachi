"""
Train LGBM (and optionally LSTM) on the hourly feature dataset.

Targets are daily-mean AQI windows — no autoregressive rollout:
  AQI_day1_mean = mean(AQI[t+1h .. t+24h])
  AQI_day2_mean = mean(AQI[t+25h .. t+48h])
  AQI_day3_mean = mean(AQI[t+49h .. t+72h])

Train/test split:
  Test  = last 30 days of hourly rows (720 h)
  Train = everything before that minus a 72-h buffer so train targets
          do not use AQI values from the test period.

Baseline comparison (daily model, same 30-day holdout):
  LGBM  R2_t1=0.773  R2_t2=0.284  R2_t3=0.120
  LSTM  R2_t1=0.633  R2_t2=0.290  R2_t3=0.185  (best recent run)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor

# Import column lists defined in the builder so they stay in sync
from scripts.build_hourly_dataset import FEATURE_COLS, TARGET_COLS

# ── Toggles ───────────────────────────────────────────────────────────────────
RUN_LSTM = True   # flip True to also train LSTM (takes ~10-20 min on CPU)

DATA_PATH = Path(__file__).parent.parent / "data" / "hourly_features.csv"

N_TEST   = 720   # 30 days x 24 hours
N_BUFFER = 72    # 3-day gap so train AQI_day3 targets don't touch test AQI values

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

LSTM_SEQ_LEN = 168   # 1-week lookback (vs 7 days in daily model)


# ── Helpers ───────────────────────────────────────────────────────────────────
def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "R2":   round(float(r2_score(y_true, y_pred)), 4),
        "MAE":  round(float(mean_absolute_error(y_true, y_pred)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
    }


def print_results(label: str, m1: dict, m2: dict, m3: dict) -> None:
    print(f"\n{label}")
    print(f"  {'Horizon':<12}  {'R2':>7}  {'MAE':>7}  {'RMSE':>7}")
    print("  " + "-" * 38)
    for name, m in [("day1 (t+24h)", m1), ("day2 (t+48h)", m2), ("day3 (t+72h)", m3)]:
        print(f"  {name:<12}  {m['R2']:>7.3f}  {m['MAE']:>7.1f}  {m['RMSE']:>7.1f}")


# ── Load & split ──────────────────────────────────────────────────────────────
def load_split():
    df = pd.read_csv(DATA_PATH, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    feat = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  WARNING: {len(missing)} FEATURE_COLS missing from CSV: {missing}")

    df = df.dropna(subset=feat + TARGET_COLS).reset_index(drop=True)
    n = len(df)

    train_df = df.iloc[: n - N_TEST - N_BUFFER]
    test_df  = df.iloc[n - N_TEST :]

    print(f"Dataset       : {n} rows  |  {len(feat)} features")
    print(f"Train rows    : {len(train_df)}  "
          f"({train_df['time'].iloc[0].date()} to {train_df['time'].iloc[-1].date()})")
    print(f"Test rows     : {len(test_df)}   "
          f"({test_df['time'].iloc[0].date()} to {test_df['time'].iloc[-1].date()})")

    X_train = train_df[feat].values.astype(np.float32)
    X_test  = test_df[feat].values.astype(np.float32)
    y_train = train_df[TARGET_COLS].values.astype(np.float32)
    y_test  = test_df[TARGET_COLS].values.astype(np.float32)

    return X_train, X_test, y_train, y_test, feat


# ── LGBM ──────────────────────────────────────────────────────────────────────
def train_lgbm(X_train, X_test, y_train, y_test, feat):
    print("\nTraining LGBM ...")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    models, preds = [], []
    for h, name in enumerate(TARGET_COLS):
        m = LGBMRegressor(**LGBM_PARAMS)
        m.fit(X_tr, y_train[:, h])
        models.append(m)
        preds.append(m.predict(X_te))
        print(f"  {name} done")

    m1 = metrics(y_test[:, 0], preds[0])
    m2 = metrics(y_test[:, 1], preds[1])
    m3 = metrics(y_test[:, 2], preds[2])
    print_results("LGBM (hourly features, daily-mean targets)", m1, m2, m3)

    # Feature importance — top 15 by day3 model
    imp = pd.Series(models[2].feature_importances_, index=feat).sort_values(ascending=False)
    print("\n  Top 15 features (day3 model):")
    for fname, score in imp.head(15).items():
        print(f"    {fname:<30}  {score}")

    return models, scaler, (m1, m2, m3)


# ── LSTM ──────────────────────────────────────────────────────────────────────
def train_lstm(X_train, X_test, y_train, y_test):
    import tensorflow as tf

    print(f"\nTraining LSTM (seq_len={LSTM_SEQ_LEN}) ...")

    x_sc = StandardScaler()
    y_sc = StandardScaler()
    X_tr_sc = x_sc.fit_transform(X_train)
    X_te_sc = x_sc.transform(X_test)
    y_tr_sc = y_sc.fit_transform(y_train)

    # Build sequences — allow test sequences to look back into train
    X_all = np.vstack([X_tr_sc, X_te_sc])
    y_all = np.vstack([y_tr_sc, y_sc.transform(y_test)])

    def build_seqs(X, y, seq_len):
        Xs, ys = [], []
        for i in range(len(X) - seq_len + 1):
            Xs.append(X[i : i + seq_len])
            ys.append(y[i + seq_len - 1])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    X_seq, y_seq = build_seqs(X_all, y_all, LSTM_SEQ_LEN)
    n_tr_seq = len(X_train) - LSTM_SEQ_LEN + 1
    X_s_tr, y_s_tr = X_seq[:n_tr_seq], y_seq[:n_tr_seq]
    X_s_te, y_s_te = X_seq[n_tr_seq:], y_seq[n_tr_seq:]

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(LSTM_SEQ_LEN, X_train.shape[1])),
        tf.keras.layers.LSTM(128, return_sequences=True,
                             kernel_regularizer=tf.keras.regularizers.L2(1e-4)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64, kernel_regularizer=tf.keras.regularizers.L2(1e-4)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="huber")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=5, min_lr=1e-5),
    ]
    model.fit(X_s_tr, y_s_tr, validation_split=0.1,
              epochs=50, batch_size=64, callbacks=callbacks, verbose=1)

    preds_sc = model.predict(X_s_te, verbose=0)
    preds    = y_sc.inverse_transform(preds_sc)
    y_true   = y_sc.inverse_transform(y_s_te)

    m1 = metrics(y_true[:, 0], preds[:, 0])
    m2 = metrics(y_true[:, 1], preds[:, 1])
    m3 = metrics(y_true[:, 2], preds[:, 2])
    print_results("LSTM (hourly features, daily-mean targets)", m1, m2, m3)
    return model, (x_sc, y_sc), (m1, m2, m3)


# ── Comparison table ──────────────────────────────────────────────────────────
def print_comparison(lgbm_metrics, lstm_metrics=None):
    print("\n" + "=" * 65)
    print("COMPARISON vs DAILY MODEL BASELINE (same 30-day holdout)")
    print(f"  {'Model':<30}  {'R2_d1':>7}  {'R2_d2':>7}  {'R2_d3':>7}")
    print("  " + "-" * 55)
    # Baselines
    print(f"  {'Daily LGBM (baseline)':<30}  {0.773:>7.3f}  {0.284:>7.3f}  {0.120:>7.3f}")
    print(f"  {'Daily LSTM best run (baseline)':<30}  {0.633:>7.3f}  {0.290:>7.3f}  {0.185:>7.3f}")
    # New results
    m1, m2, m3 = lgbm_metrics
    print(f"  {'Hourly LGBM (this run)':<30}  {m1['R2']:>7.3f}  {m2['R2']:>7.3f}  {m3['R2']:>7.3f}")
    if lstm_metrics:
        m1, m2, m3 = lstm_metrics
        print(f"  {'Hourly LSTM (this run)':<30}  {m1['R2']:>7.3f}  {m2['R2']:>7.3f}  {m3['R2']:>7.3f}")
    print("=" * 65)
    print("Note: daily baselines trained on 1,234 rows; hourly models on ~32k rows.")
    print("Targets are comparable: daily mean AQI for the next 1/2/3 days.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feat = load_split()

    _, _, lgbm_m = train_lgbm(X_train, X_test, y_train, y_test, feat)

    lstm_m = None
    if RUN_LSTM:
        _, _, lstm_m = train_lstm(X_train, X_test, y_train, y_test)

    print_comparison(lgbm_m, lstm_m)
