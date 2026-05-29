"""
LSTM Architecture Experiment — exploration only.
Trains multiple LSTM variants on the Karachi AQI feature store and prints
per-horizon metrics (MAE, RMSE, R2) plus a final ranking table.

Nothing is saved to disk or MongoDB.

Usage:
    python scripts/lstm_architecture_experiment.py           # full run (epochs=100)
    python scripts/lstm_architecture_experiment.py --quick   # fast run (epochs=20, skips SEQ_LEN=14)
"""

import argparse
import os
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — allow importing from project root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

# ---------------------------------------------------------------------------
# Seeding — must happen before TF and numpy are used
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

import numpy as np
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

# Suppress TF info/warning noise
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Production data pipeline (imported, not reimplemented)
# ---------------------------------------------------------------------------
from src.train import load_data, get_feature_cols, time_split, TARGET_COLS

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="LSTM architecture benchmark — no models saved.")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Set epochs=20 and skip the SEQ_LEN=14 experiment for fast iteration.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Sequence builder (mirrors production lstm_model._build_sequences exactly)
# ---------------------------------------------------------------------------

def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Slide a window of seq_len rows over X, align each window's target."""
    Xs, ys = [], []
    for i in range(len(X) - seq_len + 1):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len - 1])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(seq_len: int = 7):
    """
    Load feature store, apply time split (last 30 days = test),
    scale X and y independently, build sequences.
    Returns (X_s_tr, y_s_tr, X_s_te, y_s_te, n_features).
    """
    df = load_data()
    feat_cols = get_feature_cols(df)
    train_df, test_df = time_split(df, test_days=30)

    X_train = train_df[feat_cols].values
    y_train = train_df[TARGET_COLS].values
    X_test  = test_df[feat_cols].values
    y_test  = test_df[TARGET_COLS].values

    x_sc = StandardScaler()
    y_sc = StandardScaler()

    X_tr_sc = x_sc.fit_transform(X_train)
    X_te_sc = x_sc.transform(X_test)
    y_tr_sc = y_sc.fit_transform(y_train)
    y_te_sc = y_sc.transform(y_test)

    # Build sequences over the combined scaled array, then re-split —
    # mirrors the exact production approach in train_lstm()
    X_all = np.vstack([X_tr_sc, X_te_sc])
    y_all = np.vstack([y_tr_sc, y_te_sc])

    X_seq, y_seq = build_sequences(X_all, y_all, seq_len)

    n_train_seq = len(X_train) - seq_len + 1
    X_s_tr, y_s_tr = X_seq[:n_train_seq], y_seq[:n_train_seq]
    X_s_te, y_s_te = X_seq[n_train_seq:], y_seq[n_train_seq:]

    return X_s_tr, y_s_tr, X_s_te, y_s_te, X_tr_sc.shape[1], y_sc


# ---------------------------------------------------------------------------
# Callbacks factory
# ---------------------------------------------------------------------------

def make_callbacks(max_epochs: int):
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=0,
    )
    return [early_stop, reduce_lr]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_per_horizon(y_true: np.ndarray, y_pred: np.ndarray):
    """Return list of (mae, rmse, r2) tuples, one per output horizon."""
    results = []
    for i in range(y_pred.shape[1]):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        mae  = float(mean_absolute_error(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        r2   = float(r2_score(yt, yp))
        results.append((mae, rmse, r2))
    return results


# ---------------------------------------------------------------------------
# Training runner
# ---------------------------------------------------------------------------

def run_experiment(
    name: str,
    build_fn,            # callable(n_features, seq_len) -> tf.keras.Model
    X_s_tr, y_s_tr,
    X_s_te, y_s_te,
    y_sc: StandardScaler,
    max_epochs: int,
    seq_len: int,
):
    """Train one model, evaluate on the held-out test sequences, print metrics."""
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)

    n_features = X_s_tr.shape[2]
    model = build_fn(n_features, seq_len)

    callbacks = make_callbacks(max_epochs)
    model.fit(
        X_s_tr, y_s_tr,
        validation_split=0.1,
        epochs=max_epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
    )

    # Predict on test set (or fall back to last training sample if test is empty)
    if len(X_s_te) == 0:
        preds_sc = model.predict(X_s_tr[-1:], verbose=0)
        y_true_sc = y_s_tr[-1:]
    else:
        preds_sc = model.predict(X_s_te, verbose=0)
        y_true_sc = y_s_te

    # Inverse-transform back to AQI scale
    preds   = y_sc.inverse_transform(preds_sc)
    y_true  = y_sc.inverse_transform(y_true_sc)

    per_horizon = compute_per_horizon(y_true, preds)

    n_params = model.count_params()

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print(f"\n=== {name} ===")
    print(f"Params: {n_params:,}")
    for i, (mae, rmse, r2) in enumerate(per_horizon, start=1):
        print(f"  d{i} MAE={mae:.1f}  RMSE={rmse:.1f}  R2={r2:.2f}")

    avg_mae  = float(np.mean([m for m, _, _ in per_horizon]))
    avg_rmse = float(np.mean([r for _, r, _ in per_horizon]))
    avg_r2   = float(np.mean([r for _, _, r in per_horizon]))
    print(f"  Avg MAE={avg_mae:.1f}  Avg RMSE={avg_rmse:.1f}  Avg R2={avg_r2:.2f}")

    return {
        "name":      name,
        "params":    n_params,
        "per_horizon": per_horizon,
        "avg_mae":   avg_mae,
        "avg_rmse":  avg_rmse,
        "avg_r2":    avg_r2,
    }


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_baseline(n_features: int, seq_len: int) -> tf.keras.Model:
    """
    Exact reproduction of production _build_model() from src/models/lstm_model.py:
      LSTM(64, return_sequences=True, L2(1e-3))
      Dropout(0.4)
      LSTM(32, L2(1e-3))
      Dropout(0.4)
      Dense(4)
      Compiled with Adam + Huber loss
    """
    from tensorflow.keras.regularizers import L2

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(seq_len, n_features)),
        tf.keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=L2(1e-3)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.LSTM(32, kernel_regularizer=L2(1e-3)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(4),
    ], name="baseline")
    model.compile(optimizer="adam", loss="huber")
    return model


def build_deeper_lstm(n_features: int, seq_len: int) -> tf.keras.Model:
    """3 stacked LSTM layers (128 -> 64 -> 32), Dropout(0.3) between each, Dense(4)."""
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(seq_len, n_features)),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(4),
    ], name="deeper_lstm")
    model.compile(optimizer="adam", loss="huber")
    return model


def build_bilstm(n_features: int, seq_len: int) -> tf.keras.Model:
    """Bidirectional LSTM: BiLSTM(64) + BiLSTM(32), Dropout(0.2), Dense(4)."""
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(seq_len, n_features)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(4),
    ], name="bilstm")
    model.compile(optimizer="adam", loss="huber")
    return model


class DotProductAttention(tf.keras.layers.Layer):
    """
    Simple trainable scaled dot-product self-attention over time steps.
    Input shape: (batch, timesteps, features)
    Output shape: (batch, timesteps, features)  — then pooled externally.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        d_model = input_shape[-1]
        # Single query vector that attends over all timesteps
        self.W_q = self.add_weight(
            name="W_q",
            shape=(d_model, d_model),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.W_k = self.add_weight(
            name="W_k",
            shape=(d_model, d_model),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.W_v = self.add_weight(
            name="W_v",
            shape=(d_model, d_model),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.scale = tf.math.sqrt(tf.cast(d_model, tf.float32))
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, T, d)
        Q = tf.matmul(inputs, self.W_q)   # (batch, T, d)
        K = tf.matmul(inputs, self.W_k)   # (batch, T, d)
        V = tf.matmul(inputs, self.W_v)   # (batch, T, d)

        # Scaled dot-product scores: (batch, T, T)
        scores = tf.matmul(Q, K, transpose_b=True) / self.scale
        weights = tf.nn.softmax(scores, axis=-1)

        # Weighted sum of values: (batch, T, d)
        attended = tf.matmul(weights, V)
        return attended

    def get_config(self):
        return super().get_config()


def build_lstm_attention(n_features: int, seq_len: int) -> tf.keras.Model:
    """
    LSTM(64, return_sequences=True) -> DotProductAttention -> GlobalAveragePooling1D
    -> Dense(32, relu) -> Dense(4)
    """
    inp = tf.keras.Input(shape=(seq_len, n_features), name="input")
    x   = tf.keras.layers.LSTM(64, return_sequences=True, name="lstm_64")(inp)
    x   = DotProductAttention(name="dot_product_attention")(x)
    x   = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    x   = tf.keras.layers.Dense(32, activation="relu", name="dense_32")(x)
    out = tf.keras.layers.Dense(4, name="output")(x)
    model = tf.keras.Model(inputs=inp, outputs=out, name="lstm_attention")
    model.compile(optimizer="adam", loss="huber")
    return model


def build_cnn_lstm(n_features: int, seq_len: int) -> tf.keras.Model:
    """
    Conv1D(64, kernel_size=3, relu) -> MaxPooling1D(2) -> LSTM(64) -> Dense(4).
    MaxPooling halves the sequence length; kernel_size=3 requires seq_len >= 3.
    """
    if seq_len < 3:
        raise ValueError(f"CNN-LSTM requires seq_len >= 3, got {seq_len}")

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(seq_len, n_features)),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation="relu", padding="causal"),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(4),
    ], name="cnn_lstm")
    model.compile(optimizer="adam", loss="huber")
    return model


def build_residual_lstm(n_features: int, seq_len: int) -> tf.keras.Model:
    """
    Residual LSTM:
      Block 1: LSTM(64, return_sequences=True)
      Block 2: LSTM(64, return_sequences=True)
      Residual add (block1_input projected to 64 dims) + block2_output
      GlobalAveragePooling -> Dense(4)

    The input projection (Dense(64) applied along time axis) matches dims so
    the residual add is valid even when n_features != 64.
    """
    inp = tf.keras.Input(shape=(seq_len, n_features), name="input")

    # Project input to 64 dims for the residual connection
    inp_proj = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(64), name="input_projection"
    )(inp)

    # Block 1
    x1 = tf.keras.layers.LSTM(64, return_sequences=True, name="lstm_block1")(inp)

    # Block 2
    x2 = tf.keras.layers.LSTM(64, return_sequences=True, name="lstm_block2")(x1)

    # Residual: add projected input to block 2 output
    residual = tf.keras.layers.Add(name="residual_add")([inp_proj, x2])

    # Collapse time dimension
    pooled = tf.keras.layers.GlobalAveragePooling1D(name="gap")(residual)
    out    = tf.keras.layers.Dense(4, name="output")(pooled)

    model = tf.keras.Model(inputs=inp, outputs=out, name="residual_lstm")
    model.compile(optimizer="adam", loss="huber")
    return model


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(results: list):
    ranked = sorted(results, key=lambda r: r["avg_r2"], reverse=True)

    print("\n" + "=" * 72)
    print("SUMMARY — ranked by Avg R2 (descending)")
    print("=" * 72)
    header = f"{'Rank':<5} {'Experiment':<30} {'Params':>10}  {'Avg MAE':>8}  {'Avg RMSE':>9}  {'Avg R2':>7}"
    print(header)
    print("-" * 72)
    for rank, r in enumerate(ranked, start=1):
        marker = "  <-- BEST" if rank == 1 else ""
        print(
            f"{rank:<5} {r['name']:<30} {r['params']:>10,}  "
            f"{r['avg_mae']:>8.1f}  {r['avg_rmse']:>9.1f}  {r['avg_r2']:>7.2f}"
            f"{marker}"
        )
    print("=" * 72)

    best = ranked[0]
    print(
        f"\nRECOMMENDATION: '{best['name']}' achieves the highest average R2 of "
        f"{best['avg_r2']:.2f} (Avg MAE={best['avg_mae']:.1f}, "
        f"Avg RMSE={best['avg_rmse']:.1f}) across all four AQI horizons. "
        f"Consider adopting this architecture as the production model, "
        f"verifying that training time and latency remain within acceptable bounds."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    max_epochs = 20 if args.quick else 100

    mode_label = "QUICK (epochs=20)" if args.quick else "FULL (epochs=100)"
    print(f"{'=' * 60}")
    print(f"  Karachi AQI — LSTM Architecture Experiment")
    print(f"  Mode  : {mode_label}")
    print(f"  Seed  : {SEED}")
    print(f"{'=' * 60}")

    # ------------------------------------------------------------------
    # Load and prepare data (SEQ_LEN=7, production split)
    # ------------------------------------------------------------------
    print("\nLoading data from MongoDB feature store ...")
    X_s_tr_7, y_s_tr_7, X_s_te_7, y_s_te_7, n_features_7, y_sc_7 = prepare_data(seq_len=7)
    print(
        f"  Train sequences : {len(X_s_tr_7):,}  |  "
        f"Test sequences : {len(X_s_te_7):,}  |  "
        f"Features : {n_features_7}"
    )

    results = []

    # ------------------------------------------------------------------
    # 0. Baseline (production architecture)
    # ------------------------------------------------------------------
    results.append(run_experiment(
        name="Baseline (production LSTM)",
        build_fn=build_baseline,
        X_s_tr=X_s_tr_7, y_s_tr=y_s_tr_7,
        X_s_te=X_s_te_7, y_s_te=y_s_te_7,
        y_sc=y_sc_7,
        max_epochs=max_epochs,
        seq_len=7,
    ))

    # ------------------------------------------------------------------
    # 1. Deeper LSTM
    # ------------------------------------------------------------------
    results.append(run_experiment(
        name="Deeper LSTM (128->64->32)",
        build_fn=build_deeper_lstm,
        X_s_tr=X_s_tr_7, y_s_tr=y_s_tr_7,
        X_s_te=X_s_te_7, y_s_te=y_s_te_7,
        y_sc=y_sc_7,
        max_epochs=max_epochs,
        seq_len=7,
    ))

    # ------------------------------------------------------------------
    # 2. Bidirectional LSTM
    # ------------------------------------------------------------------
    results.append(run_experiment(
        name="Bidirectional LSTM (64+32)",
        build_fn=build_bilstm,
        X_s_tr=X_s_tr_7, y_s_tr=y_s_tr_7,
        X_s_te=X_s_te_7, y_s_te=y_s_te_7,
        y_sc=y_sc_7,
        max_epochs=max_epochs,
        seq_len=7,
    ))

    # ------------------------------------------------------------------
    # 3. LSTM + Attention
    # ------------------------------------------------------------------
    results.append(run_experiment(
        name="LSTM + Dot-Product Attention",
        build_fn=build_lstm_attention,
        X_s_tr=X_s_tr_7, y_s_tr=y_s_tr_7,
        X_s_te=X_s_te_7, y_s_te=y_s_te_7,
        y_sc=y_sc_7,
        max_epochs=max_epochs,
        seq_len=7,
    ))

    # ------------------------------------------------------------------
    # 4. CNN-LSTM
    # ------------------------------------------------------------------
    results.append(run_experiment(
        name="CNN-LSTM (Conv64->Pool->LSTM64)",
        build_fn=build_cnn_lstm,
        X_s_tr=X_s_tr_7, y_s_tr=y_s_tr_7,
        X_s_te=X_s_te_7, y_s_te=y_s_te_7,
        y_sc=y_sc_7,
        max_epochs=max_epochs,
        seq_len=7,
    ))

    # ------------------------------------------------------------------
    # 5. Residual LSTM
    # ------------------------------------------------------------------
    results.append(run_experiment(
        name="Residual LSTM (64+64+skip)",
        build_fn=build_residual_lstm,
        X_s_tr=X_s_tr_7, y_s_tr=y_s_tr_7,
        X_s_te=X_s_te_7, y_s_te=y_s_te_7,
        y_sc=y_sc_7,
        max_epochs=max_epochs,
        seq_len=7,
    ))

    # ------------------------------------------------------------------
    # 6. Longer sequence (SEQ_LEN=14) with best architecture so far
    # ------------------------------------------------------------------
    if not args.quick:
        best_so_far = max(results, key=lambda r: r["avg_r2"])
        best_name   = best_so_far["name"]
        print(f"\nBest architecture so far: '{best_name}'  (Avg R2={best_so_far['avg_r2']:.2f})")
        print("Re-training with SEQ_LEN=14 ...")

        # Map experiment name back to its builder
        builder_map = {
            "Baseline (production LSTM)":       build_baseline,
            "Deeper LSTM (128->64->32)":          build_deeper_lstm,
            "Bidirectional LSTM (64+32)":       build_bilstm,
            "LSTM + Dot-Product Attention":     build_lstm_attention,
            "CNN-LSTM (Conv64->Pool->LSTM64)":   build_cnn_lstm,
            "Residual LSTM (64+64+skip)":       build_residual_lstm,
        }
        best_builder = builder_map.get(best_name, build_baseline)

        print("\nLoading data with SEQ_LEN=14 ...")
        X_s_tr_14, y_s_tr_14, X_s_te_14, y_s_te_14, n_features_14, y_sc_14 = prepare_data(seq_len=14)
        print(
            f"  Train sequences : {len(X_s_tr_14):,}  |  "
            f"Test sequences : {len(X_s_te_14):,}  |  "
            f"Features : {n_features_14}"
        )

        results.append(run_experiment(
            name=f"SEQ_LEN=14 ({best_name})",
            build_fn=best_builder,
            X_s_tr=X_s_tr_14, y_s_tr=y_s_tr_14,
            X_s_te=X_s_te_14, y_s_te=y_s_te_14,
            y_sc=y_sc_14,
            max_epochs=max_epochs,
            seq_len=14,
        ))
    else:
        print("\n[--quick] Skipping SEQ_LEN=14 experiment.")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print_summary(results)


if __name__ == "__main__":
    main()
