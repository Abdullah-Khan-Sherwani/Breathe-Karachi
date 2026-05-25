import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEQ_LEN = 7


def _build_sequences(X: np.ndarray, y: np.ndarray):
    """Slide a window of SEQ_LEN rows over X, aligning each window's target to y[i + SEQ_LEN - 1]."""
    Xs, ys = [], []
    for i in range(len(X) - SEQ_LEN + 1):
        Xs.append(X[i : i + SEQ_LEN])
        ys.append(y[i + SEQ_LEN - 1])
    return np.array(Xs), np.array(ys)


def train_lstm(X_train, y_train, X_test, y_test):
    """
    Train LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(3).
    Returns (model, scaler, metrics, hyperparameters).
    """
    import tensorflow as tf

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    # Combine to build sequences spanning the train/test boundary, then split back
    X_all = np.vstack([X_tr_sc, X_te_sc])
    y_all = np.vstack([np.array(y_train), np.array(y_test)])

    X_seq, y_seq = _build_sequences(X_all, y_all)

    # The first len(X_train) - SEQ_LEN + 1 sequences are fully within train data
    n_train_seq = len(X_train) - SEQ_LEN + 1
    X_s_tr, y_s_tr = X_seq[:n_train_seq], y_seq[:n_train_seq]
    X_s_te, y_s_te = X_seq[n_train_seq:], y_seq[n_train_seq:]

    n_features = X_tr_sc.shape[1]
    model = _build_model(n_features)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )
    model.fit(
        X_s_tr, y_s_tr,
        validation_split=0.1,
        epochs=100,
        batch_size=16,
        callbacks=[early_stop],
        verbose=0,
    )

    if len(X_s_te) == 0:
        # Fallback when test set is too small for even one sequence
        preds = model.predict(X_s_tr[-1:], verbose=0)
        metrics = _compute_metrics(y_s_tr[-1:], preds)
    else:
        preds = model.predict(X_s_te, verbose=0)
        metrics = _compute_metrics(y_s_te, preds)

    hyperparameters = {
        "seq_len": SEQ_LEN,
        "units_1": 64,
        "units_2": 32,
        "dropout": 0.2,
        "epochs": 100,
        "patience": 20,
        "batch_size": 16,
    }

    return model, scaler, metrics, hyperparameters


def _build_model(n_features: int):
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(SEQ_LEN, n_features)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def _compute_metrics(y_true, y_pred) -> dict:
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}
