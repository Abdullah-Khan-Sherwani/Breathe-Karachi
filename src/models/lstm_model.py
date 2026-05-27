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


def train_lstm_full(X: np.ndarray, y: np.ndarray):
    """Retrain on the complete labeled dataset (no holdout). Returns (model, (x_sc, y_sc))."""
    import tensorflow as tf
    tf.keras.backend.clear_session()

    x_sc = StandardScaler()
    y_sc = StandardScaler()
    X_sc  = x_sc.fit_transform(X)
    y_arr = np.asarray(y)
    y_scl = y_sc.fit_transform(y_arr)

    X_seq, y_seq = _build_sequences(X_sc, y_scl)

    model = _build_model(X_sc.shape[1])
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=25, restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5
    )
    model.fit(
        X_seq, y_seq,
        validation_split=0.1,
        epochs=150,
        batch_size=16,
        callbacks=[early_stop, reduce_lr],
        verbose=0,
    )
    return model, (x_sc, y_sc)


def train_lstm(X_train, y_train, X_test, y_test):
    """
    Train LSTM(64) → Dropout(0.4) → LSTM(32) → Dropout(0.4) → Dense(4).
    Uses independent x_sc and y_sc scalers for better convergence.
    Returns (model, (x_sc, y_sc), metrics, hyperparameters).
    """
    import tensorflow as tf
    from tensorflow.keras.regularizers import L2

    tf.keras.backend.clear_session()

    x_sc = StandardScaler()
    y_sc = StandardScaler()

    X_tr_sc = x_sc.fit_transform(X_train)
    X_te_sc = x_sc.transform(X_test)

    y_train_arr = np.asarray(y_train)
    y_test_arr  = np.asarray(y_test)

    y_tr_sc = y_sc.fit_transform(y_train_arr)
    y_te_sc = y_sc.transform(y_test_arr)

    X_all  = np.vstack([X_tr_sc, X_te_sc])
    y_all  = np.vstack([y_tr_sc, y_te_sc])

    X_seq, y_seq = _build_sequences(X_all, y_all)

    n_train_seq = len(X_train) - SEQ_LEN + 1
    X_s_tr, y_s_tr = X_seq[:n_train_seq], y_seq[:n_train_seq]
    X_s_te, y_s_te = X_seq[n_train_seq:], y_seq[n_train_seq:]

    n_features = X_tr_sc.shape[1]
    model = _build_model(n_features)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=25, restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5
    )
    model.fit(
        X_s_tr, y_s_tr,
        validation_split=0.1,
        epochs=150,
        batch_size=16,
        callbacks=[early_stop, reduce_lr],
        verbose=0,
    )

    if len(X_s_te) == 0:
        preds_sc = model.predict(X_s_tr[-1:], verbose=0)
        preds    = y_sc.inverse_transform(preds_sc)
        y_true   = y_sc.inverse_transform(y_s_tr[-1:])
    else:
        preds_sc = model.predict(X_s_te, verbose=0)
        preds    = y_sc.inverse_transform(preds_sc)
        y_true   = y_sc.inverse_transform(y_s_te)

    metrics = _compute_metrics(y_true, preds)

    hyperparameters = {
        "seq_len":    SEQ_LEN,
        "units_1":    64,
        "units_2":    32,
        "dropout":    0.4,
        "l2_reg":     1e-3,
        "loss":       "huber",
        "epochs":     150,
        "patience":   25,
        "batch_size": 16,
    }

    return model, (x_sc, y_sc), metrics, hyperparameters


def _build_model(n_features: int):
    import tensorflow as tf
    from tensorflow.keras.regularizers import L2

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(SEQ_LEN, n_features)),
        tf.keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=L2(1e-3)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.LSTM(32, kernel_regularizer=L2(1e-3)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(4),
    ])
    model.compile(optimizer="adam", loss="huber")
    return model


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))

    metrics: dict = {"MAE": mae, "RMSE": rmse, "R2": r2}

    for i in range(y_pred.shape[1]):
        y_h    = y_true[:, i]
        preds_h = y_pred[:, i]
        metrics[f"MAE_d{i + 1}"]  = float(mean_absolute_error(y_h, preds_h))
        metrics[f"RMSE_d{i + 1}"] = float(np.sqrt(mean_squared_error(y_h, preds_h)))
        metrics[f"R2_d{i + 1}"]   = float(r2_score(y_h, preds_h))

    return metrics
