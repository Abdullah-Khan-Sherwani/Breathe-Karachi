import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)


def train_rf(X_train, y_train, X_test, y_test):
    """
    Train RandomForestRegressor (multi-output natively) on pre-split data.
    Returns (model, scaler, metrics, hyperparameters).
    Interface is identical to train_lgbm / train_ridge for easy comparison.
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_tr, np.asarray(y_train))

    preds = model.predict(X_te)
    metrics = _compute_metrics(np.asarray(y_test), preds)
    return model, scaler, metrics, RF_PARAMS


def train_rf_full(X: np.ndarray, y: np.ndarray):
    """Retrain on the complete labeled dataset (no holdout). Returns (model, scaler)."""
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_sc, np.asarray(y))
    return model, scaler


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2, "_preds": y_pred.tolist()}
    for i in range(y_true.shape[1]):
        h = i + 1
        metrics[f"MAE_d{h}"]  = float(mean_absolute_error(y_true[:, i], y_pred[:, i]))
        metrics[f"RMSE_d{h}"] = float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
        metrics[f"R2_d{h}"]   = float(r2_score(y_true[:, i], y_pred[:, i]))
    return metrics
