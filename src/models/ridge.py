import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_ridge(X_train, y_train, X_test, y_test):
    """
    Train MultiOutputRegressor(Ridge) on pre-scaled data.
    Returns (model, scaler, metrics, hyperparameters).
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    model = MultiOutputRegressor(Ridge(alpha=1.0))
    model.fit(X_tr, y_train)

    preds = model.predict(X_te)
    metrics = _compute_metrics(y_test, preds)
    hyperparameters = {"alpha": 1.0}

    return model, scaler, metrics, hyperparameters


def train_ridge_full(X: np.ndarray, y: np.ndarray):
    """Retrain on the complete labeled dataset (no holdout). Returns (model, scaler)."""
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    model = MultiOutputRegressor(Ridge(alpha=1.0))
    model.fit(X_sc, y)
    return model, scaler


def _compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
    for i in range(y_true.shape[1]):
        h = i + 1
        metrics[f"MAE_d{h}"]  = float(mean_absolute_error(y_true[:, i], y_pred[:, i]))
        metrics[f"RMSE_d{h}"] = float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
        metrics[f"R2_d{h}"]   = float(r2_score(y_true[:, i], y_pred[:, i]))
    return metrics
