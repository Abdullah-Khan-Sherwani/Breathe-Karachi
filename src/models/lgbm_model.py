import numpy as np
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.per_horizon_wrapper import PerHorizonWrapper

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


def train_lgbm(X_train, y_train, X_test, y_test):
    """
    Train one LGBMRegressor per horizon wrapped in PerHorizonWrapper.
    Returns (model, scaler, metrics, hyperparameters).
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    models = []
    per_horizon_preds = []
    for h in range(y_train.shape[1]):
        m = LGBMRegressor(**LGBM_PARAMS)
        m.fit(X_tr, y_train[:, h])
        models.append(m)
        per_horizon_preds.append(m.predict(X_te))

    model = PerHorizonWrapper(models)

    preds = np.column_stack(per_horizon_preds)
    metrics = _compute_metrics(y_test, preds, per_horizon_preds)

    hyperparameters = {**LGBM_PARAMS, "strategy": "per_horizon"}

    return model, scaler, metrics, hyperparameters


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, per_horizon_preds: list) -> dict:
    avg_mae  = float(mean_absolute_error(y_true, y_pred))
    avg_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    avg_r2   = float(r2_score(y_true, y_pred))

    metrics: dict = {"MAE": avg_mae, "RMSE": avg_rmse, "R2": avg_r2}

    for i, preds_h in enumerate(per_horizon_preds, start=1):
        y_h = y_true[:, i - 1]
        metrics[f"MAE_d{i}"]  = float(mean_absolute_error(y_h, preds_h))
        metrics[f"RMSE_d{i}"] = float(np.sqrt(mean_squared_error(y_h, preds_h)))
        metrics[f"R2_d{i}"]   = float(r2_score(y_h, preds_h))

    return metrics
