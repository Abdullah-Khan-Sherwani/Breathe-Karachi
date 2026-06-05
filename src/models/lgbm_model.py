import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.per_horizon_wrapper import PerHorizonWrapper

# Regularized to stop the severe overfitting of the previous config
# (n_estimators=500, max_depth=7, num_leaves=63, NO regularization, and a dead
# `subsample` that never applied because subsample_freq defaulted to 0 →
# train R²≈0.999 / test R²≈0.60, test/train RMSE ratio ≈11x).
#
# n_estimators is a CEILING, not a fixed count: both trainers use early stopping
# to select the real number of trees (~300 in practice), so 2000 is never reached.
LGBM_PARAMS = dict(
    n_estimators=2000,        # early-stopping ceiling, not a fixed count
    learning_rate=0.03,       # was 0.05 — slower, smoother fit
    max_depth=4,              # was 7 — limits interaction depth
    num_leaves=15,            # was 63 — fewer leaves, less memorization
    min_child_samples=30,     # NEW — floor on leaf size
    subsample=0.8,
    subsample_freq=1,         # NEW — makes bagging actually apply (was a no-op)
    colsample_bytree=0.8,
    reg_alpha=0.1,            # NEW — L1
    reg_lambda=1.0,           # NEW — L2
    random_state=42,
    verbose=-1,
)
EARLY_STOPPING_ROUNDS = 50


def _val_tail(n: int) -> int:
    """Number of rows held out at the END of the time-ordered training data to
    monitor early stopping. Scales with data size; floored at 30, capped at 150."""
    return max(30, min(150, int(0.15 * n)))


def train_lgbm_full(X: np.ndarray, y: np.ndarray):
    """
    Retrain on the complete labeled dataset for deployment. With no external
    holdout, the tree count is chosen data-drivenly in two stages per horizon:
      1. early-stop on a time-ordered validation tail to discover the optimal
         number of rounds,
      2. refit on ALL rows with that count (scaled up for the held-out tail) so
         the deployed model uses every observation, including the most recent days.
    Returns (model, scaler).
    """
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    y = np.asarray(y)
    n = len(X_sc)
    vt = _val_tail(n)
    models = []
    for h in range(y.shape[1]):
        # Stage 1 — discover the early-stopping iteration on a recent validation tail
        finder = LGBMRegressor(**LGBM_PARAMS)
        finder.fit(
            X_sc[:-vt], y[:-vt, h],
            eval_set=[(X_sc[-vt:], y[-vt:, h])],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                       lgb.log_evaluation(0)],
        )
        best = finder.best_iteration_ or LGBM_PARAMS["n_estimators"]
        # Stage 2 — refit on all rows; scale the count up for the rows stage 1 held out
        n_full = max(50, int(round(best * n / (n - vt))))
        m = LGBMRegressor(**{**LGBM_PARAMS, "n_estimators": n_full})
        m.fit(X_sc, y[:, h])
        models.append(m)
    return PerHorizonWrapper(models), scaler


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

    n = len(X_tr)
    vt = _val_tail(n)
    X_fit, X_val = X_tr[:-vt], X_tr[-vt:]

    models = []
    per_horizon_preds = []
    best_iters = []
    for h in range(y_train.shape[1]):
        m = LGBMRegressor(**LGBM_PARAMS)
        m.fit(
            X_fit, y_train[:-vt, h],
            eval_set=[(X_val, y_train[-vt:, h])],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                       lgb.log_evaluation(0)],
        )
        models.append(m)
        best_iters.append(int(m.best_iteration_ or LGBM_PARAMS["n_estimators"]))
        per_horizon_preds.append(m.predict(X_te))   # predict() uses best_iteration_

    model = PerHorizonWrapper(models)

    preds = np.column_stack(per_horizon_preds)
    metrics = _compute_metrics(y_test, preds, per_horizon_preds)

    hyperparameters = {**LGBM_PARAMS, "strategy": "per_horizon_early_stop",
                       "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                       "best_iterations": best_iters}

    return model, scaler, metrics, hyperparameters


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, per_horizon_preds: list) -> dict:
    avg_mae  = float(mean_absolute_error(y_true, y_pred))
    avg_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    avg_r2   = float(r2_score(y_true, y_pred))

    metrics: dict = {"MAE": avg_mae, "RMSE": avg_rmse, "R2": avg_r2, "_preds": y_pred.tolist()}

    for i, preds_h in enumerate(per_horizon_preds, start=1):
        y_h = y_true[:, i - 1]
        metrics[f"MAE_d{i}"]  = float(mean_absolute_error(y_h, preds_h))
        metrics[f"RMSE_d{i}"] = float(np.sqrt(mean_squared_error(y_h, preds_h)))
        metrics[f"R2_d{i}"]   = float(r2_score(y_h, preds_h))

    return metrics
