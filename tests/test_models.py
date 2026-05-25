"""
Tests for src/models/ — unit tests for each trainer using synthetic data.
No MongoDB, no live API calls.
Run with: python -m pytest tests/test_models.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.models.ridge import train_ridge
from src.models.lgbm_model import train_lgbm
from src.models.lstm_model import train_lstm, SEQ_LEN


RNG = np.random.default_rng(0)
N_FEAT = 10


def _make_dataset(n_train: int = 80, n_test: int = 20):
    X_tr = RNG.standard_normal((n_train, N_FEAT)).astype(np.float32)
    y_tr = RNG.uniform(50, 200, (n_train, 3)).astype(np.float32)
    X_te = RNG.standard_normal((n_test,  N_FEAT)).astype(np.float32)
    y_te = RNG.uniform(50, 200, (n_test,  3)).astype(np.float32)
    return X_tr, y_tr, X_te, y_te


# ---------------------------------------------------------------------------
# Ridge
# ---------------------------------------------------------------------------

class TestRidge:
    def test_returns_four_values(self):
        result = train_ridge(*_make_dataset())
        assert len(result) == 4

    def test_metrics_keys(self):
        _, _, metrics, _ = train_ridge(*_make_dataset())
        assert {"MAE", "RMSE", "R2"} <= set(metrics.keys())

    def test_metrics_are_finite(self):
        _, _, metrics, _ = train_ridge(*_make_dataset())
        for v in metrics.values():
            assert np.isfinite(v)

    def test_rmse_nonnegative(self):
        _, _, metrics, _ = train_ridge(*_make_dataset())
        assert metrics["RMSE"] >= 0

    def test_predict_shape(self):
        model, scaler, _, _ = train_ridge(*_make_dataset())
        X_new = RNG.standard_normal((5, N_FEAT))
        preds = model.predict(scaler.transform(X_new))
        assert preds.shape == (5, 3)

    def test_scaler_transforms_correctly(self):
        _, scaler, _, _ = train_ridge(*_make_dataset())
        X = RNG.standard_normal((10, N_FEAT))
        Xsc = scaler.transform(X)
        assert Xsc.shape == X.shape

    def test_hyperparameters_present(self):
        _, _, _, hparams = train_ridge(*_make_dataset())
        assert "alpha" in hparams

    def test_single_sample_test(self):
        X_tr, y_tr, _, _ = _make_dataset()
        X_te = X_tr[:1]
        y_te = y_tr[:1]
        _, _, metrics, _ = train_ridge(X_tr, y_tr, X_te, y_te)
        assert np.isfinite(metrics["MAE"])


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

class TestLGBM:
    def test_returns_four_values(self):
        assert len(train_lgbm(*_make_dataset())) == 4

    def test_metrics_keys(self):
        _, _, metrics, _ = train_lgbm(*_make_dataset())
        assert {"MAE", "RMSE", "R2"} <= set(metrics.keys())

    def test_metrics_finite(self):
        _, _, metrics, _ = train_lgbm(*_make_dataset())
        for v in metrics.values():
            assert np.isfinite(v)

    def test_rmse_nonnegative(self):
        _, _, metrics, _ = train_lgbm(*_make_dataset())
        assert metrics["RMSE"] >= 0

    def test_predict_shape(self):
        model, scaler, _, _ = train_lgbm(*_make_dataset())
        preds = model.predict(scaler.transform(RNG.standard_normal((5, N_FEAT))))
        assert preds.shape == (5, 3)

    def test_hyperparameters_present(self):
        _, _, _, hparams = train_lgbm(*_make_dataset())
        assert "n_estimators" in hparams and "learning_rate" in hparams

    def test_better_than_random_on_linear_data(self):
        """LGBM should fit a simple linear relationship."""
        X = RNG.standard_normal((200, 5)).astype(np.float32)
        y = (X[:, :3] * 10 + 100).astype(np.float32)  # perfectly linear
        _, _, metrics, _ = train_lgbm(X[:150], y[:150], X[150:], y[150:])
        assert metrics["R2"] > 0.5


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

class TestLSTM:
    def test_returns_four_values(self):
        assert len(train_lstm(*_make_dataset())) == 4

    def test_metrics_keys(self):
        _, _, metrics, _ = train_lstm(*_make_dataset())
        assert {"MAE", "RMSE", "R2"} <= set(metrics.keys())

    def test_metrics_finite(self):
        _, _, metrics, _ = train_lstm(*_make_dataset())
        for v in metrics.values():
            assert np.isfinite(v)

    def test_rmse_nonnegative(self):
        _, _, metrics, _ = train_lstm(*_make_dataset())
        assert metrics["RMSE"] >= 0

    def test_predict_shape_single_sequence(self):
        model, scaler, _, _ = train_lstm(*_make_dataset())
        seq = RNG.standard_normal((SEQ_LEN, N_FEAT)).astype(np.float32)
        seq_sc = scaler.transform(seq)
        preds = model.predict(seq_sc[np.newaxis, ...], verbose=0)
        assert preds.shape == (1, 3)

    def test_hyperparameters_contain_seq_len(self):
        _, _, _, hparams = train_lstm(*_make_dataset())
        assert hparams["seq_len"] == SEQ_LEN

    def test_small_test_set_fallback(self):
        """Test split smaller than SEQ_LEN should not crash."""
        X_tr = RNG.standard_normal((80, N_FEAT)).astype(np.float32)
        y_tr = RNG.uniform(50, 200, (80, 3)).astype(np.float32)
        X_te = RNG.standard_normal((3, N_FEAT)).astype(np.float32)
        y_te = RNG.uniform(50, 200, (3, 3)).astype(np.float32)
        model, scaler, metrics, hparams = train_lstm(X_tr, y_tr, X_te, y_te)
        assert np.isfinite(metrics["MAE"])
