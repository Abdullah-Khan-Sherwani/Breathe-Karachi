"""
Tests for src/train.py — orchestration logic using synthetic in-memory data.
MongoDB calls are mocked so these tests run without a live connection.
Run with: python -m pytest tests/test_train.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

import src.train as train_mod


def _synthetic_df(n: int = 100) -> pd.DataFrame:
    rng   = np.random.default_rng(7)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    df    = pd.DataFrame({
        "date":             dates,
        "AQI":              rng.uniform(50, 200, n),
        "PM2_5":            rng.uniform(10, 150, n),
        "PM10":             rng.uniform(20, 200, n),
        "NO2":              rng.uniform(5, 80, n),
        "SO2":              rng.uniform(1, 40, n),
        "CO":               rng.uniform(200, 1500, n),
        "O3":               rng.uniform(10, 80, n),
        "Temperature":      rng.uniform(20, 45, n),
        "Humidity":         rng.uniform(30, 90, n),
        "Precipitation":    rng.uniform(0, 5, n),
        "log_PM2_5":        rng.uniform(0, 5, n),
        "log_CO":           rng.uniform(0, 8, n),
        "month":            rng.integers(1, 13, n),
        "season_Spring":    rng.integers(0, 2, n),
        "season_Summer":    rng.integers(0, 2, n),
        "season_Winter":    rng.integers(0, 2, n),
        "weekday_1":        rng.integers(0, 2, n),
        "weekday_2":        rng.integers(0, 2, n),
        "AQI_lag_1":        rng.uniform(50, 200, n),
        "AQI_lag_2":        rng.uniform(50, 200, n),
        "AQI_roll_mean_3":  rng.uniform(50, 200, n),
        "AQI_roll_std_3":   rng.uniform(0, 20, n),
        "AQI_diff":         rng.uniform(-30, 30, n),
        "AQI_t+1":          rng.uniform(50, 200, n),
        "AQI_t+2":          rng.uniform(50, 200, n),
        "AQI_t+3":          rng.uniform(50, 200, n),
    })
    return df


class TestGetFeatureCols:
    def test_excludes_targets_and_date(self):
        df   = _synthetic_df()
        feat = train_mod.get_feature_cols(df)
        for col in ["date", "AQI_t+1", "AQI_t+2", "AQI_t+3"]:
            assert col not in feat

    def test_includes_aqi(self):
        feat = train_mod.get_feature_cols(_synthetic_df())
        assert "AQI" in feat

    def test_count_sensible(self):
        feat = train_mod.get_feature_cols(_synthetic_df())
        assert len(feat) > 5


class TestTimeSplit:
    def test_no_overlap(self):
        df            = _synthetic_df(100)
        train, test   = train_mod.time_split(df, test_days=30)
        assert train["date"].max() < test["date"].min()

    def test_sizes(self):
        df          = _synthetic_df(100)
        train, test = train_mod.time_split(df, test_days=30)
        assert len(train) + len(test) == len(df)

    def test_test_spans_last_30_days(self):
        df          = _synthetic_df(100)
        _, test     = train_mod.time_split(df, test_days=30)
        span = (test["date"].max() - test["date"].min()).days
        assert span <= 30

    def test_empty_test_with_all_train(self):
        df          = _synthetic_df(10)
        _, test     = train_mod.time_split(df, test_days=0)
        assert len(test) == 0


class TestRunOrchestration:
    """Mocks MongoDB and model trainers to verify orchestration logic."""

    def _mock_trainers(self):
        dummy_model  = MagicMock()
        dummy_scaler = MagicMock()
        metrics = {"MAE": 5.0, "RMSE": 7.0, "R2": 0.8}
        return dummy_model, dummy_scaler, metrics, {"alpha": 1.0}

    def test_run_completes_without_error(self):
        df = _synthetic_df(100)
        with (
            patch.object(train_mod, "load_data",    return_value=df),
            patch.object(train_mod, "train_ridge",   return_value=self._mock_trainers()),
            patch.object(train_mod, "train_lgbm",    return_value=self._mock_trainers()),
            patch.object(train_mod, "train_lstm",    return_value=self._mock_trainers()),
            # patch in train_mod's own namespace (it imports save_model via `from config.db import`)
            patch.object(train_mod, "save_model",    return_value="fake_id"),
            patch.object(train_mod, "get_collection") as mock_col,
        ):
            mock_col.return_value.insert_one = MagicMock()
            train_mod.run()   # should not raise

    def test_empty_feature_store_raises(self):
        with patch.object(train_mod, "load_data", side_effect=RuntimeError("feature_store is empty")):
            with pytest.raises(RuntimeError, match="feature_store is empty"):
                train_mod.run()

    def test_all_models_failed_raises(self):
        df = _synthetic_df(100)
        with (
            patch.object(train_mod, "load_data",  return_value=df),
            patch.object(train_mod, "train_ridge", side_effect=Exception("boom")),
            patch.object(train_mod, "train_lgbm",  side_effect=Exception("boom")),
            patch.object(train_mod, "train_lstm",  side_effect=Exception("boom")),
            patch("config.db.get_collection")      as mock_col,
        ):
            mock_col.return_value.insert_one = MagicMock()
            with pytest.raises(RuntimeError, match="All models failed"):
                train_mod.run()
