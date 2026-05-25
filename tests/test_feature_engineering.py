"""
Tests for src/preprocess_daily_data.py — pure transformation functions.
No MongoDB calls needed; all tests work on in-memory DataFrames.
Run with: python -m pytest tests/test_feature_engineering.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.preprocess_daily_data import (
    cap_iqr,
    add_log_transforms,
    add_temporal,
    add_lag_rolling,
    add_targets,
    IQR_COLS,
)


def _base_df(n: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    rng   = np.random.default_rng(42)
    return pd.DataFrame({
        "date":          dates,
        "AQI":           rng.uniform(50, 200, n),
        "PM2_5":         rng.uniform(10, 150, n),
        "PM10":          rng.uniform(20, 200, n),
        "NO2":           rng.uniform(5, 80, n),
        "SO2":           rng.uniform(1, 40, n),
        "CO":            rng.uniform(200, 1500, n),
        "O3":            rng.uniform(10, 80, n),
        "Temperature":   rng.uniform(20, 45, n),
        "Humidity":      rng.uniform(30, 90, n),
        "Precipitation": rng.uniform(0, 5, n),
    })


class TestCapIQR:
    def test_no_values_beyond_bounds(self):
        df = _base_df()
        # Inject extreme outliers
        df.loc[0, "PM10"] = 99999
        df.loc[1, "SO2"]  = -99999
        result = cap_iqr(df.copy())
        for col in IQR_COLS:
            if col not in result.columns:
                continue
            q1 = result[col].quantile(0.25)
            q3 = result[col].quantile(0.75)
            iqr = q3 - q1
            assert result[col].max() <= q3 + 1.5 * iqr + 1e-6
            assert result[col].min() >= q1 - 1.5 * iqr - 1e-6

    def test_missing_col_skipped(self):
        df = _base_df().drop(columns=["PM10"])
        result = cap_iqr(df.copy())
        assert "PM10" not in result.columns

    def test_no_mutation_of_input(self):
        df = _base_df()
        before = df["PM10"].copy()
        cap_iqr(df)
        pd.testing.assert_series_equal(df["PM10"], before)


class TestLogTransforms:
    def test_log_pm25_positive(self):
        df = add_log_transforms(_base_df())
        assert (df["log_PM2_5"] >= 0).all()

    def test_log_co_positive(self):
        df = add_log_transforms(_base_df())
        assert (df["log_CO"] >= 0).all()

    def test_zero_safe(self):
        df = _base_df()
        df["PM2_5"] = 0
        df["CO"]    = 0
        result = add_log_transforms(df)
        assert (result["log_PM2_5"] == 0).all()
        assert (result["log_CO"]    == 0).all()

    def test_columns_added(self):
        df = add_log_transforms(_base_df())
        assert "log_PM2_5" in df.columns
        assert "log_CO" in df.columns


class TestTemporal:
    def test_month_range(self):
        df = add_temporal(_base_df())
        assert df["month"].between(1, 12).all()

    def test_season_dummies_sum_to_one_or_zero(self):
        # Use 400 days to guarantee all 4 seasons appear (drop_first needs >1 unique)
        df = add_temporal(_base_df(400))
        season_cols = [c for c in df.columns if c.startswith("season_")]
        assert len(season_cols) >= 1
        # drop_first=True means Autumn is reference; row sums can be 0 (Autumn) or 1
        assert df[season_cols].sum(axis=1).isin([0, 1]).all()

    def test_weekday_dummies_present(self):
        df = add_temporal(_base_df())
        weekday_cols = [c for c in df.columns if c.startswith("weekday_")]
        assert len(weekday_cols) >= 1

    def test_original_season_col_dropped(self):
        df = add_temporal(_base_df())
        assert "season" not in df.columns

    def test_original_weekday_col_dropped(self):
        df = add_temporal(_base_df())
        assert "weekday" not in df.columns


class TestLagRolling:
    def test_lag1_is_prev_aqi(self):
        df = _base_df(10)
        result = add_lag_rolling(df.copy())
        for i in range(1, len(df)):
            assert result.loc[i, "AQI_lag_1"] == pytest.approx(df.loc[i - 1, "AQI"])

    def test_lag2_is_two_prev_aqi(self):
        df = _base_df(10)
        result = add_lag_rolling(df.copy())
        for i in range(2, len(df)):
            assert result.loc[i, "AQI_lag_2"] == pytest.approx(df.loc[i - 2, "AQI"])

    def test_roll_mean_3_value(self):
        df = _base_df(10)
        result = add_lag_rolling(df.copy())
        # roll_mean_3 at index 3 = mean of AQI[0], AQI[1], AQI[2]
        expected = df["AQI"].iloc[:3].mean()
        assert result.loc[3, "AQI_roll_mean_3"] == pytest.approx(expected)

    def test_first_rows_have_nans(self):
        df = _base_df(10)
        result = add_lag_rolling(df.copy())
        assert pd.isna(result.loc[0, "AQI_lag_1"])
        assert pd.isna(result.loc[0, "AQI_lag_2"])


class TestTargets:
    def test_t1_is_next_aqi(self):
        df = _base_df(10)
        result = add_targets(df.copy())
        for i in range(len(df) - 1):
            assert result.loc[i, "AQI_t+1"] == pytest.approx(df.loc[i + 1, "AQI"])

    def test_last_rows_are_nan(self):
        df = _base_df(10)
        result = add_targets(df.copy())
        assert pd.isna(result.loc[len(df) - 1, "AQI_t+1"])
        assert pd.isna(result.loc[len(df) - 2, "AQI_t+2"])
        assert pd.isna(result.loc[len(df) - 3, "AQI_t+3"])
