"""
Tests for src/predict.py — inference logic with mocked MongoDB.
Run with: python -m pytest tests/test_predict.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from datetime import date, timedelta

import src.predict as pred_mod


FEAT_COLS = ["AQI", "PM2_5", "PM10", "NO2", "SO2", "CO", "O3",
             "Temperature", "Humidity", "Precipitation",
             "log_PM2_5", "log_CO", "month"]

RNG = np.random.default_rng(1)


def _make_store_rows(n: int = 10) -> list[dict]:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    rows  = []
    for d in dates:
        row = {"date": d.strftime("%Y-%m-%d"), "AQI": float(RNG.uniform(50, 200))}
        for col in FEAT_COLS[1:]:
            row[col] = float(RNG.uniform(0, 100))
        rows.append(row)
    return rows


class TestPredictTabular:
    def test_predict_tabular_shape(self):
        model = MagicMock()
        model.predict.return_value = np.array([[120.0, 125.0, 130.0]])
        scaler = MagicMock()
        scaler.transform.return_value = np.zeros((1, len(FEAT_COLS)))

        df = pd.DataFrame(_make_store_rows(1))
        df["date"] = pd.to_datetime(df["date"])
        result = pred_mod._predict_tabular(model, scaler, FEAT_COLS, df)
        assert len(result) == 3

    def test_predict_tabular_uses_last_row(self):
        captured = {}
        model    = MagicMock()
        model.predict.return_value = np.array([[100.0, 110.0, 120.0]])
        scaler   = MagicMock()
        scaler.transform.side_effect = lambda x: (captured.update({"X": x}) or x)

        df = pd.DataFrame(_make_store_rows(5))
        df["date"] = pd.to_datetime(df["date"])
        pred_mod._predict_tabular(model, scaler, FEAT_COLS, df)
        # transform was called with shape (1, n_feat)
        assert captured["X"].shape[0] == 1


class TestPredictLSTM:
    def test_lstm_requires_seq_len_rows(self):
        model  = MagicMock()
        scaler = MagicMock()
        scaler.transform.return_value = np.zeros((pred_mod.SEQ_LEN, len(FEAT_COLS)))
        model.predict.return_value = np.array([[100.0, 110.0, 120.0]])

        df = pd.DataFrame(_make_store_rows(pred_mod.SEQ_LEN))
        df["date"] = pd.to_datetime(df["date"])
        result = pred_mod._predict_lstm(model, scaler, FEAT_COLS, df)
        assert len(result) == 3

    def test_lstm_raises_if_too_few_rows(self):
        model  = MagicMock()
        scaler = MagicMock()
        df = pd.DataFrame(_make_store_rows(pred_mod.SEQ_LEN - 1))
        df["date"] = pd.to_datetime(df["date"])
        with pytest.raises(RuntimeError, match="Need at least"):
            pred_mod._predict_lstm(model, scaler, FEAT_COLS, df)


class TestForecastDates:
    def test_forecast_dates_sequential(self):
        """The 3 forecast dates should be consecutive days after the last stored date."""
        rows = _make_store_rows(5)
        last_date = date.fromisoformat(rows[-1]["date"])
        expected  = [(last_date + timedelta(days=i+1)).isoformat() for i in range(3)]

        model = MagicMock()
        model.predict.return_value = np.array([[100.0, 110.0, 120.0]])
        scaler = MagicMock()
        scaler.transform.return_value = np.zeros((1, len(FEAT_COLS)))

        with (
            patch.object(pred_mod, "_latest_active_type", return_value="ridge"),
            # patch in pred_mod's namespace (imported via `from config.db import load_model`)
            patch.object(pred_mod, "load_model", return_value=(
                model, scaler,
                {"features": FEAT_COLS, "_id": "fake_id", "model_type": "ridge"}
            )),
            patch.object(pred_mod, "_load_feature_store", return_value=(
                pd.DataFrame(rows).assign(date=lambda d: pd.to_datetime(d["date"]))
            )),
            patch.object(pred_mod, "get_collection") as mock_col,
        ):
            inserted = {}
            mock_col.return_value.insert_one.side_effect = lambda doc: inserted.update({"doc": doc})
            pred_mod.run()

        for i, fc in enumerate(inserted["doc"]["forecasts"]):
            assert fc["date"] == expected[i]

    def test_no_active_model_raises(self):
        with patch.object(pred_mod, "_latest_active_type", side_effect=ValueError("No active")):
            with pytest.raises(ValueError, match="No active"):
                pred_mod.run()
