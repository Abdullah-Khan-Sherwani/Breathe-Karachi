"""
Tests for config/db.py — connection, CRUD helpers, and model round-trip.
Run with: python -m pytest tests/test_db.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pickle
import numpy as np
from unittest.mock import MagicMock, patch
from bson import Binary

from config.db import (
    get_client,
    get_db,
    get_collection,
    save_model,
    load_model,
    COLLECTION_FEATURE_STORE,
    COLLECTION_MODEL_REGISTRY,
    DB_NAME,
)


# ---------------------------------------------------------------------------
# Sanity: real connection
# ---------------------------------------------------------------------------

class TestConnection:
    def test_ping(self):
        client = get_client()
        result = client.admin.command("ping")
        assert result.get("ok") == 1.0

    def test_get_db_name(self):
        db = get_db()
        assert db.name == DB_NAME

    def test_get_collection_returns_correct(self):
        col = get_collection(COLLECTION_FEATURE_STORE)
        assert col.name == COLLECTION_FEATURE_STORE

    def test_unknown_collection_accessible(self):
        # pymongo creates collections lazily — no error expected
        col = get_collection("__test_nonexistent__")
        assert col is not None


# ---------------------------------------------------------------------------
# save_model / load_model — tabular (Ridge-like sklearn object)
# ---------------------------------------------------------------------------

class TestModelRoundTrip:
    """Uses a real insert into model_registry then cleans up."""

    def _make_dummy_scaler(self):
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        sc.fit([[1, 2], [3, 4]])
        return sc

    def test_save_and_load_ridge(self):
        from sklearn.linear_model import Ridge
        from sklearn.multioutput import MultiOutputRegressor
        model = MultiOutputRegressor(Ridge(alpha=1.0))
        X = np.random.rand(20, 2)
        y = np.random.rand(20, 3)
        model.fit(X, y)
        scaler = self._make_dummy_scaler()

        model_id = save_model(
            model=model,
            scaler=scaler,
            model_type="ridge",
            metrics={"MAE": 1.0, "RMSE": 1.5, "R2": 0.8},
            feature_cols=["f1", "f2"],
            hyperparameters={"alpha": 1.0},
            extra_metadata={"train_samples": 20, "test_samples": 5},
        )
        assert model_id is not None

        loaded_model, loaded_scaler, meta = load_model("ridge")
        # predictions should be deterministic
        X_test = np.random.rand(3, 2)
        np.testing.assert_allclose(
            model.predict(scaler.transform(X_test)),
            loaded_model.predict(loaded_scaler.transform(X_test)),
            rtol=1e-5,
        )
        assert meta["model_type"] == "ridge"
        assert meta["status"] == "active"
        assert "features" in meta

        # Cleanup
        get_collection(COLLECTION_MODEL_REGISTRY).delete_one({"_id": __import__("bson").ObjectId(model_id)})

    def test_save_marks_previous_inactive(self):
        from sklearn.linear_model import Ridge
        from sklearn.multioutput import MultiOutputRegressor
        model = MultiOutputRegressor(Ridge())
        model.fit(np.random.rand(10, 2), np.random.rand(10, 3))
        scaler = self._make_dummy_scaler()

        id1 = save_model(model, scaler, "ridge",
                         {"MAE": 1, "RMSE": 2, "R2": 0.5}, ["f1", "f2"], {})
        id2 = save_model(model, scaler, "ridge",
                         {"MAE": 0.9, "RMSE": 1.8, "R2": 0.6}, ["f1", "f2"], {})

        import bson
        doc1 = get_collection(COLLECTION_MODEL_REGISTRY).find_one({"_id": bson.ObjectId(id1)})
        doc2 = get_collection(COLLECTION_MODEL_REGISTRY).find_one({"_id": bson.ObjectId(id2)})

        assert doc1["status"] == "inactive"
        assert doc2["status"] == "active"

        # Cleanup
        get_collection(COLLECTION_MODEL_REGISTRY).delete_many(
            {"_id": {"$in": [bson.ObjectId(id1), bson.ObjectId(id2)]}}
        )

    def test_load_nonexistent_type_raises(self):
        with pytest.raises(ValueError, match="No active"):
            load_model("__nonexistent_type__")
