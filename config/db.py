import os
import pickle
import tempfile
from datetime import datetime, timezone

from bson import Binary
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

_client = None

DB_NAME = "karachi_aqi"

COLLECTION_FEATURE_STORE  = "feature_store"
COLLECTION_MODEL_REGISTRY = "model_registry"
COLLECTION_PREDICTIONS    = "predictions"
COLLECTION_MODEL_LOGS     = "model_logs"


def get_client():
    global _client
    if _client is None:
        username = os.getenv("MONGODB_USERNAME")
        password = os.getenv("MONGODB_PASSWORD")
        cluster  = os.getenv("MONGODB_CLUSTER")
        uri = f"mongodb+srv://{username}:{password}@{cluster}/?appName=KarachiAQI"
        _client = MongoClient(uri)
    return _client


def get_db():
    return get_client()[DB_NAME]


def get_collection(name: str):
    return get_db()[name]


def save_model(model, scaler, model_type: str, metrics: dict, feature_cols: list, hyperparameters: dict, extra_metadata: dict = None):
    """
    Serialize model + scaler into MongoDB model_registry.
    Marks all previous models of the same type as inactive before inserting.
    metrics must contain keys: MAE, RMSE, R2.
    """
    # Serialize model via temp file (BytesIO not supported by Keras 3 .save())
    if model_type == "lstm":
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            model.save(tmp_path)
            with open(tmp_path, "rb") as f:
                model_binary = f.read()
        finally:
            os.unlink(tmp_path)
    else:
        model_binary = pickle.dumps(model)

    now = datetime.now(timezone.utc)

    # Retire previous active models of this type
    get_collection(COLLECTION_MODEL_REGISTRY).update_many(
        {"model_type": model_type, "status": "active"},
        {"$set": {"status": "inactive"}},
    )

    doc = {
        "model_type":      model_type,
        "version":         now.strftime("%Y%m%d_%H%M%S"),
        "trained_at":      now,
        "model_binary":    Binary(model_binary),
        "scaler_binary":   Binary(pickle.dumps(scaler)),
        "features":        feature_cols,
        "hyperparameters": hyperparameters,
        "status":          "active",
        **metrics,           # expects: MAE, RMSE, R2
    }
    if extra_metadata:
        doc.update(extra_metadata)

    result = get_collection(COLLECTION_MODEL_REGISTRY).insert_one(doc)
    return result.inserted_id


def load_model(model_type: str = "lstm"):
    """
    Load the latest active model of the given type from model_registry.
    Returns (model, scaler, metadata).
    """
    import tensorflow as tf

    doc = get_collection(COLLECTION_MODEL_REGISTRY).find_one(
        {"model_type": model_type, "status": "active"},
        sort=[("trained_at", -1)],
    )
    if doc is None:
        raise ValueError(f"No active '{model_type}' model found in model_registry.")

    # Deserialize model via temp file
    if model_type == "lstm":
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp.write(bytes(doc["model_binary"]))
            tmp_path = tmp.name
        try:
            model = tf.keras.models.load_model(tmp_path)
        finally:
            os.unlink(tmp_path)
    else:
        model = pickle.loads(doc["model_binary"])

    scaler = pickle.loads(doc["scaler_binary"])
    metadata = {k: v for k, v in doc.items() if k not in ("model_binary", "scaler_binary", "_id")}
    metadata["_id"] = str(doc["_id"])

    return model, scaler, metadata
