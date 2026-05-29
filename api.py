"""
Flask REST API — exposes the latest AQI forecast from MongoDB.

Endpoints:
  GET  /health    — liveness check
  GET  /predict   — latest 4-day ensemble forecast
  POST /predict   — same result (convenience for clients that POST)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from flask import Flask, jsonify

from config.db import get_collection, COLLECTION_PREDICTIONS

app = Flask(__name__)


def _latest_forecast() -> dict:
    doc = get_collection(COLLECTION_PREDICTIONS).find_one(
        {}, sort=[("predicted_at", -1)]
    )
    if doc is None:
        return None

    return {
        "predicted_at":  doc["predicted_at"].isoformat() if hasattr(doc["predicted_at"], "isoformat") else str(doc["predicted_at"]),
        "model_type":    doc.get("model_type", "ensemble"),
        "anchor_date":   doc.get("anchor_date"),
        "forecasts":     doc.get("forecasts", []),
    }


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/predict")
@app.post("/predict")
def predict():
    forecast = _latest_forecast()
    if forecast is None:
        return jsonify({"error": "No forecast available. Run predict.py first."}), 503
    return jsonify(forecast)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
