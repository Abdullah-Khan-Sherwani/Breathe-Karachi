# Breathe Karachi

> End-to-end serverless AQI forecasting pipeline for Karachi — predicts US Air Quality Index four days ahead using automated ML training, live Open-Meteo data, and an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?logo=tensorflow&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?logo=mongodb&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)
![Render](https://img.shields.io/badge/Deployed_on-Render-46E3B7?logo=render&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**Live App** → [breathe-karachi.onrender.com](https://breathe-karachi.onrender.com)

---

## About

Karachi is consistently ranked among the most polluted megacities in the world, yet real-time air quality forecasting tools tailored to the city remain scarce. **Breathe Karachi** addresses this by building a fully automated, serverless ML system that:

- Fetches live weather and pollution data every hour from [Open-Meteo](https://open-meteo.com/) — no API key required
- Engineers 141 time-series features per day (lags, rolling statistics, lead features, seasonality)
- Retrains three model families daily (Ridge Regression, LightGBM, LSTM) and selects the best via rolling-origin cross-validation
- Serves 4-day AQI forecasts through a public Streamlit dashboard
- Stores all data, models, and SHAP explanations in MongoDB Atlas — no files committed to the repo

The rolling-CV-selected model (currently LightGBM) achieves **8.8 MAE** and **12.0 RMSE** on a 90-day holdout, and a **3.6 AQI mean error** against live Open-Meteo reference values. Full methodology and results are documented in [`report.tex`](report.tex).

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   GitHub Actions                     │
│                                                      │
│  ⏱ Hourly                    📅 Daily (7:19 AM PKT) │
│  update_daily_data.py        train.py                │
│  update_hourly_data.py       predict.py              │
│  preprocess_daily_data.py    create_shap.py          │
│         │                    create_lime.py          │
│         └──────────┬─────────────────────────────────┘
│                    │
│                    ▼
│            MongoDB Atlas (karachi_aqi)
│    feature_store  model_registry  predictions
│    model_logs     shap_explanations  ensemble_config
│                    │
│                    ▼
│            Streamlit Dashboard (Render)
└─────────────────────────────────────────────────────┘
```

**Data flow:**
1. Open-Meteo Archive/Forecast/Air Quality APIs → raw daily row → `feature_store`
2. Feature engineering (lags, rolling stats, log transforms, lead features, seasonality) → `feature_store`
3. Three models trained on 90-day holdout eval → rolling-origin CV selects best → full retrain → `model_registry`
4. 4-day inference from latest feature-store row → `predictions`
5. SHAP/LIME explanations generated → `shap_explanations`
6. Dashboard reads everything live from MongoDB

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Data Source | Open-Meteo Archive, Forecast, and Air Quality APIs (free, no key) |
| Feature Store & Model Registry | MongoDB Atlas (serverless) |
| Models | Ridge Regression, LightGBM (PerHorizonWrapper), LSTM (TensorFlow/Keras) |
| Model Selection | Rolling-origin cross-validation (3 folds, full dataset) |
| Explainability | SHAP (TreeExplainer / LinearExplainer / Expected Gradients) + LIME |
| Dashboard | Streamlit + Plotly |
| Orchestration | GitHub Actions + cron-job.org |
| Hosting | Render |

---

## Getting Started

### Prerequisites

- Python 3.11
- A free [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) account

### Installation

```bash
git clone https://github.com/Abdullah-Khan-Sherwani/Breathe-Karachi.git
cd Breathe-Karachi
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
```

Open `.env` and fill in your Atlas credentials:

```
MONGODB_USERNAME=your_username
MONGODB_PASSWORD=your_password
MONGODB_CLUSTER=cluster0.xxxxx.mongodb.net
```

### Run Locally

```bash
# Step 1 — one-time historical backfill (run before first training)
python src/fetch_data.py

# Step 2 — incremental update + feature engineering
python src/update_daily_data.py
python src/preprocess_daily_data.py

# Step 3 — train models + generate forecast + explainability
python src/train.py
python src/predict.py
python src/create_shap.py
python src/create_lime.py

# Step 4 — launch dashboard
streamlit run app.py
```

---

## Key Files

| File | Purpose |
|------|---------|
| `src/fetch_data.py` | One-time historical backfill from 2018-01-01 |
| `src/update_daily_data.py` | Hourly: fetch latest daily row from Open-Meteo |
| `src/update_hourly_data.py` | Hourly: fetch intraday hourly data |
| `src/preprocess_daily_data.py` | Hourly: engineer all 141 features per row |
| `src/train.py` | Daily: train Ridge/LightGBM/LSTM, rolling-CV model selection, save to MongoDB |
| `src/predict.py` | Daily: generate 4-day forecast from latest feature-store row |
| `src/create_shap.py` | Daily: compute SHAP for all three models, persist to MongoDB |
| `src/create_lime.py` | Daily: compute LIME explanation for latest prediction |
| `src/models/lgbm_model.py` | LightGBM with PerHorizonWrapper + two-stage early stopping |
| `src/models/lstm_model.py` | Stacked LSTM (64→32) with expected-gradients SHAP support |
| `src/models/ridge.py` | Ridge with StandardScaler + MultiOutputRegressor |
| `src/models/per_horizon_wrapper.py` | Wraps per-horizon sub-models for LightGBM |
| `scripts/predict_pm25_shadow.py` | Research shadow pipeline: trains with PM2.5 leads (excluded from prod) |
| `config/db.py` | MongoDB connection, model serialization, collection helpers |
| `app.py` | Streamlit dashboard entry point |
| `report.tex` | Full methodology and results report |

---

## MongoDB Collections

| Collection | Contents |
|------------|---------|
| `feature_store` | One document per calendar day; 141+ engineered features + AQI targets |
| `model_registry` | Serialized model binaries + scaler + metrics + feature list |
| `predictions` | 4-day AQI forecasts per daily run |
| `model_logs` | Lightweight per-run training metrics for all three models |
| `ensemble_config` | Rolling-CV-selected model order and weights |
| `shap_explanations` | Per-model and ensemble SHAP feature importances |

---

## CI/CD

Two GitHub Actions workflows triggered via `workflow_dispatch` from cron-job.org:

| Workflow | Schedule | Steps |
|----------|----------|-------|
| `feature_pipeline.yml` | Every hour | Fetch daily → fetch hourly → preprocess → store to MongoDB |
| `training_pipeline.yml` | Daily 7:19 AM PKT | Train → predict → shadow forecast → LIME → SHAP |

**Required GitHub Secrets:**

```
MONGODB_USERNAME
MONGODB_PASSWORD
MONGODB_CLUSTER
```

---

## License

MIT © [Abdullah Khan Sherwani](https://github.com/Abdullah-Khan-Sherwani)
