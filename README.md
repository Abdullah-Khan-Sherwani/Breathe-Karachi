# 🌫️ Breathe Karachi

> End-to-end serverless AQI forecasting pipeline for Karachi — predicts air quality 3 days ahead using a live LSTM model, automated CI/CD, and an interactive web dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?logo=tensorflow&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?logo=mongodb&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)
![Render](https://img.shields.io/badge/Deployed_on-Render-46E3B7?logo=render&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**Live App** → *(coming soon)*

---

## Table of Contents

- [About](#about)
- [Architecture](#architecture)
- [Dashboard](#dashboard)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Pipeline Scripts](#pipeline-scripts)
- [MongoDB Schema](#mongodb-schema)
- [CI/CD](#cicd)
- [Project Structure](#project-structure)

---

## About

Karachi is consistently ranked among the most polluted megacities in the world, yet real-time air quality forecasting tools tailored to the city remain scarce. **Breathe Karachi** addresses this by building a fully automated, serverless ML system that:

- Fetches live weather and pollution data every hour from [Open-Meteo](https://open-meteo.com/) — no API key required
- Engineers time-series features and retrains a deep learning model daily
- Serves 3-day AQI forecasts through a public Streamlit dashboard
- Stores all data and models in MongoDB Atlas — no files committed to the repo

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   GitHub Actions                     │
│                                                      │
│  ⏱ Hourly                    📅 Daily               │
│  update_daily_data.py        lstm_model_training.py  │
│         │                          │                 │
│  preprocess_daily_data.py    predict.py              │
│         │                          │                 │
│         └──────────┬───────────────┘                 │
└──────────────────── │ ────────────────────────────────┘
                      │
                      ▼
              MongoDB Atlas (karachi_aqi)
          ┌───────────┬──────────────┐
          │           │              │
    feature_store  model_registry  predictions
                      │
                      ▼
              Streamlit Dashboard
              (hosted on Render)
```

**Data Flow:**
1. Open-Meteo API → raw daily AQI + weather row → `feature_store`
2. Feature engineering (lags, rolling stats, log transforms, seasonality) → `feature_store`
3. LSTM trains on full history → serialized binary stored in `model_registry`
4. Autoregressive 3-step inference → forecast written to `predictions`
5. LIME explanation generated on the latest data point
6. Dashboard reads everything live from MongoDB

---

## Dashboard

| Tab | Description |
|-----|-------------|
| **Overview** | AQI gauge, today's pollutants and weather readings, 3-day forecast cards with category labels |
| **AQI Trends** | Time-filtered line chart with EPA AQI category bands (Good → Hazardous) |
| **Pollutants & LIME** | WHO safe-limit radar chart, pollutant composition pie, LIME feature contribution bar chart |
| **General Insights** | Worst season, worst recorded day, weekday vs. weekend AQI, WHO exceedance percentage |
| **Logs** | Full training run history — MAE, RMSE, R² per run |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Data Source | Open-Meteo API (free, no key) |
| Feature Store & Model Registry | MongoDB Atlas M0 (free tier) |
| Model | LSTM — TensorFlow / Keras |
| Explainability | LIME |
| Dashboard | Streamlit + Plotly |
| Orchestration | GitHub Actions |
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

# Step 3 — train model + generate forecast + explainability
python src/lstm_model_training.py
python src/predict.py
python src/create_lime.py

# Step 4 — launch dashboard
streamlit run app.py
```

---

## Pipeline Scripts

| Script | Trigger | Description |
|--------|---------|-------------|
| `src/fetch_data.py` | Manual (once) | Backfills historical data from Jan 2023 |
| `src/update_daily_data.py` | Hourly (CI/CD) | Fetches the latest daily row from Open-Meteo |
| `src/preprocess_daily_data.py` | Hourly (CI/CD) | Computes all engineered features |
| `src/lstm_model_training.py` | Daily (CI/CD) | Trains LSTM, saves model binary to MongoDB |
| `src/predict.py` | Daily (CI/CD) | Generates 3-day forecast, writes to MongoDB |
| `src/create_lime.py` | Daily (CI/CD) | Produces LIME explanation for the latest prediction |

---

## MongoDB Schema

**Database:** `karachi_aqi`

<details>
<summary><code>feature_store</code> — one document per calendar day</summary>

```json
{
  "date": "2024-01-15",
  "AQI": 145.3,
  "PM2.5": 67.2, "PM10": 89.1, "NO2": 34.5,
  "SO2": 12.3, "CO": 890.0, "O3": 45.6,
  "Temperature": 28.4, "Humidity": 65.0, "Precipitation": 0.0,
  "AQI_lag_1": 138.2, "AQI_lag_2": 142.1,
  "AQI_roll_mean_3": 141.9, "AQI_roll_std_3": 3.6, "AQI_diff": 7.1,
  "log_PM2.5": 4.21, "log_CO": 6.79,
  "month": 1, "season_Winter": 1, "weekday_0": 1,
  "AQI_t+1": 150.1, "AQI_t+2": 148.7, "AQI_t+3": 143.2,
  "processed_at": "2024-01-15T08:00:00Z"
}
```
</details>

<details>
<summary><code>model_registry</code> — serialized model per training run</summary>

```json
{
  "model_type": "lstm",
  "version": "20240115_083000",
  "trained_at": "2024-01-15T08:30:00Z",
  "status": "active",
  "model_binary": "BinData(...)",
  "scaler_binary": "BinData(...)",
  "features": ["AQI", "PM10", "NO2", "..."],
  "hyperparameters": { "seq_len": 7, "units_1": 64, "units_2": 32, "dropout": 0.2 },
  "MAE": 12.4, "RMSE": 18.3, "R2": 0.87,
  "train_samples": 320, "test_samples": 80
}
```
</details>

<details>
<summary><code>predictions</code> — 3-day forecast per daily run</summary>

```json
{
  "predicted_at": "2024-01-15T09:00:00Z",
  "model_id": "ObjectId(...)",
  "forecasts": [
    { "date": "2024-01-16", "predicted_AQI": 152.3 },
    { "date": "2024-01-17", "predicted_AQI": 148.1 },
    { "date": "2024-01-18", "predicted_AQI": 144.7 }
  ]
}
```
</details>

<details>
<summary><code>model_logs</code> — lightweight metrics per training run</summary>

```json
{
  "timestamp": "2024-01-15T08:30:00Z",
  "status": "success",
  "model_id": "ObjectId(...)",
  "MAE": 12.4, "RMSE": 18.3, "R2": 0.87,
  "train_samples": 320, "test_samples": 80
}
```
</details>

---

## CI/CD

Two GitHub Actions workflows run automatically:

| Workflow | Schedule | Steps |
|----------|----------|-------|
| `feature_pipeline.yml` | Every hour | Fetch → Preprocess → Store to MongoDB |
| `training_pipeline.yml` | Every day at 08:00 PKT | Train → Predict → LIME → Trigger Render deploy |

**Required GitHub Secrets:**

```
MONGODB_USERNAME
MONGODB_PASSWORD
MONGODB_CLUSTER
RENDER_DEPLOY_HOOK
```

---

## Project Structure

```
Breathe-Karachi/
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml
│       └── training_pipeline.yml
├── config/
│   ├── __init__.py
│   └── db.py                    # MongoDB connection + model serialization
├── src/
│   ├── fetch_data.py
│   ├── update_daily_data.py
│   ├── preprocess_daily_data.py
│   ├── lstm_model_training.py
│   ├── predict.py
│   └── create_lime.py
├── lime_explanations/
├── docs/
├── app.py
├── requirements.txt
├── render.yaml
├── .env.example
└── CLAUDE.md
```

---

## License

MIT © [Abdullah Khan Sherwani](https://github.com/Abdullah-Khan-Sherwani)
