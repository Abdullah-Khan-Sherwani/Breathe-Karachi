# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

This is the **Karachi AQI Predictor** — a Pearls bootcamp submission. The goal is to predict Karachi's Air Quality Index for the next 3 days using a fully serverless, automated ML pipeline deployed as an interactive Streamlit web app.

**Always read the project spec before starting work:**
- `docs/Pearls_AQI_Predictor.md` — project requirements and key features
- `docs/Pearls_AQI_Predictor_Slides_Content.md` — architecture breakdown (5 components)

---

## Repository Layout

```
Karachi-AQI-Predictor/
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml    # runs hourly: fetch → preprocess
│       └── training_pipeline.yml  # runs daily: train → predict → lime → deploy
├── config/
│   ├── __init__.py                 # exports: get_db, get_collection, save_model, load_model
│   └── db.py                       # MongoDB connection + model serialization helpers
├── src/
│   ├── __init__.py
│   ├── fetch_data.py               # one-time historical backfill (run manually once)
│   ├── update_daily_data.py        # incremental fetch → upsert raw row to feature_store
│   ├── preprocess_daily_data.py    # feature engineering → write processed rows to feature_store
│   ├── lstm_model_training.py      # train LSTM → save to model_registry + log to model_logs
│   ├── predict.py                  # load model → 3-day forecast → write to predictions
│   └── create_lime.py              # LIME on latest row → save HTML/CSV/PNG to lime_explanations/
├── lime_explanations/              # gitignored; populated by create_lime.py at runtime
├── docs/                           # project spec (read-only, never modify)
├── app.py                          # Streamlit dashboard (5 tabs, entry point)
├── requirements.txt                # all deps pinned
├── render.yaml                     # Render deployment config
├── .env.example                    # env var template (no secrets)
├── .gitignore
└── CLAUDE.md
```

---

## Architecture: Five-Stage Pipeline

### 1. Data Source
- **API**: Open-Meteo (free, no API key needed)
  - Air quality (historical): `https://air-quality-api.open-meteo.com/v1/air-quality`
  - Weather (historical): `https://archive-api.open-meteo.com/v1/archive`
  - Weather (forecast): `https://api.open-meteo.com/v1/forecast`
- **Location**: Karachi — `LAT=24.8607, LON=67.0011`, `TIMEZONE=Asia/Karachi`
- **Pollutants fetched**: `us_aqi, pm2_5, pm10, nitrogen_dioxide, sulphur_dioxide, carbon_monoxide, ozone`
- **Weather fetched**: `temperature_2m, relative_humidity_2m, precipitation`
- **Granularity**: daily (hourly data fetched, averaged to one row per day)

### 2. Feature Engineering (`src/preprocess_daily_data.py`)
- Lag features: `AQI_lag_1`, `AQI_lag_2`
- Rolling stats: `AQI_roll_mean_3`, `AQI_roll_std_3`, `AQI_diff`
- Lead targets: `AQI_t+1`, `AQI_t+2`, `AQI_t+3`
- Log transforms: `log_PM2.5`, `log_CO`
- Cyclical: `month`, `season_*` dummies, `weekday_*` dummies

### 3. Model (`src/lstm_model_training.py`)
- LSTM with `SEQ_LEN=7` (7-day lookback)
- Autoregressive inference: each of 3 forecast steps feeds its predicted AQI back as input
- Evaluation metrics: MAE, RMSE, R²
- Serialized to MongoDB via `config/db.py:save_model()` (temp file → binary — see db.py notes)

### 4. Explainability (`src/create_lime.py`)
- LIME on the most recent processed data point
- Outputs written to `lime_explanations/`: `lime_explanation.html`, `.csv`, `.png`

### 5. Dashboard (`app.py`)
Five tabs via `st.tabs`. All data loaded from MongoDB — never from local files.
- **Overview**: AQI gauge + today's pollutants + 3-day forecast cards
- **AQI Trends**: time-filtered line chart with EPA AQI bands
- **Pollutants & LIME**: radar vs WHO limits + pie chart + LIME bar chart
- **General Insights**: stats cards (worst season, worst day, WHO exceedance %)
- **Logs**: training run history from `model_logs` collection

**Mandatory caching in app.py:**
- `@st.cache_resource` — model and scaler (loaded from MongoDB once per session)
- `@st.cache_data` — feature data and predictions (loaded from MongoDB, TTL optional)

### 6. CI/CD (GitHub Actions) — two workflows per project spec

| Workflow | Schedule | Steps |
|---|---|---|
| `feature_pipeline.yml` | Every hour | `update_daily_data.py` → `preprocess_daily_data.py` |
| `training_pipeline.yml` | Every day | `lstm_model_training.py` → `predict.py` → `create_lime.py` → trigger Render deploy hook |

Both workflows use `[skip ci]` in auto-commit messages to prevent re-trigger loops.

---

## MongoDB: Database & Document Schemas

**DB name**: `karachi_aqi`

---

### Collection: `feature_store`
One document per calendar day. Written by `preprocess_daily_data.py`. Raw rows (pre-feature-engineering) are overwritten in-place via upsert on `date`.

```json
{
  "date":             "2024-01-15",
  "AQI":              145.3,
  "PM2.5":            67.2,
  "PM10":             89.1,
  "NO2":              34.5,
  "SO2":              12.3,
  "CO":               890.0,
  "O3":               45.6,
  "Temperature":      28.4,
  "Humidity":         65.0,
  "Precipitation":    0.0,
  "AQI_lag_1":        138.2,
  "AQI_lag_2":        142.1,
  "AQI_roll_mean_3":  141.9,
  "AQI_roll_std_3":   3.6,
  "AQI_diff":         7.1,
  "log_PM2.5":        4.21,
  "log_CO":           6.79,
  "month":            1,
  "season_Spring":    0,
  "season_Summer":    0,
  "season_Winter":    1,
  "weekday_0":        1,
  "weekday_1":        0,
  "AQI_t+1":          150.1,
  "AQI_t+2":          148.7,
  "AQI_t+3":          143.2,
  "processed_at":     "2024-01-15T08:00:00Z"
}
```

---

### Collection: `model_registry`
One document per training run. Written by `save_model()` in `config/db.py`.
Previous active models are marked `"status": "inactive"` before each new insert.

```json
{
  "_id":             "ObjectId(...)",
  "model_type":      "lstm",
  "version":         "20240115_083000",
  "trained_at":      "2024-01-15T08:30:00Z",
  "status":          "active",
  "model_binary":    "BinData(...)",
  "scaler_binary":   "BinData(...)",
  "features":        ["AQI", "PM10", "NO2", "SO2", "O3", "Temperature", "Humidity", "Precipitation", "month", "log_PM2.5", "log_CO", "..."],
  "hyperparameters": { "seq_len": 7, "units_1": 64, "units_2": 32, "dropout": 0.2, "epochs": 100, "patience": 20 },
  "MAE":             12.4,
  "RMSE":            18.3,
  "R2":              0.87,
  "train_samples":   320,
  "test_samples":    80
}
```

---

### Collection: `predictions`
One document per daily forecast run. Written by `predict.py`.

```json
{
  "_id":          "ObjectId(...)",
  "predicted_at": "2024-01-15T09:00:00Z",
  "model_id":     "ObjectId(...)",
  "forecasts": [
    { "date": "2024-01-16", "predicted_AQI": 152.3 },
    { "date": "2024-01-17", "predicted_AQI": 148.1 },
    { "date": "2024-01-18", "predicted_AQI": 144.7 }
  ]
}
```

---

### Collection: `model_logs`
One document per training run (lightweight, no binaries). Written by `lstm_model_training.py`.

```json
{
  "_id":           "ObjectId(...)",
  "timestamp":     "2024-01-15T08:30:00Z",
  "status":        "success",
  "model_id":      "ObjectId(...)",
  "MAE":           12.4,
  "RMSE":          18.3,
  "R2":            0.87,
  "train_samples": 320,
  "test_samples":  80
}
```

---

## Environment Variables

```
MONGODB_USERNAME=...
MONGODB_PASSWORD=...
MONGODB_CLUSTER=cluster0.xxxxx.mongodb.net
```

GitHub Actions secrets required: `MONGODB_USERNAME`, `MONGODB_PASSWORD`, `MONGODB_CLUSTER`, `RENDER_DEPLOY_HOOK`

Local: copy `.env.example` → `.env` and fill in values.

---

## Git Rules

- **Never push to the remote repository.** All pushes are done manually by the user.
- **Never add Claude as a co-author or contributor** in any commit message. No `Co-Authored-By` lines, no attribution to Claude or Anthropic in commits, tags, or any git metadata.

---

## Reference Projects

The two sibling folders (`AQI-Prediction/` and `Pearls-Karachi-AQI-Prediction-for-next-3-days-/`) exist **for your eyes only** as architectural inspiration. They must never surface in the actual codebase in any form:

- No imports from or paths referencing those folders
- No comments, docstrings, or variable names citing them
- No copy-pasted code that retains their author names, project titles, or repo-specific strings
- No `sys.path` manipulation pointing outside `Karachi-AQI-Predictor/`

If you derive logic from a reference, rewrite it cleanly as original code for this project.

---

## Key Implementation Notes

- **Model serialization**: Keras 3 `.save()` does not support `BytesIO`. `db.py:save_model()` writes to a `tempfile.NamedTemporaryFile(suffix=".h5")`, reads the bytes, then deletes the file. Same pattern in reverse for `load_model()`.
- **Datetime**: always use `datetime.now(timezone.utc)` — `datetime.utcnow()` is deprecated in Python 3.12+.
- **RMSE**: compute as `mean_squared_error(y_true, y_pred, squared=False)` — the reference project omits `squared=False`, which returns MSE not RMSE. Do not copy that bug.
- **No local data files**: all reads and writes go through `config/db.py`. No CSVs, no `.pkl`, no `.keras` files are committed or expected at runtime.
- **Python version**: 3.12.4 (local) — `tensorflow==2.17.0` is the minimum that supports Python 3.12. Do not downgrade TF below 2.17.
- **Render start command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

---

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# One-time historical backfill (run before first training)
python src/fetch_data.py

# Full pipeline (manual run, in order)
python src/update_daily_data.py
python src/preprocess_daily_data.py
python src/lstm_model_training.py
python src/predict.py
python src/create_lime.py

# Launch dashboard
streamlit run app.py
```
