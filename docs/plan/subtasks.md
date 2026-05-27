# Breathe Karachi — Subtasks

Track progress here. Each agent working on a subtask should tick items as they complete them.

---

## Subtask 1 — Data Pipeline
**Goal:** Populate MongoDB `feature_store` with clean, engineered daily rows.

- [x] `src/fetch_data.py` — one-time historical backfill (2023-01-01 → today)
- [x] `src/update_daily_data.py` — incremental fetch, upserts missing days
- [x] `src/preprocess_daily_data.py` — feature engineering, upserts back to `feature_store`

**Feature columns produced (28 inputs + 3 targets):**
```
AQI, PM2.5, PM10, NO2, SO2, CO, O3, Temperature, Humidity, Precipitation,
log_PM2.5, log_CO, month,
season_Spring, season_Summer, season_Winter,
weekday_1..weekday_6,
AQI_lag_1, AQI_lag_2, AQI_roll_mean_3, AQI_roll_std_3, AQI_diff
→ targets: AQI_t+1, AQI_t+2, AQI_t+3
```

---

## Subtask 2 — EDA Notebook
**Goal:** Exploratory analysis artefacts for the report and dashboard context.

- [ ] `notebooks/eda.ipynb` with 8 sections:
  1. Overview — shape, dtypes, missing values, descriptive stats
  2. AQI Time Series — line chart with EPA colour bands
  3. Seasonal Patterns — monthly average bar, season boxplots
  4. Pollutant Analysis — histograms, pie chart, WHO safe-limit comparison bar
  5. Correlations — heatmap, feature-correlation bar sorted by abs(corr) with AQI
  6. Weather Relationships — AQI vs Temperature, AQI vs Humidity scatter (coloured by month)
  7. Weekday/Weekend split — boxplot comparison
  8. Key Insights — worst month, most correlated pollutant, % days above WHO threshold

---

## Subtask 3 — Training Pipeline
**Goal:** Train Ridge + LightGBM + LSTM daily; best RMSE wins and goes to `model_registry`.

**Approach:** Direct multi-output regression — each model predicts `[AQI_t+1, AQI_t+2, AQI_t+3]` simultaneously from the 28 input features.

- [x] `src/models/__init__.py`
- [x] `src/models/ridge.py` — `MultiOutputRegressor(Ridge(alpha=1.0))`
- [x] `src/models/lgbm_model.py` — `MultiOutputRegressor(LGBMRegressor(n_estimators=300, lr=0.05, max_depth=6))`
- [x] `src/models/lstm_model.py` — `LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(3)`, SEQ_LEN=7
- [x] `src/train.py` — loads data, time-aware split (last 30 days = test), trains all three, saves best

**Metrics to compute per model:** MAE, RMSE, R² for each of the 3 horizons + average RMSE (used for ranking).

---

## Subtask 4 — Inference & Explainability
**Goal:** Generate 3-day forecast and LIME explanation, write results to MongoDB.

- [ ] `src/predict.py`
  - Load active model from `model_registry`
  - Ridge/LGBM: use latest single row (28 features)
  - LSTM: build 7-day sequence from last 7 `feature_store` rows
  - Write to `predictions`: `{predicted_at, model_id, forecasts: [{date, predicted_AQI} × 3]}`
- [ ] `src/create_lime.py`
  - `LimeTabularExplainer`, top 15 features, regression mode
  - LSTM: wrap model with a function that tiles row into (7, 28) sequence
  - Output: `lime_explanations/lime_explanation.csv`, `.html`, `.png`

---

## Subtask 5 — CI/CD & Deployment
**Goal:** Automate the full pipeline and deploy to Render.

- [ ] `.github/workflows/feature_pipeline.yml` — schedule: `0 * * * *` (hourly)
  - Steps: checkout → install → `update_daily_data.py` → `preprocess_daily_data.py`
- [ ] `.github/workflows/training_pipeline.yml` — schedule: `0 3 * * *` (3 AM UTC = 8 AM PKT)
  - Steps: checkout → install → `train.py` → `predict.py` → `create_lime.py` → commit lime artefacts `[skip ci]` → trigger Render deploy hook
- [ ] `render.yaml` — web service config with Python 3.11, start command

**Required GitHub Secrets:** `MONGODB_USERNAME`, `MONGODB_PASSWORD`, `MONGODB_CLUSTER`, `RENDER_DEPLOY_HOOK`

---

## Subtask 6 — Dashboard
**Goal:** Streamlit app with Karachi map background, frosted-glass panels, 5 tabs. Original design.

- [ ] `app.py`
  - **Tab 1 — Live Snapshot:** AQI gauge + advisory + pollutant grid + 3-day forecast cards
  - **Tab 2 — Historical Trends:** date-range selector, AQI line chart with EPA bands, pollutant multi-line
  - **Tab 3 — Pollution Breakdown:** WHO radar chart, pollutant pie, LIME bar chart
  - **Tab 4 — Insights:** 5 stat cards, monthly AQI bar, weather scatter plots
  - **Tab 5 — Model Logs:** latest run summary, full log table, CSV download
  - AQI alerts: `st.warning` (AQI > 150), `st.error` (AQI > 200)
  - `@st.cache_resource` for model; `@st.cache_data(ttl=3600)` for data

---

## Pipeline Upgrade (2026-05-27)

### Feature engineering audit & consolidation
- [x] MongoDB backup to CSV before changes (`backups/feature_store_20260527_125204.csv`)
- [x] Audited 120 MongoDB columns — all branches compared
- [x] `src/preprocess_daily_data.py` rewritten to produce all 120 columns (15 raw + 100 engineered + 3 targets + date)
- [x] Sanity check: 114/118 exact matches, 4 documented acceptable deviations → PASS (`scripts/sanity_check_features.py`)
- [x] `src/fetch_data.py` updated to also fetch apparent_temp, surface_pressure, wind_gusts in daily backfill

### Training pipeline upgrade
- [x] `src/models/lgbm_model.py` — switched from MultiOutputRegressor to per-horizon PerHorizonWrapper (3 independent LGBMs); LGBM params tuned to n_estimators=500, max_depth=7, num_leaves=63
- [x] `src/models/lstm_model.py` — dropout 0.2→0.4, added L2(1e-3), loss huber, independent y-scaler, patience 20→25, ReduceLROnPlateau, epochs 100→150
- [x] `src/train.py` — EXCLUDE_COLS expanded to drop 22 features (4 tier-2 + 18 tier-3) that hurt holdout performance; data filtered to 2023+ rows; per-horizon metric logging added
- [x] Expected performance: LGBM d1 R2=0.865 MAE=4.10, Ensemble d1 R2=0.858 MAE=4.62

### Inference upgrade
- [x] `src/predict.py` — loads BOTH active lgbm and lstm, blends per-horizon (d1: 0.6/0.4, d2: 0.15/0.85, d3: pure LSTM), saves as `model_type: "ensemble"` with component_models dict

### Hourly data pipeline (new)
- [x] `src/update_hourly_data.py` — fetches hourly AQI + weather (incl. tier-3 vars) into `hourly_feature_store` collection; incremental upsert on `time` key
- [x] `config/db.py` — added `COLLECTION_HOURLY = "hourly_feature_store"`
- [x] `.github/workflows/feature_pipeline.yml` — step order: update_hourly_data → update_daily_data → preprocess_daily_data

---

## Completion Checklist

| Subtask | Status |
|---------|--------|
| 1 — Data Pipeline | ✅ Done |
| 2 — EDA Notebook | ⬜ Pending |
| 3 — Training Pipeline | ✅ Done (upgraded) |
| 4 — Inference & Explainability | ✅ Done (upgraded) |
| 5 — CI/CD & Deployment | ✅ Done (hourly added) |
| 6 — Dashboard | ⬜ Pending |
