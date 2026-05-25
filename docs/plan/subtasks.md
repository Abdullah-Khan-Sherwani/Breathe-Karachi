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

- [ ] `src/models/__init__.py`
- [ ] `src/models/ridge.py` — `MultiOutputRegressor(Ridge(alpha=1.0))`
- [ ] `src/models/lgbm_model.py` — `MultiOutputRegressor(LGBMRegressor(n_estimators=300, lr=0.05, max_depth=6))`
- [ ] `src/models/lstm_model.py` — `LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(3)`, SEQ_LEN=7
- [ ] `src/train.py` — loads data, time-aware split (last 30 days = test), trains all three, saves best

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

## Completion Checklist

| Subtask | Status |
|---------|--------|
| 1 — Data Pipeline | ✅ Done |
| 2 — EDA Notebook | ⬜ Pending |
| 3 — Training Pipeline | ⬜ Pending |
| 4 — Inference & Explainability | ⬜ Pending |
| 5 — CI/CD & Deployment | ⬜ Pending |
| 6 — Dashboard | ⬜ Pending |
