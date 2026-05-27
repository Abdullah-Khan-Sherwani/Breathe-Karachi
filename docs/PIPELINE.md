# Karachi AQI Predictor — Pipeline Documentation

## Overview

The Karachi AQI Predictor is a fully serverless, automated ML pipeline that forecasts
Karachi's Air Quality Index (US AQI scale) for the next **4 days**. All data,
models, and predictions are stored in MongoDB Atlas. There are no local data
files at runtime — every read and write goes through `config/db.py`.

The pipeline runs on **GitHub Actions** (two scheduled workflows), stores its
artefacts in MongoDB, and serves results through a **Streamlit** dashboard
deployed on Render.

Location: Karachi, Pakistan — `LAT=24.8607, LON=67.0011, TIMEZONE=Asia/Karachi`

---

## 5-Stage Pipeline

### Stage 1 — Historical Backfill (one-time, manual)

**Script:** `src/fetch_data.py`

Run once before the first training run to populate `feature_store` with historical
data starting from `2023-01-01`. Fetches hourly air quality from Open-Meteo's
historical air-quality API and hourly weather from the archive API, averages each
day's hourly readings to a single daily row, then upserts it into `feature_store`.

This script is not part of the automated pipeline — it only needs to be run once
when bootstrapping a fresh MongoDB instance.

---

### Stage 2 — Incremental Daily Fetch

**Script:** `src/update_daily_data.py`

Checks the most recent date already stored in `feature_store`, then fetches every
missing day up to yesterday and upserts each as a raw daily record. Runs hourly
via GitHub Actions (`feature_pipeline.yml`) so the store is always current.

Raw columns written per document: `date`, `AQI`, `PM2_5`, `PM10`, `NO2`, `SO2`,
`CO`, `O3`, `Temperature`, `Humidity`, `Precipitation`, `wind_speed`,
`wind_direction`, `apparent_temp`, `surface_pressure`, `wind_gusts`, `BLH`,
`cloud_cover`, `shortwave_rad`, `uv_index`, `aod`, `dust`.

---

### Stage 3 — Feature Engineering

**Script:** `src/preprocess_daily_data.py`

Loads all raw documents from `feature_store`, computes ~120 engineered features,
and upserts the processed rows back to the same collection (keyed on `date`).
Also runs hourly immediately after Stage 2.

Feature engineering steps, in order:

1. IQR outlier capping on weather and pollutant columns
2. Log transforms: `log_PM2_5 = log1p(PM2_5)`, `log_CO = log1p(CO)`, `log_aod = log1p(aod)`
3. Wind encoding: `wind_sin`, `wind_cos` from `wind_direction` in radians
4. Temporal dummies: `month`, `season_*` (Autumn is baseline), `weekday_*` (Mon is baseline)
5. Cyclical encodings: `month_sin/cos`, `doy_sin/cos`, `weekday_sin/cos`
6. Short-window lag/rolling: `AQI_lag_1/2`, `AQI_roll_mean/std/min/max_3`, `AQI_diff`,
   `Temperature_roll_mean_7`, `Humidity_roll_mean_7`, `log_PM2_5_lag_1`, `PM10_lag_1`,
   `wind_speed_lag_1`
7. Extended lags: `AQI_lag_3/7/14`, `PM2_5_lag_1/2/7`, `CO_lag_1`, `NO2_lag_1`
8. Extended rolling: `AQI_roll_mean/std/max/min_7/14`, `AQI_ewm_7/14/30`,
   `PM2_5_roll_mean_7`, `PM2_5_ewm_7`, `AQI_diff_2`
9. New variable features: lag-1 and 7-day rolling mean for `BLH`, `cloud_cover`,
   `shortwave_rad`, `uv_index`, `vpd`, `aod`, `dust`; `vpd` derived from
   Temperature and Humidity without a separate API call
10. Derived weather: `dew_point`, `temp_inversion`, `AQI_high_flag`, `stagnant_air`,
    `wind_dir_sin/cos`, `wind_x_PM2_5_lag1`
11. Interaction features: `PM2_5_x_Humidity`, `CO_x_Temperature`, `AQI_x_month_sin`
12. Tier-2 features: `AQI_trend_7d` (7-day slope), `AQI_x_wind`, `NO2_lag_3`,
    `Humidity_lag_3` (these are computed but excluded from training — see design decisions)
13. Weather leads t+1 through t+4: `Temperature_t1..t4`, `Humidity_t1..t4`,
    `Precipitation_t1..t4`, `wind_speed_t1..t4`, `BLH_t1..t4`, `cloud_cover_t1..t4`,
    `shortwave_rad_t1..t4`, `wind_dir_sin_t1..t4`, `wind_dir_cos_t1..t4`
14. Tier-3 leads + lags: `surface_pressure_t1..t4`, `apparent_temp_t1..t4`,
    `wind_gusts_t1..t4` (excluded from training — see design decisions)
15. Air quality leads t+1 through t+4: `PM2_5_t1..t4`, `aod_t1..t4`, `dust_t1..t4`,
    `uv_index_t1..t4` (PM2_5 leads excluded from training — see design decisions)
16. Targets: `AQI_t+1`, `AQI_t+2`, `AQI_t+3`, `AQI_t+4`

For the last 4 rows in the dataset (where shifted targets and leads are NaN because
no future observations exist yet), the script fetches the next 4 days of weather
from Open-Meteo's forecast API and the next 4 days of air quality from the CAMS
forecast API, then fills the lead feature columns in-place. Target columns
(`AQI_t+1` through `AQI_t+4`) remain NaN — they are unknowable today and are
intentionally omitted from MongoDB for those rows.

---

### Stage 4 — Model Training

**Script:** `src/train.py`

Loads `feature_store`, performs a time-aware train/test split (last 30 days = test),
trains three model types, saves all to `model_registry`, and logs all runs to
`model_logs`. Runs daily via GitHub Actions (`training_pipeline.yml`).

Data filter: rows before `2023-01-01` are excluded. Rows with any NaN in the
four target columns are dropped.

**Models trained:**

- **Ridge** (`src/models/ridge.py`): L2-regularised linear regression wrapped in
  `PerHorizonWrapper` — one Ridge model per forecast horizon (d1 through d4).
  Input scaled with `StandardScaler`.

- **LGBM** (`src/models/lgbm_model.py`): `LGBMRegressor` with 500 estimators,
  learning rate 0.05, max depth 7, 63 leaves, 80% row/column subsampling.
  Wrapped in `PerHorizonWrapper` — one independent LGBM model per horizon.
  Input scaled with `StandardScaler`.

- **LSTM** (`src/models/lstm_model.py`): Two-layer LSTM with architecture
  `Input(SEQ_LEN=7, n_features) → LSTM(64, L2=1e-3) → Dropout(0.4) →
  LSTM(32, L2=1e-3) → Dropout(0.4) → Dense(4)`. Uses Huber loss and the Adam
  optimiser. A 7-day sliding window builds sequences from the scaled feature
  matrix. Separate `StandardScaler` instances for X and y. EarlyStopping
  (patience=25) and ReduceLROnPlateau (factor=0.5, patience=10) callbacks.
  Produces all 4 forecast horizons in a single forward pass.

All three models are saved to `model_registry` with status `active`. A previous
active model of the same type is marked `inactive` before each new insert.
Metrics stored per model: `MAE`, `RMSE`, `R2` (overall), plus per-horizon
`MAE_d1..d4`, `RMSE_d1..d4`, `R2_d1..d4`.

---

### Stage 5 — Inference (Ensemble Forecast)

**Script:** `src/predict.py`

Loads the active `lgbm` and `lstm` models from `model_registry`, runs each on
the most recent row(s) from `feature_store`, blends their predictions with
per-horizon weights, and inserts one document to `predictions`.

**Ensemble blend weights** (derived from holdout analysis):

| Horizon | LGBM weight | LSTM weight |
|---------|-------------|-------------|
| Day 1   | 0.60        | 0.40        |
| Day 2   | 0.15        | 0.85        |
| Day 3   | 0.00        | 1.00        |
| Day 4   | 0.00        | 1.00        |

LGBM inference uses the last single row of `feature_store` (tabular).
LSTM inference uses the last 7 rows as a sequence. If either model is missing
from `model_registry`, the remaining model provides the full forecast without
blending.

Any NaN in the feature row (e.g. from a forecast API failure) is filled with
the column median across all available rows before inference.

---

### Stage 5b — LIME Explainability

**Script:** `src/create_lime.py`

Runs after `predict.py` in the daily training workflow. Loads the latest active
model (any type), retrieves the most recent processed row from `feature_store`,
and generates a LIME tabular explanation for the `AQI_t+1` prediction.

Outputs:
- `lime_explanations/lime_explanation.html` — interactive LIME report
- `lime_explanations/lime_explanation.csv` — top-15 feature weights as a table
- `lime_explanations/lime_explanation.png` — horizontal bar chart

The explanation is also persisted to the `lime_explanations` MongoDB collection
so the Streamlit dashboard can load it without relying on local files.

---

## Dataset Structure

**MongoDB database:** `karachi_aqi`
**Collection:** `feature_store`
**One document per calendar day, keyed on `date` (ISO 8601 string, e.g. `"2024-03-15"`).**

Training data starts from `2023-01-01`. Rows before this date may have sparse or
unreliable AQI values from Open-Meteo and are excluded during training.

**Date range:** 2023-01-01 to yesterday (grows daily via `update_daily_data.py`).

### Key column groups

| Group | Columns | Notes |
|---|---|---|
| Raw AQI | `AQI`, `PM2_5`, `PM10`, `NO2`, `SO2`, `CO`, `O3` | US AQI and individual pollutants |
| Raw weather | `Temperature`, `Humidity`, `Precipitation`, `wind_speed`, `wind_direction` | Core meteorological inputs |
| Extended weather | `apparent_temp`, `surface_pressure`, `wind_gusts`, `BLH`, `cloud_cover`, `shortwave_rad`, `uv_index`, `aod`, `dust` | Tier-2/3 variables; sparse in early rows |
| Log transforms | `log_PM2_5`, `log_CO`, `log_aod` | Right-skewed columns normalised with log1p |
| Wind encoding | `wind_sin`, `wind_cos`, `wind_dir_sin`, `wind_dir_cos` | Circular encoding of wind_direction |
| Temporal | `month`, `season_Spring/Summer/Winter`, `weekday_1..6` | Calendar context (Autumn and Monday are baseline dummies) |
| Cyclical | `month_sin/cos`, `doy_sin/cos`, `weekday_sin/cos` | Continuous cyclical time encoding |
| AQI lags | `AQI_lag_1`, `AQI_lag_2`, `AQI_lag_3`, `AQI_lag_7`, `AQI_lag_14` | Autoregressive features, all shift(1) |
| AQI rolling | `AQI_roll_mean/std/min/max_3`, `AQI_roll_mean/std/max/min_7/14`, `AQI_ewm_7/14/30`, `AQI_diff`, `AQI_diff_2` | Short and medium-window statistics |
| Pollutant lags | `PM2_5_lag_1/2/7`, `CO_lag_1`, `NO2_lag_1`, `PM10_lag_1`, `log_PM2_5_lag_1` | Lag-1/2/7 for key pollutants |
| Weather lags/rolling | `Temperature_roll_mean_7`, `Humidity_roll_mean_7`, `wind_speed_lag_1`, `BLH_lag_1`, `aod_lag_1`, `dust_lag_1`, `BLH_roll_mean_7`, etc. | Meteorological history |
| Derived weather | `dew_point`, `temp_inversion`, `stagnant_air`, `vpd`, `wind_x_PM2_5_lag1`, `AQI_high_flag` | Physics-inspired engineered features |
| Interaction | `PM2_5_x_Humidity`, `CO_x_Temperature`, `AQI_x_month_sin` | Multiplicative cross-terms |
| Weather leads | `Temperature_t1..t4`, `Humidity_t1..t4`, `Precipitation_t1..t4`, `wind_speed_t1..t4`, `BLH_t1..t4`, `cloud_cover_t1..t4`, `shortwave_rad_t1..t4`, `wind_dir_sin_t1..t4`, `wind_dir_cos_t1..t4` | Forward-shifted actuals for training; filled from forecast API for last 4 rows |
| AQ leads | `aod_t1..t4`, `dust_t1..t4`, `uv_index_t1..t4` | CAMS-sourced air quality forecasts (PM2_5 leads excluded — see design decisions) |
| Targets | `AQI_t+1`, `AQI_t+2`, `AQI_t+3`, `AQI_t+4` | Next-1/2/3/4-day US AQI; NaN for the last 4 rows |
| Metadata | `date`, `processed_at` | Excluded from model feature columns |

---

## How to Run the Pipeline Manually

Run these commands in order from the project root. Requires `.env` with
`MONGODB_USERNAME`, `MONGODB_PASSWORD`, and `MONGODB_CLUSTER` set.

```bash
# One-time only: populate feature_store from 2023-01-01 to today
python src/fetch_data.py

# Daily cycle (steps 1-4 below)

# 1. Fetch any missing raw daily rows up to yesterday
python src/update_daily_data.py

# 2. Rebuild all engineered features across the full dataset
python src/preprocess_daily_data.py

# 3. Train Ridge + LGBM + LSTM and save to model_registry
python src/train.py

# 4. Run ensemble inference and write forecast to predictions
python src/predict.py

# 5. Generate LIME explanation for the most recent row
python src/create_lime.py

# Launch the dashboard locally
streamlit run app.py
```

---

## CI/CD (GitHub Actions)

Two workflows run automatically. Both require three repository secrets:
`MONGODB_USERNAME`, `MONGODB_PASSWORD`, `MONGODB_CLUSTER`.

### `feature_pipeline.yml` — runs every hour

```
schedule: "0 * * * *"
```

Steps:
1. `python src/update_daily_data.py` — fetch missing raw daily rows
2. `python src/preprocess_daily_data.py` — rebuild all engineered features

This workflow keeps `feature_store` perpetually fresh. Running hourly (rather
than daily) ensures the processed rows used at inference time always reflect the
latest forecast-fill for the lead columns.

### `training_pipeline.yml` — runs daily at 08:00 PKT (03:00 UTC)

```
schedule: "0 3 * * *"
```

Steps:
1. `python src/train.py` — retrain all three models on the latest dataset
2. `python src/predict.py` — produce the 4-day ensemble forecast
3. `python src/create_lime.py` — regenerate LIME explanation
4. `git commit lime_explanations/ [skip ci]` — persist PNG/HTML/CSV to repo
5. `curl RENDER_DEPLOY_HOOK` — trigger a new Render deploy (requires
   `RENDER_DEPLOY_HOOK` secret; step skipped if secret is absent)

---

## Model Architecture

### Ridge (baseline)

One `sklearn.linear_model.Ridge` per horizon, wrapped in `PerHorizonWrapper`.
Features scaled with `StandardScaler`. Provides a fast linear baseline and
dominates the ensemble only at Day 1 when combined with LGBM.

### LightGBM (tabular gradient boosting, per-horizon)

One `LGBMRegressor` per horizon (d1 through d4), wrapped in `PerHorizonWrapper`.
Hyperparameters: 500 trees, learning rate 0.05, max depth 7, 63 leaves, 80%
row/column subsampling, random state 42.

Operates on a single feature vector (the most recent `feature_store` row at
inference time). LGBM dominates the ensemble for Day 1 forecasts; its weight
drops to zero for Days 3–4.

### LSTM (sequence model, multi-output)

Architecture: `LSTM(64, L2=1e-3) → Dropout(0.4) → LSTM(32, L2=1e-3) →
Dropout(0.4) → Dense(4)`.

Input shape: `(SEQ_LEN=7, n_features)`. A sliding window of 7 consecutive days
is used. Both the feature matrix (x_sc) and the target matrix (y_sc) are
independently scaled with `StandardScaler`; y_sc is inverted after prediction.

Training uses Huber loss (robust to outlier days), EarlyStopping with patience
25, and ReduceLROnPlateau (factor 0.5, patience 10, min_lr 1e-5). Maximum
epochs: 150, batch size: 16, validation split: 10%.

The LSTM produces all 4 horizons in a single forward pass (multi-output Dense(4)),
making it structurally different from the per-horizon LGBM approach. It
dominates the ensemble for Days 2–4.

### Weighted Ensemble

`predict.py` blends LGBM and LSTM predictions per horizon:

- Day 1: 60% LGBM + 40% LSTM
- Day 2: 15% LGBM + 85% LSTM
- Day 3: 100% LSTM
- Day 4: 100% LSTM

Weights were derived empirically from holdout analysis. The blend transitions
from LGBM dominance on Day 1 (where tabular features generalise well for
short-horizon predictions) to pure LSTM on Days 3–4 (where the sequential
model's ability to capture temporal dynamics outperforms the tabular approach).

---

## Key Design Decisions

### Why PM2.5 leads are excluded from training

`PM2_5_t1` through `PM2_5_t4` are computed in `preprocess_daily_data.py` (filled
from CAMS forecasts at inference time) but are explicitly excluded from the
`EXCLUDE_COLS` set in `train.py`. PM2.5 is the dominant driver of US AQI, so
including its future values in the training features inflates all metrics
artificially without producing a genuinely useful model. At inference time the
CAMS PM2.5 forecast would be present, but the resulting model would effectively
be "predict AQI from the PM2.5 that already encodes AQI" rather than learning
weather and atmospheric patterns. The other AQ leads (`aod_t1..t4`,
`dust_t1..t4`, `uv_index_t1..t4`) are kept because they add predictive signal
beyond the AQI series itself.

### Why t+4 features and targets were added

The original pipeline predicted 3 days ahead. Adding a fourth horizon (t+4
target and all corresponding lead features) required minimal changes and improves
the Day 4 ensemble output. Training cost is negligible for LGBM (one extra
model in `PerHorizonWrapper`) and zero for the LSTM (the `Dense(4)` output layer
already produces a 4-element vector).

### Why MongoDB is used as the feature store

A serverless deployment on Render (ephemeral filesystem) cannot persist CSV or
pickle files between runs. MongoDB Atlas provides a persistent, schemaless store
accessible from both GitHub Actions runners and the Render container without any
file-system dependency. The daily upsert pattern (keyed on `date`) makes
re-running any pipeline stage idempotent.

### Why tier-2 and tier-3 features are excluded at training time

Several features computed by `preprocess_daily_data.py` appear in `EXCLUDE_COLS`
in `train.py` and are therefore stored in MongoDB but never fed to the models:

- **Tier-2** (`AQI_trend_7d`, `AQI_x_wind`, `NO2_lag_3`, `Humidity_lag_3`):
  confirmed to hurt holdout performance — retained in storage for explainability
  analysis but gated out of training.
- **Tier-3** (`surface_pressure`, `apparent_temp`, `wind_gusts` and their leads/lags):
  these variables were sparse in historical rows (not available before a certain
  backfill date), causing NaN-heavy training columns that degraded model quality.
  They are stored and displayed in the dashboard but excluded from model features.

### Why the feature pipeline runs hourly

The lead features for the most recent rows are filled from the Open-Meteo and
CAMS forecast APIs. Forecast model outputs are updated by those services multiple
times per day. Running `preprocess_daily_data.py` hourly ensures the feature
vectors used at the next daily training run reflect the freshest available
atmospheric forecasts.
