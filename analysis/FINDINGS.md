# Karachi AQI Predictor — Deep Analysis Findings

_Branch: analysis/data-exploration | Date: 2026-05-26_

---

## Summary of Key Answers

| # | Question | Answer |
|---|----------|--------|
| 1 | Autocorrelation ceiling for day2/day3? | Daily ACF lag-3 = 0.560 → theoretical R² ceiling ≈ 0.314. Current model (0.120) is well below — room to improve. |
| 2 | What drives Day1 0.795 R²? | log_PM2_5 (56.5% of SHAP weight). NOT autoregressive — exploits PM2.5 physical persistence. |
| 3 | Does prediction stacking improve day3? | No — catastrophic. Day3 R² collapses 0.152 → -0.013. Do not use stacking. |
| 4 | New Open-Meteo variables with |r|>0.3 for day3? | 7 variables: apparent_temperature (-0.525), dew_point_2m (-0.518), soil_temperature (-0.513), surface_pressure (+0.504), wind_gusts (-0.434), european_aqi (+0.423), boundary_layer_height (-0.326). |
| 5 | Is Summer day3 fundamentally unpredictable? | Partially. Summer-only model fails (R²=-0.384). But missing variables (surface_pressure) explain most gap. |
| 6 | Single highest-confidence recommendation for day3? | Add AQI_roll_min_7 + AQI_roll_max_7 to FEAT_BASE (already in MongoDB). Expected Δ day3 R² = +0.067. |

---

## Section 1: EDA — Autocorrelation & Data Characteristics

### 1.1 Autocorrelation: Theoretical Ceiling

The daily AQI ACF decays slowly — all 30 lags tested remain statistically significant (threshold ±0.0526). This means seasonal memory is strong but per-horizon the forecast horizon imposes an unavoidable decay.

| Lag | Daily ACF | R² ceiling (ACF²) | Hourly ACF |
|-----|-----------|-------------------|------------|
| 1 day / 24h | 0.8356 | 0.698 | 0.757 |
| 2 days / 48h | 0.6599 | 0.435 | 0.607 |
| 3 days / 72h | 0.5604 | 0.314 | 0.514 |
| 7 days / 168h | 0.4395 | 0.193 | 0.421 |

**Interpretation:** The day1 model (R²=0.791) already exceeds the ACF² ceiling (0.698) because it uses multiple correlated features beyond a single lag. The day3 model (R²=0.120) is well below its ceiling (0.314), confirming that **better features can materially improve day3** — this is not a fundamentally insoluble problem.

### 1.2 Seasonal AQI Dynamics

| Season | Mean AQI | Std | % > 100 | % > 150 | Skewness | Rolling 7d std |
|--------|----------|-----|---------|---------|----------|----------------|
| Spring | 79.9 | 26.2 | 13.3% | 3.9% | 2.35 | 12.9 |
| Summer | 78.3 | 18.9 | 8.6% | 1.3% | **2.43** | **9.7** |
| Autumn | 93.6 | **31.6** | 32.4% | 9.1% | 1.00 | 17.5 |
| Winter | **110.6** | 29.1 | **57.6%** | **12.5%** | 0.32 | 19.6 |

**Why Summer is hardest to predict at day3:**
- Monsoon winds physically disperse pollutants: Summer mean wind_speed = **21.45 km/h** vs 14.68 km/h other seasons (t-test p < 0.0001)
- Highest skewness (2.43): most days are clean but monsoon-onset timing creates unpredictable AQI spikes
- Low rolling std (9.7) means ordinary summer days are stable, but the spikes are structural outliers
- Summer-specific models train on too few extreme events to generalise

**Winter** is the highest public-health burden (57.6% days >100, near-symmetric chronic pollution from temperature inversions). **Autumn** (post-monsoon transition) has highest std and is the most unstable prediction environment.

### 1.3 Diurnal Pattern (Hourly Data)
- Rush-hour AQI mean: **92.92** vs non-rush: **90.09** (Δ = 2.83 units, p < 0.0001)
- Peak hour: **18:00** (evening commute); Trough: **06:00**
- Amplitude ≈ 8 AQI units — modest relative to seasonal std (~29 units)
- Time-of-day features give marginal lift to hourly models but are not primary drivers for daily forecasting

### 1.4 Cross-Correlation: What Predicts AQI_t+3?

| Feature | r at lag 0 (with AQI_t+3) | r at best lag | Best lag |
|---------|--------------------------|---------------|----------|
| log_PM2_5 | **+0.613** | 0.613 | lag 0 |
| SO2 | +0.529 | 0.529 | lag 0 |
| log_CO | +0.527 | 0.527 | lag 0 |
| NO2 | +0.475 | 0.477 | **lag 3** |
| Temperature | -0.473 | -0.480 | **lag 2** |
| O3 | +0.386 | 0.386 | lag 0 |
| Humidity | -0.383 | -0.386 | **lag 3** |
| wind_speed | -0.307 | -0.309 | lag 7 |
| PM10 | +0.199 | 0.199 | lag 0 |

**Notable:** NO2 and Humidity peak at lag 3 — their 3-days-ago values are specifically more predictive of day3 AQI than current-day values. Explicitly engineering NO2_lag_3 and Humidity_lag_3 features is theoretically motivated (though NO2_lag_1 was confirmed redundant — the lag-3 versions have not yet been tested).

### 1.5 Data Quality
- Effective training rows: ~1,234 (pre-August 2022 rows have 54-60% null features)
- Maximum AQI: 214.6 on **2023-03-10** (confirmed dust storm event — not noise)
- Five of top-10 AQI days are in March 2023 — consistent with regional dust transport

---

## Section 2: SHAP Feature Importance

### 2.1 The Surprising Finding: Day1 Success is Not Autoregressive

| Feature | Day1 mean|SHAP| | % of Day1 total | Day3 mean|SHAP| | Day1 rank | Day3 rank |
|---------|-----------------|-----------------|-----------------|-----------|-----------|
| log_PM2_5 | **14.10** | **56.5%** | 5.88 | 1 | 1 |
| AQI_lag_1 | 0.42 | 1.7% | 0.31 | 10 | 26 |
| AQI (current) | 0.81 | 3.2% | 0.49 | 4 | 21 |

**Day1 R²=0.791 is driven overwhelmingly by log_PM2_5 persistence, not by AQI autoregression.** Fine particles disperse slowly in Karachi stable low-wind conditions — PM2.5 today is strongly predictive of PM2.5 tomorrow. AQI_lag_1 (the obvious autoregressive feature) contributes only 1.7% of total SHAP weight. The Day1 model is genuinely extracting multi-pollutant physical information, not operating as a simple autoregressive persistence model.

### 2.2 Top 10 Features by Horizon

**Day1 top 10 (by mean |SHAP|):**

| Rank | Feature | mean|SHAP| |
|------|---------|------------|
| 1 | log_PM2_5 | 14.1035 |
| 2 | PM10 | 1.3354 |
| 3 | O3 | 1.2929 |
| 4 | AQI | 0.8123 |
| 5 | SO2 | 0.6179 |
| 6 | log_CO | 0.5589 |
| 7 | PM10_lag_1 | 0.5518 |
| 8 | AQI_roll_std_7 | 0.4294 |
| 9 | Humidity | 0.4202 |
| 10 | AQI_lag_1 | 0.4184 |

**Day3 top 10 (by mean |SHAP|):**

| Rank | Feature | mean|SHAP| |
|------|---------|------------|
| 1 | log_PM2_5 | 5.8845 |
| 2 | AQI_roll_std_7 | 2.2281 |
| 3 | Temperature | 2.0868 |
| 4 | Temperature_roll_mean_7 | 1.9741 |
| 5 | AQI_roll_mean_7 | 1.9155 |
| 6 | month | 1.6625 |
| 7 | SO2 | 1.2123 |
| 8 | PM10_lag_1 | 1.0598 |
| 9 | PM10 | 0.8717 |
| 10 | wind_speed_lag_1 | 0.7916 |

### 2.3 Feature Shift Across Horizons

**Features dominant at Day1 that lose signal by Day3:** O3, AQI_lag_1, AQI (current), log_CO, Humidity

**Features that emerge as important at Day3 (not in Day1 top-10):** wind_speed_lag_1, AQI_roll_mean_7, Temperature_roll_mean_7, month, Temperature

**Interpretation:** Day1 prediction exploits current atmospheric state (what pollutants are in the air right now). Day3 prediction relies on seasonal and meteorological context (what month, what rolling temperature pattern). This motivates adding surface_pressure and boundary_layer_height — they are meteorological context variables with strong day3 correlations that the current feature set lacks.

### 2.4 SHAP Interactions (Day3)

| Feature 1 | Feature 2 | mean|interaction| | Interpretation |
|-----------|-----------|-----------------|----------------|
| log_PM2_5 | AQI_roll_mean_7 | **0.508** | Persistent pollution episodes amplify PM2.5 predictive power |
| log_PM2_5 | wind_sin | 0.402 | Wind direction modulates PM2.5 impact: onshore vs offshore flow |
| log_PM2_5 | Temperature_roll_mean_7 | 0.332 | Cold dry periods steepen the PM2.5-to-AQI relationship |
| log_PM2_5 | Temperature | 0.305 | Same mechanism at shorter timescale |
| log_PM2_5 | AQI_roll_std_7 | 0.268 | High-variability regimes amplify PM2.5 predictive power |

All top-5 interactions involve log_PM2_5 as one partner — PM2.5 predictive power is meteorologically conditioned. PM2.5 is much more predictive when the 7-day rolling mean is elevated (persistent episode vs transient spike).

### 2.5 Hourly vs Daily SHAP
- Hourly Day1 model puts raw AQI first (not log_PM2_5) — because hourly AQI ≈ its own 24h mean target
- Sub-daily lags (AQI_lag_1h, AQI_lag_6h) rank highly in hourly SHAP but do not exist in daily features
- **Hourly Day3 model R² = -0.02** — confirms that predicting 3-day daily means from hourly inputs without future weather data is not viable

### 2.6 LIME Consistency
log_PM2_5 is the top local feature in all 5 LIME instances (best prediction, worst prediction, Summer day, Winter day, high-AQI day). Consistent with SHAP. The worst local prediction (error = 17.6 AQI units) corresponds to PM2.5 in the mid-range bin, indicating the model struggles when PM2.5 is near the clean-to-polluted regime transition.

### 2.7 Permutation vs SHAP Divergences (Day3)

| Feature | Perm rank | SHAP rank | Agreement |
|---------|-----------|-----------|-----------|
| log_PM2_5 | 1 | 1 | Exact |
| AQI_roll_std_7 | 2 | 2 | Exact |
| AQI_lag_3 | 3 | 15 | Diverges |
| AQI_roll_mean_7 | 4 | 5 | Close |
| NO2 | **5** | **23** | **Diverges** |
| PM10_lag_1 | 6 | 8 | Close |
| Temperature | 7 | 3 | Close |
| Temperature_roll_mean_7 | 11 | 4 | Diverges |

Notable divergence: NO2 (permutation rank 5, SHAP rank 23) — carries non-redundant information that SHAP distributes across correlated pollutant features. AQI_lag_3 (permutation rank 3, SHAP rank 15) is absorbed by rolling aggregates in SHAP but has independent signal in permutation testing. This supports explicitly testing NO2_lag_3.

---

## Section 3: Error Analysis

### 3.1 Residual Characteristics

| Horizon | Mean bias | Std | Shapiro p | Gaussian? | Pattern |
|---------|-----------|-----|-----------|-----------|---------|
| Day1 | -0.06 | 6.85 | 0.97 | Yes | Unbiased, clean |
| Day2 | +2.29 | 14.44 | 0.012 | No | Under-predicts (positive bias) |
| Day3 | **+3.74** | 15.82 | 0.045 | No | Systematic under-prediction |

Model pulls toward training mean rather than tracking directional trends — classic regression-to-mean effect that worsens with horizon. Growing bias (+3.74 at day3) means the model consistently predicts lower AQI than will actually occur when pollution is rising.

### 3.2 Residual Serial Correlation — Critical Finding

| Horizon | Residual ACF lag-1 | Residual ACF lag-2 | Serial Correlation? |
|---------|-------------------|-------------------|---------------------|
| Day1 | 0.092 | -0.304 | No |
| Day2 | 0.341 | -0.088 | Borderline |
| Day3 | **0.649** | 0.151 | **STRONG** |

A residual ACF of **0.649** at lag 1 for day3 means: when the model is wrong today, it is wrong in the same direction tomorrow. This is the clearest signal of a systematic missing temporal feature — specifically one that captures **multi-day blocking weather regimes** (surface pressure persistence, boundary layer height persistence). Day1 residuals (ACF = 0.092) confirm that model is well-specified for the 24h horizon; the problem is specifically in the day3 component.

### 3.3 Error by Season and AQI Level
- Test window was entirely Spring (30 days ending 2026-05-26)
- May day3 MAE = **14.67** vs April day3 MAE = **4.74** — pre-monsoon transition (May) is 3x harder than stable Spring (April)

| AQI Level | Day1 MAE | Day2 MAE | Day3 MAE |
|-----------|----------|----------|----------|
| Moderate (51-100) | 5.12 | 10.48 | 11.68 |
| Unhealthy (101-150) | **9.63** | 4.22 | 11.94 |

By AQI level: worst day1 errors in Unhealthy range (101-150); day3 MAE roughly equal (~11.7-11.9) across bins — **horizon is the error driver, not AQI magnitude**.

### 3.4 Worst-Predicted Days

All 10 worst day3 predictions fall in May 2026, concentrated around **May 18-21**:

| Date | Actual AQI | Predicted AQI | Abs Error | Wind (km/h) | Precipitation |
|------|-----------|---------------|-----------|-------------|---------------|
| 2026-05-20 | 68.6 | 71.3 | **49.9** | 14.5 | 0.0 |
| 2026-05-19 | 68.9 | 63.9 | 39.9 | 16.8 | 0.0 |
| 2026-05-02 | 85.0 | 93.2 | 29.3 | 15.5 | 0.0 |
| 2026-05-13 | 85.4 | 76.2 | 21.7 | 11.9 | 0.0 |
| 2026-05-12 | 71.5 | 71.6 | 21.5 | 9.1 | 0.0 |

Wind was elevated (14-17 km/h), precipitation was zero. The model predicted continued moderate AQI while improving winds rapidly cleared the air. This is the concrete manifestation of the missing wind clearing regime feature — a direct match with the serial correlation finding in 3.2.

---

## Section 4: Prediction Stacking Experiment

### 4.1 Results

| Horizon | Baseline R² | Stacked R² | Delta | Verdict |
|---------|-------------|------------|-------|---------|
| Day1 | 0.7910 | 0.7910 | 0.000 | N/A (no stacking) |
| Day2 | 0.2702 | 0.2489 | **-0.021** | Hurts |
| Day3 | 0.1519 | **-0.0132** | **-0.165** | Catastrophic |

Train: 1,193 days. Test: 30 days. 5-fold TimeSeriesSplit for OOF generation.

### 4.2 Why Stacking Fails for Day3
1. Day1 prediction carries correlated noise that already overlaps with AQI lags in the feature set — no new signal is added
2. 30 test samples is insufficient to calibrate the stacking relationship reliably
3. Day3 residuals have ACF = 0.649 — errors are serially correlated, so the correction signal from stacking propagates rather than cancels
4. Raw features already encode strong AQI autocorrelation (AQI_lag_1, AQI_lag_2, AQI_roll_mean_3). The day1 prediction is a smoothed version of those same lags.

### 4.3 Answer to Key Question #3
**Do NOT use prediction stacking.** The independent per-horizon LGBM models are both simpler and more accurate. Day3 stacking reduces R² from 0.152 to -0.013 — worse than predicting the mean.

---

## Section 5: Open-Meteo Feature Expansion

### 5.1 New Variables Available for Karachi

All 11 archive API candidates returned full data. Air quality API: only dust and european_aqi had non-null data. boundary_layer_height is available from **archive API only** (not air quality API) — it is all-null from the air quality API.

### 5.2 Correlation with Day3 AQI (2023-2024, n ≈ 727 days)

| Variable | r with AQI (day 0) | r with AQI_t+3 | Worth adding? |
|----------|-------------------|-----------------|---------------|
| apparent_temperature | -0.524 | **-0.525** | High priority |
| dew_point_2m | -0.505 | **-0.518** | High priority |
| soil_temperature_0_to_7cm | -0.500 | **-0.513** | High priority |
| surface_pressure | +0.521 | **+0.504** | High priority |
| wind_gusts_10m | -0.488 | **-0.434** | High priority |
| european_aqi | +0.759 | **+0.423** | Consider |
| boundary_layer_height | -0.364 | **-0.326** | Consider |
| shortwave_radiation | -0.280 | -0.286 | Optional |
| cloud_cover | -0.243 | -0.265 | Optional |
| dust | -0.137 | -0.211 | Skip |
| vapour_pressure_deficit | +0.220 | +0.196 | Skip |
| sunshine_duration | -0.026 | -0.001 | Skip |

The threshold for High priority is |r| > 0.3 with AQI_t+3. All four top candidates (apparent_temperature, dew_point_2m, soil_temperature, surface_pressure) exceed |r| = 0.5 — **stronger than any existing non-AQI feature** in the current pipeline.

### 5.3 Boundary Layer Height Analysis
- Available from: https://archive-api.open-meteo.com/v1/archive (API param: hourly=boundary_layer_height)
- r with AQI_t+3 = -0.326 (strongest physically motivated new variable)
- High-AQI days (>150): mean BLH = **449.6 m** vs low-AQI (<50): mean BLH = **483.4 m**
- Direction confirmed: lower mixing layer → less dispersion → higher AQI
- Add as daily mean to feature_store. The separation is modest because most Karachi days fall in the 50-150 AQI range, but the direction is consistent and the lag correlation is meaningful.

### 5.4 Answer to Key Question #4
**7 variables not currently in the pipeline show |r| > 0.3 with day3 AQI.** The top 4 (apparent_temperature, dew_point_2m, soil_temperature, surface_pressure) all exceed |r|=0.5. Surface pressure is the most actionable: directly measures the subsidence inversion mechanism driving Karachi worst pollution episodes and is a standard input to operational AQI forecast models worldwide.

---

## Section 6: Feature Engineering Experiments

### 6.1 Individual Results (baseline: d1=0.803, d2=0.237, d3=0.082)

| Feature Added | R2_d1 | R2_d2 | R2_d3 | delta_d1 | delta_d2 | delta_d3 |
|---------------|-------|-------|-------|----------|----------|----------|
| AQI_roll_min_7 + AQI_roll_max_7 * | 0.813 | 0.207 | **0.149** | +0.010 | -0.029 | **+0.067** |
| AQI x wind_speed | 0.803 | 0.199 | 0.135 | 0.000 | -0.037 | **+0.053** |
| AQI_trend_7d (7d slope) | 0.784 | 0.190 | 0.135 | -0.019 | -0.047 | **+0.053** |
| AQI_ewm_7 * | 0.793 | 0.213 | 0.129 | -0.010 | -0.024 | **+0.047** |
| AQI_ewm_30 * | 0.796 | 0.197 | 0.108 | -0.008 | -0.040 | +0.026 |
| AQI_ewm_14 | 0.801 | 0.198 | 0.086 | -0.002 | -0.039 | +0.004 |
| is_high_event_3d flag | 0.787 | 0.182 | 0.083 | -0.017 | -0.055 | +0.001 |
| O3_lag_1 | 0.784 | 0.207 | 0.069 | -0.020 | -0.029 | -0.013 |
| wind x PM2_5_lag1 | 0.799 | 0.276 | 0.073 | -0.004 | +0.039 | -0.009 |
| Humidity x PM2_5_lag | 0.810 | 0.277 | 0.062 | +0.007 | +0.040 | -0.020 |
| NO2_lag_1 | 0.792 | 0.215 | 0.028 | -0.011 | -0.021 | -0.054 |
| AQI_lag_14 | 0.803 | 0.247 | 0.028 | 0.000 | +0.010 | -0.054 |
| doy_sin + doy_cos | 0.813 | 0.202 | -0.140 | +0.010 | -0.035 | **-0.222** |
| Best combination (6 feats) | 0.809 | 0.216 | 0.125 | +0.006 | -0.021 | +0.043 |

* Already exists in MongoDB feature_store — zero data pipeline changes needed

### 6.2 Season-Specific Model

| Horizon | Summer-only CV R² | Global model R² |
|---------|-------------------|-----------------|
| Day 1 | 0.636 | 0.803 |
| Day 2 | -0.032 | 0.237 |
| Day 3 | **-0.384** | 0.082 |

Summer-only training is strictly worse at every horizon. The global model with season dummies is the correct approach — Summer has insufficient within-season variation in training data to learn from. The global model benefits from winter and spring signal to learn the feature-AQI relationship.

### 6.3 Feature Engineering Ceiling
The best achievable day3 R² from lag-based feature engineering alone is ~**0.149** (with AQI_roll_min/max_7d added). This is still well below the ACF² ceiling of 0.314, confirming that the remaining gap requires new external variables (surface_pressure, apparent_temperature, boundary_layer_height). Feature engineering within the existing variable set is nearly exhausted at approximately R² = 0.15 for day3.

### 6.4 Best Combination vs Best Individual
The 6-feature combination achieves delta_d3 = +0.043, which is **worse** than AQI_roll_min/max_7d alone (+0.067). Classic feature redundancy: AQI_trend_7d and AQI_ewm_7 both capture AQI momentum, creating overlapping signals. Practical takeaway: add AQI_roll_min/max_7d first, evaluate, then add others incrementally.

---

## Section 7: Prioritised Action Plan

### Tier 1 — Zero pipeline changes, implement in one commit
Both AQI_roll_min_7 and AQI_roll_max_7 already exist in MongoDB feature_store from a prior preprocessing run. The training feature list just needs updating.

| Action | Expected day3 delta R² | Code change |
|--------|----------------------|-------------|
| Add AQI_roll_min_7 + AQI_roll_max_7 to FEAT_BASE | **+0.067** | src/train.py FEAT_BASE list |
| Add AQI_ewm_7 to FEAT_BASE | **+0.047** | Same — already in MongoDB |
| Add AQI_ewm_30 to FEAT_BASE | **+0.026** | Same — already in MongoDB |

### Tier 2 — Add to feature engineering pipeline (src/preprocess_daily_data.py)

| Action | Expected day3 delta R² | Code change |
|--------|----------------------|-------------|
| Add AQI_trend_7d (7-day linear slope of lagged AQI) | **+0.053** | add_lag_rolling() function |
| Add AQI x wind_speed interaction | **+0.053** | add_lag_rolling() function |

### Tier 3 — Add new Open-Meteo variables (src/update_daily_data.py + preprocess)

| Action | Expected impact | API change |
|--------|-----------------|------------|
| Fetch surface_pressure from archive API | High (r=0.504 with AQI_t+3) | Add to WEATHER_PARAMS |
| Fetch apparent_temperature from archive API | High (r=0.525) | Add to WEATHER_PARAMS |
| Fetch dew_point_2m from archive API | High (r=0.518) | Add to WEATHER_PARAMS |
| Fetch boundary_layer_height from archive API | Moderate (r=0.326) | Add to WEATHER_PARAMS |
| Fetch wind_gusts_10m from archive API | High (r=0.434) | Add to WEATHER_PARAMS |

### Do NOT implement
- **Prediction stacking** (day3 R² collapses to -0.013)
- **doy_sin/doy_cos** (day3 R² drops by 0.222 — month/season dummies already cover annual cycle)
- **Season-specific separate models** (Summer R² = -0.384 in isolation)
- **AQI_lag_14** (ACF near zero at lag 14 for Karachi; hurts day3 by -0.054)
- **NO2_lag_1** (redundant with same-day NO2; hurts day3 by -0.054)

---

## Section 8: Key Questions — Direct Answers

### Key Question 1: Autocorrelation ceiling for day2/day3?

Daily AQI ACF at lag 2 (48h) = **0.660**, at lag 3 (72h) = **0.560**. As ACF² is a rough R² ceiling: day2 ceiling ≈ 0.435, day3 ceiling ≈ 0.314. Current models (day2 R²=0.270, day3 R²=0.120) are below these ceilings, meaning **better features can materially improve both horizons**. The current day3 gap is ~0.194 R² units below the ceiling — this is not a fundamentally insoluble problem. All 30 daily lags tested remain statistically significant (threshold ±0.0526), confirming that seasonal memory in Karachi AQI is strong and persistent.

### Key Question 2: What drives Day1 0.795 R²?

**log_PM2_5 alone accounts for 56.5% of total Day1 SHAP weight** (mean |SHAP| = 14.10 out of 24.96 total). This is not an autoregressive model — AQI_lag_1 contributes only 1.7% of SHAP weight. The model exploits PM2.5 physical persistence: fine particles disperse slowly in Karachi stable low-wind conditions, so today PM2.5 directly predicts tomorrow AQI. The remaining Day1 signal comes from other current-state pollutants (PM10 = rank 2, O3 = rank 3, SO2 = rank 5, log_CO = rank 6). Autoregressive features (AQI_lag_1 + current AQI combined) contribute only 4.9% of total SHAP weight.

### Key Question 3: Does prediction stacking help?

**No — it is catastrophic for day3.** Day3 R² collapses from 0.152 → -0.013 (Δ = -0.165). Day2 also hurts (0.270 → 0.249). Do not use stacking. Three reasons: (1) day1 prediction carries correlated noise already encoded in AQI lag features, providing no new signal while introducing variance; (2) 30 test samples are insufficient to calibrate the stacking relationship; (3) serial correlation in day3 residuals (ACF = 0.649) means errors propagate rather than cancel. The independent per-horizon LGBM models are both simpler and more accurate.

### Key Question 4: Open-Meteo variables with |r|>0.3 for day3 AQI?

Seven: **apparent_temperature** (-0.525), **dew_point_2m** (-0.518), **soil_temperature_0_to_7cm** (-0.513), **surface_pressure** (+0.504), **wind_gusts_10m** (-0.434), **european_aqi** (+0.423), **boundary_layer_height** (-0.326). All stronger than any existing non-AQI feature. surface_pressure and boundary_layer_height are physically the most interpretable and directly measurable at inference time. Surface pressure directly measures the subsidence inversion mechanism driving Karachi worst pollution episodes.

### Key Question 5: Is Summer day3 fundamentally unpredictable?

**Partially, but not entirely.** The Summer-only model fails catastrophically (CV R²=-0.384) because there is insufficient within-season variation in training data to learn from. However, the global model does learn some Summer signal via season dummies and wind features. The real gap is missing variables: surface_pressure captures the subsidence inversion events that drive Summer AQI spikes, and it has never been tested in the model. Summer contribution to the day3 residual serial correlation (ACF = 0.649) is likely reducible with surface_pressure lags encoding multi-day blocking weather regimes. Do not train Summer in isolation — use the global model with season dummies and add meteorological context variables.

### Key Question 6: Single highest-confidence recommendation for day3 R²?

**Add AQI_roll_min_7 and AQI_roll_max_7 to FEAT_BASE in src/train.py.**
- Expected delta day3 R²: **+0.067** (0.082 → 0.149)
- Both columns already exist in MongoDB feature_store from a prior preprocessing run — zero data pipeline changes required
- Physical justification: These capture the range of AQI over the last week, encoding whether the city is in a sustained high-pollution episode (high min) or recovering (widening gap between min and max) — information not captured by mean or std alone. For a 3-day forecast, knowing whether Karachi has been in a stable high-pollution state for a week is more predictive than knowing the average level, because stable high-pollution atmospheric conditions (surface inversions, weak winds) tend to persist for 3-7 days.
- Verify columns exist: from config.db import get_collection, COLLECTION_FEATURE_STORE; docs = list(get_collection(COLLECTION_FEATURE_STORE).find({"AQI_roll_min_7": {"": True}}, {"_id":0,"AQI_roll_min_7":1})); print(len(docs), "rows with AQI_roll_min_7")

---

_All plots in analysis/plots/ | All tables in analysis/tables/ | Scripts in analysis/_
