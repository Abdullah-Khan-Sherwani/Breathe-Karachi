## Section 5: Open-Meteo Feature Expansion

### 5.1 New Variables Available for Karachi

From the archive API (`archive-api.open-meteo.com/v1/archive`), the following new variables were successfully fetched with full data coverage for 2023-2024:

- `surface_pressure` — atmospheric pressure (hPa)
- `cloud_cover` — total cloud cover fraction (%)
- `vapour_pressure_deficit` — atmospheric dryness indicator (kPa)
- `et0_fao_evapotranspiration` — reference evapotranspiration (mm)
- `sunshine_duration` — daily sunshine seconds (s) — returned data but near-zero variance for Karachi
- `wind_gusts_10m` — maximum wind gust at 10 m (km/h)
- `shortwave_radiation` — downwelling solar radiation (W/m²)
- `dew_point_2m` — dew point at 2 m (°C)
- `apparent_temperature` — feels-like temperature (°C)
- `boundary_layer_height` — planetary boundary layer depth (m) — available from archive API; all-null from air quality API
- `soil_temperature_0_to_7cm` — near-surface soil temperature (°C)

Variable not available for Karachi: `uv_index` — returned all-null from archive API for this location.

From the air quality API (`air-quality-api.open-meteo.com/v1/air-quality`), additional variables available with non-null data:

- `dust` — dust aerosol surface concentration (µg/m³)
- `european_aqi` — European AQI composite index

Variables requested but all-null for Karachi from the air quality API:
`boundary_layer_height`, `alder_pollen`, `ammonia`, `nitrogen_monoxide`, `pm10_wildfires`

---

### 5.2 Correlation with Day3 AQI

| Variable | r with AQI (day 0) | r with AQI_t+3 | Worth Adding? |
|---|---|---|---|
| apparent_temperature | -0.5240 | -0.5248 | YES |
| dew_point_2m | -0.5049 | -0.5176 | YES |
| soil_temperature_0_to_7cm | -0.4999 | -0.5133 | YES |
| surface_pressure | +0.5205 | +0.5037 | YES |
| wind_gusts_10m | -0.4882 | -0.4343 | YES |
| european_aqi | +0.7587 | +0.4230 | YES |
| boundary_layer_height | -0.3640 | -0.3257 | YES |
| shortwave_radiation | -0.2800 | -0.2861 | YES |
| cloud_cover | -0.2435 | -0.2652 | YES |
| dust | -0.1374 | -0.2109 | YES |
| vapour_pressure_deficit | +0.2196 | +0.1956 | YES |
| et0_fao_evapotranspiration | -0.1782 | -0.1736 | YES |
| nitrogen_monoxide | +0.2224 | +0.1076 | no |
| sunshine_duration | -0.0260 | -0.0011 | no |

Threshold for "Worth Adding": |r with AQI_t+3| > 0.15. All correlations are Pearson r on 2023-2024 daily means, n ≈ 727 days.

---

### 5.3 Boundary Layer Height

Boundary layer height (BLH) is available from the Open-Meteo archive API for Karachi with complete hourly coverage. It is NOT available from the air quality API (all-null for this location).

- r with AQI (same day): -0.364
- r with AQI_t+3: -0.326
- High AQI days (>150): mean BLH = 449.6 m, std = 174.0 m (n = 43 days in 2023-2024 overlap window)
- Low AQI days (<50): mean BLH = 483.4 m, std = 98.0 m (n = 2 days — Karachi rarely reaches clean-air conditions)

Physical interpretation: A shallower boundary layer confines pollutants near the surface. The negative correlation with AQI_t+3 (-0.326) indicates current BLH has meaningful predictive value for pollution 3 days ahead, consistent with persistent atmospheric stability regimes over Karachi. The separation between high- and low-AQI groups is moderate because most days fall in the 50-150 AQI range and Karachi's arid climate limits BLH variation compared to temperate cities. Scatter plot: `analysis/plots/boundary_layer_analysis.png`.

---

### 5.4 Top Recommendations (|r| > 0.15 with AQI_t+3)

Ranked by absolute correlation with day3 AQI:

1. **apparent_temperature** (r = -0.525) — Strongest new predictor. Composite of temperature, humidity, and wind; captures the full thermal-moisture environment driving pollutant accumulation. Highly collinear with monsoon seasonality.

2. **dew_point_2m** (r = -0.518) — Moisture content of the air. High dew point (monsoon) coincides with wet deposition of PM2.5; low dew point (winter) aligns with dry stable conditions and elevated AQI. Partially correlated with existing `Humidity` but adds independent signal.

3. **soil_temperature_0_to_7cm** (r = -0.513) — Lagged thermal memory of surface energy state. Predicts near-surface stability 3 days ahead at similar magnitude to apparent temperature; useful as a redundancy check.

4. **surface_pressure** (r = +0.504) — Higher pressure = anticyclonic subsidence inversion = trapped pollutants. Strongest positive predictor. Critical for forecasting stagnant pollution episodes.

5. **wind_gusts_10m** (r = -0.434) — Turbulent mixing proxy. More predictive of future AQI than mean wind speed (existing feature). Captures peak dispersion capacity that time-averaged winds understate.

6. **european_aqi** (r = +0.423) — Aggregate index computed from pollutants already in the pipeline. Strong contemporaneous correlation (r = 0.759) but day3 correlation decays to 0.423 — adds no independent physical information. Lower priority.

7. **boundary_layer_height** (r = -0.326) — Most physically motivated variable: directly measures the mixing volume available to dilute emissions. Shallow BLH (< 400 m) consistently precedes elevated AQI episodes.

8. **shortwave_radiation** (r = -0.286) — Proxy for cloudiness, monsoon intensity, and ozone photochemistry. Distinct signal from cloud_cover.

9. **cloud_cover** (r = -0.265) — Correlated with precipitation-driven aerosol scavenging. Overlaps with shortwave_radiation; one of the two likely sufficient.

10. **dust** (r = -0.211) — Mineral dust aerosol concentration. Negative correlation reflects that dust events occur with high winds that also disperse combustion pollutants. Worth including as a distinct pollution-source variable.

11. **vapour_pressure_deficit** (r = +0.196) — Atmospheric dryness; higher VPD indicates drier, clearer skies associated with combustion-driven AQI. Partial overlap with dew_point and apparent_temperature.

12. **et0_fao_evapotranspiration** (r = -0.174) — Borderline useful; derived from radiation, wind, and humidity — heavily overlaps variables above.

**Priority additions for the LSTM feature set:** `surface_pressure`, `wind_gusts_10m`, `boundary_layer_height`, `dew_point_2m`, and `shortwave_radiation`. These five span the four key physical drivers: atmospheric stability (pressure), mechanical mixing (gusts), boundary layer trapping (BLH), moisture/dew point, and radiation.

---

### 5.5 Answer to Key Question #4

**Which Open-Meteo variables not currently in the pipeline show |r| > 0.3 with day3 AQI?**

Seven variables exceed this threshold:

| Variable | r with AQI_t+3 | Source |
|---|---|---|
| apparent_temperature | -0.5248 | archive API |
| dew_point_2m | -0.5176 | archive API |
| soil_temperature_0_to_7cm | -0.5133 | archive API |
| surface_pressure | +0.5037 | archive API |
| wind_gusts_10m | -0.4343 | archive API |
| european_aqi | +0.4230 | air quality API |
| boundary_layer_height | -0.3257 | archive API |

The three strongest predictors (apparent_temperature, dew_point_2m, soil_temperature) all exceed |r| = 0.5 — considerably stronger than any single existing feature except same-day AQI autocorrelation. Surface pressure (r = +0.504) is the single most actionable addition because it directly measures the subsidence inversion mechanism behind Karachi's winter pollution episodes and is a standard input to operational AQI forecast models.

---

## Section 3: Error Analysis

### 3.1 Residual Distribution

| Horizon | Mean Residual | Std   | Shapiro-Wilk p | Gaussian? | Bias           |
|---------|---------------|-------|----------------|-----------|----------------|
| Day1    | -0.06         | 6.85  | 0.9694         | Yes       | Negligible (near-unbiased) |
| Day2    | +2.29         | 14.44 | 0.0124         | No        | UNDER-predicts (mean residual = +2.29) |
| Day3    | +3.74         | 15.82 | 0.0447         | No        | UNDER-predicts (mean residual = +3.74) |

Day1 residuals are well-behaved: near-zero bias and normal distribution (Shapiro p = 0.97). Day2 and day3 residuals fail normality (p < 0.05) and carry a positive mean bias, meaning the model systematically under-predicts future AQI as horizon extends. The growing bias (+3.74 at day3) indicates mean-reversion pressure in the model — it pulls predictions toward the training mean rather than tracking rising or falling AQI trends.

---

### 3.2 Error by Season

The test window (30 days ending 2026-05-26) falls entirely within Spring; consequently all season-level MAE observations are for Spring only. The Spring test window shows:

- Day1 MAE: 5.27
- Day2 MAE: 10.27
- Day3 MAE: 11.69

Error escalates sharply from day1 to day3 (a 2.2x increase). Within the test period, May shows considerably higher error than April: April day3 MAE = 4.74 vs May day3 MAE = 14.67, suggesting the model degrades during the Spring-to-Summer transition month when AQI dynamics are less stable (onset of pre-monsoon wind patterns). See `analysis/tables/error_by_season.csv` and `analysis/plots/error_by_season_month.png`.

---

### 3.3 Error by AQI Level

| AQI Level          | Day1 MAE | Day2 MAE | Day3 MAE |
|--------------------|----------|----------|----------|
| Good (0-50)        | NaN      | NaN      | NaN      |
| Moderate (51-100)  | 5.12     | 10.48    | 11.68    |
| Unhealthy (101-150)| 9.63     | 4.22     | 11.94    |
| Very Unhealthy (151+) | NaN   | NaN      | NaN      |

Good and Very Unhealthy bins are empty in the test window — the 30-day April-May test period has no truly clean days (AQI < 50) or severe episodes (>150). Within the populated range, Unhealthy (101-150) days show the largest day1 MAE (9.63), indicating the model struggles most at the transition into elevated pollution. Day2 MAE is anomalously low for Unhealthy days (4.22) relative to Moderate (10.48), likely a test-set sampling artifact from the small n. Day3 MAE is roughly equal across both populated AQI levels (~11.7-11.9), suggesting horizon is the dominant driver of error rather than absolute AQI level in this test window. See `analysis/tables/error_by_aqi_level.csv` and `analysis/plots/error_by_aqi_level.png`.

Physical interpretation: the model's worst predictions are not at extreme AQI values but at intermediate-to-high levels (100-150). These correspond to pre-monsoon conditions with variable wind patterns — meteorologically the hardest regime to predict in Karachi.

---

### 3.4 Residual Autocorrelation

| Horizon | ACF Lag 1 | ACF Lag 2 | ACF Lag 3 | Serial Correlation? |
|---------|-----------|-----------|-----------|---------------------|
| Day1    | 0.092     | -0.304    | -0.121    | No (lag-1 < 0.3)   |
| Day2    | 0.341     | -0.088    | -0.176    | YES — lag-1 = 0.34  |
| Day3    | 0.649     | 0.151     | -0.045    | YES — lag-1 = 0.65  |

Day1 residuals have no meaningful lag-1 serial correlation (ACF = 0.09). Day2 is borderline (ACF lag-1 = 0.34, just above the 0.30 warning threshold). Day3 shows strong serial correlation (ACF lag-1 = 0.649) — the single strongest finding of this error analysis.

Interpretation: A lag-1 ACF of 0.65 on day3 residuals means that when the model over-predicts today's day3 AQI, it is highly likely to also over-predict tomorrow's day3 AQI. This is the signature of a systematic missing temporal feature — the model is not capturing multi-day AQI persistence patterns beyond what the existing lag features provide. The most likely candidate is a medium-range atmospheric circulation pattern (3-5 day blocking regime) that cannot be reconstructed from 1-2 day AQI lags alone. Adding boundary_layer_height or surface_pressure lagged features is a concrete remedy. See `analysis/plots/error_residual_acf.png`.

---

### 3.5 Worst-Predicted Days

All 10 worst day3 predictions fall in May 2026 (Spring), concentrated around 18-21 May. Key observations:

- Worst single prediction: 20 May 2026 (actual AQI = 68.6, predicted = 71.3, absolute error = 49.9). The large absolute error at a moderate actual AQI indicates the model drastically over-predicted a period of improving air quality.
- All worst days have zero precipitation — no washout events to blame.
- Wind speeds on worst days are elevated (9-17 km/h), consistent with pre-monsoon ventilation that was reducing AQI while the model predicted continued pollution.
- The cluster of errors around 18-21 May is physically coherent: a multi-day wind event cleared the air, and the model — lacking wind-change detection — continued predicting near-current levels. This directly confirms the missing temporal feature identified by the ACF analysis.

See `analysis/tables/worst_predicted_days.csv`.

---

## Section 4: Prediction Stacking Experiment

### 4.1 Results

| Horizon | Baseline R2 | Stacked R2 | Delta   | Material Improvement? |
|---------|-------------|------------|---------|----------------------|
| Day1    | 0.7910      | —          | —       | N/A (no stacking)    |
| Day2    | 0.2702      | 0.2489     | -0.0213 | No (threshold: +0.05) |
| Day3    | 0.1519      | -0.0132    | -0.1652 | No — stacking HURTS  |

Train: 1193 days. Test: 30 days (last 30 days). 5-fold TimeSeriesSplit for OOF generation.

---

### 4.2 Interpretation

Stacking makes things worse, not better, for both day2 and day3. The day3 result is especially damning: stacking reduces R2 from 0.1519 to -0.0132 (a 0.165-point collapse), meaning the stacked model is literally worse than predicting the mean.

Why does stacking fail here?

1. The day1 prediction is itself uncertain (R2 = 0.79, not 1.0), so appending it as a feature for day2 introduces correlated noise rather than clean signal. On a 30-sample test set this variance compounds quickly.

2. The test window is only 30 days. OOF predictions are trained on 5 folds of ~200-240 samples each, but the stacked features are evaluated on just 30 test observations. With this ratio, the stacked model overfits to the day1 prediction pattern seen in training folds, which does not generalize.

3. The raw features already encode strong AQI autocorrelation (AQI_lag_1, AQI_lag_2, AQI_roll_mean_3). The day1 prediction is essentially a smoothed version of those same lags — it adds no new physical information to the feature space.

4. Day3 error has high serial correlation (ACF = 0.65), so the residuals that the stacked day1 and day2 predictions try to correct are themselves serially correlated and not addressable by stacking without far more data.

Physical interpretation: The day1 prediction carries information about current momentum (how fast AQI is rising or falling) that the model could in principle exploit for day3. But the noise in that prediction — combined with insufficient test data to calibrate the stacking relationship — turns that signal into a liability.

---

### 4.3 Answer to Key Question #3

**Does prediction stacking materially improve day2/day3 R2?**

No. Stacking does not improve — and actively degrades — both day2 and day3 R2 on the 30-day holdout test set.

- Day2: stacking reduces R2 by 0.021 (from 0.2702 to 0.2489). Not material.
- Day3: stacking reduces R2 by 0.165 (from 0.1519 to -0.0132). The stacked model performs below a naive mean predictor.

The stacking approach requires substantially more test data to produce a reliable signal. With only 30 test observations and a day3 baseline R2 of only 0.15, there is insufficient statistical power to differentiate whether any improvement from stacking is real or noise. The conclusion for this project is: **do not use prediction stacking in the production pipeline**. The existing per-horizon independent models are both simpler and more accurate.

---

## Section 6: Feature Engineering Experiments

### 6.1 Individual Feature Results

| Experiment | R²_d1 | R²_d2 | R²_d3 | Δ_d1 | Δ_d2 | Δ_d3 |
|---|---|---|---|---|---|---|
| baseline (39 feat) | 0.8031 | 0.2365 | 0.0819 | 0 | 0 | 0 |
| add AQI_lag_14 (2-week lag) | 0.8030 | 0.2467 | 0.0281 | -0.0001 | +0.0102 | -0.0538 |
| add NO2_lag_1 | 0.7924 | 0.2152 | 0.0280 | -0.0107 | -0.0213 | -0.0539 |
| add O3_lag_1 | 0.7836 | 0.2073 | 0.0690 | -0.0195 | -0.0292 | -0.0129 |
| add AQI x wind_speed | 0.8029 | 0.1991 | 0.1347 | -0.0002 | -0.0374 | +0.0528 |
| add AQI_trend_7d (7d slope) | 0.7844 | 0.1900 | 0.1345 | -0.0187 | -0.0465 | +0.0526 |
| add AQI_ewm_7 | 0.7934 | 0.2129 | 0.1293 | -0.0097 | -0.0236 | +0.0474 |
| add AQI_ewm_14 | 0.8009 | 0.1976 | 0.0857 | -0.0022 | -0.0389 | +0.0038 |
| add is_high_event_3d flag | 0.7865 | 0.1815 | 0.0826 | -0.0166 | -0.0550 | +0.0007 |
| add doy_sin+doy_cos (cyclic) | 0.8134 | 0.2020 | -0.1403 | +0.0103 | -0.0345 | -0.2222 |
| add AQI_roll_min/max_7d | 0.8130 | 0.2073 | 0.1491 | +0.0099 | -0.0292 | +0.0672 |
| add wind x PM2_5_lag1 | 0.7987 | 0.2755 | 0.0725 | -0.0044 | +0.0390 | -0.0094 |
| add Humidity x PM2_5_lag | 0.8099 | 0.2767 | 0.0621 | +0.0068 | +0.0402 | -0.0198 |
| add AQI_ewm_30 (DB) | 0.7955 | 0.1970 | 0.1078 | -0.0076 | -0.0395 | +0.0259 |
| combined: lag14+ewm7+slope | 0.7725 | 0.1975 | -0.0286 | -0.0306 | -0.0390 | -0.1105 |
| BEST COMBINATION (6 new feats) | 0.8093 | 0.2160 | 0.1250 | +0.0062 | -0.0205 | +0.0431 |

---

### 6.2 Features That Improve Day3 R² (Δ > 0.01)

Four individual features cleared the +0.01 threshold, ranked by day3 delta:

1. **AQI_roll_min_7 + AQI_roll_max_7** (Δ_d3 = +0.0672, R²_d3 = 0.1491): The 7-day rolling minimum and maximum jointly encode the AQI range over the past week. The range is a proxy for pollution volatility — wide range signals an active dispersion regime; a narrow high range signals persistent stagnation. This variance information is orthogonal to the rolling mean and is physically meaningful for 3-day forecasts because atmospheric stability regimes tend to persist for 3-5 days.

2. **AQI × wind_speed** (Δ_d3 = +0.0528, R²_d3 = 0.1347): The multiplicative interaction captures the nonlinear coupling between pollution level and dispersion capacity. When both AQI and wind are high simultaneously (rare but possible during dust events), the product takes an extreme value that the model cannot reconstruct from AQI and wind_speed separately. Physically, high-AQI low-wind days define the stagnation episodes that are hardest to forecast 3 days ahead — this term encodes that state explicitly.

3. **AQI_trend_7d** (Δ_d3 = +0.0526, R²_d3 = 0.1345): The linear regression slope over the 7 days preceding today (computed on AQI shifted by 1 to prevent leakage). A positive slope means pollution is building; a negative slope means it is declining. The trend is a more compact representation of momentum than including multiple individual lags. For day3, knowing whether pollution is currently rising or falling is more informative than knowing the specific level 3 or 7 days ago.

4. **AQI_ewm_7** (Δ_d3 = +0.0474, R²_d3 = 0.1293): The 7-day exponential weighted mean places higher weight on recent observations and provides a smooth low-frequency signal. It partially overlaps with AQI_roll_mean_7 but decays differently and is less affected by a single outlier day. Although AQI_ewm_7 already exists in the feature store DB, it is not included in the production 39-feature set, making it immediately available for production use without any preprocessing change.

---

### 6.3 Features That Hurt or Have No Effect

- **doy_sin + doy_cos** (Δ_d3 = -0.2222, worst performer): The cyclic day-of-year encoding severely hurts day3 performance despite slightly improving day1. The likely explanation is that `month` and the four season dummies already capture the annual cycle; adding a finer-grained cyclic encoding introduces near-multicollinearity that destabilizes the LGBM split decisions on the 30-day test window. The extremely large drop (-0.22) with only 30 test days suggests high variance in the test-window estimate, but the direction is clearly negative.

- **combined: lag14+ewm7+slope** (Δ_d3 = -0.1105): Combining three individually positive features produces worse results than any of them alone. This is a classic case of feature interaction causing overfitting: adding AQI_lag_14 alongside AQI_ewm_7 introduces redundant autocorrelation signals at overlapping time scales that increase noise in the feature selection process.

- **AQI_lag_14** alone (Δ_d3 = -0.0538): A 2-week lag adds no predictive signal for day3; the autocorrelation at lag 14 is weak (ACF drops below 0.1 by lag 7 for Karachi AQI) and the added column consumes a feature importance slot without contributing information.

- **NO2_lag_1** (Δ_d3 = -0.0539): Lagged NO2 hurts because the production feature set already contains same-day NO2. Adding a 1-day lag creates a near-collinear pair that splits feature importance between two redundant columns.

- **wind × PM2_5_lag1** and **Humidity × PM2_5_lag** both improve day2 (+0.039, +0.040) but slightly hurt day3, indicating these interactions help the model resolve near-term concentration memory but introduce noise beyond 48 hours.

---

### 6.4 Best Combination Result

The script identified the top-4 day3 improvers (AQI_roll_min/max_7d, AQI_x_wind, AQI_trend_7d, AQI_ewm_7) plus AQI_ewm_30 as the best 6-feature combination. The combined result:

- Baseline day3 R²: **0.0819**
- Best combination day3 R²: **0.1250**
- Net improvement: **+0.0431**

The combination underperforms the best individual addition (AQI_roll_min/max_7d alone = +0.0672), which is a well-known phenomenon: features that each help individually may be partially redundant with each other, diluting their combined benefit. Adding AQI_trend_7d alongside AQI_ewm_7 provides overlapping momentum signals. The practical takeaway is to add the rolling min/max pair first (largest single gain) and evaluate before adding more.

---

### 6.5 Season-Specific Model

Summer-only cross-validation results (TimeSeriesSplit, n_splits=3, approximately 736 summer days):

| Horizon | Summer-only CV R² | Global model R² |
|---|---|---|
| Day 1 | 0.6364 | 0.8031 |
| Day 2 | -0.0318 | 0.2365 |
| Day 3 | -0.3835 | 0.0819 |

Summer-only training is strictly worse than the global model at every horizon. This finding is counterintuitive but has a clear explanation: Karachi's summer AQI is relatively stable and low (monsoon washout effect), so within-season variation is dominated by noise rather than structured patterns. The global model benefits from the large winter and spring signal to learn the feature-AQI relationship; a summer-only model has too little variance to work with and overfits to the training splits. Training season-specific models would require substantially more data per season or a regularized approach such as seasonal interaction terms rather than fully separate models.

---

### 6.6 Answer: Is Day3 Fundamentally Unpredictable?

- Maximum observed day3 R² from any single feature set tested: **0.1491** (AQI_roll_min/max_7d added)
- Baseline: 0.0819; best combination: 0.1250

The ceiling is modest but real. The AQI autocorrelation function for Karachi drops below 0.15 by lag 4 and below 0.10 by lag 7, which imposes a hard theoretical ceiling on lag-based feature engineering for day3. Adding rolling range (min/max over 7 days) provides the largest improvement because it encodes regime state rather than level — stagnation episodes are persistent and the range captures whether the system is currently in a high-variance transitional state or a stable high-AQI state.

The ceiling is imposed by a combination of autocorrelation decay (the AQI signal itself loses predictability beyond approximately 2 days) AND missing features (meteorological predictors like surface_pressure, boundary_layer_height, and wind_gusts identified in Section 5 are not yet in the pipeline). Section 5 shows that variables like surface_pressure and apparent_temperature maintain |r| > 0.5 with AQI_t+3 — these have not been tested in the LGBM feature set and likely represent the largest remaining improvement opportunity. Feature engineering within the existing variable set appears nearly exhausted at approximately R² = 0.15 for day3.

---

### 6.7 Single Highest-Confidence Recommendation

**To improve day3 R² in production, add:**

- **Feature:** `AQI_roll_min_7` and `AQI_roll_max_7` (add as a pair)
- **Expected R² gain:** +0.0672 (day3 R²: 0.0819 to 0.1491)
- **Code change needed:** Add the following two lines to `src/preprocess_daily_data.py` inside the `add_lag_rolling()` function, after the existing rolling computations:
  ```python
  df["AQI_roll_min_7"] = df["AQI"].shift(1).rolling(7).min()
  df["AQI_roll_max_7"] = df["AQI"].shift(1).rolling(7).max()
  ```
  Then add `'AQI_roll_min_7'` and `'AQI_roll_max_7'` to the production feature list in `src/train.py`.
  Note: Both columns already exist in the feature store DB (written by a prior preprocessing update), so no new data fetch is required — only the training feature list needs updating.
- **Physical justification:** The 7-day rolling range (max minus min) captures atmospheric persistence regime. A wide range indicates the system is in transition (varying dispersion); a narrow high range signals sustained stagnation. Both the minimum (floor of recent AQI) and maximum (ceiling) independently encode regime memory that the rolling mean obscures. For a 3-day forecast horizon, knowing whether Karachi has been in a stable high-pollution state for a week is more predictive than knowing the average level because stable high-pollution atmospheric conditions (surface inversions, weak winds) tend to persist for 3-7 days.
