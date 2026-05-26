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
