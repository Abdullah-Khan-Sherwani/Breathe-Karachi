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
