# EDA Findings — Karachi AQI

Source: `notebooks/eda.ipynb`, run against the live MongoDB `feature_store`.
Window analysed: **2022-08-05 → 2026-05-31** (1,396 days with an AQI reading).
Focus is on the **actual observed measurements** (pollutants + weather + AQI), not
engineered or forecast columns.

> All numbers below are reproducible by re-running the notebook; they will drift
> slightly as new data arrives.

---

## 1. Data coverage

| Fact | Value |
|------|-------|
| Full backfill span | 2018-01-01 → 2026-05-31 (3,073 days) |
| AQI / pollutant span | 2022-08-05 → 2026-05-31 (1,396 days, 45% of backfill) |
| Always-null variables | `cape`, `visibility`, `wind_speed_80m` |

The weather archive reaches back to 2018, but air-quality data only begins in
**Aug 2022**. Three weather variables never return data for Karachi at all.

**→ Code decisions this supports**
- `src/train.py` filters `df[df["date"] >= "2023-01-01"]` — keeps the dense,
  fully-populated window and drops the weather-only backfill era. ✔ matches the data.
- `EXCLUDE_COLS` drops `cape`, `visibility`, `wind_speed_80m` (+ their lead/lag
  variants). ✔ EDA confirms these are 100% null — the exclusion is fully justified.

---

## 2. Target distribution (AQI)

| Stat | Value |
|------|-------|
| Mean / Median | 91 / 82 (US-AQI, "Moderate") |
| Std / Max | 30 / 215 |
| Skewness / Kurtosis | 1.17 / 1.04 |
| Days AQI > 100 | 29% |

Right-skewed: most days are Moderate, with a long tail of pollution spikes
(max 214.6 — a confirmed March-2023 dust event).

**→ Code decisions this supports**
- Skewed pollutant inputs are log-transformed (`log_PM2_5`, `log_CO`) before
  modelling, compressing the heavy right tail.
- Tree models (LGBM) + robust LSTM/Ridge ensemble handle the non-Gaussian target
  better than a single linear fit.

---

## 3. Stationarity & autocorrelation

- **ADF test:** statistic −5.26, p ≈ 6.7e-06 → the daily AQI series is **stationary**.
- **Autocorrelation decays with horizon:**

| Horizon | ACF | Rough R² ceiling (ACF²) |
|---------|-----|--------------------------|
| +1 day | 0.84 | 0.70 |
| +2 day | 0.66 | 0.44 |
| +3 day | 0.56 | 0.31 |

Strong short-term memory, but predictability falls off sharply by day 3.

**→ Code decisions this supports**
- Lag and rolling features (`AQI_lag_1/2/3`, `AQI_roll_mean/std/min/max_7`,
  `AQI_ewm_*`) exploit the strong short-range autocorrelation.
- **Per-horizon modelling**: `src/train.py` trains a separate head for each of
  `AQI_t+1..t+4` (`PerHorizonWrapper` for LGBM, `MultiOutputRegressor` for Ridge,
  a 4-unit output for the LSTM). The ACF ceiling explains why day-3 metrics are
  structurally lower than day-1 — it is a property of the data, not a model bug.

---

## 4. Seasonality

| Season | Mean AQI | Std | Max |
|--------|----------|-----|-----|
| Winter | **110.6** | 29.1 | 198 |
| Autumn | 93.6 | 31.6 | 203 |
| Spring | 79.8 | 25.9 | 215 |
| Summer | 78.3 | 18.9 | 200 |

Winter is the clear high-pollution season (temperature inversions, weak winds);
the additive seasonal decomposition shows a strong, stable annual cycle.

**→ Code decisions this supports**
- `month` / season encodings and cyclical features feed the model the annual cycle.
- The dashboard surfaces the seasonal pattern (worst-season insight, monthly heatmap).

---

## 5. Drivers of AQI (same-day correlation)

Top contemporaneous correlations with AQI (observed variables only):

| Variable | r | | Variable | r |
|----------|-----|--|----------|-----|
| PM2_5 | **+0.91** | | apparent_temp | −0.45 |
| SO2 | +0.73 | | surface_pressure | +0.45 |
| CO | +0.72 | | Temperature | −0.44 |
| NO2 | +0.69 | | wind_speed | −0.38 |
| O3 | +0.54 | | Humidity | −0.37 |
| PM10 | +0.52 | | BLH | −0.22 |
| wind_gusts | −0.50 | | Precipitation | +0.08 |
| dew_point | −0.48 | | aod | −0.06 |

- **PM2.5 dominates** (r = 0.91) — it essentially *is* the AQI driver in Karachi.
- Pollutants (SO2/CO/NO2/O3/PM10) form the next tier.
- Wind (gusts/speed negative) disperses pollution; surface pressure (positive)
  tracks subsidence/inversion episodes — physically sensible.

**→ Code decisions this supports**
- **PM2.5 lead-targets excluded to prevent leakage:** `EXCLUDE_COLS` drops
  `PM2_5_t1..t4`. Because PM2.5 is so dominant, leaking its future value would
  inflate metrics; the live forecast uses the CAMS PM2.5 forecast at inference
  instead. ✔ directly motivated by the 0.91 correlation.
- Air-quality context variables with usable signal (`BLH`, `dew_point`,
  `cloud_cover`, `shortwave_rad`, `uv_index`, `dust`, `aod`) are **kept** as model
  inputs.

**→ One honest tension to revisit**
- `surface_pressure` (+0.45), `apparent_temp` (−0.45) and `wind_gusts` (−0.50)
  are **moderately correlated and now fully populated** in the AQI window — yet
  `EXCLUDE_COLS` currently drops them, with the comment *"not available at backfill
  time for most rows."* That was true for the early 2023 training window but is no
  longer true today. These three are good **candidates to re-introduce and re-test**
  (the archive `feature_experiments.py` / `error_analysis.py` are the tools for that;
  see `docs/ANALYSIS_RERUN_GUIDE.txt`). The all-null trio (`cape`, `visibility`,
  `wind_speed_80m`) should stay excluded.

---

## Summary

The EDA confirms the modelling choices already in the code: a clean post-2023
training window, log-transformed skewed pollutants, lag/rolling features for the
strong autocorrelation, per-horizon heads reflecting the ACF decay, seasonal
encodings for the Winter peak, and exclusion of PM2.5 leads (leakage) and the
all-null weather variables. The one open item is the conservative exclusion of
`surface_pressure` / `apparent_temp` / `wind_gusts`, which the current data no
longer justifies and which are worth re-testing.
