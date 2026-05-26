# Karachi AQI Predictor — Deep Analysis Findings

_Branch: analysis/data-exploration | Date: 2026-05-26_

---

## Summary of Key Answers

| # | Question | Answer |
|---|----------|--------|
| 1 | Autocorrelation ceiling for day2/day3? | Daily ACF lag-3 = 0.560 → theoretical R² ceiling ≈ 0.314. Current model (0.120) is well below — room to improve. |
| 2 | What drives Day1's 0.795 R²? | log_PM2_5 (56.5% of SHAP weight). NOT autoregressive — exploits PM2.5 physical persistence. |
| 3 | Does prediction stacking improve day3? | No — catastrophic. Day3 R² collapses 0.152 → -0.013. Do not use stacking. |
| 4 | New Open-Meteo variables with \|r\|>0.3 for day3? | 7 variables: apparent_temperature (-0.525), dew_point_2m (-0.518), soil_temperature (-0.513), surface_pressure (+0.504), wind_gusts (-0.434), european_aqi (+0.423), boundary_layer_height (-0.326). |
| 5 | Is Summer day3 fundamentally unpredictable? | Partially. Summer-only model fails (R²=-0.384). But missing variables (surface_pressure) explain most gap. |
| 6 | Single highest-confidence recommendation for day3? | Add AQI_roll_min_7 + AQI_roll_max_7 to FEAT_BASE (already in MongoDB). Expected Δ day3 R² = +0.067. |

---
