# DengueWatch: Viva Preparation Notes

Every member should be able to answer all of these. Numbers are taken from `reports/metrics.json`, `reports/eda_summary.json`, `reports/model_selection.json` and `data/processed/quality_report.json`.

---

## Problem and data

**Q: What problem does the project solve, and for whom?**
It forecasts weekly lab-confirmed dengue cases 1–8 weeks ahead and raises graded alerts, so public-health officers, vector-control teams and hospital planners can act *before* an outbreak, not after cases are already high.

**Q: Where does the data come from? Is it real?**
It comes from the official CDC/NOAA Dengue Forecasting Project 2015 repository (github.com/cdcepi/dengue-forecasting-project-2015), licensed CC BY 4.0. The cases come from the Puerto Rico Department of Health with CDC (San Juan) and from NAMRU-6 with UC Davis (Iquitos). The climate data come from NOAA: GHCN station data, CFSR reanalysis, PERSIANN satellite rainfall and NDVI-CDR. The DrivenData DengAI competition used the same data.

**Q: Frequency, duration and size?**
The data are weekly. San Juan runs from 1990-04-30 to 2013-04-23 (1,196 weeks); Iquitos runs from 2000-07-01 to 2013-06-25 (676 weeks). The train/test split is the original challenge's: training covers seasons up to 2008/09 and testing covers 2009/10–2012/13 (208 weeks per city).

**Q: What data-quality problems did you find and how did you fix them?**
1. **Irregular weeks.** Weeks restart every 1 January, so the last week of each year is 8–9 days. Each year therefore has exactly 52 rows, which makes a 52-week season exact. Rainfall was rescaled to a 7-day equivalent.
2. **The Iquitos weather station has big gaps.** Daily missing rates are 58.5% for TAVG, 42.2% for TMAX, 25.4% for TMIN and 17.3% for PRCP. We rebuilt TAVG as (TMAX+TMIN)/2 where possible (184 weeks), then filled the rest with a linear fit on reanalysis temperature or satellite rainfall, estimated on overlapping weeks (r = 0.63 for average temperature, 0.33 for rainfall).
3. **NDVI.** Two of San Juan's four pixels have mean NDVI below 0.1 (sea or urban surface), so they were dropped.
4. **Missing codes and range checks.** The value -9999 was converted to NaN; temperatures outside 5–45 °C and rainfall outside 0–500 mm/day were treated as errors.
5. **No duplicates and no missing case values.** We also checked that total = sum of the serotype columns.

**Q: Why didn't you remove outliers in the case counts?**
Because the outliers *are* the outbreaks, which are exactly what we must predict. Capping them with an IQR rule would delete the signal. Climate outliers were only flagged (robust z > 5), and they were plausible heavy-rain weeks.

---

## EDA

**Q: Is the series stationary? How do you know?**
Yes, on the log scale. For San Juan, ADF gives −7.00 (p < 0.001), which rejects a unit root, and KPSS gives 0.29 (p > 0.10), which does not reject stationarity. Iquitos is similar (ADF −5.21, KPSS 0.35). Because both tests agree, we set d = 0 and D = 0.

**Q: Why log(1 + cases)?**
The counts are over-dispersed (variance/mean = 76 in San Juan, 18 in Iquitos) and right-skewed (skewness 3.6 and 4.1). The log stabilises the variance, keeps forecasts non-negative, and "+1" handles zero weeks (127 in Iquitos).

**Q: What seasonality did you find?**
Using STL with period 52, seasonal strength is 0.51 in San Juan and 0.41 in Iquitos. Peaks fall around season-week 23 (about October) in San Juan and week 24 (about December) in Iquitos. The *size* of the peak varies by two orders of magnitude, so seasonality tells you *when*, not *how many*.

**Q: What do the ACF and PACF show?**
The ACF decays slowly from 0.93 and oscillates with a 52-week period. The PACF cuts off after about lag 3. This suggests a low-order AR plus a seasonal component, consistent with the chosen SARIMA(1,0,1)(1,0,1)52.

**Q: How did you find the climate lags, and why remove seasonality first?**
Rain and dengue peak in the same months, so raw correlation mostly measures shared seasonality. We subtracted each variable's week-of-year mean (training only) and then computed lagged correlations. In San Juan, temperature anomalies lead cases by about 10–16 weeks (r ≈ 0.25). In Iquitos, humidity leads at 4–10 weeks (r ≈ 0.31–0.34). After deseasonalising, rainfall had no significant link in San Juan.

---

## Models

**Q: Which models did you compare and why each?**
- **Baselines:** persistence, seasonal naive and climatology. A model must beat these to be useful.
- **Holt-Winters:** exponential smoothing.
- **SARIMA:** the classical seasonal model.
- **SARIMAX:** tests whether climate adds value.
- **Prophet:** a popular additive model.
- **XGBoost:** non-linear effects of lags, season and climate.
- **Hybrid SARIMA + XGBoost:** a linear part plus a non-linear correction of the residuals.
- **Ensemble:** the average of SARIMAX, XGBoost and the Hybrid.

**Q: How did you choose the SARIMA orders?**
By AIC over 6 candidates on training data. (1,0,1)(1,0,1)52 won in both cities (San Juan AIC 900.9 vs 955.7 without the seasonal MA term).

**Q: How does SARIMAX avoid using future climate?**
Every climate input is lagged by at least 8 weeks, which is the maximum horizon. So every value an 8-week forecast needs has already been observed when the forecast is made.

**Q: How does the hybrid work?**
SARIMA makes its h-step forecast. XGBoost is trained on the *error* y(o+h) − SARIMA_h(o), using lags, season and climate features. The final forecast is SARIMA plus XGBoost's predicted correction.

**Q: Why one XGBoost model per horizon (direct strategy)?**
A recursive model feeds its own forecasts back in, so errors compound. Direct models avoid that, and each can learn which features matter at its horizon.

**Q: How did you tune XGBoost without leakage?**
With expanding-window walk-forward CV on training origins only: 4 folds, a gap so no target overlaps the validation fold, and a grid of depth {2, 3, 4} × trees {200, 500}. The chosen depth was 2 (San Juan) and 3 (Iquitos), with 200 trees.

**Q: Where do the prediction intervals come from?**
SARIMA and SARIMAX intervals come from the state-space model. Prophet uses its own simulation. XGBoost and the Hybrid use split-conformal intervals: residual quantiles (10%, 90%) on the last two training seasons, from a model fitted without them.

---

## Evaluation and results

**Q: Explain your evaluation method.**
It is a rolling origin. From every week of the 208 test weeks, each model forecasts the next 1–8 weeks using only data up to that week. Parameters are fixed from training. Statistical models are updated with each new observation (Kalman filter), and Prophet is refitted every 4 weeks. This is what an operational system would do, and it matches the original challenge's rule.

**Q: Which model is best?**
- **San Juan:** XGBoost has the lowest average MAE (21.27 cases/week), effectively tied with the Ensemble (21.28). Seasonal naive is 63.0 and SARIMA is 23.2.
- **Iquitos:** the Ensemble is best at every horizon, with an average MAE of 4.60 (seasonal naive 9.40, SARIMA 4.90).
- **Skill vs seasonal naive:** 66% in San Juan and 51% in Iquitos.

**Q: Did climate help?**
A little, but consistently: SARIMAX beat SARIMA at all 8 horizons in both cities. The non-linear models gain more. Past cases carry most of the signal, and climate is secondary. This is an honest finding.

**Q: Why is persistence so strong at 1 week ahead?**
Lag-1 autocorrelation is 0.93. Next week's count is very close to this week's, so beating persistence at h = 1 is hard (11.9 vs 12.1 MAE in San Juan). From h = 2 onwards, the climate-aware models win, and at h = 8 the Hybrid beats persistence by about 24%.

**Q: Why did Prophet do badly?**
It has no autoregressive term. It cannot "see" that an epidemic is already under way, only the average seasonal shape. Its 80% intervals covered only 67% of outcomes in San Juan.

**Q: Why not MAPE?**
Iquitos has 127 zero-case weeks, which make MAPE undefined or explode. We used MAE (the original challenge's point metric), RMSE, log-scale MAE, and skill relative to seasonal naive.

---

## Decision support

**Q: How is an "epidemic" defined?**
With an endemic channel. Per season-week, the training seasons give a mean (typical), mean + 1 SD (warning) and mean + 2 SD (epidemic threshold), smoothed over 5 weeks. About 5% of training weeks exceed the threshold.

**Q: What are the alert levels?**
- **Normal:** at or below the typical level.
- **Watch:** above the typical level.
- **Warning:** above mean + 1 SD, **or** the upper 80% bound above the epidemic threshold.
- **Epidemic alert:** the central forecast is above the epidemic threshold.

**Q: Why use the upper 80% bound for warnings?**
Point forecasts on the log scale behave like medians and stay below a mean + 2 SD threshold until the outbreak has already started. Point-only alerts gave about 0 weeks of warning. Alerting when there is roughly a 10% or greater chance of epidemic-level cases flagged all 3 San Juan outbreaks (Ensemble: 3.7 weeks earlier on average) at precision 0.79. It also flagged both Iquitos outbreaks (7.5 weeks average), but precision there was only about 0.11.

**Q: What about the false alarms in Iquitos?**
They are real. Counts are small and spiky, so the interval often touches the low off-season threshold. That is why the app has an alert-sensitivity switch, and why we recommend cheap actions for Warning and costly actions only for Epidemic alert. A future fix is to tune the alert probability per city on a separate validation period.

**Q: Walk us through a use case.**
Take Iquitos, with a forecast made in the week of 19 November 2010 (Ensemble model):
1. The central forecast is 7–12 cases a week through December.
2. The upper bound crosses the epidemic threshold for the week of 1 January 2011, 6 weeks ahead, so the app shows **Warning**.
3. In reality, cases jumped to 66, 101 and 78 in the first three weeks of January, the largest test-period outbreak.
4. The warning gave 6 weeks to prepare. The size was badly under-forecast, though, so expected-case numbers are a floor.

---

## Limitations (be ready to say these yourself)

- The data are *final* counts. A live system sees incomplete recent weeks because of reporting delays.
- Epidemic peaks are under-forecast (San Juan 2012–13: 110–150 forecast vs 236 actual at 4 weeks ahead).
- Many false alarms in Iquitos.
- Serotype shifts and immunity are not modelled.
- Climate effects are correlational, so the what-if page is a sensitivity check, not causal.
- There are two non-Indian cities and the data end in 2013.
- The 10% hospital-care share is an adjustable assumption, not a sourced rate.

---

## App and code

**Q: Why are frontend and backend separate?**
The backend (FastAPI) owns the data, the models and the logic, and exposes 10 REST endpoints with automatic docs at /docs. The frontend (Streamlit) only displays and collects input. Another frontend (a mobile app, or a dashboard at a health department) could reuse the same API.

**Q: Is anything in the app static?**
No. Every chart and number comes from API calls. Forecasts from past weeks are precomputed by the pipeline for speed and reproducibility. What-if scenarios and uploaded data are computed live.

**Q: What does the upload page do with a new dataset?**
1. It validates the columns and the minimum of 3 years of data.
2. It parses dates, merges duplicates and regularises to a weekly grid, interpolating gaps.
3. It builds that series' own endemic channel.
4. It back-tests 5 univariate models (seasonal naive, persistence, Holt-Winters, SARIMA, XGBoost) on the last 52 weeks.
5. It forecasts 8 weeks with the best of them and returns the alert and actions.
