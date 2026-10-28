# DengueWatch: Climate-Driven Dengue Outbreak Forecasting and Early-Warning System

TSDA Mini Project. DengueWatch forecasts weekly laboratory-confirmed dengue cases **1–8 weeks ahead** for San Juan (Puerto Rico) and Iquitos (Peru). It does this from past cases and lagged climate data (temperature, humidity, rainfall, vegetation). It compares ten forecasting approaches and turns the forecasts into graded outbreak alerts with recommended actions. Everything is delivered through a FastAPI backend and a Streamlit frontend.

---

## 1. Quick start

**Requirements:** Python 3.10 or newer is recommended. The project was built and tested on Python 3.13; see `requirements-lock.txt`.

```bash
# 1. create an environment (recommended)
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt      # or: pip install -r requirements-lock.txt  (exact tested versions)

# 3. start the app (backend + frontend)
./run_app.sh                         # Windows: run_app.bat
```

Then open **http://localhost:8501** (Streamlit). The API's interactive documentation is at **http://localhost:8000/docs**.

To run the two parts manually in two terminals, from the project root:

```bash
python -m uvicorn backend.api:app --port 8000
python -m streamlit run app/streamlit_app.py
```

The trained models and all forecasts are already included, so the app starts immediately. To rebuild everything from the raw data (about 5–10 minutes):

```bash
./run_pipeline.sh                    # = data_build -> eda -> train_evaluate
```

> **Prophet note:** `prophet` needs a working CmdStan backend, which `pip install prophet` normally provides. If installing it fails on your machine, the app still runs, because Prophet's forecasts are precomputed. Only `run_pipeline.sh` needs Prophet.

---

## 2. Dataset

Full documentation, including a column-by-column data dictionary, is in [`DATASET.md`](DATASET.md).

| | |
|---|---|
| **Source** | CDC / NOAA *Dengue Forecasting Project* (2015), official repository: https://github.com/cdcepi/dengue-forecasting-project-2015 (DOI badge on the repo, via Zenodo). The DrivenData "DengAI" competition was built from these same data. |
| **Publishers** | Puerto Rico Department of Health + CDC (San Juan cases); U.S. Naval Medical Research Unit 6 (NAMRU-6) and University of California, Davis (Iquitos cases); NOAA (climate) |
| **License** | Creative Commons Attribution 4.0 (CC BY 4.0) |
| **Frequency** | Weekly (cases, NDVI); daily (climate, aggregated to weeks here) |
| **Duration** | San Juan: 1990-04-30 to 2013-04-23 (1,196 weeks). Iquitos: 2000-07-01 to 2013-06-25 (676 weeks) |
| **Train / test** | Same split as the original challenge: train = seasons up to 2008/09, test = 2009/10–2012/13 (208 weeks per city) |

**Files used** (in `data/raw/`, copied unchanged from the repository):

| File | Content | Attributes |
|---|---|---|
| `Dengue_data/*_testing_data.csv` | weekly lab-confirmed cases | season, season_week, week_start_date, denv1–4_cases, other_positive_cases, (additional_cases), **total_cases** (target) |
| `Environmental_data/<City><station>.csv` | NOAA GHCN daily station data | TMAX, TMIN, TAVG, TDTR (°C), PRCP (mm) |
| `Environmental_data/<City>.xlsx` | NCEP CFSR reanalysis, daily | air/dew-point temperature (K), relative & specific humidity, min/max temperature, precipitation |
| `Environmental_data/<City>_Precip.xlsx` | PERSIANN-CDR satellite rainfall, daily | precipitation (mm) |
| `Environmental_data/*_noqc.csv` | NDVI-CDR vegetation index, weekly, 4 pixels | NDVI |
| `Population_data/*.csv` | yearly population estimates | used for incidence per 100,000 |

The `*_testing_data.csv` files contain the complete series, including the test seasons. The `*_training_data.csv` files are the same data truncated at 2008/09.

---

## 3. Results at a glance (test seasons 2009/10–2012/13, rolling origin)

| | San Juan | Iquitos |
|---|---|---|
| Best model (average MAE, horizons 1–8) | XGBoost: 21.3 cases/week (Ensemble 21.3) | Ensemble: 4.6 cases/week |
| Seasonal-naive MAE (reference) | 63.0 | 9.4 |
| Skill vs seasonal naive | 66% | 51% |
| SARIMA → SARIMAX (adding lagged climate) | 23.2 → 22.7 | 4.90 → 4.86 |
| 80% interval coverage (Ensemble) | 0.82 | 0.78 |
| Test outbreaks warned, Ensemble, alert = upper 80% bound | 3 of 3, 3.7 weeks ahead on average | 2 of 2, 7.5 weeks ahead on average |
| Alert precision (same rule) | 0.79 | 0.11 (many false alarms) |

Models compared: Persistence, Seasonal naive, Climatology, Holt-Winters, SARIMA(1,0,1)(1,0,1)52, SARIMAX (+ climate lagged ≥ 8 weeks), Prophet, XGBoost (direct, one model per horizon), Hybrid SARIMA+XGBoost, and an Ensemble. Full numbers are in `reports/metrics.json`; the full write-up is in the project report.

---

## 4. Project structure

```
DengueWatch/
├── data/
│   ├── raw/                      original CDC/NOAA files (unchanged)
│   ├── processed/                sj_weekly.csv, iq_weekly.csv, quality_report.json
│   └── sample_upload.csv         example file for the "Upload your data" page
├── src/
│   ├── config.py                 paths, cities, horizons
│   ├── data_build.py             B: cleaning, weekly alignment, gap filling, merging
│   ├── eda.py                    C: decomposition, stationarity, ACF/PACF, CCF, figures
│   ├── features.py               lagged climate anomalies, feature selection, endemic channel
│   ├── models.py                 D: baselines, Holt-Winters, SARIMA, SARIMAX, Prophet, XGBoost, Hybrid
│   ├── train_evaluate.py         D+G: rolling-origin evaluation, metrics, outbreak lead time
│   ├── recommend.py              F: alert levels and recommended actions
│   └── inference.py              live forecasting for the API (what-if, uploads)
├── backend/api.py                E: FastAPI backend
├── app/streamlit_app.py          E: Streamlit frontend (7 pages)
├── models/                       fitted parameters / XGBoost models (*.joblib)
├── reports/
│   ├── figures/                  all EDA + evaluation figures (PNG)
│   ├── metrics.json              accuracy + alert metrics
│   ├── model_selection.json      AIC tables, CV results, chosen features
│   ├── eda_summary.json          EDA statistics
│   └── forecasts_<city>.csv.gz   every model's forecast from every week
├── screenshots/                  13 screenshots of the running app + capture.py (regenerates them)
├── DATASET.md                    dataset source, coverage, data dictionary, quality summary
├── VIVA_NOTES.md                 likely viva questions with answers
├── requirements.txt, requirements-lock.txt
├── run_app.sh / run_app.bat      start backend + frontend
└── run_pipeline.sh               rebuild data, EDA, models from scratch
```

---

## 5. API endpoints (FastAPI)

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/cities` | available cities, date ranges |
| GET | `/data/{city}?start=&end=` | processed weekly data |
| GET | `/quality` | data-quality report |
| GET | `/eda/{city}` | STL, ACF/PACF, CCF, seasonal profile, serotypes, anomalies |
| GET | `/metrics/{city}` | test-period accuracy, alert metrics, lead times |
| GET | `/selection/{city}` | AIC tables, cross-validation, selected climate features |
| GET | `/forecast/{city}?origin=YYYY-MM-DD&model=&hosp_rate=` | 8-week forecast, alert level, actions, actuals |
| GET | `/compare/{city}?origin=` | all models' forecasts from one week |
| POST | `/whatif/{city}` | re-forecast with modified recent climate |
| POST | `/upload` | forecast a user-supplied weekly case CSV |

---

## 6. App pages

1. **Overview**: the problem, the users, and the headline results.
2. **Data & quality**: interactive series, a date-range filter, the data-quality report and a CSV download.
3. **Exploratory analysis**: trend, STL decomposition, seasonality, ACF/PACF, ADF/KPSS tests, lagged climate correlation, serotypes and anomalies.
4. **Forecast & alerts**: pick a city, week and model. The page shows the 8-week forecast with its 80% interval against the endemic-channel thresholds, the alert level, recommended actions and a latest-week surge check. *Replay* buttons jump to 6 weeks before each real test-period outbreak.
5. **Model comparison**: metrics table, MAE and alert F1 by horizon, lead time per outbreak, all models from one week, and AIC/CV/feature-importance details.
6. **Climate what-if**: change recent temperature, humidity or rainfall and re-run SARIMAX, XGBoost and the Hybrid live.
7. **Upload your data**: forecast any weekly case series. The page validates and cleans it, back-tests five univariate models, and alerts from that series' own endemic channel.

---

## 7. Reproducibility notes

* All randomness is seeded (`random_state=42` for XGBoost). Prophet's interval sampling is stochastic, so its interval bounds can vary slightly between runs.
* The models never see test-period data when their parameters are estimated. During the test period they are *updated* with each new week, which mirrors real operational use.
* Climate inputs to SARIMAX and Prophet are lagged by at least 8 weeks, so every value an 8-week forecast needs is already observed when the forecast is made.
* `python -m src.train_evaluate --score-only` recomputes all metrics and evaluation figures from the saved forecasts in a few seconds, without retraining.
* Screenshots: with the app running, `python screenshots/capture.py` (needs `pip install playwright` and `playwright install chromium`).

## 8. Limitations (short)

The data are final counts, so a live system would see incomplete recent weeks. Epidemic peaks are under-forecast. The early-warning rule gives many false alarms in Iquitos. Serotype and immunity effects are not modelled. Climate effects are correlational, so treat the what-if page as a sensitivity check. The hospital-care share is an adjustable planning assumption, not a sourced rate. The project report has the full discussion.
