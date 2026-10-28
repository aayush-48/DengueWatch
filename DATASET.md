# DengueWatch: Dataset Documentation

## 1. Source

| | |
|---|---|
| **Name** | Dengue Forecasting Project data (2015), US Government interagency project (HHS/CDC, DoD, Commerce/NOAA, DHS) |
| **Repository** | https://github.com/cdcepi/dengue-forecasting-project-2015 (archived on Zenodo; DOI badge in the repository README) |
| **License** | Creative Commons Attribution 4.0 (CC BY 4.0), as stated in the dataset metadata files |
| **Case data publishers** | San Juan: Puerto Rico Department of Health and CDC. Iquitos: U.S. Naval Medical Research Unit 6 (NAMRU-6) and University of California, Davis |
| **Climate data publisher** | NOAA (GHCN-Daily, CFSR reanalysis, PERSIANN-CDR, NDVI-CDR) |
| **Related** | The DrivenData *DengAI* competition was built from these data; this repository additionally contains the test-period case counts and raw daily climate files |
| **Retrieved** | October 2026, via `git clone` of the repository (commit 6ab36c9, dated 25 Oct 2019) |

The original files are kept unchanged in `data/raw/`, together with the publishers' own metadata (`*_metadata.md`, `README.md`).

## 2. Coverage

| City | First week | Last week | Weeks | Frequency | Training seasons | Test seasons |
|---|---|---|---|---|---|---|
| San Juan, Puerto Rico | 1990-04-30 | 2013-04-23 | 1,196 | weekly | 1990/91–2008/09 (988 weeks) | 2009/10–2012/13 (208 weeks) |
| Iquitos, Peru | 2000-07-01 | 2013-06-25 | 676 | weekly | 2000/01–2008/09 (468 weeks) | 2009/10–2012/13 (208 weeks) |

A *season* starts after the week with the historically lowest incidence (late April in San Juan, July in Iquitos). Weeks restart on 1 January, so the last week of each calendar year is 8–9 days long and every year has exactly 52 rows.

## 3. Raw files used

| File | Content | Native frequency | Units / notes |
|---|---|---|---|
| `Dengue_data/<city>_testing_data.csv` | lab-confirmed dengue cases, full series incl. test seasons | weekly | counts; `total_cases` = sum of serotype + other (+ additional) columns |
| `Environmental_data/SanJuanRQW00011641.csv`, `IquitosPE000084377.csv` | GHCN weather station | daily | °C, mm; −9999 = missing |
| `Environmental_data/SanJuan.xlsx`, `Iquitos.xlsx` | NCEP CFSR reanalysis (humidity, temperature) | daily | Kelvin, %, g/kg |
| `Environmental_data/<City>_Precip.xlsx` | PERSIANN-CDR satellite precipitation | daily | mm |
| `Environmental_data/<city>_noqc.csv` | NDVI-CDR vegetation index, 4 pixels near the city | weekly | unitless, −1 to 1 |
| `Population_data/<city>_population_data.csv` | population estimates | yearly | persons |

## 4. Processed data dictionary (`data/processed/<city>_weekly.csv`)

| Column | Description | Unit | How it was built |
|---|---|---|---|
| `week_start_date` | first day of the reporting week | date | from source |
| `season` | transmission season, e.g. 2010/2011 | — | from source |
| `season_week` | week of the season, 1–52 | — | from source |
| `total_cases` | **target**: lab-confirmed dengue cases in the week | cases | from source |
| `denv1_cases` … `denv4_cases` | cases by serotype | cases | from source |
| `other_positive_cases` | lab-positive, serotype not identified | cases | from source |
| `additional_cases` | estimated positives among untested specimens (San Juan only) | cases | from source |
| `week_days` | length of the week window (7, 8 or 9) | days | from dates |
| `temp_avg_c`, `temp_min_c`, `temp_max_c` | station temperature, weekly mean | °C | station; Iquitos gaps filled from reanalysis via linear fit |
| `temp_dtr_c` | diurnal temperature range = max − min | °C | derived |
| `precip_mm` | station rainfall, 7-day equivalent | mm/week | station; gaps filled from satellite via linear fit |
| `precip_sat_mm` | satellite rainfall, 7-day equivalent | mm/week | PERSIANN-CDR |
| `rel_humidity_pct` | relative humidity, weekly mean | % | reanalysis |
| `spec_humidity_gkg` | specific humidity, weekly mean | g/kg | reanalysis |
| `dew_point_c` | dew point, weekly mean | °C | reanalysis, converted from K |
| `ndvi` | vegetation index, mean of retained pixels | — | NDVI-CDR; San Juan keeps 2 of 4 pixels |
| `cal_year`, `cal_week` | calendar year and k-th week of that year | — | derived, used to align NDVI |
| `population` | population for the calendar year | persons | interpolated between estimates |
| `incidence_100k` | cases per 100,000 population | cases/100k | derived |
| `split` | `train` or `test` (original challenge split) | — | derived |
| `t` | regular week index 0…n−1 | — | derived |

Observed ranges (processed): San Juan average temperature 22.1–30.2 °C, weekly rainfall 0–329 mm, incidence 0–19.9 per 100k. Iquitos average temperature 24.9–29.8 °C, weekly rainfall 0–543 mm, incidence 0–28.2 per 100k.

## 5. Data-quality summary

The full log is in `data/processed/quality_report.json`, and the app shows it on the *Data & quality* page.

| Check | San Juan | Iquitos |
|---|---|---|
| Duplicate dates | 0 | 0 |
| Missing case values | 0 | 0 |
| Week lengths (days: count) | 7: 1172, 8: 17, 9: 6 | 7: 662, 8: 9, 9: 4 |
| Station days missing (TMAX / TMIN / TAVG / PRCP) | 0 / 0 / 0 / 0 % | 42.2 / 25.4 / 58.5 / 17.3 % |
| Weekly temperature gaps filled | 0 | average 272 (184 more rebuilt from max/min), max 207, min 102 |
| Weekly rainfall gaps filled | 0 | 160 (proxy fit r = 0.33) |
| NDVI pixels kept | 2 of 4 (2 had mean NDVI < 0.1: sea or urban) | 4 of 4 |
| Remaining gaps after merge (interpolated / week-of-year mean) | 16 NDVI, 7 satellite rain | 1 NDVI |
| Case outliers removed | none: outbreaks are the signal | none |

## 6. Known caveats (from the publishers' metadata and our checks)

- Counts are **final** counts compiled after all reports arrived. A real-time system would see incomplete recent weeks.
- San Juan laboratory testing became more sensitive in 2005 (PCR), 2007 (improved IgM) and 2009 (more sensitive PCR), according to the dataset metadata (citing Sharp et al.).
- Climate sources have different spatial scales: station = point, reanalysis = 0.5° grid, PERSIANN = 0.25° grid, NDVI = 0.05° pixels.
- The publishers note that environmental data "may still potentially contain quality issues" and should be inspected before use. Our range checks and gap-filling are described above.
