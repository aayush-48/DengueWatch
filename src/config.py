"""Central configuration: paths, city metadata, and evaluation settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGURES = ROOT / "reports" / "figures"
RESULTS = ROOT / "reports"

for p in (PROCESSED, MODELS, FIGURES):
    p.mkdir(parents=True, exist_ok=True)

CITIES = {
    "sj": {
        "name": "San Juan, Puerto Rico",
        "cases": "Dengue_data/san_juan_testing_data.csv",
        "station": "Environmental_data/SanJuanRQW00011641.csv",
        "reanalysis": "Environmental_data/SanJuan.xlsx",
        "rean_temp_sheet": "ReanalysisTemp",
        "persiann": "Environmental_data/SanJuan_Precip.xlsx",
        "ndvi": "Environmental_data/sanJuan_noqc.csv",
        "population": "Population_data/san_juan_population_data.csv",
        # Same split as the original 2015 CDC/NOAA forecasting challenge:
        # train on seasons up to 2008/2009, test on 2009/2010 - 2012/2013.
        "test_start_season": "2009/2010",
    },
    "iq": {
        "name": "Iquitos, Peru",
        "cases": "Dengue_data/iquitos_testing_data.csv",
        "station": "Environmental_data/IquitosPE000084377.csv",
        "reanalysis": "Environmental_data/Iquitos.xlsx",
        "rean_temp_sheet": "RanalysisTemp",  # sheet name is misspelled in the source file
        "persiann": "Environmental_data/Iquitos_Precip.xlsx",
        "ndvi": "Environmental_data/iquitos_noqc.csv",
        "population": "Population_data/iquitos_population_data.csv",
        "test_start_season": "2009/2010",
    },
}

SEASON_LENGTH = 52          # every calendar year has exactly 52 rows (see report, Sec. B)
MAX_HORIZON = 8             # weeks ahead
EVAL_HORIZONS = [1, 4, 8]   # horizons reported in the comparison table
CLIMATE_COLS = [
    "temp_avg_c", "temp_min_c", "temp_max_c", "temp_dtr_c",
    "precip_mm", "precip_sat_mm",
    "rel_humidity_pct", "spec_humidity_gkg", "dew_point_c",
    "ndvi",
]
