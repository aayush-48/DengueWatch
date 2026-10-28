"""
Feature engineering shared by the models, the evaluation and the live API.

Key rule (no look-ahead): everything a forecast made at origin week `o` uses must be
known at `o`. Climatologies and thresholds are computed on TRAINING seasons only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CLIMATE_COLS, PROCESSED, SEASON_LENGTH

# Variables that measure the same thing are grouped so that feature selection picks
# at most one per group (avoids feeding near-duplicate, collinear inputs to SARIMAX).
CLIMATE_GROUPS = {
    "temperature": ["temp_avg_c", "temp_min_c", "temp_max_c"],
    "absolute_humidity": ["spec_humidity_gkg", "dew_point_c"],
    "relative_humidity": ["rel_humidity_pct"],
    "rainfall": ["precip_mm", "precip_sat_mm"],
    "diurnal_range": ["temp_dtr_c"],
    "vegetation": ["ndvi"],
}


def load(city: str) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / f"{city}_weekly.csv", parse_dates=["week_start_date"])
    df["y"] = np.log1p(df.total_cases)
    return df


def train_mask(df: pd.DataFrame) -> np.ndarray:
    return (df.split == "train").values


def extend_future(df: pd.DataFrame, steps: int) -> pd.DataFrame:
    """Append `steps` placeholder weeks after the last observation (cases and climate
    unknown). Needed so that lagged climate inputs for future target weeks - which
    refer to already-observed weeks because every lag is >= the horizon - line up."""
    last = df.iloc[-1]
    rows = []
    sw, cw, cy = int(last.season_week), int(last.cal_week), int(last.cal_year)
    for k in range(1, steps + 1):
        sw = sw % 52 + 1
        cw = cw + 1
        if cw > 52:
            cw, cy = 1, cy + 1
        rows.append({"week_start_date": last.week_start_date + pd.Timedelta(weeks=k),
                     "season_week": sw, "cal_week": cw, "cal_year": cy, "split": "future"})
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


# ----------------------------------------------------------------------------- climatology
def climate_climatology(df: pd.DataFrame) -> pd.DataFrame:
    """Week-of-year mean of each climate variable, from training rows only."""
    return df[train_mask(df)].groupby("cal_week")[CLIMATE_COLS].mean()


def climate_anomaly_r4(df: pd.DataFrame, clim: pd.DataFrame) -> pd.DataFrame:
    """Deseasonalised climate, smoothed with a trailing 4-week mean (week t uses t-3..t)."""
    base = clim.reindex(df.cal_week.values)[CLIMATE_COLS].values
    anom = pd.DataFrame(df[CLIMATE_COLS].values - base, columns=CLIMATE_COLS, index=df.index)
    return anom.rolling(4, min_periods=1).mean()


def case_climatology(df: pd.DataFrame) -> pd.Series:
    """Mean log(1+cases) per season_week over training seasons (the 'typical year')."""
    return df[train_mask(df)].groupby("season_week").y.mean()


# ----------------------------------------------------------------------------- selection
def lagged_climate_ccf(df: pd.DataFrame, clim: pd.DataFrame, min_lag: int, max_lag: int) -> pd.DataFrame:
    tr = train_mask(df)
    r4 = climate_anomaly_r4(df, clim)
    y_anom = df.y - df.season_week.map(case_climatology(df))
    rows = []
    for v in CLIMATE_COLS:
        for lag in range(min_lag, max_lag + 1):
            x = r4[v].shift(lag)
            ok = tr & x.notna().values
            rows.append({"variable": v, "lag": lag,
                         "corr": float(np.corrcoef(x[ok], y_anom[ok])[0, 1])})
    return pd.DataFrame(rows)


def select_climate_features(df: pd.DataFrame, clim: pd.DataFrame, min_lag: int = 8,
                            max_lag: int = 20, k: int = 3) -> list[dict]:
    """Pick the best (variable, lag) per climate group, keep the k strongest groups whose
    correlation clears the approximate 95% significance bound 2/sqrt(n)."""
    ccf = lagged_climate_ccf(df, clim, min_lag, max_lag)
    ccf["abs"] = ccf["corr"].abs()
    n = int(train_mask(df).sum())
    bound = 2 / np.sqrt(n)
    picks = []
    for g, vars_ in CLIMATE_GROUPS.items():
        sub = ccf[ccf.variable.isin(vars_)]
        best = sub.loc[sub["abs"].idxmax()]
        if best["abs"] > bound:
            picks.append({"group": g, "variable": best.variable, "lag": int(best.lag),
                          "corr": round(float(best["corr"]), 3)})
    picks.sort(key=lambda d: -abs(d["corr"]))
    return picks[:k]


# ----------------------------------------------------------------------------- SARIMAX exog
def exog_matrix(df: pd.DataFrame, clim: pd.DataFrame, feats: list[dict]) -> pd.DataFrame:
    """Column for (variable v, lag L) at week t = smoothed anomaly of v at t-L.
    With L >= 8 every value needed for an 8-week-ahead forecast is already observed."""
    r4 = climate_anomaly_r4(df, clim)
    cols = {f"{f['variable']}_L{f['lag']}": r4[f["variable"]].shift(f["lag"]) for f in feats}
    return pd.DataFrame(cols, index=df.index).fillna(0.0)


# ----------------------------------------------------------------------------- XGBoost
def xgb_feature_frame(df: pd.DataFrame, clim: pd.DataFrame, climate_vars: list[str],
                      h: int, y: np.ndarray | None = None,
                      case_clim: pd.Series | None = None) -> pd.DataFrame:
    """One row per forecast origin o. All inputs are known at o. Target = y[o+h] - y[o]
    (log growth ratio), which keeps tree models from having to extrapolate levels.
    Pass `case_clim` when df is a truncated / modified copy (live API use)."""
    y = df.y.values if y is None else y
    s = pd.Series(y, index=df.index)
    case_clim = case_climatology(df) if case_clim is None else case_clim
    tgt_week = df.season_week.shift(-h)
    # season_week of the target row; near the end of the data, roll the calendar forward
    tgt_week = tgt_week.fillna((df.season_week + h - 1) % SEASON_LENGTH + 1)
    F = pd.DataFrame(index=df.index)
    F["y0"] = s
    for k in (1, 2, 3):
        F[f"y_lag{k}"] = s.shift(k)
    F["mean4"] = s.rolling(4).mean()
    F["mean8"] = s.rolling(8).mean()
    F["mean13"] = s.rolling(13).mean()
    F["std8"] = s.rolling(8).std()
    F["slope4"] = s - s.shift(4)
    F["slope8"] = s - s.shift(8)
    F["seasonal_lag"] = s.shift(SEASON_LENGTH - h)            # same week last year (target)
    F["clim_target"] = tgt_week.map(case_clim).values           # typical level for target week
    F["clim_now"] = df.season_week.map(case_clim).values
    F["above_clim"] = F["y0"] - F["clim_now"]                   # how unusual is now?
    F["sin_w"] = np.sin(2 * np.pi * tgt_week / SEASON_LENGTH)
    F["cos_w"] = np.cos(2 * np.pi * tgt_week / SEASON_LENGTH)
    if climate_vars:  # uploaded series have no climate -> univariate features only
        r4 = climate_anomaly_r4(df, clim)
        for v in climate_vars:
            for lag in (0, 4, 8, 12):
                F[f"{v}_a{lag}"] = r4[v].shift(lag)
    F["target"] = s.shift(-h) - s
    return F


XGB_FEATURES_EXCLUDE = {"target", "base"}  # 'base' = SARIMA forecast the hybrid corrects


def feature_columns(F: pd.DataFrame) -> list[str]:
    return [c for c in F.columns if c not in XGB_FEATURES_EXCLUDE]


# ----------------------------------------------------------------------------- outbreak threshold
def endemic_channel(df: pd.DataFrame) -> pd.DataFrame:
    """Classic endemic-channel style thresholds per season_week from training seasons:
         typical = mean, warning = mean + 1 SD, epidemic = mean + 2 SD,
       each smoothed with a centred circular 5-week moving average."""
    tr = df[train_mask(df)]
    g = tr.groupby("season_week").total_cases
    out = pd.DataFrame({"typical": g.mean(), "warning": g.mean() + g.std(),
                        "epidemic": g.mean() + 2 * g.std()})

    def smooth(v):
        a = v.values
        return np.convolve(np.r_[a[-2:], a, a[:2]], np.ones(5) / 5, "valid")

    return out.apply(smooth).set_axis(out.index)
