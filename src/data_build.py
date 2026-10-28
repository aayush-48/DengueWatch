"""
Step B - Dataset collection and preparation.

Builds one clean weekly table per city by merging five raw sources from the
CDC/NOAA Dengue Forecasting Project (2015) repository:

  1. weekly laboratory-confirmed dengue cases (target)          - CDC / PR Dept. of Health / NAMRU-6
  2. daily GHCN station temperature + precipitation             - NOAA NCEI
  3. daily CFSR reanalysis humidity / temperature               - NOAA NCEP
  4. daily PERSIANN-CDR satellite precipitation                 - NOAA CDR
  5. weekly NDVI-CDR vegetation index (4 pixels near the city)  - NOAA CDR
  6. yearly population estimates (for incidence per 100k)

Every data-quality decision is logged to data/processed/quality_report.json so
the report and the app can show exactly what was done.

Run:  python -m src.data_build
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import CITIES, PROCESSED, RAW

K_TO_C = 273.15


# ----------------------------------------------------------------------------- loaders
def load_cases(city: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / CITIES[city]["cases"], parse_dates=["week_start_date"])
    return df.sort_values("week_start_date").reset_index(drop=True)


def load_station(city: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / CITIES[city]["station"], skipinitialspace=True)
    df["date"] = pd.to_datetime(dict(year=df.YYYY, month=df.MM, day=df.DD))
    df = df.drop(columns=["YYYY", "MM", "DD"]).set_index("date")
    return df.replace(-9999.0, np.nan)


def _read_noaa_sheet(path, sheet, names):
    """NOAA xlsx sheets have a 1-row units header under the column names."""
    df = pd.read_excel(path, sheet_name=sheet, skiprows=2, header=None)
    df = df.iloc[:, : len(names)]
    df.columns = names
    df = df.dropna(subset=["Year"])
    df["date"] = pd.to_datetime(
        dict(year=df.Year.astype(int), month=df.Month.astype(int), day=df.Day.astype(int))
    )
    df = df.drop(columns=["Year", "Month", "Day"]).set_index("date")
    df = df.apply(pd.to_numeric, errors="coerce").replace(-9999.0, np.nan)
    return df[~df.index.duplicated(keep="first")]


def load_reanalysis(city: str) -> pd.DataFrame:
    path = RAW / CITIES[city]["reanalysis"]
    hum = _read_noaa_sheet(path, "ReanalysisHumidity",
                           ["Year", "Month", "Day", "air_temp_k", "dew_point_k", "rh", "q"])
    tmp = _read_noaa_sheet(path, CITIES[city]["rean_temp_sheet"],
                           ["Year", "Month", "Day", "tmin_k", "tmax_k", "prcp", "tdtr_k", "tavg_k"])
    out = hum.join(tmp, how="outer")
    for c in ["air_temp_k", "dew_point_k", "tmin_k", "tmax_k", "tavg_k"]:
        out[c.replace("_k", "_c")] = out[c] - K_TO_C
    return out


def load_persiann(city: str) -> pd.Series:
    path = RAW / CITIES[city]["persiann"]
    df = pd.read_excel(path, skiprows=2, header=None).iloc[:, :4]
    df.columns = ["Year", "Month", "Day", "prcp"]
    df = df.dropna(subset=["Year"])
    df["date"] = pd.to_datetime(
        dict(year=df.Year.astype(int), month=df.Month.astype(int), day=df.Day.astype(int))
    )
    s = pd.to_numeric(df.set_index("date")["prcp"], errors="coerce").replace(-9999.0, np.nan)
    return s[~s.index.duplicated()]


def load_ndvi(city: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(RAW / CITIES[city]["ndvi"], skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(how="all")
    pix = [c for c in df.columns if c not in ("Year", "Week")]
    df[pix] = df[pix].replace(-9999.0, np.nan)
    means = df[pix].mean()
    # Pixels whose long-run mean NDVI is < 0.1 are dominated by water / urban surface
    # (NDVI near zero or negative), so they say nothing about mosquito habitat vegetation.
    keep = [p for p in pix if means[p] >= 0.1]
    df["ndvi"] = df[keep].mean(axis=1)
    info = {"pixels_total": len(pix), "pixels_kept": len(keep),
            "pixel_means": {p: round(float(m), 3) for p, m in means.items()}}
    return df[["Year", "Week", "ndvi"]].astype({"Year": int, "Week": int}), info


def load_population(city: str) -> pd.Series:
    df = pd.read_csv(RAW / CITIES[city]["population"])
    return df.set_index("Year")["Estimated_population"]


# ----------------------------------------------------------------------------- weekly alignment
def weekly_windows(weeks: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Each case-week runs from its start date up to (not incl.) the next start date.
    The last week of each calendar year is 8-9 days long in the source data."""
    starts = list(weeks)
    ends = starts[1:] + [starts[-1] + pd.Timedelta(days=7)]
    return list(zip(starts, ends))


def aggregate_daily(daily: pd.DataFrame, windows, how: dict) -> pd.DataFrame:
    """how = {col: 'mean' | 'sum7'}. 'sum7' rescales sums to a 7-day equivalent so that
    the 8-9 day year-end weeks are not biased upwards. Weeks with < 4 valid days -> NaN."""
    rows = []
    for start, end in windows:
        chunk = daily.loc[(daily.index >= start) & (daily.index < end)]
        n_days = (end - start).days
        rec = {}
        for col, agg in how.items():
            vals = chunk[col].dropna() if col in chunk else pd.Series(dtype=float)
            if len(vals) < 4:
                rec[col] = np.nan
            elif agg == "mean":
                rec[col] = vals.mean()
            else:  # sum7
                rec[col] = vals.mean() * 7.0  # mean daily amount * 7 = weekly equivalent
        rec["_days"] = n_days
        rows.append(rec)
    return pd.DataFrame(rows)


def regression_fill(target: pd.Series, proxy: pd.Series) -> tuple[pd.Series, dict]:
    """Fill gaps in a station series using a linear fit on the reanalysis/satellite proxy,
    estimated only on weeks where both are observed (bias-corrects the proxy)."""
    both = target.notna() & proxy.notna()
    info = {"missing_before": int(target.isna().sum())}
    if both.sum() < 30:
        info["method"] = "insufficient overlap"
        return target, info
    a, b = np.polyfit(proxy[both], target[both], 1)
    r = float(np.corrcoef(proxy[both], target[both])[0, 1])
    filled = target.copy()
    gap = target.isna() & proxy.notna()
    filled[gap] = a * proxy[gap] + b
    info.update({"method": "linear regression on proxy", "slope": round(float(a), 3),
                 "intercept": round(float(b), 3), "r": round(r, 3),
                 "filled": int(gap.sum())})
    return filled, info


# ----------------------------------------------------------------------------- main build
def build_city(city: str) -> tuple[pd.DataFrame, dict]:
    q: dict = {"city": CITIES[city]["name"]}
    cases = load_cases(city)
    q["rows"] = len(cases)
    q["date_range"] = [str(cases.week_start_date.min().date()), str(cases.week_start_date.max().date())]
    q["duplicate_dates"] = int(cases.week_start_date.duplicated().sum())
    gaps = cases.week_start_date.diff().dt.days.value_counts().to_dict()
    q["week_length_days_counts"] = {str(int(k)): int(v) for k, v in gaps.items()}
    cases = cases.drop_duplicates("week_start_date")
    q["missing_case_values"] = int(cases.total_cases.isna().sum())
    serotype_cols = [c for c in cases.columns if c.endswith("_cases") and c != "total_cases"]
    q["total_equals_sum_of_components"] = bool((cases[serotype_cols].sum(axis=1) == cases.total_cases).all())

    win = weekly_windows(cases.week_start_date)

    # --- station data
    st = load_station(city)
    q["station_daily_missing_pct"] = {c: round(float(st[c].isna().mean() * 100), 1)
                                      for c in ["TMAX", "TMIN", "TAVG", "PRCP"]}
    # physically implausible values -> NaN (range checks)
    for c in ["TMAX", "TMIN", "TAVG"]:
        bad = (st[c] < 5) | (st[c] > 45)
        st.loc[bad, c] = np.nan
    bad_p = (st.PRCP < 0) | (st.PRCP > 500)
    st.loc[bad_p, "PRCP"] = np.nan
    # TAVG is often missing while TMAX and TMIN exist: reconstruct it as their midpoint,
    # and recompute the diurnal range the same way.
    recon = st.TAVG.isna() & st.TMAX.notna() & st.TMIN.notna()
    st.loc[recon, "TAVG"] = (st.TMAX + st.TMIN)[recon] / 2
    st["TDTR"] = st.TMAX - st.TMIN
    q["station_tavg_reconstructed_days"] = int(recon.sum())
    st_w = aggregate_daily(st, win, {"TAVG": "mean", "TMIN": "mean", "TMAX": "mean",
                                     "TDTR": "mean", "PRCP": "sum7"})
    # Same idea at weekly level: TMAX/TMIN often have >= 4 valid days in weeks where TAVG does not.
    recon_w = st_w.TAVG.isna() & st_w.TMAX.notna() & st_w.TMIN.notna()
    st_w.loc[recon_w, "TAVG"] = (st_w.TMAX + st_w.TMIN)[recon_w] / 2
    q["station_tavg_reconstructed_weeks"] = int(recon_w.sum())

    # --- reanalysis + satellite
    re = load_reanalysis(city)
    re_w = aggregate_daily(re, win, {"air_temp_c": "mean", "dew_point_c": "mean", "rh": "mean",
                                     "q": "mean", "tmin_c": "mean", "tmax_c": "mean",
                                     "tavg_c": "mean"})
    sat = load_persiann(city).to_frame("prcp")
    sat_w = aggregate_daily(sat, win, {"prcp": "sum7"})

    df = cases[["week_start_date", "season", "season_week", "total_cases"] + serotype_cols].copy()
    df = df.reset_index(drop=True)
    df["week_days"] = st_w["_days"].values

    # station temperature: fill gaps from reanalysis (bias-corrected by regression)
    fills = {}
    df["temp_avg_c"], fills["temp_avg_c"] = regression_fill(st_w["TAVG"], re_w["tavg_c"])
    df["temp_min_c"], fills["temp_min_c"] = regression_fill(st_w["TMIN"], re_w["tmin_c"])
    df["temp_max_c"], fills["temp_max_c"] = regression_fill(st_w["TMAX"], re_w["tmax_c"])
    df["temp_dtr_c"] = df["temp_max_c"] - df["temp_min_c"]
    df["precip_mm"], fills["precip_mm"] = regression_fill(st_w["PRCP"], sat_w["prcp"])
    df["precip_mm"] = df["precip_mm"].clip(lower=0)
    df["precip_sat_mm"] = sat_w["prcp"].values
    df["rel_humidity_pct"] = re_w["rh"].values
    df["spec_humidity_gkg"] = re_w["q"].values
    df["dew_point_c"] = re_w["dew_point_c"].values
    q["station_weekly_gap_fill"] = fills

    # --- NDVI: case weeks restart every 1 Jan, so (year, k-th week of year) matches NDVI (Year, Week)
    ndvi, ndvi_info = load_ndvi(city)
    df["cal_year"] = df.week_start_date.dt.year
    df["cal_week"] = df.groupby("cal_year").cumcount() + 1
    df = df.merge(ndvi, left_on=["cal_year", "cal_week"], right_on=["Year", "Week"], how="left")
    df = df.drop(columns=["Year", "Week"])
    q["ndvi"] = ndvi_info

    # --- population -> incidence per 100k (linear interpolation between census years)
    pop = load_population(city)
    years = np.arange(df.cal_year.min(), df.cal_year.max() + 1)
    pop_full = pop.reindex(years).interpolate(limit_direction="both")
    df["population"] = df.cal_year.map(pop_full).round().astype(int)
    df["incidence_100k"] = df.total_cases / df.population * 1e5

    # --- remaining climate gaps: time interpolation for short gaps (<= 4 weeks), then
    # week-of-year climatology for anything longer.
    clim_cols = ["temp_avg_c", "temp_min_c", "temp_max_c", "temp_dtr_c", "precip_mm",
                 "precip_sat_mm", "rel_humidity_pct", "spec_humidity_gkg", "dew_point_c", "ndvi"]
    q["missing_after_merge"] = {c: int(df[c].isna().sum()) for c in clim_cols}
    for c in clim_cols:
        df[c] = df[c].interpolate(limit=4, limit_direction="both")
        clim = df.groupby("cal_week")[c].transform("mean")
        df[c] = df[c].fillna(clim)
    q["missing_final"] = {c: int(df[c].isna().sum()) for c in clim_cols}

    # --- outliers: we deliberately do NOT cap the case counts (outbreak peaks ARE the signal).
    # Climate outliers are flagged with a robust z-score (median/MAD) for the EDA, not removed.
    out_flags = {}
    for c in clim_cols:
        med = df[c].median()
        mad = (df[c] - med).abs().median() or 1e-9
        z = 0.6745 * (df[c] - med) / mad
        out_flags[c] = int((z.abs() > 5).sum())
    q["climate_robust_outliers_flagged"] = out_flags

    # --- train/test flag (original challenge split)
    test_start = CITIES[city]["test_start_season"]
    df["split"] = np.where(df.season >= test_start, "test", "train")
    q["train_rows"] = int((df.split == "train").sum())
    q["test_rows"] = int((df.split == "test").sum())
    df["t"] = np.arange(len(df))  # regular weekly step index (52 per year)
    return df, q


def main():
    report = {}
    for city in CITIES:
        df, q = build_city(city)
        df.to_csv(PROCESSED / f"{city}_weekly.csv", index=False)
        report[city] = q
        print(f"[{city}] {len(df)} weeks -> {PROCESSED / f'{city}_weekly.csv'}")
    with open(PROCESSED / "quality_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
