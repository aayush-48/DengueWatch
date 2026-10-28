"""
Live inference used by the backend API.

  * Stored forecasts   - every model's rolling-origin forecast is precomputed by
                         train_evaluate.py; the API just looks them up (fast, reproducible).
  * What-if scenarios  - SARIMAX, XGBoost and the Hybrid are re-run on the fly with
                         user-modified recent climate.
  * Uploaded data      - a univariate pipeline (no climate needed) that fits models on a
                         user's own weekly case series and returns an 8-week forecast,
                         a back-test, and alerts from that series' own endemic channel.
"""
from __future__ import annotations

import io
import warnings
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src import features as fe
from src import models as M
from src.config import CITIES, MAX_HORIZON, MODELS, RESULTS, SEASON_LENGTH

warnings.filterwarnings("ignore")
H = MAX_HORIZON


@lru_cache(maxsize=None)
def data(city: str) -> pd.DataFrame:
    return fe.load(city)


@lru_cache(maxsize=None)
def artifacts(city: str) -> dict:
    return joblib.load(MODELS / f"{city}_artifacts.joblib")


@lru_cache(maxsize=None)
def stored_forecasts(city: str) -> pd.DataFrame:
    fc = pd.read_csv(RESULTS / f"forecasts_{city}.csv.gz", parse_dates=["target_date", "origin_date"])
    return fc


def channel_for(city: str, season_weeks) -> pd.DataFrame:
    ch = artifacts(city)["channel"]
    return ch.reindex(np.asarray(season_weeks)).reset_index(drop=True)


def origin_index(city: str, origin: str | None) -> int:
    df = data(city)
    if origin is None:
        return len(df) - 1
    ts = pd.Timestamp(origin)
    idx = int(np.searchsorted(df.week_start_date.values, ts.to_datetime64(), side="right") - 1)
    return int(np.clip(idx, M.MIN_ORIGIN, len(df) - 1))


def target_meta(city: str, o: int) -> pd.DataFrame:
    """Dates and season weeks of o+1..o+H (extending past the data when needed)."""
    dfx = fe.extend_future(data(city), H)
    t = dfx.iloc[o + 1: o + 1 + H]
    return pd.DataFrame({"h": np.arange(1, H + 1), "target": np.arange(o + 1, o + 1 + H),
                         "target_date": t.week_start_date.values, "season_week": t.season_week.values})


def lookup(city: str, o: int, model: str | None = None) -> pd.DataFrame:
    fc = stored_forecasts(city)
    g = fc[fc.origin == o]
    if model:
        g = g[g.model == model]
    g = g.copy()
    g["cases"] = np.expm1(g.pred).clip(lower=0)
    g["cases_lo"] = np.expm1(g.lo).clip(lower=0)
    g["cases_hi"] = np.expm1(g.hi).clip(lower=0)
    df = data(city)
    g["actual"] = [df.total_cases.iloc[t] if t < len(df) else np.nan for t in g.target]
    return g


# ----------------------------------------------------------------------------- what-if
def perturb(df: pd.DataFrame, o: int, weeks: int, temp_delta: float, humidity_pct: float,
            rain_pct: float) -> pd.DataFrame:
    """Apply a climate scenario to the `weeks` weeks ending at origin o."""
    d = df.copy()
    sl = slice(max(0, o - weeks + 1), o + 1)
    for c in ["temp_avg_c", "temp_min_c", "temp_max_c"]:
        d.loc[d.index[sl], c] += temp_delta
    d.loc[d.index[sl], "dew_point_c"] += temp_delta  # warmer air at similar RH holds more moisture
    for c in ["spec_humidity_gkg", "rel_humidity_pct"]:
        d.loc[d.index[sl], c] *= (1 + humidity_pct / 100)
    d["rel_humidity_pct"] = d["rel_humidity_pct"].clip(upper=100)
    for c in ["precip_mm", "precip_sat_mm"]:
        d.loc[d.index[sl], c] *= (1 + rain_pct / 100)
    d["temp_dtr_c"] = d["temp_max_c"] - d["temp_min_c"]
    return d


def live_forecasts(city: str, o: int, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Re-run SARIMAX, XGBoost and Hybrid from origin o on (possibly modified) data."""
    art = artifacts(city)
    df = data(city) if df is None else df
    hist = df.iloc[: o + 1].reset_index(drop=True)
    clim = art["clim"]
    case_clim = fe.case_climatology(data(city))
    out = {}

    # SARIMAX
    exog = fe.exog_matrix(fe.extend_future(hist, H), clim, art["sarimax_feats"]).values
    res = M.sarima_filter(hist.y.values, art["sarimax"], exog[: o + 1])
    fc = res.get_forecast(H, exog=exog[o + 1: o + 1 + H])
    ci = fc.conf_int(alpha=0.2)
    out["SARIMAX"] = (fc.predicted_mean, ci[:, 0], ci[:, 1])

    # SARIMA (needed as the hybrid's base)
    rs = M.sarima_filter(hist.y.values, art["sarima"])
    sar = rs.get_forecast(H).predicted_mean

    # XGBoost + Hybrid: last row of the feature frame = origin o
    for name, a in (("XGBoost", art["xgb"]), ("Hybrid SARIMA+XGB", art["hybrid"])):
        P, L, U = np.zeros(H), np.zeros(H), np.zeros(H)
        for h in range(1, H + 1):
            F = fe.xgb_feature_frame(hist, clim, a["climate_vars"], h, case_clim=case_clim)
            row = F.iloc[[-1]].copy()
            base = row["y0"].values[0]
            if name.startswith("Hybrid"):
                row["sarima_h"] = sar[h - 1]
                base = sar[h - 1]
            d = float(a["models"][h].predict(row[a["cols"][h]])[0])
            q_lo, q_hi = a["quantiles"][h]
            P[h - 1], L[h - 1], U[h - 1] = base + d, base + d + q_lo, base + d + q_hi
        out[name] = (P, L, U)

    meta = target_meta(city, o)
    rows = []
    for m, (p, lo, hi) in out.items():
        for i in range(H):
            rows.append({"model": m, "h": i + 1, "target_date": meta.target_date.iloc[i],
                         "cases": max(0.0, float(np.expm1(p[i]))),
                         "cases_lo": max(0.0, float(np.expm1(lo[i]))),
                         "cases_hi": max(0.0, float(np.expm1(hi[i])))})
    return pd.DataFrame(rows)


def whatif(city: str, o: int, temp_delta=0.0, humidity_pct=0.0, rain_pct=0.0, weeks=8) -> dict:
    base = live_forecasts(city, o)
    mod_df = perturb(data(city), o, weeks, temp_delta, humidity_pct, rain_pct)
    scen = live_forecasts(city, o, mod_df)
    m = base.merge(scen, on=["model", "h", "target_date"], suffixes=("_base", "_scenario"))
    m["change_pct"] = (m.cases_scenario - m.cases_base) / m.cases_base.replace(0, np.nan) * 100
    art = artifacts(city)
    return {"table": m, "used_climate": {
        "SARIMAX": [f"{f['variable']} (lag {f['lag']} wk)" for f in art["sarimax_feats"]],
        "XGBoost / Hybrid": art["xgb_vars"]}}


# ----------------------------------------------------------------------------- uploaded data
class UploadError(ValueError):
    pass


def parse_upload(raw: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise UploadError(f"Could not read the CSV: {e}")
    cols = {c.lower().strip(): c for c in df.columns}
    date_col = next((cols[c] for c in cols if c in ("week_start_date", "date", "week", "ds")), None)
    case_col = next((cols[c] for c in cols if c in ("total_cases", "cases", "count", "y")), None)
    if date_col is None or case_col is None:
        raise UploadError("CSV needs a date column (week_start_date/date/week) and a case column "
                          "(total_cases/cases/count).")
    d = pd.DataFrame({"date": pd.to_datetime(df[date_col], errors="coerce"),
                      "cases": pd.to_numeric(df[case_col], errors="coerce")})
    bad_dates = int(d.date.isna().sum())
    d = d.dropna(subset=["date"]).sort_values("date")
    dups = int(d.date.duplicated().sum())
    d = d.groupby("date", as_index=False).cases.sum(min_count=1)
    # regularise to a weekly grid anchored on the first date
    span = (d.date.max() - d.date.min()).days
    if span / 7 < 3 * SEASON_LENGTH:
        raise UploadError(f"Need at least 3 years of weekly data (got about {span / 7:.0f} weeks).")
    grid = pd.date_range(d.date.min(), d.date.max(), freq="7D")
    s = d.set_index("date").cases
    s = s.reindex(s.index.union(grid)).interpolate("time").reindex(grid)
    missing = int(s.isna().sum())
    s = s.ffill().bfill().clip(lower=0)
    out = pd.DataFrame({"week_start_date": grid, "total_cases": s.values})
    out.attrs["quality"] = {"rows_read": int(len(df)), "unparseable_dates": bad_dates,
                            "duplicate_dates_merged": dups, "weeks": int(len(out)),
                            "gaps_after_regularising": missing,
                            "negative_values_clipped": int((d.cases < 0).sum())}
    return out


def upload_forecast(raw: bytes, backtest_weeks: int = 52, hosp_rate: float = 0.1) -> dict:
    from src.recommend import assess

    d = parse_upload(raw)
    q = d.attrs["quality"]
    n = len(d)
    y = np.log1p(d.total_cases.values)
    d["season_week"] = (np.arange(n) % SEASON_LENGTH) + 1
    cut = n - backtest_weeks          # parameters estimated on data before the back-test
    tr = np.arange(n) < cut

    # endemic channel from the history before the back-test window
    g = d[tr].groupby("season_week").total_cases
    ch = pd.DataFrame({"typical": g.mean(), "warning": g.mean() + g.std(),
                       "epidemic": g.mean() + 2 * g.std()})

    preds = {}
    # Seasonal naive & persistence
    origins = np.arange(cut - 1, n)
    def stack(f):
        return np.array([[f(o, h) for h in range(1, H + 1)] for o in origins])
    preds["Seasonal naive"] = stack(lambda o, h: y[o + h - SEASON_LENGTH])
    preds["Persistence"] = stack(lambda o, h: y[o])

    # Holt-Winters with fixed params
    hw = ExponentialSmoothing(y[:cut], trend=None, seasonal="add", seasonal_periods=SEASON_LENGTH,
                              initialization_method="estimated").fit()
    spec = {"trend": None, "params": dict(hw.params)}
    preds["Holt-Winters"] = np.array([M.holt_winters_forecast(y[: o + 1], spec) for o in origins])

    # SARIMA(1,0,1)(1,0,0)52 - fixed, light specification for speed
    r = SARIMAX(y[:cut], order=(1, 0, 1), seasonal_order=(1, 0, 0, 52), trend="c").fit(disp=False)
    sspec = {"order": (1, 0, 1), "seasonal_order": (1, 0, 0, 52), "params": r.params.tolist()}
    sres = M.sarima_filter(y, sspec)
    S = []
    for o in origins:
        if o + H <= n - 1:
            S.append(sres.get_prediction(start=o + 1, end=o + H, dynamic=True).predicted_mean)
        else:
            S.append(M.sarima_filter(y[: o + 1], sspec).get_forecast(H).predicted_mean)
    preds["SARIMA"] = np.array(S)

    # XGBoost direct, univariate features
    dd = d.assign(y=y, split=np.where(tr, "train", "test"), cal_week=d.season_week)
    case_clim = dd[tr].groupby("season_week").y.mean()
    X_models = {}
    xp = np.zeros((len(origins), H))
    for h in range(1, H + 1):
        F = fe.xgb_feature_frame(dd, pd.DataFrame(), [], h, case_clim=case_clim)
        cols = fe.feature_columns(F)
        rows = np.arange(n)
        fit_rows = rows[(rows >= M.MIN_ORIGIN) & (rows + h < cut)]
        m = xgb.XGBRegressor(max_depth=3, n_estimators=300, min_child_weight=5, **M.XGB_FIXED)
        m.fit(F.loc[fit_rows, cols], F.loc[fit_rows, "target"])
        xp[:, h - 1] = F.loc[origins, "y0"].values + m.predict(F.loc[origins, cols])
        X_models[h] = m
    preds["XGBoost"] = xp

    # back-test scoring on the held-out window
    actual = d.total_cases.values
    scores = []
    for name, P in preds.items():
        errs = []
        for i, o in enumerate(origins):
            for h in range(1, H + 1):
                t = o + h
                if t < n:
                    errs.append(np.expm1(P[i, h - 1]) - actual[t])
        errs = np.array(errs)
        scores.append({"model": name, "MAE": float(np.mean(np.abs(errs))),
                       "RMSE": float(np.sqrt(np.mean(errs ** 2)))})
    scores = sorted(scores, key=lambda s: s["MAE"])
    best = scores[0]["model"]

    # forecast from the last observed week with the best model
    last = preds[best][-1]
    fdates = [d.week_start_date.iloc[-1] + pd.Timedelta(weeks=k) for k in range(1, H + 1)]
    fweeks = [(d.season_week.iloc[-1] + k - 1) % SEASON_LENGTH + 1 for k in range(1, H + 1)]
    chf = ch.reindex(fweeks).reset_index(drop=True)
    resid = np.array([np.expm1(preds[best][i, 0]) - actual[o + 1] for i, o in enumerate(origins) if o + 1 < n])
    q_lo, q_hi = np.quantile(resid, [0.1, 0.9])
    widen = np.sqrt(np.arange(1, H + 1))  # one-step error quantiles, widened ~sqrt(h) with horizon
    lo_q, hi_q = q_lo * widen, q_hi * widen
    f = pd.DataFrame({"target_date": fdates, "h": np.arange(1, H + 1),
                      "cases": np.expm1(last).clip(0), "typical": chf.typical, "warning": chf.warning,
                      "epidemic": chf.epidemic})
    f["lo"] = (f.cases + lo_q).clip(lower=0)
    f["hi"] = f.cases + hi_q
    return {"quality": q, "history": d[["week_start_date", "total_cases"]], "channel": ch.reset_index(),
            "backtest": scores, "best_model": best, "forecast": f,
            "assessment": assess(f, hosp_rate), "all_last": {k: np.expm1(v[-1]).clip(0).tolist()
                                                               for k, v in preds.items()}}
