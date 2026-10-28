"""
Step D - Model development.

Every model produces forecasts for weeks o+1 ... o+8 from EVERY forecast origin o,
using only information available at o (rolling-origin / walk-forward evaluation).
Parameters are estimated on the training seasons only (1990/91-2008/09 San Juan,
2000/01-2008/09 Iquitos); during the test seasons the models are updated with new
observations as they "arrive", exactly as an operational system would be.

All models work on y = log(1 + cases): it stabilises the variance of the
over-dispersed counts (variance/mean ratio ~ 18-76) and keeps forecasts >= 0.

Models
  0  Persistence          y(o+h) = y(o)
  0  Seasonal naive       y(o+h) = y(o+h-52)
  0  Climatology          mean log cases of that season-week in training
  1  Holt-Winters (ETS)   additive seasonality, period 52, trend chosen by AIC
  2  SARIMA               (p,0,q)(P,0,Q)52, orders chosen by AIC
  3  SARIMAX              SARIMA + lagged climate anomalies (lag >= 8 weeks)
  4  Prophet              yearly seasonality + the same climate regressors
  5  XGBoost (direct)     one gradient-boosted model per horizon h
  6  Hybrid SARIMA+XGB    XGBoost learns the SARIMA forecast error (Zhang-style hybrid)
"""
from __future__ import annotations

import itertools
import logging
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.config import MAX_HORIZON, SEASON_LENGTH
from src import features as fe

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

H = MAX_HORIZON
MIN_ORIGIN = 2 * SEASON_LENGTH  # need two years of history for seasonal features
Z80 = 1.2816  # 80% central interval


def _frame(model, origins, preds, lo=None, hi=None):
    """Long format: one row per (origin, horizon)."""
    rows = []
    for i, o in enumerate(origins):
        for h in range(1, H + 1):
            rows.append((model, o, h, o + h, preds[i, h - 1],
                         np.nan if lo is None else lo[i, h - 1],
                         np.nan if hi is None else hi[i, h - 1]))
    return pd.DataFrame(rows, columns=["model", "origin", "h", "target", "pred", "lo", "hi"])


def all_origins(n: int) -> np.ndarray:
    """Every week from the third year on, including the last observed week (whose
    forecasts are the genuine 'next 8 weeks' beyond the data)."""
    return np.arange(MIN_ORIGIN, n)


# ----------------------------------------------------------------------------- baselines
def baselines(df: pd.DataFrame) -> pd.DataFrame:
    y = df.y.values
    n = len(y)
    origins = all_origins(n)
    clim = fe.case_climatology(df)
    sw = df.season_week.values
    pers = np.zeros((len(origins), H)); snaive = np.zeros_like(pers); cl = np.zeros_like(pers)
    for i, o in enumerate(origins):
        for h in range(1, H + 1):
            pers[i, h - 1] = y[o]
            snaive[i, h - 1] = y[o + h - SEASON_LENGTH]
            w = sw[o + h] if o + h < n else (sw[o] + h - 1) % SEASON_LENGTH + 1
            cl[i, h - 1] = clim[w]
    return pd.concat([_frame("Persistence", origins, pers),
                      _frame("Seasonal naive", origins, snaive),
                      _frame("Climatology", origins, cl)])


# ----------------------------------------------------------------------------- Holt-Winters
def fit_holt_winters(df: pd.DataFrame) -> dict:
    y = df.y.values[fe.train_mask(df)]
    cands = []
    for trend in (None, "add"):
        m = ExponentialSmoothing(y, trend=trend, damped_trend=trend is not None, seasonal="add",
                                 seasonal_periods=SEASON_LENGTH,
                                 initialization_method="estimated").fit()
        cands.append((m.aic, trend, m))
    aic, trend, m = min(cands, key=lambda c: c[0])
    p = {k: v for k, v in m.params.items()}
    return {"trend": trend, "aic": float(aic), "params": p,
            "aic_table": {str(t): round(float(a), 1) for a, t, _ in cands}}


def holt_winters_forecast(y_hist: np.ndarray, spec: dict, steps: int = H) -> np.ndarray:
    """Re-run the HW recursion on y_hist with the parameters fixed at their training
    estimates (no re-optimisation => fast, and no information leakage)."""
    p, trend = spec["params"], spec["trend"]
    kw = dict(initialization_method="known", initial_level=p["initial_level"],
              initial_seasonal=p["initial_seasons"])
    fit_kw = dict(smoothing_level=p["smoothing_level"], smoothing_seasonal=p["smoothing_seasonal"],
                  optimized=False)
    if trend:
        kw["initial_trend"] = p["initial_trend"]
        fit_kw.update(smoothing_trend=p["smoothing_trend"], damping_trend=p["damping_trend"])
    m = ExponentialSmoothing(y_hist, trend=trend, damped_trend=trend is not None, seasonal="add",
                             seasonal_periods=SEASON_LENGTH, **kw).fit(**fit_kw)
    return m.forecast(steps)


def holt_winters_rolling(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    y = df.y.values
    origins = all_origins(len(y))
    preds = np.array([holt_winters_forecast(y[: o + 1], spec) for o in origins])
    return _frame("Holt-Winters", origins, preds)


# ----------------------------------------------------------------------------- SARIMA / SARIMAX
SARIMA_GRID = [((1, 0, 1), (1, 0, 0, 52)), ((2, 0, 1), (1, 0, 0, 52)),
               ((1, 0, 1), (1, 0, 1, 52)), ((2, 0, 1), (1, 0, 1, 52)),
               ((1, 0, 0), (1, 0, 1, 52)), ((2, 0, 0), (0, 0, 0, 0))]


def select_sarima(df: pd.DataFrame, exog: pd.DataFrame | None = None) -> dict:
    """AIC grid search on training data. d = 0 because ADF and KPSS both say log(1+cases)
    is stationary in both cities (see EDA); D = 0 for the same reason."""
    tr = fe.train_mask(df)
    y = df.y.values[tr]
    X = None if exog is None else exog.values[: len(df)][tr]
    table, best = [], None
    for order, sorder in SARIMA_GRID:
        t = time.time()
        try:
            r = SARIMAX(y, exog=X, order=order, seasonal_order=sorder, trend="c").fit(disp=False, maxiter=200)
        except Exception as e:  # pragma: no cover
            table.append({"order": str(order), "seasonal": str(sorder), "aic": None, "error": str(e)})
            continue
        table.append({"order": str(order), "seasonal": str(sorder), "aic": round(float(r.aic), 1),
                      "bic": round(float(r.bic), 1), "fit_s": round(time.time() - t, 1)})
        if best is None or r.aic < best[0]:
            best = (r.aic, order, sorder, r)
    aic, order, sorder, r = best
    return {"order": order, "seasonal_order": sorder, "aic": float(aic),
            "params": r.params.tolist(), "param_names": list(r.model.param_names),
            "table": table, "summary": str(r.summary())}


def sarima_filter(y: np.ndarray, spec: dict, exog: np.ndarray | None = None):
    """Build the state-space model on the full history and run the Kalman filter with
    the training parameters fixed (no refit)."""
    m = SARIMAX(y, exog=exog, order=tuple(spec["order"]),
                seasonal_order=tuple(spec["seasonal_order"]), trend="c")
    return m.filter(np.array(spec["params"]))


def sarima_rolling(df: pd.DataFrame, spec: dict, name: str, exog: pd.DataFrame | None = None,
                   origins: np.ndarray | None = None) -> pd.DataFrame:
    """`exog`, if given, must cover the observed weeks AND the next H future weeks
    (built on features.extend_future), so that forecasts past the end of the data still
    receive their already-observed lagged climate inputs."""
    y = df.y.values
    n = len(y)
    X = None if exog is None else exog.values
    res = sarima_filter(y, spec, None if X is None else X[:n])
    origins = all_origins(n) if origins is None else origins
    P, L, U = (np.full((len(origins), H), np.nan) for _ in range(3))
    for i, o in enumerate(origins):
        end = min(o + H, n - 1)
        if end > o:
            pr = res.get_prediction(start=o + 1, end=end, dynamic=True)
            k = end - o
            P[i, :k] = pr.predicted_mean
            ci = pr.conf_int(alpha=0.2)
            L[i, :k], U[i, :k] = ci[:, 0], ci[:, 1]
        if end < o + H:  # beyond the data: genuine out-of-sample forecast
            res_o = sarima_filter(y[: o + 1], spec, None if X is None else X[: o + 1])
            Xf = None if X is None else X[o + 1: o + 1 + H]
            fc = res_o.get_forecast(H, exog=Xf)
            P[i], ci = fc.predicted_mean, fc.conf_int(alpha=0.2)
            L[i], U[i] = ci[:, 0], ci[:, 1]
    return _frame(name, origins, P, L, U)


def sarima_in_sample_h(df: pd.DataFrame, spec: dict, exog=None) -> np.ndarray:
    """Matrix [origin, h] of dynamic h-step SARIMA forecasts for every origin (used as an
    input by the hybrid model)."""
    fr = sarima_rolling(df, spec, "tmp", exog)
    n = len(df)
    M = np.full((n, H), np.nan)
    for o, h, p in fr[["origin", "h", "pred"]].itertuples(index=False):
        M[o, h - 1] = p
    return M


# ----------------------------------------------------------------------------- Prophet
def prophet_rolling(df: pd.DataFrame, exog: pd.DataFrame, refit_every: int = 4) -> pd.DataFrame:
    """Prophet cannot condition on the latest observation, so it is refitted every
    `refit_every` weeks during the test period (a realistic monthly retraining cycle).
    Training-period origins use the single model fitted on all training data (in-sample)."""
    from prophet import Prophet

    n = len(df)
    tr = fe.train_mask(df)
    t0 = int(np.argmax(~tr))  # first test index
    dfx = fe.extend_future(df, H)
    base = pd.DataFrame({"ds": dfx.week_start_date, "y": dfx.y})
    regs = list(exog.columns)
    base = pd.concat([base, exog.reset_index(drop=True)], axis=1)  # exog has n + H rows

    def fit(upto: int):
        m = Prophet(yearly_seasonality=10, weekly_seasonality=False, daily_seasonality=False,
                    interval_width=0.8, changepoint_prior_scale=0.05, uncertainty_samples=200)
        for r in regs:
            m.add_regressor(r)
        m.fit(base.iloc[: upto + 1])
        return m

    def future_frame(o):
        return base.iloc[o + 1: o + H + 1].drop(columns="y")

    origins = all_origins(n)
    P, L, U = (np.full((len(origins), H), np.nan) for _ in range(3))
    m_train = fit(t0 - 1)
    model, fitted_at = m_train, t0 - 1
    for i, o in enumerate(origins):
        if o >= t0 and (o - fitted_at) >= refit_every:
            model, fitted_at = fit(o), o
        fc = model.predict(future_frame(o))
        P[i], L[i], U[i] = fc.yhat.values, fc.yhat_lower.values, fc.yhat_upper.values
    return _frame("Prophet", origins, P, L, U)


# ----------------------------------------------------------------------------- XGBoost
XGB_GRID = {"max_depth": [2, 3, 4], "n_estimators": [200, 500], "min_child_weight": [5]}
XGB_FIXED = dict(learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                 objective="reg:squarederror", n_jobs=4, random_state=42)


def _train_rows(F: pd.DataFrame, df: pd.DataFrame, h: int, last_target: int) -> np.ndarray:
    """Origins whose target o+h lies in the allowed period and whose features exist."""
    idx = np.arange(len(F))
    ok = (idx >= MIN_ORIGIN) & (idx + h <= last_target) & F["target"].notna().values
    return idx[ok]


def tune_xgb(df, clim, climate_vars, h=4, extra=None, n_splits=4) -> tuple[dict, list]:
    """Walk-forward (expanding window) cross-validation on TRAINING origins only."""
    F = fe.xgb_feature_frame(df, clim, climate_vars, h)
    if extra is not None:
        F = extra(F, h)
    t0 = int(np.argmax(~fe.train_mask(df)))
    rows = _train_rows(F, df, h, t0 - 1)
    cols = fe.feature_columns(F)
    folds = np.array_split(rows, n_splits + 1)
    results = []
    for md, ne, mcw in itertools.product(*XGB_GRID.values()):
        maes = []
        for k in range(1, n_splits + 1):
            tr_idx = np.concatenate(folds[:k])
            tr_idx = tr_idx[tr_idx + h < folds[k][0]]  # gap: no target overlap with val fold
            va_idx = folds[k]
            m = xgb.XGBRegressor(max_depth=md, n_estimators=ne, min_child_weight=mcw, **XGB_FIXED)
            m.fit(F.loc[tr_idx, cols], F.loc[tr_idx, "target"])
            maes.append(float(np.mean(np.abs(m.predict(F.loc[va_idx, cols]) - F.loc[va_idx, "target"]))))
        results.append({"max_depth": md, "n_estimators": ne, "min_child_weight": mcw,
                        "cv_mae_log": round(float(np.mean(maes)), 4)})
    best = min(results, key=lambda r: r["cv_mae_log"])
    return {k: best[k] for k in XGB_GRID}, results


def fit_xgb_direct(df, clim, climate_vars, params, extra=None, calib_weeks=104):
    """One model per horizon. Also estimates 80% prediction intervals from residuals on the
    last two training seasons (held out from a calibration fit) - a split-conformal approach."""
    t0 = int(np.argmax(~fe.train_mask(df)))
    models, quant, cols_by_h = {}, {}, {}
    for h in range(1, H + 1):
        F = fe.xgb_feature_frame(df, clim, climate_vars, h)
        if extra is not None:
            F = extra(F, h)
        cols = fe.feature_columns(F)
        rows = _train_rows(F, df, h, t0 - 1)
        cal_cut = t0 - 1 - calib_weeks
        fit_rows = rows[rows + h <= cal_cut]
        cal_rows = rows[rows > cal_cut]
        m_cal = xgb.XGBRegressor(**params, **XGB_FIXED).fit(F.loc[fit_rows, cols], F.loc[fit_rows, "target"])
        resid = F.loc[cal_rows, "target"].values - m_cal.predict(F.loc[cal_rows, cols])
        quant[h] = (float(np.quantile(resid, 0.1)), float(np.quantile(resid, 0.9)))
        models[h] = xgb.XGBRegressor(**params, **XGB_FIXED).fit(F.loc[rows, cols], F.loc[rows, "target"])
        cols_by_h[h] = cols
    return {"models": models, "quantiles": quant, "cols": cols_by_h, "params": params,
            "climate_vars": climate_vars}


def xgb_predict_frame(df, clim, art, extra=None, y=None) -> dict[int, pd.DataFrame]:
    """Predictions for every origin; returns {h: DataFrame(origin, pred, lo, hi)}."""
    out = {}
    for h in range(1, H + 1):
        F = fe.xgb_feature_frame(df, clim, art["climate_vars"], h, y=y)
        if extra is not None:
            F = extra(F, h)
        cols = art["cols"][h]
        base = F["y0"].values if extra is None else F["base"].values
        d = art["models"][h].predict(F[cols])
        q_lo, q_hi = art["quantiles"][h]
        out[h] = pd.DataFrame({"pred": base + d, "lo": base + d + q_lo, "hi": base + d + q_hi},
                              index=F.index)
    return out


def xgb_rolling(df, clim, art, name, extra=None) -> pd.DataFrame:
    n = len(df)
    origins = all_origins(n)
    pf = xgb_predict_frame(df, clim, art, extra)
    P, L, U = (np.zeros((len(origins), H)) for _ in range(3))
    for h in range(1, H + 1):
        P[:, h - 1] = pf[h].loc[origins, "pred"].values
        L[:, h - 1] = pf[h].loc[origins, "lo"].values
        U[:, h - 1] = pf[h].loc[origins, "hi"].values
    return _frame(name, origins, P, L, U)


def hybrid_extra(sarima_matrix: np.ndarray):
    """For the hybrid: the base forecast is SARIMA's h-step forecast and XGBoost learns
    the residual  y(o+h) - SARIMA_h(o)  from lags, season and climate."""
    def add(F: pd.DataFrame, h: int) -> pd.DataFrame:
        F = F.copy()
        base = sarima_matrix[:, h - 1]
        F["sarima_h"] = base
        F["base"] = base
        F["target"] = (F["target"] + F["y0"]) - base   # y(o+h) - SARIMA_h(o)
        return F
    return add
