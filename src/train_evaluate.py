"""
Steps D + G - Train every model, run the rolling-origin evaluation on the four test
seasons (2009/10 - 2012/13), score forecasts and outbreak alerts, and save:

  reports/forecasts_<city>.csv.gz   every model's forecast from every origin (used by the app)
  reports/metrics.json              accuracy + alert metrics per city / model / horizon
  reports/model_selection.json      AIC tables, CV results, selected climate features
  models/<city>_artifacts.joblib    fitted parameters + XGBoost models for live forecasting
  reports/figures/1x_*.png          evaluation figures for the report

Run:  python -m src.train_evaluate        (about 5-10 minutes on a laptop)
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd

from src import features as fe
from src import models as M
from src.config import CITIES, FIGURES, MAX_HORIZON, MODELS, RESULTS
from src.recommend import outbreak_episodes

MODEL_ORDER = ["Persistence", "Seasonal naive", "Climatology", "Holt-Winters", "SARIMA",
               "SARIMAX", "Prophet", "XGBoost", "Hybrid SARIMA+XGB", "Ensemble"]
ENSEMBLE_OF = ["SARIMAX", "XGBoost", "Hybrid SARIMA+XGB"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- scoring
ALERT_RULES = {
    "point": "point forecast above the epidemic threshold",
    "upper80": "upper 80% bound above the epidemic threshold (>= ~10% chance)",
}


def add_ensemble(fc: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight average (in log space) of the three climate-aware models. Averaging
    models whose errors are not perfectly correlated usually reduces error."""
    parts = fc[fc.model.isin(ENSEMBLE_OF)]
    key = ["origin", "h", "target"]
    ens = parts.groupby(key)[["pred", "lo", "hi"]].mean().reset_index()
    meta = parts.drop_duplicates(key).set_index(key)[["target_date", "origin_date", "in_sample"]]
    ens = ens.join(meta, on=key)
    ens["model"] = "Ensemble"
    return pd.concat([fc[fc.model != "Ensemble"], ens[fc.columns]], ignore_index=True)


def _alerts(cases, hi_cases, thr, rule):
    if rule == "upper80":
        bound = np.where(np.isnan(hi_cases), cases, hi_cases)
        return bound > thr
    return cases > thr


def score(fc: pd.DataFrame, df: pd.DataFrame, channel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    n = len(df)
    t0 = int(np.argmax(df.split.values == "test"))
    actual = df.total_cases.values.astype(float)
    sw = df.season_week.values
    thr_by_week = channel["epidemic"]
    test = fc[(fc.target >= t0) & (fc.target < n)].copy()
    test["actual"] = actual[test.target.values]
    test["cases"] = np.expm1(test.pred).clip(lower=0)
    test["cases_hi"] = np.expm1(test.hi)
    test["epi_thr"] = thr_by_week.reindex(sw[test.target.values]).values
    test["outbreak"] = test.actual > test.epi_thr
    for rule in ALERT_RULES:
        test[f"alert_{rule}"] = _alerts(test.cases.values, test.cases_hi.values, test.epi_thr.values, rule)
    has_pi = test.lo.notna()
    test["covered"] = np.where(has_pi, (np.log1p(test.actual) >= test.lo) & (np.log1p(test.actual) <= test.hi), np.nan)

    rows = []
    for (m, h), g in test.groupby(["model", "h"]):
        err = g.cases - g.actual
        r = {"model": m, "h": h, "n": len(g), "MAE": err.abs().mean(),
             "RMSE": np.sqrt((err ** 2).mean()), "MAE_log": (g.pred - np.log1p(g.actual)).abs().mean(),
             "bias": err.mean(), "PI80_coverage": g.covered.mean() if g.covered.notna().any() else np.nan}
        for rule in ALERT_RULES:
            a, o = g[f"alert_{rule}"].values, g.outbreak.values
            tp, fp, fn = int((a & o).sum()), int((a & ~o).sum()), int((~a & o).sum())
            sfx = "" if rule == "point" else "_p10"
            r[f"alert_precision{sfx}"] = tp / (tp + fp) if tp + fp else np.nan
            r[f"alert_recall{sfx}"] = tp / (tp + fn) if tp + fn else np.nan
            r[f"alert_F1{sfx}"] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            r[f"false_alarms{sfx}"] = fp
        rows.append(r)
    met = pd.DataFrame(rows)
    sn = met[met.model == "Seasonal naive"].set_index("h")["MAE"]
    met["skill_vs_seasonal_naive"] = 1 - met.MAE / met.h.map(sn)

    # Episode lead time: the largest horizon h (<= 8) whose forecast, made h weeks before the
    # outbreak's first above-threshold week, already put that week above the threshold.
    is_out = actual[t0:] > thr_by_week.reindex(sw[t0:]).values
    episodes = [(a + t0, b + t0) for a, b in outbreak_episodes(is_out)]
    ep_rows = []
    for (a, b) in episodes:
        thr = float(thr_by_week.reindex([sw[a]]).values[0])
        for m in fc.model.unique():
            g = fc[(fc.model == m) & (fc.target == a)]
            row = {"model": m, "onset_index": int(a), "onset": str(df.week_start_date[a].date()),
                   "end": str(df.week_start_date[b].date()), "peak": int(actual[a:b + 1].max()),
                   "threshold_at_onset": round(thr, 1)}
            for rule in ALERT_RULES:
                ok = g[_alerts(np.expm1(g.pred.values), np.expm1(g.hi.values), thr, rule)]
                row["lead_weeks" if rule == "point" else "lead_weeks_p10"] = int(ok.h.max()) if len(ok) else 0
            ep_rows.append(row)
    return met, pd.DataFrame(ep_rows), [(int(a), int(b)) for a, b in episodes]


# ----------------------------------------------------------------------------- per city
def run_city(city: str) -> dict:
    log(f"==== {CITIES[city]['name']} ====")
    df = fe.load(city)
    clim = fe.climate_climatology(df)
    channel = fe.endemic_channel(df)
    sel = {}

    sarimax_feats = fe.select_climate_features(df, clim, min_lag=MAX_HORIZON, max_lag=16, k=3)
    short = fe.select_climate_features(df, clim, min_lag=1, max_lag=16, k=4)
    xgb_vars = list(dict.fromkeys([f["variable"] for f in sarimax_feats + short]))
    exog = fe.exog_matrix(fe.extend_future(df, MAX_HORIZON), clim, sarimax_feats)  # n + 8 rows
    sel["climate_features_sarimax"] = sarimax_feats
    sel["climate_features_short_lag"] = short
    sel["xgb_climate_vars"] = xgb_vars
    log(f"climate features (lag>=8): {sarimax_feats}")

    frames = [M.baselines(df)]

    log("Holt-Winters")
    hw = M.fit_holt_winters(df)
    sel["holt_winters"] = {"trend": hw["trend"], "aic_table": hw["aic_table"],
                           "smoothing": {k: (None if v is None or (isinstance(v, float) and np.isnan(v))
                                             else round(float(v), 4))
                                         for k, v in hw["params"].items()
                                         if k in ("smoothing_level", "smoothing_trend",
                                                  "smoothing_seasonal", "damping_trend")}}
    frames.append(M.holt_winters_rolling(df, hw))

    log("SARIMA order selection")
    sarima = M.select_sarima(df)
    sel["sarima"] = {k: sarima[k] for k in ("order", "seasonal_order", "aic", "table")}
    frames.append(M.sarima_rolling(df, sarima, "SARIMA"))
    log(f"  SARIMA {sarima['order']}x{sarima['seasonal_order']} AIC={sarima['aic']:.1f}")

    log("SARIMAX order selection")
    sarimax = M.select_sarima(df, exog)
    sel["sarimax"] = {k: sarimax[k] for k in ("order", "seasonal_order", "aic", "table")}
    # statsmodels names numpy exog columns x1, x2, ... -> map back to feature names
    xmap = {f"x{i + 1}": c for i, c in enumerate(exog.columns)}
    sel["sarimax"]["exog_coefficients"] = {xmap[n]: round(float(p), 4) for n, p in
                                           zip(sarimax["param_names"], sarimax["params"]) if n in xmap}
    frames.append(M.sarima_rolling(df, sarimax, "SARIMAX", exog))
    log(f"  SARIMAX {sarimax['order']}x{sarimax['seasonal_order']} AIC={sarimax['aic']:.1f}")

    log("Prophet (rolling refits)")
    frames.append(M.prophet_rolling(df, exog))

    log("XGBoost tuning + fitting")
    xgb_params, xgb_cv = M.tune_xgb(df, clim, xgb_vars)
    xgb_art = M.fit_xgb_direct(df, clim, xgb_vars, xgb_params)
    sel["xgb"] = {"params": xgb_params, "cv": xgb_cv}
    frames.append(M.xgb_rolling(df, clim, xgb_art, "XGBoost"))

    log("Hybrid SARIMA + XGBoost")
    S = M.sarima_in_sample_h(df, sarima)
    extra = M.hybrid_extra(S)
    hyb_params, hyb_cv = M.tune_xgb(df, clim, xgb_vars, extra=extra)
    hyb_art = M.fit_xgb_direct(df, clim, xgb_vars, hyb_params, extra=extra)
    sel["hybrid"] = {"params": hyb_params, "cv": hyb_cv}
    frames.append(M.xgb_rolling(df, clim, hyb_art, "Hybrid SARIMA+XGB", extra=extra))

    fc = pd.concat(frames, ignore_index=True)
    fc["target_date"] = fc.target.map(lambda t: df.week_start_date.iloc[t] if t < len(df)
                                      else df.week_start_date.iloc[-1] + pd.Timedelta(weeks=int(t - len(df) + 1)))
    fc["origin_date"] = df.week_start_date.values[fc.origin.values]
    fc["in_sample"] = df.split.values[np.minimum(fc.target.values, len(df) - 1)] == "train"
    fc = add_ensemble(fc)
    fc.to_csv(RESULTS / f"forecasts_{city}.csv.gz", index=False, float_format="%.4f")

    log("Scoring")
    met, eps, episodes = score(fc, df, channel)

    # feature importance (gain) of the h=4 models
    def imp(art):
        b = art["models"][4].get_booster().get_score(importance_type="gain")
        s = pd.Series(b).sort_values(ascending=False)
        return (s / s.sum()).round(4).head(15).to_dict()
    sel["xgb_importance_h4"] = imp(xgb_art)
    sel["hybrid_importance_h4"] = imp(hyb_art)

    joblib.dump({"city": city, "hw": hw, "sarima": sarima, "sarimax": sarimax,
                 "sarimax_feats": sarimax_feats, "xgb": xgb_art, "hybrid": hyb_art,
                 "xgb_vars": xgb_vars, "clim": clim, "channel": channel},
                MODELS / f"{city}_artifacts.joblib")
    return {"metrics": met, "episodes": eps, "episode_windows": episodes, "selection": sel,
            "fc": fc, "df": df, "channel": channel}


# ----------------------------------------------------------------------------- figures
def figures(results: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})
    colors = {"Seasonal naive": "#8a9ba8", "Persistence": "#b8c4cc", "Climatology": "#d0d8de",
              "Holt-Winters": "#6c757d", "SARIMA": "#3d7ea6", "SARIMAX": "#1f3b4d",
              "Prophet": "#e09f3e", "XGBoost": "#2a9d8f", "Hybrid SARIMA+XGB": "#c8553d",
              "Ensemble": "#7b2cbf"}

    for city, r in results.items():
        met, fc, df, ch = r["metrics"], r["fc"], r["df"], r["channel"]
        # MAE by horizon
        fig, ax = plt.subplots(figsize=(9.5, 3.8))
        for m in MODEL_ORDER:
            g = met[met.model == m].sort_values("h")
            ax.plot(g.h, g.MAE, marker="o", ms=3, color=colors[m], label=m,
                    lw=2.2 if m in ("Ensemble", "XGBoost", "SARIMAX") else 1)
        ax.set_xlabel("forecast horizon (weeks ahead)"); ax.set_ylabel("MAE (cases/week)")
        ax.set_title(f"{CITIES[city]['name']} - test-period MAE by horizon", loc="left")
        ax.legend(fontsize=7.5, ncol=1, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
        fig.tight_layout(); fig.savefig(FIGURES / f"11_mae_horizon_{city}.png"); plt.close(fig)

        # actual vs predicted at h=4 for 3 models
        t0 = int(np.argmax(df.split.values == "test"))
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        for ax, m in zip(axes, ["SARIMAX", "XGBoost", "Ensemble"]):
            g = fc[(fc.model == m) & (fc.h == 4) & (fc.target >= t0) & (fc.target < len(df))]
            dates = df.week_start_date.values[g.target.values]
            ax.plot(df.week_start_date[t0:], df.total_cases[t0:], color="black", lw=1, label="actual")
            ax.plot(dates, np.expm1(g.pred), color=colors[m], lw=1.4, label=f"{m} (4 weeks ahead)")
            if g.lo.notna().any():
                ax.fill_between(dates, np.expm1(g.lo).clip(0), np.expm1(g.hi), color=colors[m], alpha=0.15,
                                label="80% interval")
            ax.plot(df.week_start_date[t0:], ch["epidemic"].reindex(df.season_week[t0:]).values,
                    color="#c8553d", ls="--", lw=0.9, label="epidemic threshold")
            ax.legend(fontsize=7, frameon=False, loc="upper left"); ax.set_ylabel("cases/week")
        axes[0].set_title(f"{CITIES[city]['name']} - 4-week-ahead forecasts vs actual (test seasons)", loc="left")
        fig.tight_layout(); fig.savefig(FIGURES / f"12_actual_vs_pred_{city}.png"); plt.close(fig)

        # lead times
        eps = r["episodes"]
        if len(eps):
            piv = eps.pivot_table(index="model", columns="onset", values="lead_weeks_p10").reindex(MODEL_ORDER[::-1])
            fig, ax = plt.subplots(figsize=(9.5, 4.2))
            piv.plot(kind="barh", ax=ax, width=0.8, color=["#1f3b4d", "#c8553d", "#e09f3e", "#2a9d8f"][: piv.shape[1]])
            ax.set_xlabel("warning lead time before outbreak onset (weeks, max 8)")
            ax.set_xlim(0, 8.5)
            ax.set_title(f"{CITIES[city]['name']} - early-warning lead time per test outbreak\n"
                         f"(alert = upper 80% bound above the epidemic threshold)", loc="left")
            ax.legend(title="outbreak onset", fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
            fig.tight_layout(); fig.savefig(FIGURES / f"13_lead_time_{city}.png"); plt.close(fig)


def rescore_city(city: str) -> dict:
    """Score saved forecasts without retraining (python -m src.train_evaluate --score-only)."""
    df = fe.load(city)
    channel = fe.endemic_channel(df)
    fc = pd.read_csv(RESULTS / f"forecasts_{city}.csv.gz", parse_dates=["target_date", "origin_date"])
    fc = add_ensemble(fc)
    fc.to_csv(RESULTS / f"forecasts_{city}.csv.gz", index=False, float_format="%.4f")
    met, eps, episodes = score(fc, df, channel)
    return {"metrics": met, "episodes": eps, "episode_windows": episodes, "fc": fc, "df": df,
            "channel": channel}


def summarise(results: dict) -> dict:
    out = {}
    for c, r in results.items():
        met = r["metrics"]
        cols = ["MAE", "RMSE", "MAE_log", "PI80_coverage", "alert_precision", "alert_recall", "alert_F1",
                "alert_precision_p10", "alert_recall_p10", "alert_F1_p10", "skill_vs_seasonal_naive"]
        avg = met.groupby("model")[cols].mean().reindex(MODEL_ORDER)
        eps = r["episodes"]
        lead = eps.groupby("model").agg(mean_lead_weeks=("lead_weeks", "mean"),
                                        episodes_warned=("lead_weeks", lambda s: int((s > 0).sum())),
                                        mean_lead_weeks_p10=("lead_weeks_p10", "mean"),
                                        episodes_warned_p10=("lead_weeks_p10", lambda s: int((s > 0).sum())),
                                        ).reindex(MODEL_ORDER)
        ranked = avg.sort_values("MAE")
        out[c] = {
            "city": CITIES[c]["name"],
            "best_model_by_mean_MAE": ranked.index[0],
            "alert_rules": ALERT_RULES,
            "average_over_h1_8": avg.round(3).reset_index().to_dict("records"),
            "by_horizon": met.round(4).to_dict("records"),
            "episodes": eps.to_dict("records"),
            "episode_count": len(r["episode_windows"]),
            "lead_time": lead.round(2).reset_index().to_dict("records"),
            "best_model_per_horizon": met.loc[met.groupby("h").MAE.idxmin(), ["h", "model", "MAE"]]
                                       .round(2).to_dict("records"),
        }
    return out


def main(score_only: bool = False):
    t = time.time()
    if score_only:
        results = {c: rescore_city(c) for c in CITIES}
    else:
        results = {c: run_city(c) for c in CITIES}
        with open(RESULTS / "model_selection.json", "w") as f:
            json.dump({c: r["selection"] for c, r in results.items()}, f, indent=2, default=str)
    out = summarise(results)
    with open(RESULTS / "metrics.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    figures(results)
    log(f"done in {time.time() - t:.0f}s")
    pd.set_option("display.width", 200)
    for c in out:
        print("\n", out[c]["city"], "| best by mean MAE:", out[c]["best_model_by_mean_MAE"])
        print(pd.DataFrame(out[c]["average_over_h1_8"]).to_string(index=False))
        print(pd.DataFrame(out[c]["lead_time"]).to_string(index=False))


if __name__ == "__main__":
    import sys
    main(score_only="--score-only" in sys.argv)
