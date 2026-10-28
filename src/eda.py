"""
Step C - Exploratory data analysis.

The functions here return plain numbers / DataFrames so they can be reused by the
API (interactive charts) and by `main()`, which writes the static figures and
summary statistics used in the report.

Run:  python -m src.eda
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from src.config import CITIES, CLIMATE_COLS, FIGURES, PROCESSED, RESULTS, SEASON_LENGTH

warnings.filterwarnings("ignore")


def load(city: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED / f"{city}_weekly.csv", parse_dates=["week_start_date"])


# ----------------------------------------------------------------------------- statistics
def stationarity(series: pd.Series) -> dict:
    """ADF (H0: unit root) and KPSS (H0: stationary). Agreement makes the verdict credible."""
    s = series.dropna()
    adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
    kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
    return {
        "adf_stat": round(float(adf_stat), 3), "adf_p": round(float(adf_p), 4),
        "kpss_stat": round(float(kpss_stat), 3), "kpss_p": round(float(kpss_p), 4),
        "verdict": ("stationary" if adf_p < 0.05 and kpss_p > 0.05 else
                    "non-stationary" if adf_p >= 0.05 and kpss_p <= 0.05 else
                    "inconclusive (trend/level shifts or near-unit-root)"),
    }


def stl_decompose(df: pd.DataFrame) -> pd.DataFrame:
    y = np.log1p(df.total_cases.values)
    res = STL(y, period=SEASON_LENGTH, robust=True).fit()
    return pd.DataFrame({"week_start_date": df.week_start_date, "observed": y,
                         "trend": res.trend, "seasonal": res.seasonal, "resid": res.resid})


def seasonal_strength(dec: pd.DataFrame) -> float:
    """Hyndman's strength of seasonality: 1 - Var(R) / Var(S + R). 0 = none, 1 = strong."""
    return float(max(0, 1 - dec.resid.var() / (dec.seasonal + dec.resid).var()))


def trend_strength(dec: pd.DataFrame) -> float:
    return float(max(0, 1 - dec.resid.var() / (dec.trend + dec.resid).var()))


def stl_anomalies(dec: pd.DataFrame, k: float = 3.0) -> pd.DataFrame:
    """Historical anomaly detection: STL residual beyond k robust standard deviations."""
    r = dec.resid
    mad = 1.4826 * (r - r.median()).abs().median()
    z = (r - r.median()) / mad
    out = dec.assign(robust_z=z)
    return out[np.abs(z) > k]


def climate_anomalies(df: pd.DataFrame, col: str) -> pd.Series:
    """Remove each variable's week-of-year climatology so that shared seasonality does
    not create spurious correlation (both rain and dengue peak in the same months)."""
    return df[col] - df.groupby("cal_week")[col].transform("mean")


def cross_correlation(df: pd.DataFrame, max_lag: int = 20, use_anomalies: bool = True) -> pd.DataFrame:
    """corr( climate(t - k), log cases(t) ) for k = 0..max_lag, computed on TRAINING data only."""
    tr = df[df.split == "train"].reset_index(drop=True)
    y = np.log1p(tr.total_cases)
    if use_anomalies:
        y = y - y.groupby(tr.cal_week).transform("mean")
    rows = []
    for col in CLIMATE_COLS:
        x = climate_anomalies(tr, col) if use_anomalies else tr[col]
        for k in range(0, max_lag + 1):
            rows.append({"variable": col, "lag": k, "corr": float(y.corr(x.shift(k)))})
    return pd.DataFrame(rows)


def best_lags(ccf: pd.DataFrame, min_lag: int = 1, top: int = 5) -> list[dict]:
    """Strongest |corr| lag per variable (lag >= min_lag), ranked."""
    c = ccf[ccf.lag >= min_lag].copy()
    c["abs"] = c["corr"].abs()
    best = c.loc[c.groupby("variable")["abs"].idxmax()].sort_values("abs", ascending=False)
    return best.head(top)[["variable", "lag", "corr"]].round(3).to_dict("records")


def seasonal_profile(df: pd.DataFrame) -> pd.DataFrame:
    tr = df[df.split == "train"]
    g = tr.groupby("season_week").total_cases
    return pd.DataFrame({"season_week": g.mean().index, "mean": g.mean().values,
                         "median": g.median().values, "q90": g.quantile(0.9).values})


def serotype_share(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["denv1_cases", "denv2_cases", "denv3_cases", "denv4_cases"] if c in df]
    s = df.groupby("season")[cols].sum()
    return s.div(s.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).round(3)


def summary(city: str) -> dict:
    df = load(city)
    y = np.log1p(df.total_cases)
    dec = stl_decompose(df)
    ccf = cross_correlation(df)
    tr = df[df.split == "train"]
    yearly = df.groupby("season").total_cases.agg(["sum", "max"])
    return {
        "city": CITIES[city]["name"],
        "weeks": len(df), "train_weeks": int((df.split == "train").sum()),
        "test_weeks": int((df.split == "test").sum()),
        "cases_mean": round(float(df.total_cases.mean()), 2),
        "cases_median": float(df.total_cases.median()),
        "cases_max": int(df.total_cases.max()),
        "cases_max_week": str(df.loc[df.total_cases.idxmax(), "week_start_date"].date()),
        "zero_weeks": int((df.total_cases == 0).sum()),
        "variance_to_mean": round(float(df.total_cases.var() / df.total_cases.mean()), 1),
        "skewness": round(float(df.total_cases.skew()), 2),
        "stationarity_log_cases": stationarity(y),
        "stationarity_log_cases_diff1": stationarity(y.diff()),
        "seasonal_strength": round(seasonal_strength(dec), 3),
        "trend_strength": round(trend_strength(dec), 3),
        "peak_season_week": int(seasonal_profile(df).sort_values("mean").iloc[-1].season_week),
        "biggest_seasons": yearly.sort_values("sum", ascending=False).head(3)["sum"].to_dict(),
        "stl_anomaly_weeks": int(len(stl_anomalies(dec))),
        "best_climate_lags_anomaly": best_lags(ccf),
        "best_climate_lags_raw": best_lags(cross_correlation(df, use_anomalies=False)),
        "acf_lag1": round(float(acf(y, nlags=1)[1]), 3),
        "acf_lag52": round(float(acf(y, nlags=52)[52]), 3),
        "train_mean_cases": round(float(tr.total_cases.mean()), 2),
    }


# ----------------------------------------------------------------------------- static figures
def make_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})
    INK, ACCENT, MUTED = "#1f3b4d", "#c8553d", "#8a9ba8"

    # 1. time series with train/test split
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2))
    for ax, city in zip(axes, CITIES):
        df = load(city)
        ax.plot(df.week_start_date, df.total_cases, color=INK, lw=0.8)
        ts = df[df.split == "test"].week_start_date.min()
        ax.axvspan(ts, df.week_start_date.max(), color=ACCENT, alpha=0.08)
        ax.text(ts, ax.get_ylim()[1] * 0.92, "  test period", color=ACCENT, fontsize=8)
        ax.plot(df.week_start_date, df.total_cases.rolling(52, center=True).mean(), color=ACCENT,
                lw=1.4, label="52-week rolling mean")
        ax.set_title(f"{CITIES[city]['name']} - weekly lab-confirmed dengue cases", loc="left")
        ax.set_ylabel("cases / week")
        ax.legend(loc="upper left", frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "01_timeseries.png"); plt.close(fig)

    for city in CITIES:
        df = load(city)
        # 2. STL
        dec = stl_decompose(df)
        fig, axes = plt.subplots(4, 1, figsize=(10, 6.5), sharex=True)
        for ax, col in zip(axes, ["observed", "trend", "seasonal", "resid"]):
            ax.plot(dec.week_start_date, dec[col], color=INK, lw=0.8)
            ax.set_ylabel(col)
        an = stl_anomalies(dec)
        axes[3].scatter(an.week_start_date, an.resid, color=ACCENT, s=12, zorder=3,
                        label=f"|robust z| > 3 ({len(an)} weeks)")
        axes[3].legend(frameon=False, loc="upper left")
        axes[0].set_title(f"{CITIES[city]['name']} - STL decomposition of log(1+cases), period 52",
                          loc="left")
        fig.tight_layout(); fig.savefig(FIGURES / f"02_stl_{city}.png"); plt.close(fig)

        # 3. seasonal box plot by season week
        fig, ax = plt.subplots(figsize=(10, 3.4))
        tr = df[df.split == "train"]
        sns.boxplot(data=tr, x="season_week", y="total_cases", ax=ax, color="#d9e2e8",
                    fliersize=1.5, linewidth=0.6)
        ax.set_yscale("symlog", linthresh=10)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("cases / week (symlog)"); ax.set_xlabel("week of transmission season")
        ax.set_xticks(range(0, 52, 4)); ax.set_xticklabels(range(1, 53, 4))
        ax.set_title(f"{CITIES[city]['name']} - cases by week of transmission season (training data)",
                     loc="left")
        fig.tight_layout(); fig.savefig(FIGURES / f"03_season_box_{city}.png"); plt.close(fig)

        # 4. heatmap season x week
        piv = df.pivot_table(index="season", columns="season_week", values="total_cases")
        fig, ax = plt.subplots(figsize=(10, 0.28 * len(piv) + 1.2))
        sns.heatmap(np.log1p(piv), cmap="rocket_r", ax=ax, cbar_kws={"label": "log(1 + cases)"})
        ax.set_title(f"{CITIES[city]['name']} - season x week heatmap", loc="left")
        fig.tight_layout(); fig.savefig(FIGURES / f"04_heatmap_{city}.png"); plt.close(fig)

        # 5. ACF / PACF
        y = np.log1p(df[df.split == "train"].total_cases)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        plot_acf(y, lags=110, ax=axes[0], color=INK, vlines_kwargs={"colors": INK})
        plot_pacf(y, lags=60, ax=axes[1], color=INK, vlines_kwargs={"colors": INK}, method="ywm")
        axes[0].set_title(""); axes[1].set_title("")
        axes[0].set_title("ACF of log(1+cases), training data", loc="left")
        axes[1].set_title("PACF", loc="left")
        axes[0].set_xlabel("lag (weeks)"); axes[1].set_xlabel("lag (weeks)")
        fig.suptitle(CITIES[city]["name"], x=0.01, ha="left", fontsize=9)
        fig.tight_layout(); fig.savefig(FIGURES / f"05_acf_pacf_{city}.png"); plt.close(fig)

        # 6. cross-correlation (anomalies) for key climate variables
        ccf = cross_correlation(df)
        fig, ax = plt.subplots(figsize=(10, 3.4))
        for var, color in zip(["temp_avg_c", "temp_min_c", "precip_mm", "spec_humidity_gkg", "ndvi"],
                              [ACCENT, "#e09f3e", "#3d7ea6", "#2a9d8f", MUTED]):
            c = ccf[ccf.variable == var]
            ax.plot(c.lag, c["corr"], marker="o", ms=3, lw=1.2, color=color, label=var)
        n = int((df.split == "train").sum())
        ax.axhspan(-1.96 / np.sqrt(n), 1.96 / np.sqrt(n), color="grey", alpha=0.12,
                   label="approx. 95% null band")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("lag k (weeks): climate at t-k vs cases at t")
        ax.set_ylabel("correlation of anomalies")
        ax.set_title(f"{CITIES[city]['name']} - lagged cross-correlation (seasonality removed, training data)",
                     loc="left")
        ax.legend(ncol=1, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
        fig.tight_layout(); fig.savefig(FIGURES / f"06_ccf_{city}.png"); plt.close(fig)

        # 7. serotype composition by season
        sh = serotype_share(df)
        fig, ax = plt.subplots(figsize=(10, 3.4))
        sh.plot(kind="bar", stacked=True, ax=ax, width=0.85,
                color=["#1f3b4d", "#c8553d", "#e09f3e", "#2a9d8f"])
        ax.set_ylabel("share of typed cases"); ax.set_xlabel("")
        ax.set_title(f"{CITIES[city]['name']} - dengue serotype mix per season", loc="left")
        ax.legend(["DENV-1", "DENV-2", "DENV-3", "DENV-4"], ncol=1, frameon=False, fontsize=8,
                  loc="upper left", bbox_to_anchor=(1.01, 1))
        fig.tight_layout(); fig.savefig(FIGURES / f"07_serotype_{city}.png"); plt.close(fig)

        # 8. correlation heatmap
        cols = ["total_cases"] + CLIMATE_COLS
        fig, ax = plt.subplots(figsize=(7, 5.5))
        sns.heatmap(df[cols].corr(method="spearman"), cmap="RdBu_r", vmin=-1, vmax=1, annot=True,
                    fmt=".2f", annot_kws={"size": 6}, ax=ax)
        ax.set_title(f"{CITIES[city]['name']} - Spearman correlations", loc="left")
        fig.tight_layout(); fig.savefig(FIGURES / f"08_corr_{city}.png"); plt.close(fig)

    # 9. distribution of cases (skew)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for ax, city in zip(axes, CITIES):
        df = load(city)
        ax.hist(df.total_cases, bins=60, color=INK)
        ax.set_title(f"{CITIES[city]['name']} - distribution of weekly cases", loc="left")
        ax.set_xlabel("cases / week")
    fig.tight_layout(); fig.savefig(FIGURES / "09_distribution.png"); plt.close(fig)

    # 10. Iquitos station missing-data pattern (raw daily)
    from src.data_build import load_station
    st = load_station("iq")
    miss = st[["TMAX", "TMIN", "TAVG", "PRCP"]].isna().groupby(st.index.year).mean() * 100
    miss = miss.loc[1999:2013]
    fig, ax = plt.subplots(figsize=(10, 2.8))
    sns.heatmap(miss.T, cmap="Greys", ax=ax, annot=True, fmt=".0f", annot_kws={"size": 6},
                cbar_kws={"label": "% days missing"})
    ax.set_title("Iquitos GHCN station - % of days missing per year (raw)", loc="left")
    fig.tight_layout(); fig.savefig(FIGURES / "10_missing_iq.png"); plt.close(fig)


def main():
    out = {city: summary(city) for city in CITIES}
    with open(RESULTS / "eda_summary.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))
    make_figures()
    print("figures ->", FIGURES)


if __name__ == "__main__":
    main()
