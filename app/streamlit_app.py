"""
DengueWatch - Streamlit frontend.

Start the backend first (uvicorn backend.api:app --port 8000), then:
    streamlit run app/streamlit_app.py
The backend URL can be changed with the DENGUEWATCH_API environment variable.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API = os.environ.get("DENGUEWATCH_API", "http://localhost:8000")
INK, ACCENT, MUTED, TEAL, AMBER = "#1f3b4d", "#c8553d", "#8a9ba8", "#2a9d8f", "#e09f3e"
MODEL_COLORS = {"Seasonal naive": "#8a9ba8", "Persistence": "#b8c4cc", "Climatology": "#c9d3da",
                "Holt-Winters": "#6c757d", "SARIMA": "#3d7ea6", "SARIMAX": "#1f3b4d",
                "Prophet": "#e09f3e", "XGBoost": "#2a9d8f", "Hybrid SARIMA+XGB": "#c8553d",
                "Ensemble": "#7b2cbf"}

st.set_page_config(page_title="DengueWatch", page_icon="🦟", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; max-width: 1250px;}
  .dw-banner {border-radius: 10px; padding: 16px 20px; color: #fff; margin: 4px 0 14px;}
  .dw-banner h3 {margin: 0 0 4px; color: #fff; font-size: 1.25rem;}
  .dw-banner p {margin: 0; font-size: 0.98rem; opacity: 0.95;}
  .dw-note {background: #f3f5f7; border-left: 4px solid #8a9ba8; padding: 10px 14px;
            border-radius: 4px; font-size: 0.9rem; margin: 6px 0 12px;}
  div[data-testid="stMetricValue"] {font-size: 1.6rem;}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- API helpers
@st.cache_data(show_spinner=False, ttl=3600)
def get(path: str, **params):
    r = requests.get(f"{API}{path}", params={k: v for k, v in params.items() if v is not None}, timeout=120)
    r.raise_for_status()
    return r.json()


def post(path: str, **kw):
    r = requests.post(f"{API}{path}", timeout=300, **kw)
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(detail)
    return r.json()


def api_ok() -> bool:
    try:
        return requests.get(f"{API}/health", timeout=3).ok
    except Exception:
        return False


def layout(fig, h=380, title=None, ytitle=None):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40 if title else 10, b=10),
                      title=dict(text=title, x=0, font=dict(size=15)) if title else None,
                      plot_bgcolor="white", legend=dict(orientation="h", y=-0.3, yanchor="top"),
                      hovermode="x unified", yaxis_title=ytitle)
    fig.update_xaxes(showgrid=True, gridcolor="#eef1f3")
    fig.update_yaxes(showgrid=True, gridcolor="#eef1f3", rangemode="tozero")
    return fig


def threshold_traces(fig, x, d, show_typical=True):
    if show_typical:
        fig.add_trace(go.Scatter(x=x, y=d["typical"], name="typical (mean)", mode="lines",
                                 line=dict(color=TEAL, dash="dot", width=1)))
    fig.add_trace(go.Scatter(x=x, y=d["warning"], name="warning (mean+1 SD)", mode="lines",
                             line=dict(color=AMBER, dash="dash", width=1)))
    fig.add_trace(go.Scatter(x=x, y=d["epidemic"], name="epidemic (mean+2 SD)", mode="lines",
                             line=dict(color=ACCENT, dash="dash", width=1.4)))


def banner(a: dict):
    st.markdown(f"""<div class="dw-banner" style="background:{a['color']}">
        <h3>{a['label']}</h3><p>{a['headline']}</p></div>""", unsafe_allow_html=True)


def note(text: str):
    st.markdown(f'<div class="dw-note">{text}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("🦟 DengueWatch")
st.sidebar.caption("Climate-driven dengue outbreak forecasting & early warning")
PAGES = ["Overview", "Data & quality", "Exploratory analysis", "Forecast & alerts",
         "Model comparison", "Climate what-if", "Upload your data"]
page = st.sidebar.radio("Page", PAGES, label_visibility="collapsed")

if not api_ok():
    st.error(f"Cannot reach the backend at {API}. Start it with: `uvicorn backend.api:app --port 8000`")
    st.stop()

CITIES = get("/cities")
city_names = {c["id"]: c["name"] for c in CITIES}
if page != "Upload your data":
    city = st.sidebar.selectbox("City", list(city_names), format_func=city_names.get)
    cinfo = next(c for c in CITIES if c["id"] == city)
st.sidebar.markdown("---")
st.sidebar.caption("Data: CDC / NOAA Dengue Forecasting Project 2015 (CC BY 4.0). "
                   "Decision aid only - not medical advice.")


# ============================================================================= pages
def page_overview():
    st.title("DengueWatch")
    st.markdown("#### Forecast dengue cases up to 8 weeks ahead and warn health teams *before* an outbreak starts.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**The problem**  \nDengue control works best *before* case numbers explode: removing "
                "mosquito breeding sites and preparing hospitals takes weeks. Health departments usually "
                "react to case counts that are already high.")
    c2.markdown("**Who it is for**  \nMunicipal public-health officers, vector-control teams and hospital "
                "planners who must decide *when* to start intensified mosquito control and surge preparations.")
    c3.markdown("**What it does**  \nCombines past cases with lagged temperature, humidity, rainfall and "
                "vegetation data, compares 10 forecasting approaches, and turns forecasts into graded alerts "
                "with recommended actions.")
    st.markdown("---")
    st.subheader("Results on the four held-out test seasons (2009/10 - 2012/13)")
    cols = st.columns(len(CITIES))
    for col, c in zip(cols, CITIES):
        m = get(f"/metrics/{c['id']}")
        avg = pd.DataFrame(m["average_over_h1_8"]).set_index("model")
        lead = pd.DataFrame(m["lead_time"]).set_index("model")
        best = m["best_model_by_mean_MAE"]
        with col:
            st.markdown(f"**{c['name']}**")
            a, b, cc = st.columns(3)
            a.metric("Best model (avg MAE)", best.replace(" SARIMA+XGB", ""))
            b.metric("Avg MAE, 1-8 wk", f"{avg.loc[best, 'MAE']:.1f}",
                     f"{avg.loc[best, 'skill_vs_seasonal_naive'] * 100:+.0f}% vs seasonal naive",
                     delta_color="normal")
            if m["episode_count"]:
                cc.metric("Test outbreaks warned", f"{int(lead.loc[best, 'episodes_warned_p10'])}/{m['episode_count']}",
                          f"avg lead {lead.loc[best, 'mean_lead_weeks_p10']:.1f} wk", delta_color="off",
                          help="Early-warning rule: alert when the upper 80% bound crosses the epidemic threshold")
    st.markdown("---")
    st.subheader("How to use the app")
    st.markdown(
        "1. **Data & quality** - see the merged weekly dataset and every cleaning step.\n"
        "2. **Exploratory analysis** - trend, seasonality, stationarity tests and climate lags.\n"
        "3. **Forecast & alerts** - pick a city and a week, see the 8-week forecast, the alert level and "
        "recommended actions. Use *replay* to check whether the system would have warned before a real outbreak.\n"
        "4. **Model comparison** - accuracy, interval coverage and outbreak-warning skill of all models.\n"
        "5. **Climate what-if** - change recent temperature / humidity / rainfall and see the effect.\n"
        "6. **Upload your data** - forecast your own weekly case series.")


def page_data():
    st.title("Data & quality")
    st.caption(f"{cinfo['name']}: {cinfo['start']} to {cinfo['end']} - {cinfo['weeks']} weeks. "
               f"Training seasons end before {cinfo['test_start']}.")
    df = pd.DataFrame(get(f"/data/{city}"))
    df["week_start_date"] = pd.to_datetime(df.week_start_date)
    rng = st.slider("Date range", min_value=df.week_start_date.min().date(),
                    max_value=df.week_start_date.max().date(),
                    value=(df.week_start_date.min().date(), df.week_start_date.max().date()))
    d = df[(df.week_start_date.dt.date >= rng[0]) & (df.week_start_date.dt.date <= rng[1])]
    metric = st.radio("Show", ["total_cases", "incidence_100k"], horizontal=True,
                      format_func={"total_cases": "Weekly cases", "incidence_100k": "Incidence per 100,000"}.get)
    fig = go.Figure(go.Scatter(x=d.week_start_date, y=d[metric], line=dict(color=INK, width=1.2), name=metric))
    st.plotly_chart(layout(fig, 300), use_container_width=True)
    clim_cols = ["temp_avg_c", "temp_min_c", "temp_max_c", "temp_dtr_c", "precip_mm", "precip_sat_mm",
                 "rel_humidity_pct", "spec_humidity_gkg", "dew_point_c", "ndvi"]
    v = st.selectbox("Climate variable", clim_cols)
    fig = go.Figure(go.Scatter(x=d.week_start_date, y=d[v], line=dict(color=TEAL, width=1), name=v))
    fig.update_yaxes(rangemode="normal")
    st.plotly_chart(layout(fig, 240), use_container_width=True)
    with st.expander("Preview / download the processed weekly table"):
        st.dataframe(d, use_container_width=True, height=260)
        st.download_button("Download CSV", d.to_csv(index=False), f"{city}_weekly_processed.csv", "text/csv")

    st.subheader("Data-quality report")
    q = get("/quality")[city]
    a, b, c, e = st.columns(4)
    a.metric("Weekly rows", q["rows"])
    b.metric("Duplicate dates", q["duplicate_dates"])
    c.metric("Missing case values", q["missing_case_values"])
    e.metric("Week lengths (days)", ", ".join(f"{k}:{v}" for k, v in q["week_length_days_counts"].items()))
    st.markdown(
        f"- **Irregular weeks**: weeks restart every 1 January, so the last week of each year is 8-9 days "
        f"long. Each year therefore has exactly 52 rows, which makes a 52-week seasonal period exact. "
        f"Weekly rainfall totals are rescaled to a 7-day equivalent so long weeks are not inflated.\n"
        f"- **Raw station gaps (% of days missing)**: "
        + ", ".join(f"{k} {v}%" for k, v in q["station_daily_missing_pct"].items()) +
        f"\n- **Outliers in cases are kept on purpose**: outbreak peaks *are* the signal. "
        f"Climate values are range-checked; robust z-scores flag extreme weeks for inspection only.")
    rows = []
    for var, info in q["station_weekly_gap_fill"].items():
        rows.append({"variable": var, "weekly gaps": info.get("missing_before"),
                     "filled": info.get("filled", 0), "method": info.get("method"),
                     "proxy fit r": info.get("r")})
    st.markdown("**Station gaps filled from a bias-corrected proxy** (reanalysis temperature / satellite rainfall, "
                "linear fit on weeks where both exist):")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown(f"**NDVI**: kept {q['ndvi']['pixels_kept']} of {q['ndvi']['pixels_total']} satellite pixels "
                f"(pixels with mean NDVI < 0.1 are mostly water/urban and were dropped).")


def page_eda():
    st.title("Exploratory analysis")
    e = get(f"/eda/{city}")
    s = e["summary"]
    a, b, c, d_ = st.columns(4)
    a.metric("Mean cases / week", s["cases_mean"])
    b.metric("Peak week", s["cases_max"], s["cases_max_week"], delta_color="off")
    c.metric("Seasonal strength", s["seasonal_strength"])
    d_.metric("Variance / mean", s["variance_to_mean"])
    tabs = st.tabs(["Trend", "Decomposition", "Seasonality", "Autocorrelation & stationarity",
                    "Climate lags", "Serotypes", "Anomalies"])
    stl = pd.DataFrame(e["stl"]); stl["week_start_date"] = pd.to_datetime(stl.week_start_date)
    roll = pd.DataFrame(e["rolling"]); roll["week_start_date"] = pd.to_datetime(roll.week_start_date)
    df = pd.DataFrame(get(f"/data/{city}")); df["week_start_date"] = pd.to_datetime(df.week_start_date)

    with tabs[0]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.week_start_date, y=df.total_cases, name="weekly cases",
                                 line=dict(color=INK, width=1)))
        fig.add_trace(go.Scatter(x=roll.week_start_date, y=roll.mean52, name="52-week rolling mean",
                                 line=dict(color=ACCENT, width=2)))
        fig.add_trace(go.Scatter(x=roll.week_start_date, y=roll.std52, name="52-week rolling SD",
                                 line=dict(color=AMBER, width=1.5, dash="dot")))
        st.plotly_chart(layout(fig, 360), use_container_width=True)
        bs = ", ".join(f"{k} ({v:,} cases)" for k, v in s["biggest_seasons"].items())
        note(f"No steady long-term trend: the level rises and falls with multi-year epidemic waves. "
             f"The rolling SD moves with the mean - larger outbreaks are also more variable - which is "
             f"why models are fitted on log(1 + cases). Largest seasons: {bs}.")
    with tabs[1]:
        fig = go.Figure()
        for i, (col, color) in enumerate([("observed", INK), ("trend", ACCENT), ("seasonal", TEAL), ("resid", MUTED)]):
            fig.add_trace(go.Scatter(x=stl.week_start_date, y=stl[col], name=col, line=dict(color=color, width=1),
                                     yaxis=f"y{i + 1 if i else ''}"))
        fig.update_layout(grid=dict(rows=4, columns=1, pattern="independent"),
                          yaxis=dict(domain=[0.77, 1]), yaxis2=dict(domain=[0.52, 0.74]),
                          yaxis3=dict(domain=[0.27, 0.49]), yaxis4=dict(domain=[0, 0.24]))
        layout(fig, 560); fig.update_yaxes(rangemode="normal")
        st.plotly_chart(fig, use_container_width=True)
        note(f"STL (period 52, robust) on log(1 + cases). Strength of seasonality = {s['seasonal_strength']}, "
             f"of trend = {s['trend_strength']} (0 = none, 1 = dominant). The 'trend' here is the slow epidemic "
             f"cycle, and large positive residuals mark outbreak years.")
    with tabs[2]:
        sp = pd.DataFrame(e["seasonal_profile"])
        fig = go.Figure()
        fig.add_trace(go.Bar(x=sp.season_week, y=sp["mean"], name="mean", marker_color=INK))
        fig.add_trace(go.Scatter(x=sp.season_week, y=sp.q90, name="90th percentile", line=dict(color=ACCENT)))
        fig.update_xaxes(title="week of transmission season")
        st.plotly_chart(layout(fig, 340, ytitle="cases / week"), use_container_width=True)
        piv = df.pivot_table(index="season", columns="season_week", values="total_cases")
        fig = go.Figure(go.Heatmap(z=np.log1p(piv.values), x=piv.columns, y=piv.index, colorscale="OrRd",
                                   colorbar=dict(title="log(1+cases)")))
        st.plotly_chart(layout(fig, 420), use_container_width=True)
        note(f"Cases peak around season-week {s['peak_season_week']} every year, but the size of the peak "
             f"varies enormously between seasons - so a purely seasonal model cannot tell a normal year from an epidemic one.")
    with tabs[3]:
        n = e["n_train"]; band = 1.96 / np.sqrt(n)
        c1, c2 = st.columns(2)
        for col, key, title in ((c1, "acf", "ACF of log(1+cases)"), (c2, "pacf", "PACF")):
            vals = e[key]
            fig = go.Figure(go.Bar(x=list(range(len(vals))), y=vals, marker_color=INK))
            fig.add_hrect(y0=-band, y1=band, fillcolor="grey", opacity=0.15, line_width=0)
            fig.update_yaxes(rangemode="normal")
            col.plotly_chart(layout(fig, 280, title), use_container_width=True)
        st_rows = []
        for label, k in (("log(1+cases)", "stationarity_log_cases"), ("first difference", "stationarity_log_cases_diff1")):
            r = s[k]
            st_rows.append({"series": label, "ADF stat": r["adf_stat"], "ADF p": r["adf_p"],
                            "KPSS stat": r["kpss_stat"], "KPSS p": r["kpss_p"], "verdict": r["verdict"]})
        st.dataframe(pd.DataFrame(st_rows), hide_index=True, use_container_width=True)
        note(f"Lag-1 autocorrelation = {s['acf_lag1']} (strong persistence: next week looks like this week); "
             f"lag-52 = {s['acf_lag52']} (yearly echo). ADF rejects a unit root and KPSS does not reject "
             f"stationarity, so no differencing is needed (d = 0, D = 0). KPSS p-values are capped at 0.1 by "
             f"the test's lookup table.")
    with tabs[4]:
        ccf = pd.DataFrame(e["ccf"])
        vars_ = st.multiselect("Variables", sorted(ccf.variable.unique()),
                               default=["temp_avg_c", "temp_min_c", "precip_mm", "spec_humidity_gkg", "rel_humidity_pct"])
        fig = go.Figure()
        for v in vars_:
            g = ccf[ccf.variable == v]
            fig.add_trace(go.Scatter(x=g.lag, y=g["corr"], name=v, mode="lines+markers"))
        band = 1.96 / np.sqrt(n)
        fig.add_hrect(y0=-band, y1=band, fillcolor="grey", opacity=0.15, line_width=0)
        fig.update_xaxes(title="lag (weeks): climate at t-k vs cases at t"); fig.update_yaxes(rangemode="normal")
        st.plotly_chart(layout(fig, 380, ytitle="correlation of anomalies"), use_container_width=True)
        best = ", ".join(f"{b['variable']} at lag {b['lag']} (r = {b['corr']})" for b in s["best_climate_lags_anomaly"][:3])
        note(f"Both series have their week-of-year average removed first, otherwise shared seasonality creates "
             f"misleading correlations. Strongest lagged links: {best}. Climate leads cases by weeks - "
             f"this lead is what makes early warning possible.")
    with tabs[5]:
        sero = pd.DataFrame(e["serotypes"])
        fig = go.Figure()
        for col, name, color in zip(["denv1_cases", "denv2_cases", "denv3_cases", "denv4_cases"],
                                    ["DENV-1", "DENV-2", "DENV-3", "DENV-4"], [INK, ACCENT, AMBER, TEAL]):
            fig.add_trace(go.Bar(x=sero.season, y=sero[col], name=name, marker_color=color))
        fig.update_layout(barmode="stack")
        st.plotly_chart(layout(fig, 340, ytitle="share of typed cases"), use_container_width=True)
        note("When a serotype that has not circulated for years returns, many people lack immunity to it and "
             "large epidemics can follow. Serotype data are not used by the forecasting models (they are "
             "reported with a delay), which is one documented limitation.")
    with tabs[6]:
        an = pd.DataFrame(e["anomalies"]); an["week_start_date"] = pd.to_datetime(an.week_start_date)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stl.week_start_date, y=stl.resid, name="STL residual", line=dict(color=MUTED, width=1)))
        fig.add_trace(go.Scatter(x=an.week_start_date, y=an.resid, mode="markers", name="|robust z| > 3",
                                 marker=dict(color=ACCENT, size=6)))
        fig.update_yaxes(rangemode="normal")
        st.plotly_chart(layout(fig, 320), use_container_width=True)
        note(f"{len(an)} weeks have residuals more than 3 robust standard deviations from zero. Positive "
             f"clusters line up with known epidemic seasons; isolated negative spikes are unusually quiet weeks.")


def origin_picker(key: str, default_latest=True):
    df = pd.DataFrame(get(f"/data/{city}"))
    dates = pd.to_datetime(df.week_start_date).dt.date.tolist()[104:]
    met = get(f"/metrics/{city}")
    onsets = sorted({e["onset"] for e in met["episodes"]})
    if f"{key}_origin" not in st.session_state:
        st.session_state[f"{key}_origin"] = dates[-1]
    cols = st.columns(1 + len(onsets))
    if cols[0].button("Latest week", key=f"{key}_latest"):
        st.session_state[f"{key}_origin"] = dates[-1]
    for c, on in zip(cols[1:], onsets):
        target = pd.Timestamp(on) - pd.Timedelta(weeks=6)
        if c.button(f"Replay: 6 wk before {on}", key=f"{key}_{on}"):
            st.session_state[f"{key}_origin"] = min(dates, key=lambda d: abs(pd.Timestamp(d) - target))
    return st.select_slider("Forecast made in the week starting", options=dates, key=f"{key}_origin")


def page_forecast():
    st.title("Forecast & alerts")
    met = get(f"/metrics/{city}")
    models = get("/models")["models"]
    origin = origin_picker("fc")
    c1, c2, c3, c4 = st.columns([1.6, 1.2, 1.1, 0.9])
    model = c1.selectbox("Model", models, index=models.index(met["best_model_by_mean_MAE"]),
                         help="Default = lowest average test MAE for this city")
    sens = c2.radio("Alert sensitivity", ["early", "standard"], horizontal=True,
                    format_func={"early": "Early warning", "standard": "Standard"}.get,
                    help="Early warning: also raise 'Warning' when the upper 80% bound crosses the epidemic "
                         "threshold (more lead time, more false alarms). Standard: only the central forecast counts.")
    hosp = c3.number_input("Needing hospital care (%)", 0.0, 100.0, 10.0, 1.0,
                           help="Planning assumption - set it to your local figure.") / 100
    reveal = c4.toggle("Show actuals", value=True)
    r = get(f"/forecast/{city}", origin=str(origin), model=model, hosp_rate=hosp, sensitivity=sens)
    a = r["assessment"]
    if r["in_sample"]:
        st.warning("This week is inside the training period, so the model has already seen these data. "
                   "Pick a date from the test period (2009 onwards) for an honest check.")
    if r["beyond_data"]:
        st.info("Part of this forecast is beyond the end of the data (June 2013): a genuine forecast.")
    banner(a)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Forecast peak", f"{a['peak_cases']:.0f} cases/wk", a["peak_week"], delta_color="off")
    k2.metric("Weeks until epidemic level", a["weeks_until_epidemic"] if a["weeks_until_epidemic"] else "-")
    k3.metric("Expected cases, next 8 wk", f"{a['expected_cases_total']:.0f}")
    k4.metric("Est. needing hospital care", f"{a['expected_hospital_care']:.0f}",
              f"at {hosp * 100:.0f}% (assumption)", delta_color="off")

    h = pd.DataFrame(r["history"]); h["week_start_date"] = pd.to_datetime(h.week_start_date)
    f = pd.DataFrame(r["forecast"]); f["target_date"] = pd.to_datetime(f.target_date)
    fig = go.Figure()
    allx = pd.concat([h.week_start_date, f.target_date])
    allth = pd.concat([h[["typical", "warning", "epidemic"]], f[["typical", "warning", "epidemic"]]])
    threshold_traces(fig, allx, allth)
    fig.add_trace(go.Scatter(x=h.week_start_date, y=h.cases, name="observed (known at forecast time)",
                             line=dict(color=INK, width=1.6)))
    if f.lo.notna().any():
        fig.add_trace(go.Scatter(x=pd.concat([f.target_date, f.target_date[::-1]]),
                                 y=pd.concat([f.hi, f.lo[::-1]]), mode="lines", fill="toself", fillcolor="rgba(200,85,61,0.15)",
                                 line=dict(width=0), name="80% prediction interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=f.target_date, y=f.cases, name=f"forecast ({model})", mode="lines+markers",
                             line=dict(color=ACCENT, width=2.4)))
    if reveal and f.actual.notna().any():
        fig.add_trace(go.Scatter(x=f.target_date, y=f.actual, name="what actually happened", mode="markers",
                                 marker=dict(color="black", size=7, symbol="x")))
    fig.add_vline(x=pd.Timestamp(r["origin_date"]), line_dash="dot", line_color=MUTED)
    st.plotly_chart(layout(fig, 430, ytitle="cases / week"), use_container_width=True)

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.subheader("Recommended actions")
        for act in a["actions"]:
            st.markdown(f"- {act}")
        st.caption("Rule-based decision aid; adapt to the local outbreak response plan.")
        chk = r.get("latest_week_check")
        if chk:
            st.subheader("Latest-week check")
            if chk["is_surge"]:
                st.error(f"Unexpected surge: {chk['observed']:.0f} cases were reported, above the model's 80% "
                         f"upper bound of {chk['upper80']:.0f} from one week earlier (expected {chk['expected']:.0f}).")
            else:
                st.success(f"Reported {chk['observed']:.0f} cases - within the expected range "
                           f"(expected {chk['expected']:.0f}, upper bound {chk['upper80']:.0f}).")
    with c2:
        st.subheader("Week-by-week")
        w = pd.DataFrame(a["weekly"])[["target_date", "h", "cases", "lo", "hi", "epidemic", "level", "actual"]]
        w.columns = ["week", "h", "forecast", "low", "high", "epidemic thr.", "level", "actual"]
        st.dataframe(w.round(1), hide_index=True, use_container_width=True, height=320)


def page_compare():
    st.title("Model comparison")
    m = get(f"/metrics/{city}")
    avg = pd.DataFrame(m["average_over_h1_8"])
    num = [c for c in avg.columns if c != "model"]
    avg[num] = avg[num].apply(pd.to_numeric, errors="coerce")   # JSON nulls -> NaN
    byh = pd.DataFrame(m["by_horizon"])
    st.markdown(f"Rolling-origin evaluation on 4 held-out seasons, {byh.n.max()} forecasts per model per horizon. "
                f"**Best average MAE: {m['best_model_by_mean_MAE']}.**")
    show = avg[["model", "MAE", "RMSE", "MAE_log", "PI80_coverage", "skill_vs_seasonal_naive",
                 "alert_F1", "alert_precision_p10", "alert_recall_p10", "alert_F1_p10"]].rename(columns={
        "MAE_log": "MAE (log)", "PI80_coverage": "80% PI coverage", "skill_vs_seasonal_naive": "skill vs seas. naive",
        "alert_F1": "alert F1 (point)", "alert_precision_p10": "precision (early)", "alert_recall_p10": "recall (early)",
        "alert_F1_p10": "alert F1 (early)"})
    st.dataframe(show.style.format(precision=3, na_rep="-").highlight_min(subset=["MAE", "RMSE"], color="#d8efe9"),
                 hide_index=True, use_container_width=True)
    st.caption("Averages over horizons 1-8 weeks. Skill = 1 - MAE/MAE(seasonal naive); positive is better. "
               "80% PI coverage should be close to 0.80. Alert metrics are per week, for epidemic-level weeks: "
               "'point' alerts when the central forecast crosses the threshold, 'early' when the upper 80% bound does.")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for mod in avg.model:
            g = byh[byh.model == mod].sort_values("h")
            fig.add_trace(go.Scatter(x=g.h, y=g.MAE, name=mod, mode="lines+markers",
                                     line=dict(color=MODEL_COLORS.get(mod), width=2.5 if "Hybrid" in mod or mod in ("XGBoost", "SARIMAX") else 1.2)))
        fig.update_xaxes(title="weeks ahead")
        st.plotly_chart(layout(fig, 380, "MAE by horizon", "cases / week"), use_container_width=True)
    with c2:
        fig = go.Figure()
        for mod in avg.model:
            g = byh[byh.model == mod].sort_values("h")
            fig.add_trace(go.Scatter(x=g.h, y=g.alert_F1_p10, name=mod, mode="lines+markers",
                                     line=dict(color=MODEL_COLORS.get(mod))))
        fig.update_xaxes(title="weeks ahead")
        st.plotly_chart(layout(fig, 380, "Epidemic-week alert F1 by horizon (early-warning rule)", "F1"),
                        use_container_width=True)

    st.subheader("Early warning: lead time before each test outbreak")
    eps = pd.DataFrame(m["episodes"])
    if len(eps):
        rule = st.radio("Alert rule", ["lead_weeks_p10", "lead_weeks"], horizontal=True,
                        format_func={"lead_weeks_p10": "Early warning (upper 80% bound)",
                                     "lead_weeks": "Point forecast"}.get)
        piv = eps.pivot_table(index="model", columns="onset", values=rule).reindex(avg.model)
        piv.columns = [f"onset {c}" for c in piv.columns]
        st.dataframe(piv, use_container_width=True)
        st.caption("Lead time = largest horizon (max 8) at which the forecast made that many weeks earlier already "
                   "put the outbreak's first week above the epidemic threshold. 0 = no advance warning.")
    else:
        st.info("No test-period outbreak crossed the epidemic threshold for 2+ weeks in this city.")

    st.subheader("All models from one forecast week")
    df = pd.DataFrame(get(f"/data/{city}"))
    dates = pd.to_datetime(df.week_start_date).dt.date.tolist()
    test_dates = [d for d, s in zip(dates, df.split) if s == "test"]
    origin = st.select_slider("Forecast made in the week starting", options=dates[104:], value=test_dates[40])
    cmp = pd.DataFrame(get(f"/compare/{city}", origin=str(origin))["rows"])
    cmp["target_date"] = pd.to_datetime(cmp.target_date)
    sel = st.multiselect("Models", list(MODEL_COLORS), default=["Seasonal naive", "SARIMA", "SARIMAX",
                                                                 "Prophet", "XGBoost", "Ensemble"])
    fig = go.Figure()
    o = pd.Timestamp(origin)
    hist = df.assign(week_start_date=pd.to_datetime(df.week_start_date))
    hist = hist[(hist.week_start_date > o - pd.Timedelta(weeks=26)) & (hist.week_start_date <= o + pd.Timedelta(weeks=8))]
    fig.add_trace(go.Scatter(x=hist.week_start_date, y=hist.total_cases, name="actual", line=dict(color="black", width=1.4)))
    for mod in sel:
        g = cmp[cmp.model == mod].sort_values("h")
        fig.add_trace(go.Scatter(x=g.target_date, y=g.cases, name=mod, mode="lines+markers",
                                 line=dict(color=MODEL_COLORS.get(mod))))
    fig.add_vline(x=o, line_dash="dot", line_color=MUTED)
    st.plotly_chart(layout(fig, 400, ytitle="cases / week"), use_container_width=True)

    with st.expander("Model selection details (AIC, cross-validation, features)"):
        s = get(f"/selection/{city}")
        st.markdown(f"**SARIMA** chosen: {tuple(s['sarima']['order'])} x {tuple(s['sarima']['seasonal_order'])} "
                    f"(AIC {s['sarima']['aic']:.1f}). **SARIMAX**: {tuple(s['sarimax']['order'])} x "
                    f"{tuple(s['sarimax']['seasonal_order'])} (AIC {s['sarimax']['aic']:.1f}).")
        st.dataframe(pd.DataFrame(s["sarima"]["table"]), hide_index=True, use_container_width=True)
        st.markdown("**Climate inputs selected for SARIMAX / Prophet** (lag >= 8 weeks so they are known when "
                    "an 8-week forecast is made):")
        st.dataframe(pd.DataFrame(s["climate_features_sarimax"]), hide_index=True)
        st.markdown(f"SARIMAX climate coefficients: `{s['sarimax'].get('exog_coefficients')}`")
        st.markdown(f"**Holt-Winters**: trend = {s['holt_winters']['trend']}, AIC by trend option "
                    f"{s['holt_winters']['aic_table']}, smoothing {s['holt_winters']['smoothing']}")
        c1, c2 = st.columns(2)
        c1.markdown(f"**XGBoost walk-forward CV** (h = 4) - chosen {s['xgb']['params']}")
        c1.dataframe(pd.DataFrame(s["xgb"]["cv"]), hide_index=True)
        c2.markdown(f"**Hybrid residual model CV** - chosen {s['hybrid']['params']}")
        c2.dataframe(pd.DataFrame(s["hybrid"]["cv"]), hide_index=True)
        imp = pd.Series(s["xgb_importance_h4"]).sort_values()
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=TEAL))
        st.plotly_chart(layout(fig, 420, "XGBoost feature importance (gain share, h = 4)"), use_container_width=True)


def page_whatif():
    st.title("Climate what-if")
    st.markdown("Change the climate of the **last few weeks before the forecast date** and see how the "
                "climate-aware models react. Useful for questions like *'what if the next month is unusually hot?'*")
    origin = origin_picker("wi")
    c1, c2, c3, c4 = st.columns(4)
    t = c1.slider("Temperature change (C)", -2.0, 2.0, 1.0, 0.25)
    hum = c2.slider("Humidity change (%)", -20, 20, 0, 1)
    rain = c3.slider("Rainfall change (%)", -50, 100, 0, 5)
    weeks = c4.slider("Weeks affected", 4, 16, 12, 1)
    with st.spinner("Re-running SARIMAX, XGBoost and Hybrid..."):
        r = post(f"/whatif/{city}", json={"origin": str(origin), "temp_delta": t, "humidity_pct": hum,
                                          "rain_pct": rain, "weeks": weeks})
    tab = pd.DataFrame(r["table"]); tab["target_date"] = pd.to_datetime(tab.target_date)
    th = pd.DataFrame(r["thresholds"]); th["target_date"] = pd.to_datetime(th.target_date)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=th.target_date, y=th.epidemic, name="epidemic threshold", mode="lines",
                             line=dict(color=ACCENT, dash="dash")))
    for mod in tab.model.unique():
        g = tab[tab.model == mod].sort_values("h")
        col = MODEL_COLORS.get(mod)
        fig.add_trace(go.Scatter(x=g.target_date, y=g.cases_base, name=f"{mod} - observed climate",
                                 line=dict(color=col, dash="dot")))
        fig.add_trace(go.Scatter(x=g.target_date, y=g.cases_scenario, name=f"{mod} - scenario",
                                 line=dict(color=col, width=2.5)))
    st.plotly_chart(layout(fig, 420, ytitle="cases / week"), use_container_width=True)
    summ = tab.groupby("model")[["cases_base", "cases_scenario"]].sum()
    summ["change %"] = (summ.cases_scenario / summ.cases_base - 1) * 100
    summ.columns = ["8-week total (observed climate)", "8-week total (scenario)", "change %"]
    st.dataframe(summ.round(1), use_container_width=True)
    uc = r["used_climate"]
    note(f"Only climate variables a model actually uses can change its forecast. SARIMAX uses "
         f"{', '.join(uc['SARIMAX'])}; XGBoost and the Hybrid use {', '.join(uc['XGBoost / Hybrid'])}. "
         f"Because SARIMAX's climate lags are long (>= 8 weeks), a change in the most recent weeks mostly "
         f"affects cases *after* the 8-week window. Treat this as a sensitivity check, not a causal estimate.")


def page_upload():
    st.title("Upload your data")
    st.markdown("Forecast **your own** weekly case series - for example a district's surveillance data. "
                "Needs a CSV with a date column (`week_start_date` or `date`) and a case column "
                "(`total_cases` or `cases`), at least 3 years of weekly data. Climate is not required: this "
                "pipeline uses univariate models (seasonal naive, persistence, Holt-Winters, SARIMA, XGBoost), "
                "back-tests them on your last 52 weeks and uses the best one.")
    try:
        with open(os.path.join(os.path.dirname(__file__), "..", "data", "sample_upload.csv")) as fh:
            st.download_button("Download a sample CSV", fh.read(), "sample_upload.csv", "text/csv")
    except FileNotFoundError:
        pass
    up = st.file_uploader("Weekly case CSV", type=["csv"])
    hosp = st.number_input("Share of cases needing hospital care (%) - planning assumption", 0.0, 100.0, 10.0) / 100
    if not up:
        return
    with st.spinner("Cleaning, fitting and back-testing models (about 10-30 s)..."):
        try:
            r = post("/upload", files={"file": (up.name, up.getvalue(), "text/csv")}, params={"hosp_rate": hosp})
        except RuntimeError as e:
            st.error(str(e)); return
    q = r["quality"]
    st.caption(f"Read {q['rows_read']} rows -> {q['weeks']} regular weeks. Unparseable dates: "
               f"{q['unparseable_dates']}, duplicate dates merged: {q['duplicate_dates_merged']}, "
               f"gaps interpolated: {q['gaps_after_regularising']}.")
    banner(r["assessment"])
    hist = pd.DataFrame(r["history"]); hist["week_start_date"] = pd.to_datetime(hist.week_start_date)
    f = pd.DataFrame(r["forecast"]); f["target_date"] = pd.to_datetime(f.target_date)
    hist = hist.tail(104)
    fig = go.Figure()
    threshold_traces(fig, f.target_date, f, show_typical=True)
    fig.add_trace(go.Scatter(x=hist.week_start_date, y=hist.total_cases, name="your data", line=dict(color=INK)))
    fig.add_trace(go.Scatter(x=pd.concat([f.target_date, f.target_date[::-1]]), y=pd.concat([f.hi, f.lo[::-1]]),
                             mode="lines", fill="toself", fillcolor="rgba(200,85,61,0.15)", line=dict(width=0),
                             name="approx. 80% interval"))
    fig.add_trace(go.Scatter(x=f.target_date, y=f.cases, name=f"forecast ({r['best_model']})",
                             line=dict(color=ACCENT, width=2.5), mode="lines+markers"))
    st.plotly_chart(layout(fig, 400, ytitle="cases / week"), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.markdown("**Back-test on your last 52 weeks**")
    c1.dataframe(pd.DataFrame(r["backtest"]).round(2), hide_index=True, use_container_width=True)
    c2.markdown("**Recommended actions**")
    for act in r["assessment"]["actions"]:
        c2.markdown(f"- {act}")


{"Overview": page_overview, "Data & quality": page_data, "Exploratory analysis": page_eda,
 "Forecast & alerts": page_forecast, "Model comparison": page_compare,
 "Climate what-if": page_whatif, "Upload your data": page_upload}[page]()
