"""
Step F - Decision support: turn forecasts into alert levels and actions.

Thresholds come from the endemic channel (features.endemic_channel), computed per
season-week from training seasons:
    typical  = mean
    warning  = mean + 1 SD
    epidemic = mean + 2 SD

The action lists are generic, rule-based examples of a graded dengue response
(surveillance -> vector control -> health-system preparedness). They are a decision
aid, not medical or public-health advice; a real deployment must map them to the
local health department's own response plan.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LEVELS = [
    {"key": "normal", "label": "Normal", "color": "#2a9d8f", "rank": 0},
    {"key": "watch", "label": "Watch", "color": "#e9c46a", "rank": 1},
    {"key": "warning", "label": "Warning", "color": "#f4a261", "rank": 2},
    {"key": "epidemic", "label": "Epidemic alert", "color": "#c8553d", "rank": 3},
]
LEVEL = {lv["key"]: lv for lv in LEVELS}


def classify(cases: float, typical: float, warning: float, epidemic: float,
             upper: float | None = None) -> str:
    """Point forecast above the epidemic threshold -> Epidemic alert.
    Point forecast above mean + 1 SD, OR the upper 80% bound above the epidemic
    threshold (roughly a >= 10% chance of epidemic-level cases) -> Warning.
    The second trigger is the probabilistic early-warning rule evaluated in the report:
    it gave 3-8 weeks of warning before the test-period outbreaks."""
    if cases > epidemic:
        return "epidemic"
    if cases > warning or (upper is not None and not np.isnan(upper) and upper > epidemic):
        return "warning"
    if cases > typical:
        return "watch"
    return "normal"


ACTIONS = {
    "normal": [
        "Continue routine weekly surveillance and laboratory confirmation.",
        "Keep up routine larval-source reduction (removing standing water) in high-risk wards.",
    ],
    "watch": [
        "Increase surveillance: review case reports twice a week and check lab turnaround times.",
        "Start community awareness messages on removing breeding sites around homes.",
        "Check stock of diagnostic test kits and IV fluids.",
    ],
    "warning": [
        "Begin intensified larval-source reduction and clean-up drives in wards reporting cases.",
        "Alert hospitals and clinics to expect more dengue patients; review triage protocols.",
        "Prepare vector-control teams and insecticide stock for possible space spraying.",
    ],
    "epidemic": [
        "Activate the outbreak response plan and the rapid-response team.",
        "Run targeted vector control (larval-source reduction + space spraying) around case clusters.",
        "Scale up hospital capacity: extra beds, staff rosters and fluid-management supplies.",
        "Issue public advisories; coordinate with schools and local media.",
    ],
}


def assess(forecast: pd.DataFrame, hosp_rate: float = 0.10, use_upper: bool = True) -> dict:
    """forecast columns: target_date, h, cases, lo, hi, typical, warning, epidemic.
    Returns the overall alert (worst level over the horizon) and supporting numbers."""
    f = forecast.copy()
    hi = f["hi"] if ("hi" in f and use_upper) else pd.Series(np.nan, index=f.index)
    f["level"] = [classify(c, t, w, e, u) for c, t, w, e, u in
                  zip(f.cases, f.typical, f.warning, f.epidemic, hi)]
    f["rank"] = f.level.map(lambda k: LEVEL[k]["rank"])
    worst = f.loc[f["rank"].idxmax()]
    first_epi = f[f.level == "epidemic"]
    first_warn = f[f["rank"] >= 2]
    upper_cross = f[f.hi > f.epidemic] if "hi" in f and f.hi.notna().any() else f.iloc[0:0]
    total = float(f.cases.sum())
    summary = {
        "level": worst.level,
        "label": LEVEL[worst.level]["label"],
        "color": LEVEL[worst.level]["color"],
        "peak_week": str(pd.Timestamp(f.loc[f.cases.idxmax(), "target_date"]).date()),
        "peak_cases": round(float(f.cases.max()), 1),
        "expected_cases_total": round(total, 0),
        "expected_hospital_care": round(total * hosp_rate, 0),
        "weeks_until_epidemic": int(first_epi.h.min()) if len(first_epi) else None,
        "weeks_until_warning": int(first_warn.h.min()) if len(first_warn) else None,
        "upper_bound_crosses_epidemic_in": int(upper_cross.h.min()) if len(upper_cross) else None,
        "actions": ACTIONS[worst.level],
        "weekly": f.assign(target_date=f.target_date.astype(str)).round(2).to_dict("records"),
    }
    summary["headline"] = headline(summary)
    return summary


def headline(s: dict) -> str:
    if s["level"] == "epidemic":
        return (f"Cases are forecast to cross the epidemic threshold in {s['weeks_until_epidemic']} "
                f"week(s), peaking near {s['peak_cases']:.0f} cases/week around {s['peak_week']}.")
    if s["level"] == "warning":
        extra = ""
        if s["upper_bound_crosses_epidemic_in"]:
            extra = (f" The upper forecast bound reaches the epidemic threshold in "
                     f"{s['upper_bound_crosses_epidemic_in']} week(s).")
        if s["upper_bound_crosses_epidemic_in"] and s["weeks_until_warning"] == s["upper_bound_crosses_epidemic_in"]:
            return (f"There is a meaningful chance (upper 80% bound) that cases reach the epidemic "
                    f"threshold in {s['upper_bound_crosses_epidemic_in']} week(s); the central forecast "
                    f"peaks near {s['peak_cases']:.0f} cases/week. Prepare now.")
        return (f"Cases are forecast to rise above the warning level (mean + 1 SD) in "
                f"{s['weeks_until_warning']} week(s).{extra}")
    if s["level"] == "watch":
        return "Cases are forecast to be above the typical level for this time of year."
    return "Cases are forecast to stay within the normal range for this time of year."


# ----------------------------------------------------------------------------- evaluation helpers
def outbreak_episodes(is_outbreak: np.ndarray, min_len: int = 2, max_gap: int = 2) -> list[tuple[int, int]]:
    """Runs of above-threshold weeks, merging gaps <= max_gap, keeping runs >= min_len weeks.
    Returns (start, end) positions."""
    idx = np.flatnonzero(is_outbreak)
    if len(idx) == 0:
        return []
    runs, s, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > max_gap + 1:
            runs.append((s, prev)); s = i
        prev = i
    runs.append((s, prev))
    return [(a, b) for a, b in runs if (b - a + 1) >= min_len]
