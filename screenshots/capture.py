"""
Capture screenshots of every app page (used for the report / submission).
Requires the backend (port 8000) and frontend (port 8501) to be running:
    python screenshots/capture.py
"""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).parent
URL = "http://localhost:8501"


def wait_ready(page, extra=2.5):
    # Streamlit shows a "Running..." status widget while a script executes
    time.sleep(1.0)
    for _ in range(120):
        if page.locator('[data-testid="stStatusWidget"]').count() == 0:
            break
        time.sleep(0.5)
    time.sleep(extra)


def errors_on(page):
    errs = page.locator('[data-testid="stException"], [data-testid="stAlertContentError"]')
    return [errs.nth(i).inner_text()[:300] for i in range(errs.count())]


def goto(page, label):
    page.locator('[data-testid="stSidebar"]').get_by_text(label, exact=True).click()
    wait_ready(page)


def shot(page, name, full=True):
    """Streamlit scrolls inside its own container, so a 'full page' capture is done by
    growing the viewport to the content height first."""
    path = OUT / f"{name}.png"
    if full:
        page.set_viewport_size({"width": 1440, "height": 900})
        time.sleep(0.8)
        h = page.evaluate("() => { const m = document.querySelector('[data-testid=\"stMainBlockContainer\"]')"
                          " || document.querySelector('[data-testid=\"stMain\"]') || document.body;"
                          " return Math.max(m.scrollHeight + 60, 900); }")
        page.set_viewport_size({"width": 1440, "height": min(int(h) + 40, 6000)})
        time.sleep(1.5)
    else:
        page.set_viewport_size({"width": 1440, "height": 900})
        time.sleep(0.8)
    page.screenshot(path=str(path), full_page=False)
    e = errors_on(page)
    print(f"{name}: {'ERRORS ' + str(e) if e else 'ok'}")
    return e


def main():
    problems = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page.goto(URL)
        wait_ready(page, 4)
        problems += shot(page, "01_overview")

        goto(page, "Data & quality")
        problems += shot(page, "02_data_quality")

        goto(page, "Exploratory analysis")
        problems += shot(page, "03_eda_trend", full=False)
        for tab, name in [("Decomposition", "04_eda_decomposition"), ("Autocorrelation & stationarity", "05_eda_stationarity"),
                          ("Climate lags", "06_eda_climate_lags")]:
            page.get_by_role("tab", name=tab).click(); wait_ready(page, 1.5)
            problems += shot(page, name, full=False)

        goto(page, "Forecast & alerts")
        problems += shot(page, "07_forecast_latest")
        # replay buttons: 6 weeks before each test-period outbreak
        btns = page.get_by_role("button", name="Replay")
        btn = page.locator("button", has_text="Replay: 6 wk before 2012-11-25")
        if btn.count():
            btn.first.click(); wait_ready(page)
            problems += shot(page, "08_forecast_replay_2012")
        btn = page.locator("button", has_text="Replay: 6 wk before 2010-02-12")
        if btn.count():
            btn.first.click(); wait_ready(page)
            problems += shot(page, "09_forecast_replay_2010")

        goto(page, "Model comparison")
        problems += shot(page, "10_model_comparison")

        goto(page, "Climate what-if")
        wait_ready(page, 6)
        problems += shot(page, "11_whatif")

        goto(page, "Upload your data")
        page.locator('input[type="file"]').set_input_files(str(OUT.parent / "data" / "sample_upload.csv"))
        wait_ready(page, 8)
        problems += shot(page, "12_upload")

        # Iquitos forecast page
        goto(page, "Forecast & alerts")
        page.locator('[data-testid="stSidebar"] [data-testid="stSelectbox"]').click()
        page.get_by_role("option", name="Iquitos, Peru").click()
        wait_ready(page)
        btn = page.locator("button", has_text="Replay: 6 wk before 2011-01-01")
        if btn.count():
            btn.first.click(); wait_ready(page)
        problems += shot(page, "13_forecast_iquitos_replay_2011")
        b.close()
    if problems:
        print("PROBLEMS:", problems)
        sys.exit(1)


if __name__ == "__main__":
    main()
