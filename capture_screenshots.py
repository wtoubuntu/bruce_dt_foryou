"""
Captures screenshots of Bruce's Data Viz Tool and generates a standalone guide.html.

Requirements:
    pip install playwright httpx
    playwright install chromium

Usage:
    python capture_screenshots.py
"""

import asyncio
import base64
import subprocess
import sys
import time
from pathlib import Path

import httpx
from playwright.async_api import async_playwright

APP_URL = "http://localhost:8501"
EXAMPLE_FILE   = Path(__file__).parent / "examplefiles" / "Test _category.csv"
EXAMPLE_FILE_2 = Path(__file__).parent / "examplefiles" / "Train_category.csv"
OUTPUT = Path(__file__).parent / "guide.html"


# ── Streamlit readiness ──────────────────────────────────────────────────────

def wait_for_streamlit(timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(APP_URL, timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("Streamlit did not start within timeout")


# ── Wait helpers ─────────────────────────────────────────────────────────────

async def wait_idle(page, timeout=20000):
    """Wait for Streamlit's running spinner to disappear."""
    try:
        await page.wait_for_selector('[data-testid="stStatusWidget"]', state="hidden", timeout=timeout)
    except Exception:
        pass


async def wait_for_chart(page, timeout=30000):
    """Scroll first Plotly chart into view (triggers lazy render) then wait for paint."""
    try:
        chart = page.locator('[data-testid="stPlotlyChart"]').first
        await chart.wait_for(state="visible", timeout=timeout)
        await chart.scroll_into_view_if_needed()
    except Exception:
        pass
    await page.wait_for_timeout(3000)


# ── Widget interaction helpers ────────────────────────────────────────────────

async def click_radio(page, label: str):
    """Click a st.radio option and wait for chart."""
    await page.get_by_test_id("stRadio").get_by_text(label, exact=True).click()
    await wait_idle(page)
    await wait_for_chart(page)


async def set_selectbox(page, label_text: str, option_text: str):
    """Change a Streamlit selectbox to a given option."""
    sb = page.locator('[data-testid="stSelectbox"]').filter(has_text=label_text).first
    await sb.scroll_into_view_if_needed()
    await sb.click()
    await page.wait_for_timeout(500)
    await page.keyboard.type(option_text, delay=40)
    await page.wait_for_timeout(500)
    option = page.get_by_role("option").filter(has_text=option_text).first
    await option.wait_for(state="visible", timeout=10000)
    await option.click()
    await page.wait_for_timeout(600)


async def add_multiselect(page, label_text: str, option_text: str):
    """Add an item to a Streamlit multiselect (option must not already be selected)."""
    ms = page.locator('[data-testid="stMultiSelect"]').filter(has_text=label_text).first
    await ms.scroll_into_view_if_needed()
    await ms.locator("input").click()
    await page.keyboard.type(option_text, delay=40)
    await page.wait_for_timeout(500)
    option = page.get_by_role("option").filter(has_text=option_text).first
    await option.wait_for(state="visible", timeout=10000)
    await option.click()
    await page.wait_for_timeout(600)


# ── Screenshot helper ─────────────────────────────────────────────────────────

async def capture(page) -> str:
    """Scroll chart into view then take a viewport screenshot, returned as base64."""
    try:
        chart = page.locator('[data-testid="stPlotlyChart"]').first
        await chart.scroll_into_view_if_needed(timeout=3000)
        await page.wait_for_timeout(800)
    except Exception:
        pass
    return base64.b64encode(await page.screenshot(full_page=False)).decode()


# ── Main flow ─────────────────────────────────────────────────────────────────

async def run():
    shots = {}

    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.headless=true", "--server.port=8501"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        print("Waiting for Streamlit...")
        wait_for_streamlit()
        print("Ready.")

        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page(viewport={"width": 1400, "height": 1080})

            # ── Step 1: Empty upload screen ──────────────────────────
            print("Step 1: upload screen")
            await page.goto(APP_URL)
            await page.wait_for_load_state("networkidle")
            await wait_idle(page)
            await page.wait_for_timeout(1500)
            shots["upload"] = base64.b64encode(await page.screenshot(full_page=True)).decode()

            # ── Upload both files ────────────────────────────────────
            print("Uploading both files...")
            await page.locator('input[type="file"]').set_input_files(
                [str(EXAMPLE_FILE), str(EXAMPLE_FILE_2)]
            )
            await wait_idle(page)
            await wait_for_chart(page)

            # ── Step 2: Data loaded (scroll to top to show sidebar summary) ──
            print("Step 2: data loaded")
            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(600)
            shots["data_loaded"] = base64.b64encode(await page.screenshot(full_page=False)).decode()

            # ── Step 3: Time Series ──────────────────────────────────
            # Default Y-axis is "GT AXIAL DISP 1" (first sorted col).
            # Add "GT1 GEN ACTIVE POWER" to show multi-Y-axis + both files overlaid.
            print("Step 3: Time Series")
            await click_radio(page, "📈 Time Series")
            await add_multiselect(page, "Y-axis (sensors / values to plot)", "GT1 GEN ACTIVE POWER")
            await wait_idle(page)
            await wait_for_chart(page)
            shots["time_series"] = await capture(page)

            # ── Step 4: Scatter Plot ─────────────────────────────────
            # Default X=Y="GT AXIAL DISP 1" → single dot. Change Y to get real scatter.
            print("Step 4: Scatter Plot")
            await click_radio(page, "📊 Scatter Plot")
            await set_selectbox(page, "Y-axis", "GT1 GEN ACTIVE POWER")
            await wait_idle(page)
            await wait_for_chart(page)
            shots["scatter"] = await capture(page)

            # ── Step 5: Statistics — Time-of-Day Box Plot ────────────
            print("Step 5: Statistics")
            await click_radio(page, "📊 Statistics")
            await set_selectbox(page, "Choose chart type", "Time-of-Day Box Plot")
            await wait_idle(page)
            await wait_for_chart(page)
            shots["statistics"] = await capture(page)

            # ── Step 6: 3D — distinct X, Y, Z ───────────────────────
            # Default X=Y=Z="GT AXIAL DISP 1". Change Y and Z to get real 3D scatter.
            print("Step 6: 3D Visualization")
            await click_radio(page, "🌐 3D Visualization")
            await set_selectbox(page, "Y-axis", "GT COMP BRG TEMP 1")
            await set_selectbox(page, "Z-axis", "GT1 GEN ACTIVE POWER")
            await wait_idle(page)
            await wait_for_chart(page)
            shots["3d"] = await capture(page)

            await browser.close()

    finally:
        proc.terminate()

    print("Building guide.html...")
    build_html(shots)
    print(f"Saved → {OUTPUT}")


# ── HTML builder ──────────────────────────────────────────────────────────────

def img(b64: str) -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="Screenshot">'


def section(step, title, desc, key, shots, tip=None):
    screenshot = img(shots[key]) if key in shots else "<p><em>(screenshot unavailable)</em></p>"
    tip_html = f'<div class="tip"><strong>Tip:</strong> {tip}</div>' if tip else ""
    return f"""
    <section>
        <div class="step-header">
            <span class="step-num">Step {step}</span>
            <h2>{title}</h2>
        </div>
        <p class="desc">{desc}</p>
        {tip_html}
        <div class="screenshot">{screenshot}</div>
    </section>"""


def build_html(shots: dict):
    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bruce's Data Viz Tool — User Guide</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f5f6fa;
    color: #222;
    line-height: 1.7;
  }}
  header {{
    background: #1E3A5F;
    color: #fff;
    padding: 2.5rem 2rem 2rem;
    text-align: center;
  }}
  header h1 {{ font-size: 2rem; margin-bottom: 0.4rem; }}
  header p  {{ font-size: 1.05rem; opacity: 0.85; }}
  .container {{
    max-width: 980px;
    margin: 0 auto;
    padding: 2rem 1.5rem 4rem;
  }}
  .intro {{
    background: #fff;
    border-radius: 10px;
    padding: 1.8rem 2rem;
    margin-bottom: 2.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,.08);
  }}
  .intro h2 {{ font-size: 1.2rem; color: #1E3A5F; margin-bottom: 0.6rem; }}
  .intro ul {{ padding-left: 1.4rem; }}
  .intro li {{ margin-bottom: 0.3rem; }}
  section {{
    background: #fff;
    border-radius: 10px;
    padding: 1.8rem 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 1px 4px rgba(0,0,0,.08);
  }}
  .step-header {{
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.8rem;
  }}
  .step-num {{
    background: #1E3A5F;
    color: #fff;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 0.25rem 0.65rem;
    border-radius: 20px;
    white-space: nowrap;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }}
  h2 {{ font-size: 1.25rem; color: #1E3A5F; }}
  .desc {{ margin-bottom: 1.2rem; color: #444; }}
  .screenshot img {{
    width: 100%;
    border-radius: 8px;
    border: 1px solid #e0e0e0;
    margin-top: 1rem;
  }}
  .tip {{
    background: #eef4ff;
    border-left: 4px solid #1E3A5F;
    padding: 0.7rem 1rem;
    border-radius: 0 6px 6px 0;
    margin-top: 0.2rem;
    margin-bottom: 0.5rem;
    font-size: 0.92rem;
    color: #333;
  }}
  footer {{
    text-align: center;
    font-size: 0.85rem;
    color: #999;
    padding-bottom: 2rem;
  }}
</style>
</head>
<body>
<header>
  <h1>📊 Bruce's Data Viz Tool</h1>
  <p>A step-by-step guide — no coding required</p>
</header>
<div class="container">

  <div class="intro">
    <h2>What is this tool?</h2>
    <p>Bruce's Data Viz Tool lets you upload data files and instantly create interactive charts. You can:</p>
    <ul>
      <li>Upload one or more CSV or Excel files</li>
      <li>Compare data from multiple files on the same chart</li>
      <li>Explore trends over time, relationships between sensors, statistics, and 3D patterns</li>
      <li>Download any chart as a file you can share with anyone</li>
    </ul>
  </div>

  {section(1, "Open the app",
    "Open your web browser and go to the address provided by your team (usually <strong>http://localhost:8501</strong>). You will see the upload screen below.",
    "upload", shots)}

  {section(2, "Upload your data files",
    "Click the upload box (or drag and drop your files into it). You can upload <strong>.csv</strong> or <strong>.xlsx</strong> files and select multiple files at once. After uploading, the left sidebar will show a summary of your data — number of rows, columns, and detected data types.",
    "data_loaded", shots)}

  {section(3, "Time Series — compare trends over time",
    "Click <strong>📈 Time Series</strong> in the navigation bar. Use the <em>Y-axis</em> dropdown to pick one or more measurements to plot. You can select both files in the <em>Select files to overlay</em> box to compare them on the same graph.",
    "time_series", shots,
    tip="You can select multiple Y-axis columns to overlay several measurements in one chart. Each file gets its own colour.")}

  {section(4, "Scatter Plot — find relationships between sensors",
    "Click <strong>📊 Scatter Plot</strong>. Choose a different sensor for the X-axis and Y-axis to see whether they are related. Each dot represents one reading — a diagonal pattern means the two sensors move together.",
    "scatter", shots,
    tip="Select both files in the overlay box to compare the same relationship across two datasets.")}

  {section(5, "Statistics — understand your data distribution",
    "Click <strong>📊 Statistics</strong>. Use the <em>Choose chart type</em> dropdown to switch between histograms, box plots, density curves, correlation heatmaps, and time-of-day box plots. Each chart helps you understand a different aspect of how your data is distributed.",
    "statistics", shots,
    tip="The Time-of-Day Box Plot shows how a measurement changes throughout the day — useful for spotting daily patterns.")}

  {section(6, "3D Visualization — explore three sensors at once",
    "Click <strong>🌐 3D Visualization</strong>. Choose three different sensors for the X, Y, and Z axes. You can rotate the chart by clicking and dragging. Clusters of dots in the same area mean those readings often appear together.",
    "3d", shots,
    tip="Overlay both files to compare how their data clusters differ in 3D space.")}

  <section>
    <div class="step-header">
      <span class="step-num">Step 7</span>
      <h2>Export and share a chart</h2>
    </div>
    <p class="desc">Every chart has a <strong>📥 Download HTML</strong> button. Clicking it saves the chart as a file you can open in any browser and send to anyone — no special software needed.</p>
    <div class="tip">
      <strong>Tip:</strong> The downloaded chart is fully interactive — the recipient can zoom, pan, and hover over data points just like in the app.
    </div>
  </section>

</div>
<footer>Bruce's Data Viz Tool &mdash; User Guide</footer>
</body>
</html>"""

    OUTPUT.write_text(body, encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(run())
