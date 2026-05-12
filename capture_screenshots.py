"""
Captures screenshots of Bruce's Data Viz Tool and generates a standalone guide.html.

Requirements:
    pip install playwright
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
EXAMPLE_FILE = Path(__file__).parent / "examplefiles" / "Test _category.csv"
EXAMPLE_FILE_2 = Path(__file__).parent / "examplefiles" / "Train_category.csv"
OUTPUT = Path(__file__).parent / "guide.html"


def wait_for_streamlit(timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(APP_URL, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("Streamlit did not start within timeout")


async def wait_for_streamlit_idle(page, timeout=20000):
    """Wait for Streamlit's running indicator to disappear."""
    try:
        await page.wait_for_selector('[data-testid="stStatusWidget"]', state="hidden", timeout=timeout)
    except Exception:
        pass


async def wait_for_chart(page, timeout=25000):
    """Scroll the chart into view to trigger lazy rendering, then wait for paint."""
    try:
        chart = page.locator('[data-testid="stPlotlyChart"]').first
        await chart.wait_for(state="visible", timeout=timeout)
        await chart.scroll_into_view_if_needed()
    except Exception:
        pass
    # Allow extra time for WebGL/SVG paint after scroll
    await page.wait_for_timeout(3000)


async def capture(page, name: str) -> str:
    """Scroll to chart if present, then take a viewport screenshot."""
    try:
        chart = page.locator('[data-testid="stPlotlyChart"]').first
        await chart.scroll_into_view_if_needed(timeout=3000)
        await page.wait_for_timeout(500)
    except Exception:
        pass
    data = await page.screenshot(full_page=False)
    return base64.b64encode(data).decode()


async def click_radio(page, label: str):
    """Click a st.radio option and wait for the chart to render."""
    radio_group = page.get_by_test_id("stRadio")
    await radio_group.get_by_text(label, exact=True).click()
    await wait_for_streamlit_idle(page)
    await wait_for_chart(page)


async def run():
    screenshots = {}

    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.headless=true", "--server.port=8501"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        print("Waiting for Streamlit to start...")
        wait_for_streamlit()
        print("Streamlit is ready.")

        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page(viewport={"width": 1400, "height": 1080})

            # ── Step 1: Upload screen (empty state) ─────────────────
            print("Capturing: upload screen")
            await page.goto(APP_URL)
            await page.wait_for_load_state("networkidle")
            await wait_for_streamlit_idle(page)
            await page.wait_for_timeout(1500)
            screenshots["upload"] = base64.b64encode(await page.screenshot(full_page=True)).decode()

            # ── Step 2: Load example file ────────────────────────────
            print("Uploading example file...")
            file_input = page.locator('input[type="file"]')
            await file_input.set_input_files(str(EXAMPLE_FILE))
            await wait_for_streamlit_idle(page)
            await wait_for_chart(page)
            screenshots["data_loaded"] = await capture(page, "data_loaded")

            # ── Step 3: Time Series tab ──────────────────────────────
            print("Capturing: Time Series")
            await click_radio(page, "📈 Time Series")
            screenshots["time_series"] = await capture(page, "time_series")

            # ── Step 4: Scatter Plot tab ─────────────────────────────
            print("Capturing: Scatter Plot")
            await click_radio(page, "📊 Scatter Plot")
            screenshots["scatter"] = await capture(page, "scatter")

            # ── Step 5: Statistics tab ───────────────────────────────
            print("Capturing: Statistics")
            await click_radio(page, "📊 Statistics")
            screenshots["statistics"] = await capture(page, "statistics")

            # ── Step 6: 3D Visualization tab ─────────────────────────
            print("Capturing: 3D Visualization")
            await click_radio(page, "🌐 3D Visualization")
            screenshots["3d"] = await capture(page, "3d")

            await browser.close()

    finally:
        proc.terminate()

    print("Generating guide.html...")
    build_html(screenshots)
    print(f"Done! Guide saved to: {OUTPUT}")


def img_tag(b64: str) -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="Screenshot">'


def build_html(shots: dict):
    def section(step, title, desc, key):
        screenshot_html = img_tag(shots[key]) if key in shots else "<p><em>(screenshot not available)</em></p>"
        return f"""
        <section>
            <div class="step-header">
                <span class="step-num">Step {step}</span>
                <h2>{title}</h2>
            </div>
            <p class="desc">{desc}</p>
            <div class="screenshot">{screenshot_html}</div>
        </section>
        """

    html = f"""<!DOCTYPE html>
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
    max-width: 960px;
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
  }}
  .tip {{
    background: #eef4ff;
    border-left: 4px solid #1E3A5F;
    padding: 0.7rem 1rem;
    border-radius: 0 6px 6px 0;
    margin-top: 1rem;
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
  <p>A step-by-step guide for getting started — no coding required</p>
</header>

<div class="container">

  <div class="intro">
    <h2>What is this tool?</h2>
    <p>Bruce's Data Viz Tool lets you upload data files and instantly create interactive charts — without writing a single line of code. You can:</p>
    <ul>
      <li>Upload one or more CSV or Excel files</li>
      <li>Compare data from multiple files on the same chart</li>
      <li>Explore trends over time, relationships between values, statistics, and 3D patterns</li>
      <li>Download any chart as a shareable HTML file</li>
    </ul>
  </div>

  {section(1, "Open the app",
    "Open your web browser and go to the address your team has provided (usually <strong>http://localhost:8501</strong>). You will see the upload screen below.",
    "upload")}

  {section(2, "Upload your data files",
    "Click the upload box (or drag and drop your files into it). You can upload <strong>.csv</strong> or <strong>.xlsx / .xls</strong> files — and you can select multiple files at once. Once uploaded, you will see a summary of your data in the left sidebar.",
    "data_loaded")}

  {section(3, "Explore the Time Series chart",
    "Click <strong>📈 Time Series</strong> in the navigation bar. This chart plots your data over time. Use the dropdown menus to pick which columns (sensors or measurements) to display. You can overlay multiple files on the same graph to compare them side by side.",
    "time_series")}

  {section(4, "Analyse relationships with the Scatter Plot",
    "Click <strong>📊 Scatter Plot</strong>. Choose one column for the X axis and another for the Y axis to see whether two measurements are related. A regression line is drawn automatically to highlight any trend.",
    "scatter")}

  {section(5, "Review statistics and distributions",
    "Click <strong>📊 Statistics</strong>. This tab gives you histograms, box plots, density curves, and a correlation heatmap — useful for understanding the spread of your data and spotting which columns move together.",
    "statistics")}

  {section(6, "Explore data in 3D",
    "Click <strong>🌐 3D Visualization</strong>. Pick three columns to use as X, Y, and Z axes and rotate the chart freely with your mouse. This is helpful for spotting clusters or patterns that are hard to see in a flat chart.",
    "3d")}

  <section>
    <div class="step-header">
      <span class="step-num">Step 7</span>
      <h2>Export and share a chart</h2>
    </div>
    <p class="desc">Every chart has a <strong>📥 Download HTML</strong> button below it. Clicking it saves the chart as a self-contained HTML file you can open in any browser and share with anyone — no special software needed.</p>
    <div class="tip">
      <strong>Tip:</strong> The downloaded chart is fully interactive — the recipient can zoom, pan, and hover over data points just like in the app.
    </div>
  </section>

</div>

<footer>Bruce's Data Viz Tool &mdash; User Guide</footer>
</body>
</html>"""

    OUTPUT.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(run())
