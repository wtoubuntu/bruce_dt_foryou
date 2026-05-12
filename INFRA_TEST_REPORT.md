# Infrastructure Test Report
**App:** Bruce's Data Viz Tool  
**Branch:** `feature/performance-improvements`  
**Tested:** 2026-05-12  
**Tester:** Automated (Claude Code)

---

## 1. Environment

| Item | Value |
|---|---|
| Python | 3.12 |
| Streamlit | installed via `venv/` |
| Key dependencies | pandas, plotly, numpy |
| Config | `.streamlit/config.toml` |
| Start command | `source venv/bin/activate && streamlit run app.py` |
| Default URL | `http://localhost:8501` |

---

## 2. Dependency Check

| Check | Result |
|---|---|
| streamlit import | PASS |
| pandas import | PASS |
| plotly import | PASS |
| numpy import | PASS |
| All modules load cleanly | PASS |

---

## 3. Upload Limits

| Scenario | Expected | Result |
|---|---|---|
| Upload 1 file | Accepted | PASS |
| Upload 3 files | Accepted | PASS |
| Upload 5 files | Accepted | PASS |
| Upload 6 files | Reject over limit, show error, accept only first 5 | PASS |
| Upload 10 files | Reject over limit, accept only first 5 | PASS |
| File > 100 MB | Rejected by Streamlit before app sees it (`maxUploadSize = 100` in config.toml) | PASS (config verified) |

---

## 4. File Handling

| Scenario | Expected | Result |
|---|---|---|
| Upload same file twice | Silently skip duplicate (keyed by file_id) | PASS |
| Upload two files with same filename | Second gets suffix: `filename (1)` | PASS |
| Upload three files with same filename | `filename`, `filename (1)`, `filename (2)` | PASS |
| Standard CSV auto-detect datetime column | Detects `Datetime`, `date`, `time`, `timestamp`, etc. | PASS |
| Turbine CSV detection (`Point Name` header) | Routes to turbine parser with metadata extraction | PASS |
| Excel file parse | Parsed correctly with datetime/numeric detection | PASS |

---

## 5. Core Feature Tests

| Feature | Input | Output | Result |
|---|---|---|---|
| Standard CSV parse | 10,000 rows × 3 cols | DataFrame with typed Datetime, float columns | PASS |
| Turbine CSV parse | 5-row metadata header + 90 data rows | DataFrame + sensor metadata dict (2 sensors) | PASS |
| Resample 1H | 10,000 rows (1-min interval) | 167 hourly buckets | PASS |
| Resample multi-column (scatter mode) | 10,000 rows, 2 value cols | 167 rows, both columns preserved | PASS |
| LTTB downsample | 10,000 → 500 target | Exactly 500 points returned | PASS |
| LTTB no-op | 100 rows, target=500 | All 100 rows returned unchanged | PASS |

---

## 6. Performance Benchmark

Tests run on: MacBook (darwin), Python 3.12, single process.

### File Parse Time & RAM per File

| File Size (on disk) | Load Time | RAM Used |
|---|---|---|
| ~2 MB (10K rows) | 0.02 s | 7 MB |
| ~20 MB (100K rows) | 0.08 s | 23 MB |
| ~99 MB (500K rows) | 0.37 s | 82 MB |
| ~197 MB (1M rows) | 0.74 s | 115 MB |

> Note: pandas stores float64 columns efficiently (8 bytes/value), so in-memory size is smaller than raw CSV text.

### Resample Performance (on 500K-row file)

| Operation | Time |
|---|---|
| 1H resample: 500K → 8,334 rows | 0.007 s |

### LTTB Downsample Performance (on 500K-row file)

| Target Points | Time |
|---|---|
| 500 pts | 0.005 s |
| 2,000 pts | 0.013 s |
| 5,000 pts | 0.030 s |

---

## 7. RAM Sizing for Deployment

Scenario: **5 concurrent users, each with 5 × 50 MB files loaded**

| Metric | Value |
|---|---|
| RAM per session (5 × 50 MB files) | ~228 MB |
| RAM for 5 concurrent sessions | ~1.1 GB |
| **Recommended server RAM** | **4 GB minimum** (includes OS, Python, Streamlit workers, headroom) |
| **Comfortable headroom** | **8 GB recommended** |

> Streamlit runs one Python process per server. Each user session holds its own copy of loaded DataFrames in memory. Sessions do not share data.

---

## 8. Known Limitations (Not Blocking)

| Item | Detail |
|---|---|
| No authentication | App is open to anyone who can reach the URL. Add a proxy/auth layer if needed. |
| No session timeout | Loaded data stays in memory until user clicks "Clear all" or closes tab. |
| Caching is per-process | If running multiple Streamlit workers (e.g. behind load balancer), cache is not shared across workers. Single-worker deployment recommended. |

---

## 9. Deployment Checklist

- [ ] Server RAM ≥ 4 GB (8 GB recommended for comfort)
- [ ] Python 3.12 installed
- [ ] `pip install -r requirements.txt` in venv
- [ ] `.streamlit/config.toml` present (enforces 100 MB upload limit + XSRF protection)
- [ ] Start with: `streamlit run app.py --server.port 8501`
- [ ] Optional: put behind nginx/reverse proxy for HTTPS and access control

---

## 10. Summary

| Category | Status |
|---|---|
| Dependencies | All OK |
| Upload limits (5 files, 100 MB) | Enforced |
| File parsing (CSV, Excel, Turbine) | All passing |
| Resample & downsampling | Correct and fast |
| RAM for 5 concurrent users | ~1.1 GB data + OS overhead → 4–8 GB server RAM recommended |
| Blocking issues | None found |
