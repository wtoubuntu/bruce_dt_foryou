"""
Bruce's Data Viz Tool
Upload multiple CSV or Excel files → overlay comparisons on the same graph.
Supports turbine sensor CSV format (5-row metadata header) or standard CSV.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
import sys
from pathlib import Path
import numpy as np
import traceback

from preprocessor import load_turbine_csv
import plotly.io as pio
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="plotly")


# ── LTTB Downsampling (Largest Triangle Three Buckets) ──────
def lttb_downsample(x, y, n_target):
    """
    Downsample (x, y) to at most n_target points using LTTB algorithm.
    Preserves visual shape better than random sampling.
    x and y must be same length, 1D arrays.
    """
    n = len(x)
    if n <= n_target:
        return x, y

    n_target = max(3, n_target)
    bucket_size = float((n - 2) / (n_target - 2))

    # ── Robust numeric conversion for x ─────────────────────
    x_dtype = getattr(x, "dtype", None)
    if x_dtype is not None and np.issubdtype(x_dtype, np.datetime64):
        # Scale to milliseconds to prevent exact precision loss
        x_num = (np.asarray(x, dtype=np.int64).ravel() / 1_000_000.0).astype(float)
    else:
        x_num = np.asarray(x, dtype=float).ravel()

    # ── Robust numeric conversion for y ─────────────────────
    y_num = np.asarray(y, dtype=float).ravel()

    # ── Verify 1D shape ────────────────────────────────────
    assert x_num.ndim == 1 and y_num.ndim == 1, \
        f"x/y must be 1D arrays, got x={x_num.shape} y={y_num.shape}"

    result_x = [x[0]]
    result_y = [y[0]]

    a_idx = 0
    ax = x_num[a_idx].item()
    ay = y_num[a_idx].item()

    for i in range(n_target - 2):
        buck_start = int((i + 0) * bucket_size) + 1
        buck_end   = min(int((i + 1) * bucket_size) + 1, n - 1)
        next_buck_start = buck_end
        next_buck_end   = min(int((i + 2) * bucket_size) + 1, n - 1)

        avg_x = float(x_num[next_buck_start:next_buck_end + 1].mean())
        avg_y = float(y_num[next_buck_start:next_buck_end + 1].mean())

        max_area = -1.0
        max_area_idx = buck_start
        for j in range(buck_start, buck_end + 1):
            dx_a = ax - avg_x
            dy_j = y_num[j].item() - ay
            dx_j = x_num[j].item() - avg_x
            area = abs(dx_a * dy_j - dx_j * (avg_y - ay))
            if area > max_area:
                max_area = area
                max_area_idx = j

        result_x.append(x[max_area_idx])
        result_y.append(y[max_area_idx])
        a_idx = max_area_idx
        ax = x_num[a_idx].item()
        ay = y_num[a_idx].item()

    result_x.append(x[-1])
    result_y.append(y[-1])
    return np.array(result_x), np.array(result_y)

# Use browser renderer (Chrome) for interactive WebGL plots
pio.renderers.default = "browser"

st.set_page_config(
    page_title="Bruce's Data Viz Tool",
    page_icon="📊",
    layout="wide"
)

# ── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6C757D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-box {
        border: 2px dashed #1E3A5F;
        border-radius: 12px;
        padding: 3rem 2rem;
        text-align: center;
        background: #F8F9FA;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
    }
    .dataset-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────
st.markdown('<p class="main-header">📊 Bruce\'s Data Viz Tool</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload multiple files · Compare & overlay on the same graph · No coding required.</p>', unsafe_allow_html=True)

# ── File Upload ─────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "📁 Upload CSV or Excel files (select multiple)",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
    help="Supported: .csv, .xlsx, .xls  |  Hold Shift or Ctrl to select multiple"
)

# ── Session state for loaded datasets ───────────────────────
if "datasets" not in st.session_state:
    st.session_state.datasets = {}  # {filename: {"df": ..., "metadata": ..., "type": ...}}

# ── Shared resample rule (synced across Time Series & Scatter tabs)
if "resample_rule" not in st.session_state:
    st.session_state.resample_rule = "All (native)"

if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name in st.session_state.datasets:
            continue  # already loaded

        try:
            if uploaded_file.name.endswith(".csv"):
                uploaded_file.seek(0)
                raw_check = pd.read_csv(uploaded_file, header=None, nrows=1)
                first_val = str(raw_check.iloc[0, 0]) if raw_check.shape[0] > 0 else ""
                uploaded_file.seek(0)

                if first_val == "Point Name":
                    data, metadata = load_turbine_csv(uploaded_file)
                    df = data.reset_index().rename(columns={"datetime": "Datetime"})
                    file_type = "turbine"
                else:
                    df = pd.read_csv(uploaded_file)
                    file_type = "standard"
            else:
                df = pd.read_excel(uploaded_file)
                file_type = "excel"
                
            if file_type in ["standard", "excel"]:
                # Auto-detect date/time column
                dt_candidates = ["date_time", "datetime", "date", "time", "timestamp", "date/time", "t"]
                dt_col = None
                for col in df.columns:
                    if str(col).lower().strip() in dt_candidates:
                        dt_col = col
                        break
                        
                # Fallback to first column if no match found
                if dt_col is None and len(df.columns) > 0:
                    dt_col = df.columns[0]
                    
                if dt_col in df.columns:
                    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
                    df = df.dropna(subset=[dt_col])
                    df = df.set_index(dt_col)

                # Convert all remaining columns to float
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                
                if dt_col is not None:
                    df = df.reset_index().rename(columns={dt_col: "Datetime"})
                    
                metadata = None

            st.session_state.datasets[uploaded_file.name] = {
                "df": df,
                "metadata": metadata,
                "type": file_type,
            }
            st.success(f"✅ Loaded `{uploaded_file.name}` — {df.shape[0]} rows × {df.shape[1]} columns")
        except Exception as e:
            st.error(f"❌ Failed to load `{uploaded_file.name}`: {e}")
            # 1. Grab the full error message and line numbers as a text string
            error_details = traceback.format_exc()
            
            # 2. Display it directly on your Streamlit web page in a nice code block
            st.code(error_details, language="python")

# ── Show loaded datasets + clear button ───────────────────────
if st.session_state.datasets:
    col = st.columns([1, 1, 1, 1, 1])
    with col[0]:
        st.markdown("**📂 Loaded datasets:**")
    for i, fname in enumerate(list(st.session_state.datasets.keys())):
        with col[(i % 5) + 1]:
            st.caption(f"• {fname}")
    with col[min(len(st.session_state.datasets), 4) + 1]:
        if st.button("🗑️ Clear all"):
            st.session_state.datasets = {}
            st.rerun()

    st.divider()

    # ── Auto-detect column types helper ──────────────────────
    def get_column_type(series):
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        elif pd.api.types.is_numeric_dtype(series):
            return "numeric"
        else:
            return "categorical"

    # Collect all column types across datasets (union)
    all_cols = set()
    for ds in st.session_state.datasets.values():
        all_cols.update(ds["df"].columns)
    all_cols = sorted(all_cols)

    col_types = {}
    for col in all_cols:
        for ds in st.session_state.datasets.values():
            if col in ds["df"].columns:
                col_types[col] = get_column_type(ds["df"][col])
                break

    num_cols  = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols  = [c for c, t in col_types.items() if t == "categorical"]
    date_cols = [c for c, t in col_types.items() if t == "datetime"]

    # ── Sidebar: Data Summary ────────────────────────────────
    with st.sidebar:
        st.header("📋 Data Summary")
        st.metric("Datasets loaded", len(st.session_state.datasets))

        total_rows = sum(ds["df"].shape[0] for ds in st.session_state.datasets.values())
        total_cols = len(all_cols)
        st.metric("Total rows (all files)", total_rows)
        st.metric("Total columns", total_cols)
        st.metric("Numeric cols", len(num_cols))
        st.metric("Date cols", len(date_cols))

        st.subheader("📂 Dataset Details")
        for fname, ds in st.session_state.datasets.items():
            with st.expander(f"📄 {fname}"):
                df_sn = ds["df"]
                st.write(f"Rows: {df_sn.shape[0]} · Cols: {df_sn.shape[1]}")
                if "Datetime" in df_sn.columns or "datetime" in df_sn.columns:
                    dt_col = "Datetime" if "Datetime" in df_sn.columns else "datetime"
                    try:
                        dt_min = df_sn[dt_col].min()
                        dt_max = df_sn[dt_col].max()
                        st.write(f"📅 {dt_min} → {dt_max}")
                    except:
                        pass
                if ds["metadata"]:
                    st.write(f"🔧 {len(ds['metadata'])} sensors")

    # ── Quick Stats ─────────────────────────────────────────
    st.subheader("📈 Quick Statistics")

    if len(st.session_state.datasets) == 1:
        # Single dataset — show normal stats
        fname = list(st.session_state.datasets.keys())[0]
        ds = st.session_state.datasets[fname]
        df = ds["df"]
        tab_stat, tab_head = st.tabs(["📊 Statistics", "🔍 Data Preview"])
        with tab_stat:
            if len(num_cols) > 0:
                st.dataframe(df[num_cols].describe(), use_container_width=True)
            else:
                st.info("No numeric columns found for statistics.")
        with tab_head:
            st.dataframe(df.head(10), use_container_width=True)
    else:
        # Multi-dataset — show comparison
        tab_stat, tab_head = st.tabs(["📊 Statistics", "🔍 Data Preview"])
        with tab_stat:
            sel = st.selectbox("Select dataset for stats", list(st.session_state.datasets.keys()), key="stats_dataset")
            ds = st.session_state.datasets[sel]
            df = ds["df"]
            if len(num_cols) > 0:
                st.dataframe(df[num_cols].describe(), use_container_width=True)
            else:
                st.info("No numeric columns found for statistics.")
        with tab_head:
            st.dataframe(df.head(10), use_container_width=True)

    st.divider()

    # ══════════════════════════════════════════════════════════
    # VISUALIZATION SECTION
    # ══════════════════════════════════════════════════════════
    st.subheader("🎨 Visualizations")

    available_files = list(st.session_state.datasets.keys())

    active_tab = st.radio(
        "Navigation",
        ["📈 Time Series", "📊 Scatter Plot", "📊 Statistics", "🌐 3D Visualization"],
        horizontal=True,
        label_visibility="collapsed",
        key="viz_active_tab"
    )

    # ── Time Series ──────────────────────────────────────────
    if active_tab == "📈 Time Series":
        st.markdown("**Compare multiple files on the same time series graph**")

        if len(num_cols) == 0 or len(date_cols) == 0:
            st.warning("⚠️ Need at least one date column and one numeric column.")
        else:
            # Y-axis sensor/column selector
            y_cols = st.multiselect(
                "Y-axis (sensors / values to plot)",
                num_cols,
                default=[num_cols[0]] if num_cols else None,
                key="ts_y_multi"
            )

            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                height = st.slider("Chart height (px)", 300, 800, 450, key="ts_h_multi")
            with col2:
                show_rangebars = st.checkbox("Show range bars", value=False, key="ts_rangebars")
            with col3:
                resample_rule = st.selectbox(
                    "⏱️ Resample to",
                    ["All (native)", "1min", "2min", "5min", "10min", "15min", "30min", "1h", "2h", "4h", "6h", "12h", "1D"],
                    index=["All (native)", "1min", "2min", "5min", "10min", "15min", "30min", "1h", "2h", "4h", "6h", "12h", "1D"].index(st.session_state.resample_rule),
                    key="ts_resample_k",
                    on_change=lambda: st.session_state.__setitem__("resample_rule", st.session_state.ts_resample_k),
                )

            # File selector for overlay
            selected_files = st.multiselect(
                "📂 Select files to overlay",
                available_files,
                default=available_files[:1] if len(available_files) == 1 else available_files,
                key="ts_files"
            )

            if not selected_files:
                st.info("Select at least one file above to plot.")
            else:
                # Build traces for each selected file
                fig = go.Figure()

                # Color palette for multiple traces
                colors = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24
                range_bar_shapes = []

                dt_col_last = "Datetime"
                for i, fname in enumerate(selected_files):
                    ds = st.session_state.datasets[fname]
                    df_plot = ds["df"]

                    if "Datetime" not in df_plot.columns and "datetime" not in df_plot.columns:
                        st.warning(f"`{fname}` has no datetime column — skipped.")
                        continue

                    dt_col = "Datetime" if "Datetime" in df_plot.columns else "datetime"
                    dt_col_last = dt_col

                    for j, y_col in enumerate(y_cols):
                        if y_col not in df_plot.columns:
                            st.warning(f"`{fname}` does not have column `{y_col}` — skipped.")
                            continue

                        # Combine colors based on file and column
                        color_idx = (i * len(y_cols) + j) % len(colors)
                        color = colors[color_idx]
                        
                        trace_name = f"{fname} - {y_col}" if len(y_cols) > 1 or len(selected_files) > 1 else fname

                        # Prepare data
                        plot_df = df_plot[[dt_col, y_col]].dropna()
                        plot_df = plot_df.sort_values(dt_col)

                        # Resample by time interval if selected
                        if resample_rule != "All (native)":
                            rule = resample_rule
                            agg_df = plot_df.set_index(dt_col)[y_col].resample(rule).mean().dropna().reset_index()
                            x_vals = agg_df[dt_col].astype(str)
                            y_vals = agg_df[y_col]
                        else:
                            # LTTB downsample only for very large native datasets
                            if len(plot_df) > 5000:
                                x_raw = plot_df[dt_col].values
                                y_raw = plot_df[y_col].values.astype(float)
                                x_ds, y_ds = lttb_downsample(x_raw, y_raw, 5000)
                                x_vals = x_ds.astype(str)
                                y_vals = y_ds
                            else:
                                x_vals = plot_df[dt_col].astype(str)
                                y_vals = plot_df[y_col]

                        # Add range bar if requested
                        if show_rangebars:
                            # Show ±5% of y range as a semi-transparent band
                            y_mid = y_vals.mean()
                            y_std = y_vals.std()
                            y_lo = y_mid - 2 * y_std
                            y_hi = y_mid + 2 * y_std
                            fig.add_trace(go.Scatter(
                                x=x_vals.tolist() + x_vals.tolist()[::-1],
                                y=y_lo + [y_hi] * len(y_vals) + ([y_lo] if len(y_vals) else []),
                                fill='toself',
                                fillcolor=f'rgba{tuple(list(px.colors.hex_to_rgb(color)) + [0.1])}',
                                line=dict(color='rgba(0,0,0,0)'),
                                name=f'{trace_name} ±2σ',
                                showlegend=True,
                                hoverinfo='skip',
                            ))

                        fig.add_trace(go.Scatter(
                            x=x_vals.tolist(),
                            y=y_vals.tolist(),
                            mode='lines',
                            name=trace_name,
                            line=dict(color=color, width=1.5),
                            hovertemplate=f"<b>{fname}</b><br>{y_col}: %{{y}}<br>{dt_col}: %{{x}}<extra></extra>",
                        ))

                title_text = "Comparison" if len(y_cols) != 1 else f"{y_cols[0]} — Comparison"
                yaxis_title = "Value" if len(y_cols) != 1 else y_cols[0]
                dl_filename = "multi_timeseries_comparison.html" if len(y_cols) != 1 else f"{y_cols[0]}_timeseries_comparison.html"

                fig.update_layout(
                    title=dict(text=title_text, font_size=18),
                    template="plotly_white",
                    height=height,
                    xaxis=dict(
                        rangeslider=dict(visible=True),
                        type="date",
                        title=dt_col_last
                    ),
                    yaxis=dict(title=yaxis_title),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    ),
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Remove fixed height/width before export to ensure fullscreen in browser
                fig.update_layout(height=None, width=None)
                buf = io.StringIO()
                fig.write_html(buf, default_width='100%', default_height='100%')
                st.download_button(
                    "📥 Download HTML",
                    buf.getvalue(),
                    dl_filename,
                    "text/html",
                    key="dl_ts_multi"
                )

    # ── Scatter Plot ─────────────────────────────────────────
    if active_tab == "📊 Scatter Plot":
        st.markdown("**Compare relationships across multiple files**")

        if len(num_cols) < 2:
            st.warning("⚠️ Need at least 2 numeric columns for scatter plot.")
        else:
            sc_x = st.selectbox("X-axis", num_cols, key="sc_x_multi")
            sc_y = st.selectbox("Y-axis", num_cols, key="sc_y_multi")

            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                height = st.slider("Chart height (px)", 300, 800, 500, key="sc_h_multi")
            with col2:
                show_reg = st.checkbox("Show regression line", value=False, key="sc_reg_multi")
            with col3:
                resample_rule_sc = st.selectbox(
                    "⏱️ Resample to",
                    ["All (native)", "1min", "2min", "5min", "10min", "15min", "30min", "1h", "2h", "4h", "6h", "12h", "1D"],
                    index=["All (native)", "1min", "2min", "5min", "10min", "15min", "30min", "1h", "2h", "4h", "6h", "12h", "1D"].index(st.session_state.resample_rule),
                    key="sc_resample_k",
                    on_change=lambda: st.session_state.__setitem__("resample_rule", st.session_state.sc_resample_k),
                )

            # Max points control — only shown when resample = "All (native)"
            if resample_rule_sc == "All (native)":
                max_points = st.selectbox(
                    "📊 Max points per file",
                    [500, 1000, 2000, 5000, 10000, "All (may be slow)"],
                    index=2,
                    key="sc_max_points",
                    format_func=lambda x: str(x) if isinstance(x, int) else x,
                )
            else:
                max_points = "All (native)"

            selected_files_sc = st.multiselect(
                "📂 Select files to overlay",
                available_files,
                default=available_files[:1] if len(available_files) == 1 else available_files,
                key="sc_files"
            )

            if not selected_files_sc:
                st.info("Select at least one file above to plot.")
            else:
                fig = go.Figure()
                colors = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24

                for i, fname in enumerate(selected_files_sc):
                    ds = st.session_state.datasets[fname]
                    df_plot = ds["df"]

                    if sc_x not in df_plot.columns or sc_y not in df_plot.columns:
                        st.warning(f"`{fname}` missing `{sc_x}` or `{sc_y}` — skipped.")
                        continue

                    # Detect datetime column before slicing
                    dt_col = None
                    for dc in ["Datetime", "datetime", "Date", "date", "Time", "time"]:
                        if dc in df_plot.columns:
                            dt_col = dc
                            break

                    # Include dt_col in slice so resample has access to it
                    if dt_col and dt_col not in [sc_x, sc_y]:
                        plot_df = df_plot[[dt_col, sc_x, sc_y]].dropna(subset=[sc_x, sc_y])
                    else:
                        plot_df = df_plot[[sc_x, sc_y]].dropna(subset=[sc_x, sc_y])

                    # Apply resampling (same logic as time series plot)
                    if resample_rule_sc != "All (native)" and dt_col:
                        rule = resample_rule_sc
                        plot_df = plot_df.set_index(dt_col)[[sc_x, sc_y]].resample(rule).mean().dropna().reset_index()
                        x_vals = plot_df[sc_x].values
                        y_vals = plot_df[sc_y].values
                    elif max_points != "All (may be slow)" and len(plot_df) > max_points:
                        # LTTB downsampling for large native datasets
                        x_raw = plot_df[sc_x].values.astype(float)
                        y_raw = plot_df[sc_y].values.astype(float)
                        x_ds, y_ds = lttb_downsample(x_raw, y_raw, max_points)
                        x_vals = x_ds
                        y_vals = y_ds
                    else:
                        x_vals = plot_df[sc_x].values
                        y_vals = plot_df[sc_y].values

                    color = colors[i % len(colors)]

                    fig.add_trace(go.Scatter(
                        x=x_vals.tolist(),
                        y=y_vals.tolist(),
                        mode='markers',
                        name=fname,
                        marker=dict(color=color, size=6, opacity=0.6),
                        hovertemplate=f"<b>{fname}</b><br>{sc_x}: %{{x}}<br>{sc_y}: %{{y}}<extra></extra>",
                    ))

                    # Regression line per file
                    if show_reg:
                        import numpy as np
                        x_clean = plot_df[sc_x].values
                        y_clean = plot_df[sc_y].values
                        mask = ~(np.isnan(x_clean) | np.isnan(y_clean))
                        x_c = x_clean[mask]
                        y_c = y_clean[mask]
                        if len(x_c) > 1:
                            slope, intercept = np.polyfit(x_c, y_c, 1)
                            x_range = [x_c.min(), x_c.max()]
                            fig.add_trace(go.Scatter(
                                x=x_range,
                                y=[slope * x_range[0] + intercept, slope * x_range[1] + intercept],
                                mode='lines',
                                name=f"{fname} trend",
                                line=dict(color=color, width=2, dash='dot'),
                                showlegend=True,
                                hoverinfo='skip',
                            ))

                fig.update_layout(
                    title=dict(text=f"{sc_y} vs {sc_x}", font_size=18),
                    template="plotly_white",
                    height=height,
                    xaxis=dict(title=sc_x),
                    yaxis=dict(title=sc_y),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    hovermode="closest",
                )
                st.plotly_chart(fig, use_container_width=True)

                fig.update_layout(height=None, width=None)
                buf = io.StringIO()
                fig.write_html(buf, default_width='100%', default_height='100%')
                st.download_button(
                    "📥 Download HTML",
                    buf.getvalue(),
                    f"{sc_y}_vs_{sc_x}_scatter.html",
                    "text/html",
                    key="dl_sc_multi"
                )

    # ── Statistics Plots ────────────────────────────────────
    if active_tab == "📊 Statistics":
        st.markdown("**Distribution & statistical visualizations**")

        stat_type = st.selectbox(
            "Choose chart type",
            [
                "📊 Histogram (Distribution)",
                "📦 Box Plot (by File)",
                "📦 Box Plot (Distribution by Category)",
                "🎯 Density Plot",
                "📈 Bar Chart (Categorical counts)",
                "🔥 Correlation Heatmap",
                "📊 Multi-file Histogram Overlay",
                "⏱️ Time-of-Day Box Plot",
            ],
            key="stat_type_multi"
        )

        col1, col2 = st.columns([1, 2])
        with col1:
            height = st.slider("Chart height (px)", 300, 700, 450, key="stat_h_multi")

        # ── Histogram ──────────────────────────────────────
        if stat_type == "📊 Histogram (Distribution)":
            hist_col = st.selectbox("Select column", num_cols, key="hist_col_multi")
            bins = st.slider("Number of bins", 5, 100, 30, key="hist_bins_multi")
            overlay = st.checkbox("Overlay by file (density)", value=False, key="hist_overlay")

            if overlay:
                fig = go.Figure()
                colors = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24
                for i, fname in enumerate(selected_files if 'selected_files' in locals() else list(st.session_state.datasets.keys())):
                    ds = st.session_state.datasets.get(fname)
                    if not ds or hist_col not in ds["df"].columns:
                        continue
                    plot_df = ds["df"][[hist_col]].dropna()
                    fig.add_trace(go.Histogram(
                        x=plot_df[hist_col],
                        nbinsx=bins,
                        name=fname,
                        marker_color=colors[i % len(colors)],
                        opacity=0.6,
                    ))
                fig.update_layout(
                    title=dict(text=f"Distribution of {hist_col} — by file", font_size=16),
                    template="plotly_white",
                    height=height,
                    barmode="overlay",
                    bargap=0.05,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                sel = st.selectbox("Select dataset", list(st.session_state.datasets.keys()), key="hist_ds")
                ds = st.session_state.datasets[sel]
                if hist_col in ds["df"].columns:
                    plot_df = ds["df"][[hist_col]].dropna()
                    fig = px.histogram(
                        plot_df, x=hist_col, nbins=bins,
                        title=f"Distribution of {hist_col} ({sel})",
                        height=height,
                        color_discrete_sequence=["#1E3A5F"]
                    )
                    fig.update_layout(template="plotly_white", bargap=0.1)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"`{sel}` does not have column `{hist_col}`")

        # ── Multi-file Histogram Overlay ──────────────────
        elif stat_type == "📊 Multi-file Histogram Overlay":
            hist_col = st.selectbox("Select column", num_cols, key="hist_col_overlay")
            bins = st.slider("Number of bins", 5, 100, 30, key="hist_bins_overlay2")
            avail = list(st.session_state.datasets.keys())
            sel_files = st.multiselect("Select files", avail, default=avail, key="hist_files")
            normalize = st.checkbox("Normalize (density mode)", value=True, key="hist_norm")

            if sel_files:
                fig = go.Figure()
                colors = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24
                for i, fname in enumerate(sel_files):
                    ds = st.session_state.datasets[fname]
                    if hist_col not in ds["df"].columns:
                        continue
                    plot_df = ds["df"][[hist_col]].dropna()
                    fig.add_trace(go.Histogram(
                        x=plot_df[hist_col],
                        nbinsx=bins,
                        name=fname,
                        marker_color=colors[i % len(colors)],
                        opacity=0.6,
                        histnorm="density" if normalize else "",
                    ))
                fig.update_layout(
                    title=dict(text=f"Distribution of {hist_col} — density overlay", font_size=16),
                    template="plotly_white",
                    height=height,
                    barmode="overlay",
                    bargap=0.05,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select at least one file.")

        # ── Box Plot by File ────────────────────────────────
        elif stat_type == "📦 Box Plot (by File)":
            val_col = st.selectbox("Value column (numeric)", num_cols, key="box_val_multi")
            avail = list(st.session_state.datasets.keys())
            sel_files = st.multiselect("Select files", avail, default=avail, key="box_files")

            if sel_files and val_col:
                fig = go.Figure()
                colors = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24
                all_labels = []
                all_vals = []
                for fname in sel_files:
                    ds = st.session_state.datasets[fname]
                    if val_col not in ds["df"].columns:
                        continue
                    vals = ds["df"][val_col].dropna().tolist()
                    all_labels.extend([fname] * len(vals))
                    all_vals.extend(vals)

                if all_vals:
                    box_df = pd.DataFrame({"file": all_labels, "value": all_vals})
                    fig = px.box(
                        box_df, x="file", y="value",
                        title=f"{val_col} by file",
                        height=height,
                        color="file",
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                    fig.update_layout(template="plotly_white", showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No data to plot.")
            else:
                st.info("Select files and a value column.")

        elif stat_type == "📦 Box Plot (Distribution by Category)":
            val_col = st.selectbox("Value column (numeric)", num_cols, key="box_val_cat")
            cat_col = st.selectbox("Category column", cat_cols if cat_cols else num_cols, key="box_cat_multi")

            # Use first dataset for single-category box plot
            sel = st.selectbox("Select dataset", list(st.session_state.datasets.keys()), key="box_ds_cat")
            ds = st.session_state.datasets[sel]
            if val_col in ds["df"].columns and cat_col in ds["df"].columns:
                fig = px.box(
                    ds["df"], x=cat_col, y=val_col,
                    title=f"{val_col} by {cat_col} ({sel})",
                    height=height,
                    color=cat_col,
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_layout(template="plotly_white", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Selected dataset missing columns.")

        elif stat_type == "🎯 Density Plot":
            dense_col = st.selectbox("Select column", num_cols, key="dense_col_multi")
            avail = list(st.session_state.datasets.keys())
            sel_files = st.multiselect("Select files for density", avail, default=avail, key="dense_files")

            if sel_files:
                fig = go.Figure()
                colors = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24
                for i, fname in enumerate(sel_files):
                    ds = st.session_state.datasets[fname]
                    if dense_col not in ds["df"].columns:
                        continue
                    plot_df = ds["df"][[dense_col]].dropna()
                    fig.add_trace(go.Scatter(
                        x=plot_df[dense_col],
                        y=[1] * len(plot_df),
                        mode='markers',
                        name=fname,
                        marker=dict(
                            color=colors[i % len(colors)],
                            size=4,
                            opacity=0.5,
                        ),
                    ))
                fig.update_layout(
                    title=dict(text=f"Density of {dense_col} — by file", font_size=16),
                    template="plotly_white",
                    height=height / 2,
                    showlegend=True,
                    xaxis_title=dense_col,
                    yaxis=dict(showticklabels=False, title=""),
                    hovermode="closest",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select at least one file.")

        elif stat_type == "📈 Bar Chart (Categorical counts)":
            bar_col = st.selectbox("Select column", cat_cols if cat_cols else num_cols, key="bar_col_multi")
            top_n = st.slider("Show top N categories", 5, 50, 20, key="bar_top_multi")

            sel = st.selectbox("Select dataset", list(st.session_state.datasets.keys()), key="bar_ds")
            ds = st.session_state.datasets[sel]
            if bar_col in ds["df"].columns:
                top_cats = ds["df"][bar_col].value_counts().head(top_n)
                fig = px.bar(
                    x=top_cats.index, y=top_cats.values,
                    title=f"Top {top_n} {bar_col} ({sel})",
                    height=height,
                    color=top_cats.values,
                    color_continuous_scale="Blues"
                )
                fig.update_layout(template="plotly_white", xaxis_title=bar_col, yaxis_title="Count", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"`{sel}` missing column `{bar_col}`")

        elif stat_type == "🔥 Correlation Heatmap":
            if len(num_cols) < 2:
                st.warning("⚠️ Need at least 2 numeric columns for correlation heatmap.")
            else:
                sel = st.selectbox("Select dataset", list(st.session_state.datasets.keys()), key="corr_ds")
                ds = st.session_state.datasets[sel]
                df_corr = ds["df"][num_cols].corr()
                fig = px.imshow(
                    df_corr,
                    text_auto=True,
                    title=f"Correlation Heatmap ({sel})",
                    height=height,
                    color_continuous_scale="RdBu_r",
                    aspect="auto"
                )
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

                fig.update_layout(height=None, width=None)
                buf = io.StringIO()
                fig.write_html(buf, default_width='100%', default_height='100%')
                st.download_button(
                    "📥 Download HTML",
                    buf.getvalue(),
                    "correlation_heatmap.html",
                    "text/html",
                    key="dl_corr_multi"
                )

        elif stat_type == "⏱️ Time-of-Day Box Plot":
            # ── Time-of-Day Box Plot ────────────────────────────────
            st.markdown("**Box plot of values grouped by time-of-day interval**")

            # Detect datetime column
            avail_ds = list(st.session_state.datasets.keys())
            tod_ds_sel = st.selectbox("Select dataset", avail_ds, key="tod_ds")
            tod_ds = st.session_state.datasets[tod_ds_sel]["df"]

            dt_col_tod = None
            for dc in ["Datetime", "datetime", "Date", "date", "Time", "time"]:
                if dc in tod_ds.columns:
                    dt_col_tod = dc
                    break

            if dt_col_tod is None:
                st.warning("⚠️ No datetime column found in selected dataset.")
            elif len(num_cols) == 0:
                st.warning("⚠️ No numeric columns found.")
            else:
                tod_val_col = st.selectbox("Value column", num_cols, key="tod_val")

                tod_interval = st.selectbox(
                    "⏱️ Time interval",
                    ["15min", "30min", "1H", "2H", "4H", "6H", "12H", "1D"],
                    index=2,
                    key="tod_interval",
                )

                # Map interval label to pandas DateOffset strings for dt.floor
                interval_map = {
                    "15min": "15min",
                    "30min": "30min",
                    "1H":    "1h",
                    "2H":    "2h",
                    "4H":    "4h",
                    "6H":    "6h",
                    "12H":   "12h",
                    "1D":    "1D",
                }
                rule = interval_map[tod_interval]

                # Build time-of-day label: floor datetime to interval, then format as HH:MM
                tod_df = tod_ds.copy()
                tod_df[dt_col_tod] = pd.to_datetime(tod_df[dt_col_tod], errors="coerce")
                tod_df = tod_df.dropna(subset=[dt_col_tod, tod_val_col])

                # Floor to interval start, then extract time label
                tod_df["_tod_bucket"] = tod_df[dt_col_tod].dt.floor(rule)
                tod_df["_tod_label"] = tod_df["_tod_bucket"].dt.strftime("%H:%M")

                # Group by time bucket
                grouped = tod_df.groupby("_tod_label")[tod_val_col]

                # Build box plot data
                box_labels = []
                box_values = []
                for label, vals in grouped:
                    if len(vals) > 0:
                        box_labels.append(label)
                        box_values.append(vals.values)

                if not box_labels:
                    st.warning("No data to display for the selected interval.")
                else:
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    for i, (label, vals) in enumerate(zip(box_labels, box_values)):
                        fig.add_trace(go.Box(
                            y=vals,
                            name=label,
                            width=0.5,
                            boxpoints= "outliers", # "all', "outliers", "suspectedoutliers", or False
                            # marker=dict(color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]),
                            marker_color='royalblue',
                            text=[label],
                            hoverinfo="y+name",
                        ))

                    fig.update_layout(
                        title=dict(text=f"{tod_val_col} — Time-of-Day ({tod_interval}) | {tod_ds_sel}", font_size=16),
                        template="plotly_white",
                        height=height,
                        xaxis=dict(title="Time of Day", tickangle=45),
                        yaxis=dict(title=tod_val_col),
                        showlegend=False,
                        boxmode="group",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    fig.update_layout(height=None, width=None)
                    buf = io.StringIO()
                    fig.write_html(buf, default_width='100%', default_height='100%')
                    st.download_button(
                        "📥 Download HTML",
                        buf.getvalue(),
                        f"time_of_day_boxplot_{tod_val_col}_{tod_interval}.html",
                        "text/html",
                        key="dl_tod_box"
                    )

    st.divider()

    # ══════════════════════════════════════════════════════════
    # 3D VISUALIZATION
    # ══════════════════════════════════════════════════════════
    if active_tab == "🌐 3D Visualization":
        st.markdown("**3D scatter & line plots — explore relationships across three dimensions**")

        if len(num_cols) < 2:
            st.warning("⚠️ Need at least 2 numeric columns for 3D visualization.")
        else:
            col1, col2 = st.columns([1, 1])
            with col1:
                plot_mode = st.selectbox(
                    "Plot mode",
                    ["3D Scatter", "3D Line / Time Series"],
                    key="3d_mode"
                )
            with col2:
                height_3d = st.slider("Chart height (px)", 400, 900, 600, key="3d_height")

            avail_files_3d = list(st.session_state.datasets.keys())

            # ── Shared marker/line style controls ──
            if plot_mode == "3D Scatter":
                st.markdown("**Marker Style**")
                m_col1, m_col2, m_col3, m_col4 = st.columns([1, 1, 1, 1])
                with m_col1:
                    marker_size = st.slider("Marker size", 1, 10, 2, key="3d_marker_size")
                with m_col2:
                    marker_opacity = st.slider("Opacity", 0.0, 1.0, 0.5, step=0.05, key="3d_marker_opacity")
                with m_col3:
                    edge_width = st.slider("Edge width", 0.0, 3.0, 0.0, step=0.1, key="3d_edge_width")
                with m_col4:
                    edge_color = st.color_picker("Edge color", "#000000", key="3d_edge_color")

            if plot_mode == "3D Scatter":
                col_x, col_y, col_z = st.columns([1, 1, 1])
                with col_x:
                    x_axis = st.selectbox("X-axis", num_cols, key="3d_x")
                with col_y:
                    y_axis = st.selectbox("Y-axis", num_cols, key="3d_y")
                with col_z:
                    z_axis = st.selectbox("Z-axis", num_cols, key="3d_z")

                # File selection for overlay
                selected_3d = st.multiselect(
                    "📂 Select files to overlay",
                    avail_files_3d,
                    default=avail_files_3d[:1] if len(avail_files_3d) == 1 else avail_files_3d,
                    key="3d_files"
                )

                # Downsample by multiplier — e.g. "2×" means halve the rows, "5×" means keep 1/5
                ref_fname = avail_files_3d[0] if avail_files_3d else None
                row_count = len(st.session_state.datasets[ref_fname]["df"]) if ref_fname else 0
                multiplier_labels = []
                for m in [2, 3, 5, 10, 20, 50]:
                    result_rows = row_count // m
                    if result_rows >= 10:
                        multiplier_labels.append(f"{m}×  →  ~{result_rows:,} pts")
                multiplier_labels.insert(0, "All (native)")

                default_idx = 0
                selected_mult = st.selectbox(
                    "🔽 Downsample multiplier",
                    multiplier_labels,
                    index=default_idx,
                    key="3d_downsample_mult",
                )

                if not selected_3d:
                    st.info("Select at least one file.")
                else:
                    # Parse multiplier from label like "2×  →  ~50,000 pts"
                    mult_val = 0
                    if selected_mult != "All (native)":
                        mult_val = int(selected_mult.split("×")[0].strip())

                    fig = go.Figure()
                    colors = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24

                    for i, fname in enumerate(selected_3d):
                        ds = st.session_state.datasets[fname]
                        df_p = ds["df"]

                        if x_axis not in df_p.columns or y_axis not in df_p.columns or z_axis not in df_p.columns:
                            st.warning(f"`{fname}` missing required columns — skipped.")
                            continue

                        plot_df = df_p[[x_axis, y_axis, z_axis]].dropna()

                        # Downsample with LTTB if multiplier is set
                        if mult_val > 0:
                            n_target = max(3, len(plot_df) // mult_val)
                            
                            def safe_to_float(serie):
                                if pd.api.types.is_datetime64_any_dtype(serie):
                                    return serie.values.astype(np.int64) / 1_000_000.0
                                return serie.values.astype(float)
                                
                            x_raw = safe_to_float(plot_df[x_axis])
                            y_raw = safe_to_float(plot_df[y_axis])
                            z_raw = safe_to_float(plot_df[z_axis])
                            idx_s, _ = lttb_downsample(
                                np.arange(len(x_raw), dtype=float),
                                x_raw,
                                n_target
                            )
                            idx_s = idx_s.astype(int)
                            x_vals = x_raw[idx_s]
                            y_vals = y_raw[idx_s]
                            z_vals = z_raw[idx_s]
                        else:
                            x_vals = plot_df[x_axis].values
                            y_vals = plot_df[y_axis].values
                            z_vals = plot_df[z_axis].values

                        color = colors[i % len(colors)]
                        fig.add_trace(go.Scatter3d(
                            x=x_vals.tolist(),
                            y=y_vals.tolist(),
                            z=z_vals.tolist(),
                            mode='markers',
                            name=fname,
                            marker=dict(
                                color=color,
                                size=marker_size,
                                opacity=marker_opacity,
                                line=dict(width=edge_width, color=edge_color)
                            ),
                            hovertemplate=(
                                f"<b>{fname}</b><br>"
                                f"{x_axis}: %{{x:.3f}}<br>"
                                f"{y_axis}: %{{y:.3f}}<br>"
                                f"{z_axis}: %{{z:.3f}}<extra></extra>"
                            ),
                        ))

                    fig.update_layout(
                        title=dict(text=f"3D Scatter: {x_axis} × {y_axis} × {z_axis}", font_size=16),
                        template="plotly",
                        height=height_3d,
                        scene=dict(
                            xaxis_title=x_axis,
                            yaxis_title=y_axis,
                            zaxis_title=z_axis,
                            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                        ),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                        margin=dict(l=0, r=0, b=0, t=40),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    fig.update_layout(height=None, width=None)
                    buf = io.StringIO()
                    fig.write_html(buf, default_width='100%', default_height='100%')
                    st.download_button(
                        "📥 Download HTML",
                        buf.getvalue(),
                        f"3d_scatter_{x_axis}_{y_axis}_{z_axis}.html",
                        "text/html",
                        key="dl_3d_scatter"
                    )

            else:  # 3D Line / Time Series
                st.markdown("**Marker Style**")
                m_col1_time, m_col2_time = st.columns([1, 1])
                with m_col1_time:
                    marker_size = st.slider("Marker size", 1, 10, 2, key="3d_marker_size")
                with m_col2_time:
                    marker_opacity = st.slider("Opacity", 0.0, 1.0, 0.5, step=0.05, key="3d_marker_opacity")

                # Detect datetime column first
                dt_col_3d = None
                for dc in ["Datetime", "datetime", "Date", "date", "Time", "time"]:
                    for ds in st.session_state.datasets.values():
                        if dc in ds["df"].columns:
                            dt_col_3d = dc
                            break
                    if dt_col_3d:
                        break

                if not dt_col_3d:
                    st.warning("⚠️ No datetime column found — 3D Line requires a time/date axis.")
                else:
                    # Let user pick two numeric axes for Y and Z (X = time)
                    col_y3, col_z3 = st.columns([1, 1])
                    with col_y3:
                        y3_axis = st.selectbox("Y-axis (numeric)", num_cols, key="3d_line_y")
                    with col_z3:
                        z3_axis = st.selectbox("Z-axis (numeric)", num_cols, key="3d_line_z")

                    # Resampling option
                    resample_3d = st.selectbox(
                        "⏱️ Resample to",
                        ["All (native)", "1min", "2min", "5min", "10min", "15min", "30min", "1h", "2h", "4h", "6h", "12h", "1D"],
                        index=0,
                        key="3d_resample",
                    )

                    selected_3d_line = st.multiselect(
                        "📂 Select files to overlay",
                        avail_files_3d,
                        default=avail_files_3d[:1] if len(avail_files_3d) == 1 else avail_files_3d,
                        key="3d_line_files"
                    )

                    if not selected_3d_line:
                        st.info("Select at least one file.")
                    else:
                        fig = go.Figure()
                        colors = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24

                        for i, fname in enumerate(selected_3d_line):
                            ds = st.session_state.datasets[fname]
                            df_p = ds["df"]

                            if dt_col_3d not in df_p.columns:
                                st.warning(f"`{fname}` has no datetime column — skipped.")
                                continue
                            if y3_axis not in df_p.columns or z3_axis not in df_p.columns:
                                st.warning(f"`{fname}` missing `{y3_axis}` or `{z3_axis}` — skipped.")
                                continue

                            plot_df = df_p[[dt_col_3d, y3_axis, z3_axis]].dropna()
                            plot_df = plot_df.sort_values(dt_col_3d)

                            # Resample if requested
                            if resample_3d != "All (native)":
                                rule = resample_3d
                                plot_df = plot_df.set_index(dt_col_3d)[[y3_axis, z3_axis]].resample(rule).mean().dropna().reset_index()

                            x_vals = pd.to_datetime(plot_df[dt_col_3d]) #.values.astype(np.int64) // 10**6
                            y_vals = plot_df[y3_axis].values
                            z_vals = plot_df[z3_axis].values

                            color = colors[i % len(colors)]
                            fig.add_trace(go.Scatter3d(
                                x=x_vals.tolist(),
                                y=y_vals.tolist(),
                                z=z_vals.tolist(),
                                mode='markers',
                                name=fname,
                                marker=dict(
                                    color=color,
                                    size=marker_size,
                                    opacity=marker_opacity,
                                ),
                                # line=dict(color=color, width=2.5),
                                hovertemplate=(
                                    f"<b>{fname}</b><br>"
                                    f"Time: %{{x}}<br>"
                                    f"{y3_axis}: %{{y:.3f}}<br>"
                                    f"{z3_axis}: %{{z:.3f}}<extra></extra>"
                                ),
                            ))

                        fig.update_layout(
                            title=dict(text=f"3D Line: {dt_col_3d} × {y3_axis} × {z3_axis}", font_size=16),
                            template="plotly",
                            height=height_3d,
                            scene=dict(
                                xaxis_title=dt_col_3d,
                                yaxis_title=y3_axis,
                                zaxis_title=z3_axis,
                                camera=dict(eye=dict(x=1.2, y=1.5, z=1.0)),
                            ),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                            margin=dict(l=0, r=0, b=0, t=40),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        fig.update_layout(height=None, width=None)
                        buf = io.StringIO()
                        fig.write_html(buf, default_width='100%', default_height='100%')
                        st.download_button(
                            "📥 Download HTML",
                            buf.getvalue(),
                            f"3d_line_{y3_axis}_{z3_axis}.html",
                            "text/html",
                            key="dl_3d_line"
                        )

    st.divider()

    # ── Export ──────────────────────────────────────────────
    with st.expander("📤 Export Data"):
        if len(st.session_state.datasets) == 1:
            fname = list(st.session_state.datasets.keys())[0]
            ds = st.session_state.datasets[fname]
            st.download_button(
                "📄 Download as CSV",
                ds["df"].to_csv(index=False).encode(),
                f"processed_{fname}.csv",
                "text/csv"
            )
        else:
            # Multi-dataset: offer individual downloads
            sel = st.selectbox("Select dataset to export", list(st.session_state.datasets.keys()), key="export_ds")
            ds = st.session_state.datasets[sel]
            st.download_button(
                "📄 Download as CSV",
                ds["df"].to_csv(index=False).encode(),
                f"processed_{sel}.csv",
                "text/csv",
                key="dl_csv_multi"
            )

    st.caption("Built with Streamlit + Plotly · Bruce's Data Viz Tool")

else:
    st.info("👆 Upload one or more files above to get started")
    st.stop()
