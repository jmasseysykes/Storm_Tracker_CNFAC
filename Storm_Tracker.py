import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import os
import gc
import certifi

# Google Analytics 4 tracking
GA_MEASUREMENT_ID = "G-FEZMNSLQ1R"
ga_script = f"""
<script async src="https://www.googletagmanager.com/gtag/js?id={GA_MEASUREMENT_ID}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());
  gtag('config', '{GA_MEASUREMENT_ID}');
</script>
"""
# Inject GA script
st.components.v1.html(ga_script, height=0)

st.set_page_config(page_title="Storm Tracker", layout="wide", page_icon="❄️")

# Version & status note
st.caption("Storm Tracker v1.0 • Last updated: March 2026 • Stable & in active use")

st.title("SNOTEL Storm Tracker — Avalanche Forecasting Tool")

st.set_page_config(page_title="CNFAC Storm Tracker", layout="wide")

# Load SNOTEL master list for name lookup
@st.cache_data
def load_snotel_list():
    csv_path = 'SNOTEL_station_list.csv'
    if not os.path.exists(csv_path):
        st.error("SNOTEL_station_list.csv not found in app folder. Download from NRCS or CUAHSI.")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df['ID'] = df['site_name'].str.extract(r'\((\d+)\)', expand=False).astype(str)
    df['display_name'] = df['site_name'] + " - " + df['state']
    df = df[['display_name', 'ID', 'state']].dropna()
    return df

snotel_list = load_snotel_list()
snotel_list['display_name'] = snotel_list['display_name'].fillna("Unknown Station").astype(str)
snotel_list = snotel_list[snotel_list['display_name'].str.strip() != ""]
snotel_list = snotel_list.reset_index(drop=True)

st.markdown("""
Look up a SNOTEL station by Name, ID, or State to fetch full historical data or upload a CSV of SNOTEL data.  
App computes storm totals, histograms, percentiles, and overlays current values.
""")

# Sidebar inputs
with st.sidebar:
    st.header("Settings")
    data_source = st.radio("Data Source", ["Fetch from SNOTEL API", "Upload CSV"])
   
    if data_source == "Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload CSV",
                type=["csv"],
                key="upload_csv",
                help="Must contain at least two columns: 'Date' (any parseable date format) and 'SWE' (numeric Snow Water Equivalent values in inches). Extra columns are ignored. Download a processed example from the app to see the expected format."
            )
            # Helpful expander for CSV format
            with st.expander("CSV Upload Format Help", expanded=False):
                st.markdown("""
                **Required columns:**
                - `Date`: Dates in any parseable format (e.g., YYYY-MM-DD, MM/DD/YYYY, etc.)
                - `SWE`: Numeric Snow Water Equivalent values (inches)

                **What happens to extra columns?**
                - Ignored automatically (PREC, TMAX, TMIN, etc. are dropped).

                **Tips:**
                - Column names are case-sensitive ('Date' and 'SWE' exactly).
                - Use the "Download Processed CSV" button (after loading a station) to see an example of the cleaned format.
                - If upload fails, check for missing columns, bad dates, or non-numeric SWE values.
                """)
            triplet = None
            station_name = "Uploaded Data"
    else:
        turnagain_display = "Turnagain Pass (954) - AK"
        default_index = 0
        if turnagain_display in snotel_list['display_name'].values:
            default_index = int(snotel_list[snotel_list['display_name'] == turnagain_display].index[0]) + 1
        
        selected_display = st.selectbox(
            "Search / Select Station",
            options=[""] + [str(name) for name in snotel_list['display_name'].tolist()],
            index=default_index,
            help="Type to search or select a SNOTEL station",
            key="station_search_selectbox"
        )
       
        if selected_display:
            selected_row = snotel_list[snotel_list['display_name'] == selected_display].iloc[0]
            station_name = selected_row['display_name'].split(' (')[0]
            station_id = str(selected_row['ID'])
            state = selected_row['state']
            triplet = f"{station_id}:{state}:SNTL"
            st.success(f"Selected: {station_name} ({triplet})")
        else:
            st.warning("Please select a station to fetch data.")
            triplet = None
            station_name = None

    ref_date = st.date_input(
        "Reference Date for Current Storm",
        value=datetime.today().date(),
        min_value=datetime(1960, 1, 1).date(),
        max_value=datetime.today().date(),
        key="ref_date_input"
    )
    ref_date_str = ref_date.strftime('%Y-%m-%d')

    use_log_scale = st.checkbox("Log Y-Axis (better for high-percentile tails)", value=False)
    single_color_mode = st.checkbox("Single-Color Test Mode", value=False, key="single_color_toggle")

# ────────────────────────────────────────────────
# CACHED DATA LOAD & PROCESSING
# ────────────────────────────────────────────────
@st.cache_data(ttl=7200, max_entries=100, show_spinner="Loading & processing station data…")
def load_and_process_station_data(data_source, uploaded_file, triplet, ref_date_str):
    if data_source == "Upload CSV" and uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Explicit check for required columns
            if 'Date' not in df.columns or 'SWE' not in df.columns:
                raise ValueError("CSV must contain 'Date' and 'SWE' columns.")
        except Exception as e:
            st.error(f"Upload failed: {str(e)}\n\n**Expected format:** CSV with columns 'Date' (dates) and 'SWE' (numeric values in inches). Extra columns are ignored. Check column names (case-sensitive) and ensure no missing/invalid data.")
            return None
        raw_mode = 'delta_SWE' not in df.columns
    else:
        st.info(f"Fetching full SNOTEL data for {triplet}...")
        if triplet is None:
            return None
        try:
            url = f"https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customSingleStationReport/daily/{triplet}%7Cid=%22%22%7Cname/POR_BEGIN,POR_END/WTEQ::value,PREC::value,TMAX::value,TMIN::value,TAVG::value,PRCP::value"
            #response = requests.get(url)
            response = requests.get(url, verify=False)   # Temporary bypass for SSL issue
            response.raise_for_status()
            df = pd.read_csv(pd.io.common.StringIO(response.text), comment='#', skip_blank_lines=True)
            df.columns = ['Date', 'SWE', 'PREC', 'TMAX', 'TMIN', 'TAVG', 'PRCP']
            raw_mode = True
        except Exception as e:
            st.error(f"API error: {e}. Check ID/state (e.g., 954:AK).")
            return None

    keep_cols = ['Date', 'SWE']
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    for col in df.columns:
        if col != 'Date':
            df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    if raw_mode:
        df = df.set_index('Date')
        df = df['SWE'].resample('D').asfreq().to_frame(name='SWE')

        df['delta_SWE'] = (df['SWE'] - df['SWE'].shift(1)).clip(lower=0)
        df['3-day'] = (df['SWE'] - df['SWE'].shift(3)).clip(lower=0)
        df['7-day'] = (df['SWE'] - df['SWE'].shift(7)).clip(lower=0)
        df['10-day'] = (df['SWE'] - df['SWE'].shift(10)).clip(lower=0)
        df = df.fillna(0)

        for window, col in zip([1, 3, 7, 10], ['delta_SWE', '3-day', '7-day', '10-day']):
            gap_mask = df['SWE'].rolling(window).count() < window
            df.loc[gap_mask, col] = 0

        df = df.reset_index()

    return df

uploaded_file = uploaded_file if data_source == "Upload CSV" else None

try:
    if (data_source == "Upload CSV" and uploaded_file is not None) or (data_source == "Fetch from SNOTEL API" and triplet is not None):
        df = load_and_process_station_data(data_source, uploaded_file, triplet, ref_date_str)
        if df is None:
            st.stop()
    else:
        df = None

    if df is not None:
        ref_date = pd.to_datetime(ref_date_str)
        historical_df = df[df['Date'] <= ref_date]
        if historical_df.empty:
            st.warning("Reference date before data record. Using latest available data.")
            historical_df = df
        current_row = historical_df.iloc[-1]

        start_year = df['Date'].dt.year.min()
        end_year = df['Date'].dt.year.max()

        st.header("Dataset Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Station:** {station_name}")
            st.write(f"**Date range:** {df['Date'].min().date()} → {df['Date'].max().date()}")
            st.write(f"**Total days:** {len(df):,}")
            st.markdown(f"**Current Storm Totals (as of {ref_date.date()})**  \n1-day: {current_row['delta_SWE']:.2f}\" | 3-day: {current_row['3-day']:.2f}\" | 7-day: {current_row['7-day']:.2f}\" | 10-day: {current_row['10-day']:.2f}\"")
        with col2:
            columns = {'1-day': 'delta_SWE', '3-day': '3-day', '7-day': '7-day', '10-day': '10-day'}
            summary_data = []
            for period, col in columns.items():
                miss = df[col].isna().sum()
                neg = (df[col] < 0).sum()
                zero = (df[col] == 0).sum()
                pos = (df[col] > 0).sum()
                summary_data.append({"Period": period, "Missing": miss, "Negative": neg, "Zero": zero, "Storm days (>0)": pos})
            st.table(pd.DataFrame(summary_data))

        st.header("Historical Storm Distributions")
        percentiles = [50, 75, 90, 95, 99]
        colors = ['#2c7bb6', '#abd9e9', '#ffed82', '#fdae61', '#d7191c']  # Updated 90th to better yellow
        bins_per_inch = 10
        rwidth = 0.99
        edge_lw = 0.2

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        axes = axes.ravel()
        legend_patches = [plt.Rectangle((0,0),1,1, color=c) for c in colors]
        legend_labels = [f'{p}th percentile' for p in percentiles]
        current_values = current_row

        for idx, (period, col) in enumerate(columns.items()):
            ax = axes[idx]
            storm = df[[col, 'Date']].dropna()
            storm = storm[storm[col] > 0].copy()
            values = storm[col]
            if len(values) < 50:
                ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center', fontsize=14)
                continue
            p_vals = np.percentile(values, percentiles)
            record_val = values.max()
            record_date = storm.loc[values.idxmax(), 'Date'].strftime('%b %d, %Y')
            n_storms = len(values)
            max_plot = values.quantile(1.0)
            bin_width = 1.0 / bins_per_inch
            bins = np.arange(0, max_plot + bin_width, bin_width)

            if single_color_mode:
                ax.hist(values, bins=bins, rwidth=rwidth,
                        color='#1f77b4', edgecolor='black', linewidth=edge_lw, alpha=0.9)
            else:
                counts, edges, patches = ax.hist(values, bins=bins, rwidth=rwidth,
                                                 color='lightgray', edgecolor='white', linewidth=0, zorder=2)
                bounds = [0] + list(p_vals) + [values.max() + 0.01]
                segment_colors = colors + [colors[-1]]
                for i, patch in enumerate(patches):
                    center = (edges[i] + edges[i+1]) / 2
                    for j in range(1, len(bounds)):
                        if center < bounds[j]:
                            patch.set_facecolor(segment_colors[j-1])
                            break
                    patch.set_edgecolor('black')
                    patch.set_linewidth(edge_lw)
                record_bin_idx = np.searchsorted(edges, record_val) - 1
                if 0 <= record_bin_idx < len(patches):
                    patches[record_bin_idx].set_facecolor('black')
                    patches[record_bin_idx].set_edgecolor('#c1121f')
                    patches[record_bin_idx].set_linewidth(1.5)

            ylim_max = ax.get_ylim()[1]

            # Draw all percentile lines first
            for p, val, c in zip(percentiles, p_vals, colors):
                ax.axvline(val, color=c, linestyle='--', linewidth=3, zorder=3)

            # Then place labels with fixed, even stagger (no current dependency)
            for i, (p, val, c) in enumerate(zip(percentiles, p_vals, colors)):
                # Consistent vertical spread: starts at 94% and steps down evenly
                y_pos = ylim_max * (0.94 - 0.15 * i)   # decreases by 15% per step

                ax.text(
                    val, y_pos,
                    f'{p}th\n{val:.2f}"',
                    rotation=90,
                    va='top',
                    ha='center',
                    fontsize=9.5,
                    bbox=dict(
                        boxstyle="round,pad=0.35",
                        facecolor='white',
                        alpha=0.92,
                        edgecolor=c,
                        linewidth=2
                    ),
                    zorder=10
                )
            current = current_values[col]
            ax.axvline(current, color='black', linestyle='-', linewidth=4, alpha=0.7, zorder=5)

            stats_text = f"N = {n_storms:,} storm days\nRecord: {record_val:.2f}\"\n{record_date}"
            ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, va='top', ha='right', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f8f8", alpha=0.92))

            ax.set_title(f"{period.upper()} SWE — Storm Days Only (>0\")",
                         fontsize=14, pad=20)
            ax.set_xlabel("Snow Water Equivalent (inches)")
            ax.set_ylabel("Number of Storm Events")
            ax.grid(True, alpha=0.3)

            if use_log_scale:
                ax.set_yscale('log')
                ax.set_ylabel("Number of Storm Events (log scale)")

        title_text = f'{station_name}'
        if single_color_mode:
            title_text += " (Single-Color Test Mode)"
        title_text += f' — Historical Storm Distributions ({start_year}–{end_year})\nOnly days with measurable snowfall (>0" SWE)'
        fig.suptitle(title_text, fontsize=18, fontweight='bold', y=0.98)

        if not single_color_mode:
            legend_patches.append(plt.Line2D([0], [0], color='black', linestyle='-', linewidth=4, alpha=0.7))
            legend_labels.append('Current Storm')
            fig.legend(legend_patches, legend_labels,
                       loc='upper center', bbox_to_anchor=(0.5, 0.935),
                       ncol=6, fancybox=True, frameon=False, shadow=False,
                       fontsize=12, title_fontsize=12)

        plt.tight_layout(rect=[0, 0.04, 1, 0.96])
        st.pyplot(fig)
        plt.close(fig)
        gc.collect()

        st.header("Downloads")
        col1, col2 = st.columns(2)
        today = datetime.now().strftime("%Y-%m-%d")
        safe_name = station_name.replace(" ", "_")
        mode = "SingleColor" if single_color_mode else "MultiColor"
        csv_data = df.to_csv(index=False).encode('utf-8')
        col1.download_button(
            label="Download Processed CSV",
            data=csv_data,
            file_name=f"{safe_name}_Storm_Totals_{start_year}-{end_year}.csv",
            mime="text/csv"
        )
        fig.savefig("temp_plot.png", dpi=200, bbox_inches='tight', facecolor='white')
        with open("temp_plot.png", "rb") as f:
            col2.download_button(
                label="Download Histogram PNG",
                data=f,
                file_name=f"{safe_name}_StormTracker_{mode}_{today}.png",
                mime="image/png"
            )

    else:
        st.info("👆 Select a SNOTEL station from the sidebar or upload a daily CSV to begin. ❄️")

    # About & Feedback expander
    with st.expander("About & Feedback", expanded=False):
        st.markdown("""
        **Storm Tracker** — Original Storm Tracker spreadsheet developed by Jason Konigsberg and Spencer Logan at CAIC and adapted for CNFAC
        by Mik Dalpes and Andrew Schauer. Web app and histogram visualizations built by John Sykes. Uses public NRCS SNOTEL data.  
        Feedback, bugs, or station requests?  
        - GitHub issues: https://github.com/jmasseysykes/Storm_Tracker_CNFAC/issues  
        - Or reach out directly if you prefer to john@chugachavalanche.org.  
        Thanks for using it! ❄️
        """)

except Exception as e:
    st.error(f"Something went wrong: {str(e)}")
    st.info("Try refreshing the page or selecting a different station. If it persists, report on GitHub: https://github.com/jmasseysykes/Storm_Tracker_CNFAC/issues")