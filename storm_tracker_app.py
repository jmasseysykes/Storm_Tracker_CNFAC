import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import os

st.set_page_config(page_title="Storm Tracker", layout="wide")
st.title("SNOTEL Storm Tracker — Avalanche Forecasting Tool")
# Load SNOTEL master list for name lookup
@st.cache_data
def load_snotel_list():
    csv_path = 'SNOTEL_station_list.csv'  # Save your attached CSV in the same folder as this script
    if not os.path.exists(csv_path):
        st.error("SNOTEL_station_list.csv not found in app folder. Download from NRCS or CUAHSI.")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    # Parse ID from site_name (e.g., "Turnagain Pass (954)" → 954)
    df['ID'] = df['site_name'].str.extract(r'\((\d+)\)', expand=False).astype(str)
    # Create display name: "Turnagain Pass (954) - AK"
    df['display_name'] = df['site_name'] + " - " + df['state']
    df = df[['display_name', 'ID', 'state']].dropna()
    return df

snotel_list = load_snotel_list()
# Force clean strings and remove any bad rows
snotel_list['display_name'] = snotel_list['display_name'].fillna("Unknown Station").astype(str)
snotel_list = snotel_list[snotel_list['display_name'].str.strip() != ""]  # Drop empty
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
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="upload_csv")
    else:
        # Station name lookup (autocomplete/search)
        # Find default index for Turnagain Pass
        turnagain_display = "Turnagain Pass (954) - AK"  # Confirm exact string from your CSV
        default_index = 0
        if turnagain_display in snotel_list['display_name'].values:
            default_index = int(snotel_list[snotel_list['display_name'] == turnagain_display].index[0]) + 1  # +1 for empty "" at start
        
        selected_display = st.selectbox(
            "Search / Select Station",
            options=[""] + [str(name) for name in snotel_list['display_name'].tolist()],  # Force plain strings
            index=default_index,
            help="Type to search or select a SNOTEL station",
            key="station_search_selectbox"
        )
        
        if selected_display:
            selected_row = snotel_list[snotel_list['display_name'] == selected_display].iloc[0]
            station_name = selected_row['display_name'].split(' (')[0]  # Clean name
            station_id = str(selected_row['ID'])
            state = selected_row['state']
            triplet = f"{station_id}:{state}:SNTL"
            st.success(f"Selected: {station_name} ({triplet})")
        else:
            st.warning("Please select a station to fetch data.")
            triplet = None  # Prevent fetch

    # Reference date for current storm values
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

if (data_source == "Upload CSV" and uploaded_file) or (data_source == "Fetch from SNOTEL API" and station_id):
    # -------------------------------
    # LOAD DATA
    # -------------------------------
    if data_source == "Upload CSV":
        try:
            df = pd.read_csv(uploaded_file)
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, header=None, names=['Date', 'SWE'])
        raw_mode = 'delta_SWE' not in df.columns
    else:
        st.info(f"Fetching full SNOTEL data for {triplet}...")
        if triplet is None:
            st.warning("Please select a station first.")
            st.stop()
        try:
            url = f"https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customSingleStationReport/daily/{triplet}%7Cid=%22%22%7Cname/POR_BEGIN,POR_END/WTEQ::value,PREC::value,TMAX::value,TMIN::value,TAVG::value,PRCP::value"
            response = requests.get(url)
            response.raise_for_status()
            df = pd.read_csv(pd.io.common.StringIO(response.text), comment='#', skip_blank_lines=True)
            df.columns = ['Date', 'SWE', 'PREC', 'TMAX', 'TMIN', 'TAVG', 'PRCP']
            df['Date'] = pd.to_datetime(df['Date'])
            df['SWE'] = pd.to_numeric(df['SWE'], errors='coerce')
            df = df.dropna(subset=['SWE'])
            raw_mode = True
        except Exception as e:
            st.error(f"API error: {e}. Check ID/state (e.g., 954:AK).")
            st.stop()

    # -------------------------------
    # CLEAN AND COMPUTE — DATE-BASED WITH GAP DETECTION (NO FILLING)
    # -------------------------------
    df.replace(['', '#NA', '#VALUE!', 'NA'], np.nan, inplace=True)
    for col in df.columns:
        if col != 'Date':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).copy()

    if raw_mode:
        df = df.sort_values('Date').reset_index(drop=True)
        df = df.set_index('Date')
        # Resample to daily (no ffill — NaN for gaps)
        df = df['SWE'].resample('D').asfreq().to_frame(name='SWE')
        # Compute net gain only if no NaN in window (reset on gaps)
        df['delta_SWE'] = (df['SWE'] - df['SWE'].shift(1)).clip(lower=0)
        df['3-day'] = (df['SWE'] - df['SWE'].shift(3)).clip(lower=0)
        df['7-day'] = (df['SWE'] - df['SWE'].shift(7)).clip(lower=0)
        df['10-day'] = (df['SWE'] - df['SWE'].shift(10)).clip(lower=0)
        df = df.fillna(0)
        # Gap detection: Set totals to 0 if any NaN in window
        for window, col in zip([1, 3, 7, 10], ['delta_SWE', '3-day', '7-day', '10-day']):
            gap_mask = df['SWE'].rolling(window).count() < window  # NaN in window
            df.loc[gap_mask, col] = 0
        df = df.reset_index()

    # -------------------------------
    # CURRENT STORM VALUES (from user reference date)
    # -------------------------------
    ref_date = pd.to_datetime(ref_date_str)
    # Filter to data up to reference date
    historical_df = df[df['Date'] <= ref_date]
    if historical_df.empty:
        st.warning("Reference date before data record. Using latest available data.")
        historical_df = df
    current_row = historical_df.iloc[-1]

    # -------------------------------
    # YEAR RANGE
    # -------------------------------
    start_year = df['Date'].dt.year.min()
    end_year = df['Date'].dt.year.max()

    # -------------------------------
    # SUMMARY
    # -------------------------------
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

    # -------------------------------
    # PLOT
    # -------------------------------
    st.header("Historical Storm Distributions")
    percentiles = [50, 75, 90, 95, 99]
    colors = ['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c']

    # Hardcoded defaults
    bins_per_inch = 10
    rwidth = 0.99
    edge_lw = 0.2

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
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
            segment_colors = colors + [colors[-1]]  # Repeat red for tail
            for i, patch in enumerate(patches):
                center = (edges[i] + edges[i+1]) / 2
                for j in range(1, len(bounds)):
                    if center < bounds[j]:
                        patch.set_facecolor(segment_colors[j-1])
                        break
                patch.set_edgecolor('black')
                patch.set_linewidth(edge_lw)

            # Force the bin with the record value to be danger-red with thick black border
            record_bin_idx = np.searchsorted(edges, record_val) - 1
            if 0 <= record_bin_idx < len(patches):
                patches[record_bin_idx].set_facecolor('black')  # Darker red for danger
                patches[record_bin_idx].set_edgecolor('#c1121f')
                patches[record_bin_idx].set_linewidth(1.5)  # Thick border to stand out

        for p, val, c in zip(percentiles, p_vals, colors):
            ax.axvline(val, color=c, linestyle='--', linewidth=3, zorder=3)
            ax.text(val, ax.get_ylim()[1] * 0.94,
                    f'{p}th\n{val:.2f}"',
                    rotation=90, va='top', ha='center', fontsize=11.5,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.98,
                              edgecolor=c, linewidth=2.5), zorder=10)

        current = current_values[col]
        ax.axvline(current, color='black', linestyle='-', linewidth=4, alpha=0.7, zorder=5)

        stats_text = f"N = {n_storms:,} storm days\nRecord: {record_val:.2f}\"\n{record_date}"
        ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, va='top', ha='right', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f8f8", alpha=0.97))

        ax.set_title(f"{period.upper()} SWE — Storm Days Only (>0\")",
                     fontsize=14, pad=20)
        ax.set_xlabel("Snow Water Equivalent (inches)")
        ax.set_ylabel("Number of Storm Events")
        ax.grid(True, alpha=0.3)

        # Apply log scale if toggled
        if use_log_scale:
            ax.set_yscale('log')
            ax.set_ylabel("Number of Storm Events (log scale)")

    # Title
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

    # -------------------------------
    # DOWNLOADS
    # -------------------------------
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

    fig.savefig("temp_plot.png", dpi=300, bbox_inches='tight', facecolor='white')
    with open("temp_plot.png", "rb") as f:
        col2.download_button(
            label="Download Histogram PNG",
            data=f,
            file_name=f"{safe_name}_StormTracker_{mode}_{today}.png",
            mime="image/png"
        )

else:
    st.info("👆 Upload a CSV or enter a SNOTEL Station ID to get started.")