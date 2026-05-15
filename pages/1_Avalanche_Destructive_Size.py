import streamlit as st
import pandas as pd
from datetime import date
import avalanche_calcs as calcs
import snotel_utils as snotel
import avalanche_db as db
import dsize_plot

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Avalanche D-Size Estimator",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === LARGER, MORE VISIBLE TAB FONTS ===
st.markdown("""
    <style>
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 20px !important;     /* Change this number if you want bigger/smaller */
            font-weight: 600 !important;
            padding: 8px 0 !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Avalanche Destructive Size Estimator")
st.caption("CNFAC Avalanche Tools — ISSW Project with Erich Peitzsch, Zach Guy, Ron Simenhois, and Bruce Jamieson")

# ====================== UNIT SYSTEM TOGGLE ======================
use_imperial = st.toggle(
    "🌎 Use Imperial units (ft / ft²)",
    value=False,
    help="Toggle ON to enter all length, width, depth, and area in feet. Calculations are always done in metric."
)

unit_length = "ft" if use_imperial else "m"
unit_area = "ft²" if use_imperial else "m²"
conv_length = 0.3048 if use_imperial else 1.0          # ft → m
conv_area = conv_length ** 2                            # ft² → m²

# === D-SIZE CLASSIFICATION CHART (collapsible) ===
with st.expander("📊 View D-Size Classification Chart — Mass Ranges and Typical Values", expanded=False):
    st.image(
        "dsize_yellow_orange_red_final.png",
        caption="Avalanche Destructive Size (D-Size) Classification — Mass Ranges and Typical Values (Log Scale)",
        use_container_width=False,
        width=900                    # comfortable size on desktop
    )

# ====================== TABS ======================
tab_quick, tab_detailed, tab_log = st.tabs([
    "Quick Field Method",
    "Detailed SNOTEL Method",
    "📋 View Saved Avalanches"
])

# ====================== QUICK FIELD METHOD ======================
with tab_quick:
    st.subheader("Quick Method — Crown Width + Slab Height + Hand Hardness & Grain Type")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        crown_width = st.number_input(f"Crown Width ({unit_length})", value=250.0 if use_imperial else 80.0, min_value=1.0, step=1.0)
        slab_height = st.number_input(f"Slab Height — crown to stauchwall ({unit_length})", value=500.0 if use_imperial else 150.0, min_value=1.0, step=1.0)
        depth = st.number_input(f"Slab thickness / burial depth ({unit_length})", value=2.5 if use_imperial else 0.8, min_value=0.1, step=0.1)
        
        area_quick_input = st.number_input(f"Slab Area ({unit_area}) — optional", value=crown_width * slab_height, min_value=10.0)

    with col2:
        hardness_options = ["F-", "F", "F+", "4F-", "4F", "4F+", "1F-", "1F", "1F+", "P-", "P", "P+", "K-", "K", "K+"]
        hardness = st.selectbox("Hand Hardness", hardness_options, index=4)
        grain_options = [
            "Precipitation Particles (PP)", "Graupel (PPgp)", "Decomposing/Fragmented (DF)",
            "Rounded Grains (RG)", "Faceted Rounded (RGxf)", "Faceted Crystals (FC)",
            "Rounding Faceted (FCxr)", "Depth Hoar (DH)", "Melt-Freeze Crust (MFcr)"
        ]
        grain = st.selectbox("Grain Type", grain_options, index=3)
    
    # Advanced uncertainty — always active
    with st.expander("🔧 Advanced Uncertainty (per input) — click to expand", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            unc_lw = st.slider("Crown Width / Slab Height / Area uncertainty %", 5, 50, 25, key="quick_lw")
            unc_depth = st.slider("Depth / Slab thickness uncertainty %", 5, 50, 25, key="quick_depth")
        with col_b:
            unc_density = st.slider("Density (hardness/grain) uncertainty %", 5, 50, 25, key="quick_density")
    
    if st.button("Calculate Quick Method", type="primary", use_container_width=True):
        crown_width_m = crown_width * conv_length
        slab_height_m = slab_height * conv_length
        depth_m = depth * conv_length
        area_m2 = area_quick_input * conv_area
        
        volume_m3 = area_m2 * depth_m
        density = calcs.get_density_from_hardness_grain(hardness, grain)
        mass_tonnes = volume_m3 * density / 1000.0
        
        # === CRITICAL FIX: Read from session_state keys directly ===
        unc_lw_val = st.session_state.get("quick_lw", 25)
        unc_depth_val = st.session_state.get("quick_depth", 25)
        unc_density_val = st.session_state.get("quick_density", 25)
        
        mid_d, low_d, high_d = calcs.get_uncertainty_mass_range(
            mass_tonnes,
            unc_length_width_pct=unc_lw_val,
            unc_depth_pct=unc_depth_val,
            unc_density_pct=unc_density_val
        )
        
        total_rel_unc = (
            (unc_lw_val / 100.0)**2 +
            (unc_depth_val / 100.0)**2 +
            (unc_density_val / 100.0)**2
        ) ** 0.5
        
        f = 1 + total_rel_unc
        low_mass = mass_tonnes / f
        high_mass = mass_tonnes * f
        
        st.success(f"**Estimated D-Size: {mid_d}**  (uncertainty range: **{low_d} – {high_d}**)")
        
        st.info(f"**Mass: {mass_tonnes:,.0f} tonnes** "
                f"(uncertainty range: **{low_mass:,.0f} – {high_mass:,.0f} tonnes**) | "
                f"Volume: {volume_m3:,.0f} m³ | Density: {density:.0f} kg/m³")
        
        st.caption(calcs.get_dsize_range_string(mass_tonnes))
        
        st.session_state.quick_inputs = {
            "method": "Quick",
            "area_m2": area_m2,
            "volume_m3": volume_m3,
            "mass_tonnes": mass_tonnes,
            "calculated_d_size": mid_d,
            "unc_low": low_d,
            "unc_high": high_d,
            "low_mass": low_mass,
            "high_mass": high_mass,
            "crown_width_m": crown_width_m,
            "slab_height_m": slab_height_m,
            "depth_m": depth_m,
            "hardness": hardness,
            "grain": grain,
            "density_kgm3": density,
        }
        
        fig = dsize_plot.plot_dsize_with_user_mass(mass_tonnes, low_mass, high_mass)
        st.pyplot(fig, use_container_width=False)
        
# ====================== DETAILED SNOTEL METHOD ======================
with tab_detailed:
    st.subheader("Detailed Method — SNOTEL Slab SWE")
    
    # ==================== STATION LOADING ====================
    try:
        stations_df = pd.read_csv("SNOTEL_station_list.csv")
        stations_df['ID'] = stations_df['site_name'].str.extract(r'\((\d+)\)', expand=False).astype(str)
        stations_df['display_name'] = stations_df['site_name'] + " - " + stations_df['state']
        stations_df = stations_df[['display_name', 'ID', 'state']].dropna()
        station_options = stations_df['display_name'].tolist()
    except Exception as e:
        st.error(f"Could not load SNOTEL_station_list.csv: {e}")
        station_options = ["Error loading stations"]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        selected_display = st.selectbox("Nearby SNOTEL Station", station_options)
        if selected_display:
            selected_row = stations_df[stations_df['display_name'] == selected_display].iloc[0]
            station_id = selected_row['ID']
            state = selected_row['state']
            station_triplet = f"{station_id}:{state}:SNTL"
            station_name = selected_display.split(" - ")[0]
        else:
            station_triplet = None
            station_name = None
        
        weak_date = st.date_input("Weak Layer Date", value=date(2025, 1, 15), min_value=date(1960, 1, 1), max_value=date.today())
        release_date = st.date_input("Avalanche Release Date", value=date(2025, 2, 10), min_value=date(1960, 1, 1), max_value=date.today())
    
    with col2:
        area_detailed_input = st.number_input(f"Slab Area ({unit_area})", value=80000.0 if use_imperial else 8000.0, min_value=100.0, step=100.0)
        swe_adjust = st.slider("Local SWE Adjustment (±%)", -30, 30, 0)
        burial_depth_ref = st.number_input(f"Slab thickness ({unit_length}) — reference only", value=2.5 if use_imperial else 0.8, min_value=0.1, step=0.1)
    
    # Advanced uncertainty — only relevant sliders for Detailed mode
    with st.expander("🔧 Advanced Uncertainty (per input)", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            unc_area = st.slider("Area uncertainty %", 5, 50, 15, key="detailed_area")
        with col_b:
            unc_swe = st.slider("SWE difference uncertainty %", 5, 50, 10, key="detailed_swe")
    
    if st.button("Calculate Detailed SNOTEL Method", type="primary", use_container_width=True):
        if not station_triplet:
            st.warning("Please select a station first.")
        else:
            try:
                slab_data = snotel.get_slab_swe(station_triplet, str(weak_date), str(release_date))
                
                if "error" in slab_data:
                    st.error(f"❌ SNOTEL error: {slab_data['error']}")
                else:
                    slab_swe_mm = slab_data["slab_swe_mm"]
                    adjusted_swe_mm = slab_swe_mm * (1 + swe_adjust / 100.0)
                    adjusted_swe_m = adjusted_swe_mm / 1000.0
                    
                    area_m2 = area_detailed_input * conv_area
                    mass_tonnes = area_m2 * adjusted_swe_m
                    
                    # === DETAILED MODE: ONLY AREA + SWE UNCERTAINTY ===
                    mid_d, low_d, high_d = calcs.get_uncertainty_mass_range(
                        mass_tonnes,
                        unc_length_width_pct=unc_area,
                        unc_depth_pct=0,      # irrelevant
                        unc_density_pct=0,    # irrelevant
                        unc_swe_pct=unc_swe
                    )
                    
                    # Root Sum-Square for total relative uncertainty
                    total_rel_unc = (
                        (unc_lw_val / 100.0)**2 +
                        (unc_depth_val / 100.0)**2 +
                        (unc_density_val / 100.0)**2
                    ) ** 0.5
                    
                    f = 1 + total_rel_unc
                    low_mass = mass_tonnes / f
                    high_mass = mass_tonnes * f
                    
                    st.success(f"**Estimated D-Size: {mid_d}**  (uncertainty range: **{low_d} – {high_d}**)")
                    
                    st.info(f"**Mass: {mass_tonnes:,.0f} tonnes** "
                            f"(uncertainty range: **{low_mass:,.0f} – {high_mass:,.0f} tonnes**) | "
                            f"Slab SWE: {adjusted_swe_mm:.1f} mm")
                    
                    # Dynamic chart with accurate uncertainty band
                    fig = dsize_plot.plot_dsize_with_user_mass(mass_tonnes, low_mass, high_mass)
                    st.pyplot(fig, use_container_width=False)
                    
                    # Store for research database
                    st.session_state.detailed_inputs = {
                        "method": "Detailed",
                        "area_m2": area_m2,
                        "volume_m3": area_m2 * (burial_depth_ref * conv_length),
                        "mass_tonnes": mass_tonnes,
                        "calculated_d_size": mid_d,
                        "unc_low": low_d,
                        "unc_high": high_d,
                        "low_mass": low_mass,
                        "high_mass": high_mass,
                        "weak_layer_date": str(weak_date),
                        "release_date": str(release_date),
                        "snotel_station": station_name,
                        "snotel_triplet": station_triplet,
                        "slab_swe_mm": slab_swe_mm,
                        "adjusted_swe_mm": adjusted_swe_mm,
                        "burial_depth_ref_m": burial_depth_ref * conv_length,
                    }
            except Exception as e:
                st.error(f"Could not fetch SNOTEL data: {e}")
            
# ====================== SAVE BUTTON ======================
st.divider()
st.subheader("Save to Research Database")

observer = st.text_input("Observer Name", value="Your Name")
location = st.text_input("Avalanche Location / Notes", placeholder="e.g. Chugach Mtns, AK — crown visible from highway")

field_d_size_options = ["D1", "D1.5", "D2", "D2.5", "D3", "D3.5", "D4", "D4.5", "D5"]
field_assessed_d_size = st.selectbox("Field-Assessed D-Size (what you think it actually was)", field_d_size_options)

if st.button("💾 Save Avalanche to Research Database", type="primary", use_container_width=True):
    inputs = st.session_state.get("quick_inputs", {}) or st.session_state.get("detailed_inputs", {})
    if inputs:
        data = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "observer": observer,
            "location": location,
            "method": inputs.get("method"),
            "area_m2": inputs.get("area_m2"),
            "volume_m3": inputs.get("volume_m3"),
            "mass_tonnes": inputs.get("mass_tonnes"),
            # Use new key, with fallback for any old test entries
            "calculated_d_size": inputs.get("calculated_d_size") or inputs.get("d_size"),
            "unc_low": inputs.get("unc_low"),
            "unc_high": inputs.get("unc_high"),
            "field_assessed_d_size": field_assessed_d_size,
            # Quick method
            "crown_width_m": inputs.get("crown_width_m"),
            "slab_height_m": inputs.get("slab_height_m"),
            "depth_m": inputs.get("depth_m"),
            "hardness": inputs.get("hardness"),
            "grain": inputs.get("grain"),
            "density_kgm3": inputs.get("density_kgm3"),
            # Detailed SNOTEL
            "weak_layer_date": inputs.get("weak_layer_date"),
            "release_date": inputs.get("release_date"),
            "snotel_station": inputs.get("snotel_station"),
            "snotel_triplet": inputs.get("snotel_triplet"),
            "slab_swe_mm": inputs.get("slab_swe_mm"),
            "adjusted_swe_mm": inputs.get("adjusted_swe_mm"),
            "burial_depth_ref_m": inputs.get("burial_depth_ref_m"),
            "notes": "",
        }
        db.save_avalanche(data)
        st.success("✅ Avalanche saved to research database!")
        # Clear session state
        for key in ["quick_inputs", "detailed_inputs"]:
            st.session_state.pop(key, None)
    else:
        st.warning("Calculate an avalanche first before saving.")
        
# ====================== VIEW LOG TAB ======================
with tab_log:
    st.subheader("📋 Research Database — Saved Avalanches")
    log_df = db.load_avalanche_log()
    
    if log_df.empty:
        st.info("No avalanches saved yet. Calculate and save some entries to build the database!")
    else:
        # Logical column order for research use
        preferred_order = [
            "timestamp", "observer", "location", "method",
            "calculated_d_size", "field_assessed_d_size",
            "mass_tonnes", "volume_m3", "area_m2",
            "crown_width_m", "slab_height_m", "depth_m",
            "hardness", "grain", "density_kgm3",
            "snotel_station", "weak_layer_date", "release_date",
            "slab_swe_mm", "adjusted_swe_mm", "burial_depth_ref_m",
            "unc_low", "unc_high", "notes"
        ]
        
        # Only show columns that actually exist
        available_cols = [col for col in preferred_order if col in log_df.columns]
        display_df = log_df[available_cols].copy()
        
        # Nice formatting
        if "timestamp" in display_df.columns:
            display_df["timestamp"] = pd.to_datetime(display_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "timestamp": st.column_config.TextColumn("Date/Time"),
                "mass_tonnes": st.column_config.NumberColumn("Mass (tonnes)", format="%.0f"),
                "volume_m3": st.column_config.NumberColumn("Volume (m³)", format="%.0f"),
                "area_m2": st.column_config.NumberColumn("Area (m²)", format="%.0f"),
                "depth_m": st.column_config.NumberColumn("Depth (m)", format="%.2f"),
                "slab_swe_mm": st.column_config.NumberColumn("Slab SWE (mm)", format="%.1f"),
                "adjusted_swe_mm": st.column_config.NumberColumn("Adjusted SWE (mm)", format="%.1f"),
            }
        )
        
        st.download_button(
            label="📥 Download Full Research Database as CSV",
            data=log_df.to_csv(index=False).encode(),
            file_name="avalanche_research_log.csv",
            mime="text/csv"
        )
        
        st.caption(f"Total records in database: **{len(log_df)}**")

# ====================== FOOTER ======================
st.divider()
st.caption("💡 Quick method uses Kim & Jamieson (2014) Table 3 densities. Detailed method reuses your Storm Tracker SNOTEL code.")
st.caption("Uncertainty sliders let you reflect real field data quality. Toggle Imperial units above for U.S. field use.")