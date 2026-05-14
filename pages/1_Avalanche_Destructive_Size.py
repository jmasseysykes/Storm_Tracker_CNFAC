import streamlit as st
import pandas as pd
from datetime import date
import avalanche_calcs as calcs
import snotel_utils as snotel
import avalanche_db as db

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Avalanche D-Size Estimator",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Avalanche Destructive Size Estimator")
st.caption("CNFAC Avalanche Tools — ISSW Project with Erich Peitzsch, Zach Guy & Ron Simenhois")

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
        crown_width = st.number_input(f"Crown Width ({unit_length})", 
                                      value=250.0 if use_imperial else 80.0, 
                                      min_value=1.0, step=10.0)
        slab_height = st.number_input(f"Slab Height — crown to stauchwall ({unit_length})", 
                                      value=500.0 if use_imperial else 150.0, 
                                      min_value=1.0, step=10.0)
        depth = st.number_input(f"Slab thickness / burial depth ({unit_length})", 
                                value=2.5 if use_imperial else 0.8, 
                                min_value=0.1, step=0.1)
        
        # === CLEAR OPTIONAL AREA FIELD ===
        area_quick_input = st.number_input(
            f"Slab Area ({unit_area}) — optional",
            value=crown_width * slab_height,
            min_value=10.0,
            step=10.0,
            help="Auto-calculated from Crown Width × Slab Height. Override this if you drew a polygon, used GPS, or have a better area measurement."
        )
    
    with col2:
        hardness_options = ["F-", "F", "F+", "4F-", "4F", "4F+", "1F-", "1F", "1F+", "P-", "P", "P+", "K-", "K", "K+"]
        hardness = st.selectbox("Average Slab Hand Hardness", hardness_options, index=4)
        
        grain_options = [
            "Precipitation Particles (PP)",
            "Graupel (PPgp)",
            "Decomposing/Fragmented (DF)",
            "Rounded Grains (RG)",
            "Faceted Rounded (RGxf)",
            "Faceted Crystals (FC)",
            "Rounding Faceted (FCxr)",
            "Depth Hoar (DH)",
            "Melt-Freeze Crust (MFcr)"
        ]
        grain = st.selectbox("Grain Type", grain_options, index=3)
        
        overall_unc = st.slider("Overall Measurement Uncertainty % (simple mode)", 5, 40, 15, help="Applied when Advanced section is closed")
    
    # Advanced uncertainty controls
    with st.expander("🔧 Advanced Uncertainty (per input) — click to expand", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            unc_lw = st.slider("Crown Width / Slab Height / Area uncertainty %", 5, 40, 15, key="quick_lw")
            unc_depth = st.slider("Depth / Slab thickness uncertainty %", 5, 40, 10, key="quick_depth")
        with col_b:
            unc_density = st.slider("Density (hardness/grain) uncertainty %", 5, 50, 20, key="quick_density")
    
    if st.button("Calculate Quick Method", type="primary", use_container_width=True):
        # Convert to metric
        crown_width_m = crown_width * conv_length
        slab_height_m = slab_height * conv_length
        depth_m = depth * conv_length
        area_m2 = area_quick_input * conv_area
        
        volume_m3 = area_m2 * depth_m
        density = calcs.get_density_from_hardness_grain(hardness, grain)
        mass_tonnes = volume_m3 * density / 1000.0
        
        # Uncertainty
        use_advanced = st.session_state.get("quick_lw", None) is not None
        if use_advanced:
            mid_d, low_d, high_d = calcs.get_uncertainty_mass_range(
                mass_tonnes,
                unc_length_width_pct=unc_lw,
                unc_depth_pct=unc_depth,
                unc_density_pct=unc_density
            )
        else:
            mid_d, low_d, high_d = calcs.get_uncertainty_mass_range(
                mass_tonnes,
                unc_length_width_pct=overall_unc,
                unc_depth_pct=overall_unc,
                unc_density_pct=overall_unc
            )
        
        st.success(f"**Estimated D-Size: {mid_d}**  (uncertainty range: **{low_d} – {high_d}**)")
        st.info(f"Mass: **{mass_tonnes:,.0f} tonnes** | Volume: {volume_m3:,.0f} m³ | Density: {density:.0f} kg/m³")
        
        if use_imperial:
            st.caption(f"Original inputs: Crown Width {crown_width:.0f} ft × Slab Height {slab_height:.0f} ft × Depth {depth:.1f} ft")
        else:
            st.caption(f"Original inputs: Crown Width {crown_width:.0f} m × Slab Height {slab_height:.0f} m × Depth {depth:.1f} m")
        st.caption(calcs.get_dsize_range_string(mass_tonnes))
        
        # === FULL DATA FOR RESEARCH DATABASE ===
        st.session_state.quick_inputs = {
            "method": "Quick",
            "area_m2": area_m2,
            "volume_m3": volume_m3,
            "mass_tonnes": mass_tonnes,
            "calculated_d_size": mid_d,
            "unc_low": low_d,
            "unc_high": high_d,
            "crown_width_m": crown_width_m,
            "slab_height_m": slab_height_m,
            "depth_m": depth_m,
            "hardness": hardness,
            "grain": grain,
            "density_kgm3": density,
        }
        
# ====================== DETAILED SNOTEL METHOD ======================
with tab_detailed:
    st.subheader("Detailed Method — SNOTEL Slab SWE")
    
    # ==================== EXACT SAME STATION LOADING AS STORM TRACKER ====================
    try:
        stations_df = pd.read_csv("SNOTEL_station_list.csv")
        
        # Build the same display names and triplet that Storm Tracker uses
        stations_df['ID'] = stations_df['site_name'].str.extract(r'\((\d+)\)', expand=False).astype(str)
        stations_df['display_name'] = stations_df['site_name'] + " - " + stations_df['state']
        
        # Drop any rows that couldn't be parsed
        stations_df = stations_df[['display_name', 'ID', 'state']].dropna()
        
        station_options = stations_df['display_name'].tolist()
        
        if "station_debug_shown" not in st.session_state:
            #st.success(f"✅ Loaded **{len(station_options):,}** SNOTEL stations")
            #st.info("Using same station list & triplet logic as Storm Tracker")
            st.session_state.station_debug_shown = True
            
    except Exception as e:
        st.error(f"Could not load SNOTEL_station_list.csv: {e}")
        station_options = ["Error loading stations"]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        selected_display = st.selectbox(
            "Nearby SNOTEL Station",
            options=station_options,
            help="Type to search — matches Storm Tracker exactly"
        )
        
        # Build triplet exactly like Storm Tracker
        if selected_display:
            selected_row = stations_df[stations_df['display_name'] == selected_display].iloc[0]
            station_id = selected_row['ID']
            state = selected_row['state']
            station_triplet = f"{station_id}:{state}:SNTL"
            station_name = selected_display.split(" - ")[0]
        else:
            station_triplet = None
            station_name = None
        
        weak_date = st.date_input(
            "Weak Layer Date (when slab started accumulating)",
            value=date(2025, 1, 15),
            min_value=date(1960, 1, 1),
            max_value=date.today()
        )
        release_date = st.date_input(
            "Avalanche Release Date",
            value=date(2025, 2, 10),
            min_value=date(1960, 1, 1),
            max_value=date.today()
        )
    
    with col2:
        area_detailed_input = st.number_input(f"Slab Area ({unit_area})", value=80000.0 if use_imperial else 8000.0, min_value=100.0, step=100.0)
        swe_adjust = st.slider("Local SWE Adjustment (±%) — your knowledge of site vs SNOTEL", -30, 30, 0)
        burial_depth_ref = st.number_input(f"Slab thickness ({unit_length}) — reference only", value=2.5 if use_imperial else 0.8, min_value=0.1, step=0.1)
    
    with st.expander("🔧 Advanced Uncertainty (per input)", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            unc_area = st.slider("Area uncertainty %", 5, 40, 15, key="detailed_area")
        with col_b:
            unc_swe = st.slider("SWE difference uncertainty %", 5, 40, 10, key="detailed_swe")
    
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
                    volume_m3 = area_m2 * burial_depth_ref * conv_length   # optional reference volume
                    mass_tonnes = area_m2 * adjusted_swe_m
                    
                    mid_d, low_d, high_d = calcs.get_uncertainty_mass_range(
                        mass_tonnes,
                        unc_length_width_pct=unc_area,
                        unc_depth_pct=5,
                        unc_density_pct=5,
                        unc_swe_pct=unc_swe
                    )
                    
                    st.success(f"**Estimated D-Size: {mid_d}**  (uncertainty range: **{low_d} – {high_d}**)")
                    st.info(f"**Slab SWE: {adjusted_swe_mm:.1f} mm** | Mass: **{mass_tonnes:,.0f} tonnes**")
                    st.caption(f"Weak: {slab_data['swe_weak_mm']:.1f} mm | Release: {slab_data['swe_release_mm']:.1f} mm")
                    
                    # === FULL DATA FOR RESEARCH DATABASE ===
                    st.session_state.detailed_inputs = {
                        "method": "Detailed",
                        "area_m2": area_m2,
                        "volume_m3": volume_m3,                    # reference only
                        "mass_tonnes": mass_tonnes,
                        "calculated_d_size": mid_d,
                        "unc_low": low_d,
                        "unc_high": high_d,
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