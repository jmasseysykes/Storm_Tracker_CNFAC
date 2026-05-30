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

# === D-SIZE CLASSIFICATION CHART (collapsible) ===
with st.expander("📊 View D-Size Classification Chart — Mass Ranges and Typical Values", expanded=False):
    st.image(
        "dsize_yellow_orange_red_final.png",
        caption="Avalanche Destructive Size (D-Size) Classification — Mass Ranges and Typical Values (Log Scale)",
        use_container_width=False,
        width=900                    # comfortable size on desktop
    )
    
# ====================== UNIT SYSTEM TOGGLE ======================
use_imperial = st.toggle(
    "🌎 Use Imperial units (ft / ft²)",
    value=False,
    help="Toggle ON to enter all length, width, depth, and area in feet. Calculations are always done in metric."
)

swe_unit = "inches" if use_imperial else "mm"
unit_length = "ft" if use_imperial else "m"
unit_area = "ft²" if use_imperial else "m²"
conv_length = 0.3048 if use_imperial else 1.0          # ft → m
conv_area = conv_length ** 2                            # ft² → m²

# ====================== TABS ======================
tab_quick, tab_detailed, tab_runout, tab_log = st.tabs([
    "Quick Field Method",
    "Detailed SNOTEL Method",
    "Runout / Debris Estimate",
    "📋 View Saved Avalanches"
])

# ====================== QUICK FIELD METHOD ======================
with tab_quick:
    st.subheader("Quick Method — Slab Dimensions + Hand Hardness & Grain Type")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        crown_width = st.number_input(f"Crown Width ({unit_length})",
                                      value=250.0 if use_imperial else 80.0,
                                      min_value=1.0, step=1.0)
        slab_length = st.number_input(f"Slab Length — crown to stauchwall ({unit_length})",
                                      value=500.0 if use_imperial else 150.0,
                                      min_value=1.0, step=1.0)
        depth = st.number_input(f"Slab Depth / Crown Thickness ({unit_length})",
                                value=2.5 if use_imperial else 0.8,
                                min_value=0.1, step=0.1)
        
        area_quick_input = st.number_input(
            f"Slab Area ({unit_area}) — optional",
            value=crown_width * slab_length,
            min_value=10.0,
            help="Auto-calculated from Crown Width × Slab Length."
        )
        
        include_entrainment = st.toggle("Include Entrainment Mass", value=False, key="quick_entr_toggle")
    
    with col2:
        hardness_options = ["F-", "F", "F+", "4F-", "4F", "4F+", "1F-", "1F", "1F+", "P-", "P", "P+", "K-", "K", "K+"]
        hardness = st.selectbox("Hand Hardness (Slab)", hardness_options, index=4)
        
        grain_options = [
            "Precipitation Particles (PP)", "Graupel (PPgp)", "Decomposing/Fragmented (DF)",
            "Rounded Grains (RG)", "Faceted Rounded (RGxf)", "Faceted Crystals (FC)",
            "Rounding Faceted (FCxr)", "Depth Hoar (DH)", "Melt-Freeze Crust (MFcr)"
        ]
        grain = st.selectbox("Grain Type (Slab)", grain_options, index=3)
        
        use_layered_density = st.toggle("Use Layered Slab Profile for Density (more accurate)", value=False)
    
    # === ADVANCED UNCERTAINTY ===
    with st.expander("🔧 Advanced Uncertainty (per input) — RSS Method", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            unc_lw = st.slider("Crown Width / Slab Length / Area uncertainty %", 0, 50, 15, key="quick_lw")
            unc_depth = st.slider("Depth / Slab thickness uncertainty %", 0, 50, 15, key="quick_depth")
        with col_b:
            unc_density = st.slider("Density uncertainty %", 0, 50, 20, key="quick_density")
            if include_entrainment:
                unc_entrainment = st.slider("Entrainment uncertainty %", 0, 50, 25, key="quick_entr")
    
    # === LAYERED DENSITY INPUT ===
    if use_layered_density:
        st.markdown("**Define Slab Layers (top to bottom)**")
        if "layers" not in st.session_state:
            st.session_state.layers = [{"thickness_cm": 30, "hardness": "4F", "grain": "Rounded Grains (RG)"}]
        
        layers = st.session_state.layers
        for i in range(len(layers)):
            col_a, col_b, col_c, col_d = st.columns([2, 2, 2, 1])
            with col_a:
                layers[i]["thickness_cm"] = st.number_input(f"Layer {i+1} Thickness (cm)",
                                                           value=layers[i]["thickness_cm"],
                                                           min_value=1, key=f"thick_{i}")
            with col_b:
                layers[i]["hardness"] = st.selectbox("Hardness", hardness_options,
                                                    index=hardness_options.index(layers[i]["hardness"]),
                                                    key=f"hard_{i}")
            with col_c:
                layers[i]["grain"] = st.selectbox("Grain Type", grain_options,
                                                 index=grain_options.index(layers[i]["grain"]),
                                                 key=f"grain_{i}")
            with col_d:
                if st.button("🗑", key=f"del_{i}"):
                    layers.pop(i)
                    st.rerun()
        
        if st.button("➕ Add Layer"):
            layers.append({"thickness_cm": 20, "hardness": "4F", "grain": "Rounded Grains (RG)"})
            st.rerun()
    
    # === UNIFIED ENTRAINMENT ===
    if include_entrainment:
        st.markdown("**Entrainment Estimation**")
        entr_method = st.radio("Entrainment Calculation Method", 
                               ["Dimensions + Hardness/Grain", "SWE-based"], horizontal=True)
        
        col_e1, col_e2 = st.columns([1, 1])
        with col_e1:
            entr_width = st.number_input(f"Entrainment Width ({unit_length})", value=crown_width * 1.5, min_value=1.0, step=1.0)
            entr_length = st.number_input(f"Entrainment Length ({unit_length})", value=slab_length * 2, min_value=1.0, step=1.0)
            entr_area = st.number_input(f"Entrainment Area ({unit_area}) — optional", value=entr_width * entr_length, min_value=10.0)
        
        with col_e2:
            if entr_method == "Dimensions + Hardness/Grain":
                entr_depth = st.number_input(f"Entrainment Depth ({unit_length})", value=1.0 if use_imperial else 0.3, min_value=0.05, step=0.05)
                entr_hardness = st.selectbox("Entrainment Hardness", hardness_options, index=1)
                entr_grain = st.selectbox("Entrainment Grain Type", grain_options, index=0)
            else:
                entr_swe = st.number_input(
                    f"Entrainment SWE ({swe_unit})", 
                    value=1.5 if use_imperial else 40.0,
                    min_value=0.0, 
                    step=10.0
                )
    if st.button("Calculate Quick Method", type="primary", use_container_width=True):
        crown_width_m = crown_width * conv_length
        slab_length_m = slab_length * conv_length
        depth_m = depth * conv_length
        slab_area_m2 = area_quick_input * conv_area
        volume_m3 = slab_area_m2 * depth_m
        
        # Slab Density
        if use_layered_density and "layers" in st.session_state and st.session_state.layers:
            total_thickness = 0
            weighted_density = 0
            for layer in st.session_state.layers:
                layer_density = calcs.get_density_from_hardness_grain(layer["hardness"], layer["grain"])
                weighted_density += layer_density * layer["thickness_cm"]
                total_thickness += layer["thickness_cm"]
            density = weighted_density / total_thickness if total_thickness > 0 else 250
        else:
            density = calcs.get_density_from_hardness_grain(hardness, grain)
        
        slab_mass = volume_m3 * density / 1000.0
        
        # ==================== ENTRAINMENT ====================
        entrainment_mass = 0.0
        if include_entrainment:
            entr_area_m2 = entr_area * conv_area
            entr_depth_m = entr_depth * conv_length
            
            if entr_method == "SWE-based":
                # Use manual SWE if entered
                if 'entr_swe' in locals() and entr_swe is not None and entr_swe > 0:
                    entr_swe_mm = entr_swe * 25.4 if use_imperial else entr_swe
                else:
                    # Fallback to hardness/grain method
                    entr_density = calcs.get_density_from_hardness_grain(entr_hardness, entr_grain)
                    entr_swe_mm = entr_density * entr_depth * conv_length * 1000
            else:
                # Dimensions + Hardness/Grain
                entr_density = calcs.get_density_from_hardness_grain(entr_hardness, entr_grain)
                entr_swe_mm = entr_density * entr_depth * conv_length * 1000
            
            entrainment_mass = entr_area_m2 * (entr_swe_mm / 1000.0)
        
        total_mass = slab_mass + entrainment_mass
        
        # Uncertainty (RSS)
        unc_lw_val = st.session_state.get("quick_lw", 15)
        unc_depth_val = st.session_state.get("quick_depth", 15)
        unc_density_val = st.session_state.get("quick_density", 20)
        unc_entr_val = st.session_state.get("quick_entr_unc", 25) if include_entrainment else 0
        
        total_rel_unc = (
            (unc_lw_val / 100.0)**2 +
            (unc_depth_val / 100.0)**2 +
            (unc_density_val / 100.0)**2
        )
        if include_entrainment:
            total_rel_unc += (unc_entr_val / 100.0)**2
        total_rel_unc = total_rel_unc ** 0.5
        
        f = 1 + total_rel_unc
        low_mass = total_mass / f
        high_mass = total_mass * f
        
        mid_d, low_d, high_d = calcs.get_uncertainty_mass_range(
            total_mass,
            unc_length_width_pct=unc_lw_val,
            unc_depth_pct=unc_depth_val,
            unc_density_pct=unc_density_val
        )
        
        st.success(f"**Estimated D-Size: {mid_d}** (uncertainty range: **{low_d} – {high_d}**)")
        
        st.info(f"**Slab Mass**: {slab_mass:,.0f} t | **Entrained Mass**: {entrainment_mass:,.0f} t | "
                f"**Total Mass**: {total_mass:,.0f} tonnes")
        
        st.caption(calcs.get_dsize_range_string(total_mass))
        
        # Store
        st.session_state.quick_inputs = {
            "method": "Quick",
            "area_m2": slab_area_m2,
            "volume_m3": volume_m3,
            "mass_tonnes": slab_mass,
            "entrainment_mass": entrainment_mass,
            "total_mass": total_mass,
            "calculated_d_size": mid_d,
            "unc_low": low_d,
            "unc_high": high_d,
            "low_mass": low_mass,
            "high_mass": high_mass,
            "crown_width_m": crown_width_m,
            "slab_length_m": slab_length_m,
            "depth_m": depth_m,
            "hardness": hardness,
            "grain": grain,
            "density_kgm3": density,
            "use_layered_density": use_layered_density,
            "include_entrainment": include_entrainment
        }
        
        fig = dsize_plot.plot_dsize_with_user_mass(total_mass, low_mass, high_mass)
        st.pyplot(fig)
        
# ====================== DETAILED SNOTEL METHOD ======================
with tab_detailed:
    st.subheader("Detailed Method — Slab Area + Slab SWE")
    
    # Station Loading
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
        selected_display = st.selectbox("Nearby SNOTEL Station (Slab)", station_options)
        if selected_display:
            selected_row = stations_df[stations_df['display_name'] == selected_display].iloc[0]
            station_id = selected_row['ID']
            state = selected_row['state']
            station_triplet = f"{station_id}:{state}:SNTL"
            station_name = selected_display.split(" - ")[0]
        else:
            station_triplet = None
            station_name = None
        
        weak_date = st.date_input("Weak Layer Date (Slab)", value=date(2025, 1, 15), min_value=date(1960, 1, 1))
        release_date = st.date_input("Avalanche Release Date (Slab)", value=date(2025, 2, 10), min_value=date(1960, 1, 1))
        
        include_entrainment = st.toggle("Include Entrainment Mass", value=False, key="detailed_entr_toggle")
    
    with col2:
        area_detailed_input = st.number_input(f"Slab Area ({unit_area})", value=80000.0 if use_imperial else 8000.0, min_value=100.0)
        
        swe_unit = "inches" if use_imperial else "mm"
        manual_swe_value = st.number_input(
            f"Manual Slab SWE Estimate ({swe_unit}) — Optional",
            value=None, min_value=0.0, step=0.1
        )
        
        swe_adjust = st.slider("Local SWE Adjustment (±%)", -50, 50, 0)
        burial_depth_ref = st.number_input(f"Slab thickness ({unit_length}) — reference only",
                                           value=2.5 if use_imperial else 0.8, min_value=0.1, step=0.1)
    
    # === ADVANCED UNCERTAINTY ===
    with st.expander("🔧 Advanced Uncertainty (per input) — RSS Method", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            unc_area = st.slider("Slab Area uncertainty %", 0, 50, 15, key="detailed_area")
            unc_swe = st.slider("Slab SWE uncertainty %", 0, 50, 10, key="detailed_swe")
        with col_b:
            if include_entrainment:
                unc_entrainment = st.slider("Entrainment uncertainty %", 0, 50, 25, key="detailed_entr")
    
    # === UNIFIED ENTRAINMENT ===
    if include_entrainment:
        st.markdown("**Entrainment Estimation**")
        entr_method = st.radio("Entrainment Calculation Method", 
                               ["Dimensions + Hardness/Grain", "SWE-based"], 
                               horizontal=True, key="detailed_entr_method")
        
        col_e1, col_e2 = st.columns([1, 1])
        with col_e1:
            entr_width = st.number_input(f"Entrainment Width ({unit_length})", value=200.0 if use_imperial else 60.0, min_value=1.0, step=1.0)
            entr_length = st.number_input(f"Entrainment Length ({unit_length})", value=800.0 if use_imperial else 250.0, min_value=1.0, step=1.0)
            entr_area = st.number_input(f"Entrainment Area ({unit_area}) — optional", value=entr_width * entr_length, min_value=10.0)
            entr_depth = st.number_input(f"Entrainment Depth ({unit_length})", value=1.0 if use_imperial else 0.3, min_value=0.05, step=0.05)
        
        with col_e2:
            if entr_method == "Dimensions + Hardness/Grain":
                entr_hardness = st.selectbox("Entrainment Hardness", hardness_options, index=1)
                entr_grain = st.selectbox("Entrainment Grain Type", grain_options, index=0)
            else:
                entr_weak_date = st.date_input("Entrainment Start Date", value=weak_date, min_value=date(1960, 1, 1))
                entr_release_date = st.date_input("Entrainment End Date", value=release_date, min_value=date(1960, 1, 1))
                entr_manual_swe = st.number_input(
                    f"Manual Entrainment SWE ({swe_unit}) — Optional",
                    value=None, min_value=0.0, step=0.1
                )
                
    
    if st.button("Calculate Detailed SNOTEL Method", type="primary", use_container_width=True):
        # ==================== SLAB MASS ====================
        if manual_swe_value is not None and manual_swe_value > 0:
            slab_swe_mm = manual_swe_value * 25.4 if use_imperial else manual_swe_value
        else:
            slab_data = snotel.get_slab_swe(station_triplet, str(weak_date), str(release_date))
            slab_swe_mm = slab_data.get("slab_swe_mm", 0) if isinstance(slab_data, dict) and "error" not in slab_data else 0
        
        adjusted_slab_swe_mm = slab_swe_mm * (1 + swe_adjust / 100.0)
        slab_area_m2 = area_detailed_input * conv_area
        slab_mass = slab_area_m2 * (adjusted_slab_swe_mm / 1000.0)
        
        # ==================== ENTRAINMENT MASS ====================
        entrainment_mass = 0.0
        if include_entrainment:
            entr_area_m2 = entr_area * conv_area
            
            if entr_method == "SWE-based":
                if entr_manual_swe is not None and entr_manual_swe > 0:
                    entr_swe_mm = entr_manual_swe * 25.4 if use_imperial else entr_manual_swe
                else:
                    entr_data = snotel.get_slab_swe(station_triplet, str(entr_weak_date), str(entr_release_date))
                    entr_swe_mm = entr_data.get("slab_swe_mm", 0) if isinstance(entr_data, dict) and "error" not in entr_data else 0
            else:
                # Dimensions + Hardness/Grain
                entr_density = calcs.get_density_from_hardness_grain(entr_hardness, entr_grain)
                entr_swe_mm = entr_density * entr_depth * conv_length * 1000
            
            entrainment_mass = entr_area_m2 * (entr_swe_mm / 1000.0)
        
        total_mass = slab_mass + entrainment_mass
        
        # ==================== RSS UNCERTAINTY ====================
        unc_area_val = st.session_state.get("detailed_area", 15)
        unc_swe_val = st.session_state.get("detailed_swe", 10)
        unc_entr_val = st.session_state.get("detailed_entr", 25) if include_entrainment else 0
        
        total_rel_unc = (
            (unc_area_val / 100.0)**2 +
            (unc_swe_val / 100.0)**2
        )
        if include_entrainment:
            total_rel_unc += (unc_entr_val / 100.0)**2
        total_rel_unc = total_rel_unc ** 0.5
        
        f = 1 + total_rel_unc
        low_mass = total_mass / f
        high_mass = total_mass * f
        
        mid_d, low_d, high_d = calcs.get_uncertainty_mass_range(
            total_mass, unc_area_val, 0, 0, unc_swe_val
        )
        
        st.success(f"**Estimated D-Size: {mid_d}** (uncertainty range: **{low_d} – {high_d}**)")
        
        st.info(f"**Slab Mass**: {slab_mass:,.0f} t | **Entrained Mass**: {entrainment_mass:,.0f} t | "
                f"**Total Mass**: {total_mass:,.0f} tonnes")
        
        st.caption(f"Uncertainty range: **{low_mass:,.0f} – {high_mass:,.0f} tonnes**")
        st.caption(calcs.get_dsize_range_string(total_mass))
        
        # Store for database
        st.session_state.detailed_inputs = {
            "method": "Detailed",
            "area_m2": slab_area_m2,
            "volume_m3": slab_area_m2 * (burial_depth_ref * conv_length),
            "mass_tonnes": slab_mass,
            "entrainment_mass": entrainment_mass,
            "total_mass": total_mass,
            "calculated_d_size": mid_d,
            "unc_low": low_d,
            "unc_high": high_d,
            "low_mass": low_mass,
            "high_mass": high_mass,
            "weak_layer_date": str(weak_date),
            "release_date": str(release_date),
            "snotel_station": station_name,
            "slab_swe_mm": slab_swe_mm,
            "adjusted_swe_mm": adjusted_slab_swe_mm,
            "burial_depth_ref_m": burial_depth_ref * conv_length,
            "include_entrainment": include_entrainment,
            "entr_width_m": entr_width if include_entrainment else None,
            "entr_length_m": entr_length if include_entrainment else None,
            "entr_area_m2": entr_area * conv_area if include_entrainment else None,
            "entr_depth_m": entr_depth * conv_length if include_entrainment else None,
            "entr_hardness": entr_hardness if include_entrainment and entr_method == "Dimensions + Hardness/Grain" else None,
            "entr_grain": entr_grain if include_entrainment and entr_method == "Dimensions + Hardness/Grain" else None,
            "entr_swe_mm": entr_swe_mm if include_entrainment else None,
            "unc_area_pct": unc_area_val,
            "unc_swe_pct": unc_swe_val,
            "unc_entrainment_pct": unc_entr_val,
        }
        
        fig = dsize_plot.plot_dsize_with_user_mass(total_mass, low_mass, high_mass)
        st.pyplot(fig)
        
# ====================== RUNOUT / DEBRIS ESTIMATE ======================
with tab_runout:
    st.subheader("Runout / Debris Estimate (Test Version)")
    st.caption("For when you can only observe the avalanche deposit / runout zone")

    col1, col2 = st.columns([1, 1])
    
    with col1:
        deposit_length = st.number_input(f"Deposit Length (along path) ({unit_length})", 
                                         value=400.0 if use_imperial else 120.0, 
                                         min_value=10.0, step=1.0)
        deposit_width = st.number_input(f"Deposit Width ({unit_length})", 
                                        value=150.0 if use_imperial else 45.0, 
                                        min_value=10.0, step=1.0)
        avg_thickness = st.number_input(f"Average Deposit Thickness ({unit_length})", 
                                        value=1.5 if use_imperial else 0.5, 
                                        min_value=0.1, step=0.05)
        
        area_runout = st.number_input(
            f"Deposit Area ({unit_area}) — optional",
            value=deposit_length * deposit_width,
            min_value=50.0,
            help="Auto-calculated from Length × Width. Override if you have a polygon measurement."
        )
    
    with col2:
        debris_type = st.selectbox(
            "Deposit / Debris Type",
            ["Dry", "Moist", "Wet", "Hard Slab Debris", "Soft Slab Debris"],
            index=2
        )
        
        density_defaults = {
            "Dry": 400,
            "Moist": 500,
            "Wet": 600,
            "Hard Slab Debris": 480,
            "Soft Slab Debris": 380
        }
        
        density = st.number_input("Deposit Density (kg/m³)", 
                                  value=density_defaults[debris_type], 
                                  min_value=200, step=10,
                                  help="Typical values based on field observations and literature")
        
        overall_unc = st.slider("Overall Uncertainty %", 5, 50, 20)
    
    if st.button("Calculate Runout / Debris Estimate", type="primary", use_container_width=True):
        length_m = deposit_length * conv_length
        width_m = deposit_width * conv_length
        thickness_m = avg_thickness * conv_length
        area_m2 = area_runout * conv_area
        
        volume_m3 = area_m2 * thickness_m
        mass_tonnes = volume_m3 * density / 1000.0
        
        # Simple uncertainty
        f = 1 + overall_unc / 100.0
        low_mass = mass_tonnes / f
        high_mass = mass_tonnes * f
        
        mid_d, low_d, high_d = calcs.get_uncertainty_mass_range(
            mass_tonnes, overall_unc, overall_unc, overall_unc
        )
        
        st.success(f"**Estimated D-Size: {mid_d}** (uncertainty range: **{low_d} – {high_d}**)")
        
        st.info(f"**Deposit Volume**: {volume_m3:,.0f} m³  |  **Estimated Mass**: {mass_tonnes:,.0f} tonnes")
        st.info(f"Uncertainty range: **{low_mass:,.0f} – {high_mass:,.0f} tonnes**")
        
        st.caption(calcs.get_dsize_range_string(mass_tonnes))
        
        # Store for database
        st.session_state.runout_inputs = {
            "method": "Runout/Debris",
            "area_m2": area_m2,
            "volume_m3": volume_m3,
            "mass_tonnes": mass_tonnes,
            "entrainment_mass": 0,
            "total_mass": mass_tonnes,
            "calculated_d_size": mid_d,
            "unc_low": low_d,
            "unc_high": high_d,
            "low_mass": low_mass,
            "high_mass": high_mass,
            "density_kgm3": density,
            "debris_type": debris_type,
            "unc_runout_pct": overall_unc
        }
        
        fig = dsize_plot.plot_dsize_with_user_mass(mass_tonnes, low_mass, high_mass)
        st.pyplot(fig)
            
# ====================== SAVE BUTTON ======================
st.divider()
st.subheader("💾 Save to Research Database")

observer = st.text_input("Observer Name", value="Your Name")
location = st.text_input("Avalanche Location", placeholder="e.g. Chugach Mtns, AK — crown visible from highway")
field_d_size_options = ["D1", "D1.5", "D2", "D2.5", "D3", "D3.5", "D4", "D4.5", "D5"]
field_assessed_d_size = st.selectbox("Field-Assessed D-Size (what you think it actually was)", field_d_size_options)
notes = st.text_area("Notes", placeholder="Optional observation notes, photos description, weather, etc.", height=100)

# === SMART SAVE LOGIC - Supports Quick, Detailed, and Runout ===
quick_data = st.session_state.get("quick_inputs")
detailed_data = st.session_state.get("detailed_inputs")
runout_data = st.session_state.get("runout_inputs")

if quick_data or detailed_data or runout_data:
    options = []
    if quick_data:
        options.append(("Quick Method", quick_data))
    if detailed_data:
        options.append(("Detailed SNOTEL Method", detailed_data))
    if runout_data:
        options.append(("Runout / Debris Estimate", runout_data))
    
    if len(options) > 1:
        save_choice = st.radio(
            "Which calculation do you want to save?",
            [name for name, _ in options],
            horizontal=True
        )
        selected_data = next(data for name, data in options if name == save_choice)
    else:
        selected_data = options[0][1] if options else None
else:
    selected_data = None

if st.button("💾 Save Avalanche to Research Database", type="primary", use_container_width=True):
    if selected_data:
        data = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "observer": observer,
            "location": location,
            "method": selected_data.get("method"),
            "area_m2": selected_data.get("area_m2"),
            "volume_m3": selected_data.get("volume_m3"),
            "mass_tonnes": selected_data.get("mass_tonnes"),
            "entrainment_mass": selected_data.get("entrainment_mass"),
            "total_mass": selected_data.get("total_mass") or selected_data.get("mass_tonnes"),
            "calculated_d_size": selected_data.get("calculated_d_size"),
            "unc_low": selected_data.get("unc_low"),
            "unc_high": selected_data.get("unc_high"),
            "field_assessed_d_size": field_assessed_d_size,
            
            # Slab / Quick fields
            "crown_width_m": selected_data.get("crown_width_m"),
            "slab_length_m": selected_data.get("slab_length_m"),
            "depth_m": selected_data.get("depth_m"),
            "hardness": selected_data.get("hardness"),
            "grain": selected_data.get("grain"),
            "density_kgm3": selected_data.get("density_kgm3"),
            "use_layered_density": selected_data.get("use_layered_density"),
            
            # Detailed fields
            "weak_layer_date": selected_data.get("weak_layer_date"),
            "release_date": selected_data.get("release_date"),
            "snotel_station": selected_data.get("snotel_station"),
            "slab_swe_mm": selected_data.get("slab_swe_mm"),
            "adjusted_swe_mm": selected_data.get("adjusted_swe_mm"),
            "burial_depth_ref_m": selected_data.get("burial_depth_ref_m"),
            
            # Entrainment fields
            "include_entrainment": selected_data.get("include_entrainment"),
            "entr_width_m": selected_data.get("entr_width_m"),
            "entr_length_m": selected_data.get("entr_length_m"),
            "entr_area_m2": selected_data.get("entr_area_m2"),
            "entr_depth_m": selected_data.get("entr_depth_m"),
            "entr_hardness": selected_data.get("entr_hardness"),
            "entr_grain": selected_data.get("entr_grain"),
            "entr_swe_mm": selected_data.get("entr_swe_mm"),
            
            # Runout/Debris specific
            "debris_type": selected_data.get("debris_type"),
            
            # Uncertainty values (trying multiple possible keys)
            "unc_lw_pct": selected_data.get("unc_lw") or selected_data.get("unc_lw_pct"),
            "unc_depth_pct": selected_data.get("unc_depth") or selected_data.get("unc_depth_pct"),
            "unc_density_pct": selected_data.get("unc_density") or selected_data.get("unc_density_pct"),
            "unc_area_pct": selected_data.get("unc_area") or selected_data.get("unc_area_pct"),
            "unc_swe_pct": selected_data.get("unc_swe") or selected_data.get("unc_swe_pct"),
            "unc_runout_pct": selected_data.get("unc_runout") or selected_data.get("unc_runout_pct"),
            "unc_entrainment_pct": selected_data.get("unc_entr") or selected_data.get("unc_entrainment") or selected_data.get("unc_entrainment_pct"),
            
            "notes": notes,
        }
        
        db.save_avalanche(data)
        st.success(f"✅ {selected_data.get('method')} saved successfully!")
        
        # Clear all session states
        for key in ["quick_inputs", "detailed_inputs", "runout_inputs"]:
            st.session_state.pop(key, None)
    else:
        st.warning("Calculate at least one method before saving.")
        
# ====================== VIEW LOG TAB ======================
with tab_log:
    st.subheader("📋 Research Database — Saved Avalanches")
    log_df = db.load_avalanche_log()
   
    if log_df.empty:
        st.info("No avalanches saved yet. Calculate and save some entries to build the database!")
    else:
        preferred_order = [
            "timestamp", "observer", "location", "method",
            "calculated_d_size", "field_assessed_d_size",
            "total_mass", "mass_tonnes", "entrainment_mass",
            "area_m2", "volume_m3",
            "crown_width_m", "slab_length_m", "depth_m",
            "hardness", "grain", "density_kgm3", "debris_type",
            "use_layered_density", "include_entrainment",
            "entr_width_m", "entr_length_m", "entr_area_m2", "entr_depth_m",
            "entr_hardness", "entr_grain", "entr_swe_mm",
            "snotel_station", "weak_layer_date", "release_date",
            "slab_swe_mm", "adjusted_swe_mm",
            "unc_lw_pct", "unc_depth_pct", "unc_density_pct",
            "unc_area_pct", "unc_swe_pct", "unc_entrainment_pct", "unc_runout_pct",
            "unc_low", "unc_high", "notes"
        ]
       
        available_cols = [col for col in preferred_order if col in log_df.columns]
        display_df = log_df[available_cols].copy()
       
        if "timestamp" in display_df.columns:
            display_df["timestamp"] = pd.to_datetime(display_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
       
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "timestamp": st.column_config.TextColumn("Date/Time"),
                "total_mass": st.column_config.NumberColumn("Total Mass (t)", format="%.0f"),
                "mass_tonnes": st.column_config.NumberColumn("Slab Mass (t)", format="%.0f"),
                "entrainment_mass": st.column_config.NumberColumn("Entrained Mass (t)", format="%.0f"),
                "volume_m3": st.column_config.NumberColumn("Volume (m³)", format="%.0f"),
                "density_kgm3": st.column_config.NumberColumn("Density (kg/m³)", format="%.0f"),
                "unc_lw_pct": st.column_config.NumberColumn("Unc. LW %", format="%.1f"),
                "unc_entrainment_pct": st.column_config.NumberColumn("Unc. Entr. %", format="%.1f"),
                "unc_runout_pct": st.column_config.NumberColumn("Unc. Runout %", format="%.1f"),
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