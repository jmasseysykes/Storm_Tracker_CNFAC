import streamlit as st
import pandas as pd
from datetime import date
import os
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
st.caption("CNFAC Avalanche Tools — V2.0 (single Start Zone + Runout methods) — ISSW Project with Erich Peitzsch, Zach Guy, Ron Simenhois, and Bruce Jamieson")

# === VIDEO WALKTHROUGH (commented out until video is uploaded) ===
# st.markdown("**New to the tool?** Open the 📹 **2-minute personal walkthrough** right below for a quick guided tour.")
#
# with st.expander("📹 Watch a 2-minute personal walkthrough (highly recommended for first-time users)", expanded=False):
#     # === HOW TO ADD YOUR VIDEO ===
#     # 1. Record a short ~2 min screen recording + voiceover
#     # 2. Upload to YouTube (unlisted is perfect) or host a direct .mp4
#     # 3. Update YOUTUBE_ID below
#     YOUTUBE_ID = "REPLACE_WITH_YOUR_YOUTUBE_VIDEO_ID"
#     DIRECT_MP4_URL = ""
#
#     if YOUTUBE_ID and "REPLACE_WITH_YOUR" not in YOUTUBE_ID:
#         st.components.v1.iframe(
#             f"https://www.youtube.com/embed/{YOUTUBE_ID}?rel=0&modestbranding=1",
#             height=420
#         )
#     elif DIRECT_MP4_URL:
#         st.video(DIRECT_MP4_URL)
#     else:
#         st.info("👆 Replace `YOUTUBE_ID` near the top of this file with your actual video.")
#
#     st.caption("A personal, short introduction to the current V2 interface — choosing density methods, using entrainment, and understanding the results.")

# === D-SIZE CLASSIFICATION CHART (collapsible) ===
with st.expander("📊 View D-Size Classification Chart — Mass Ranges and Typical Values", expanded=False):
    # Use path relative to this file so it works reliably in Docker/Render
    # regardless of working directory.
    chart_path = os.path.join(os.path.dirname(__file__), "..", "dsize_scott_binning.png")
    if os.path.exists(chart_path):
        st.image(
            chart_path,
            caption="Avalanche Destructive Size (D-Size) Classification — Mass Ranges and Typical Values (Log Scale) — Scott Log-Midpoint Binning",
            use_container_width=False,
            width=900                    # comfortable size on desktop
        )
    else:
        st.warning("Chart image not found. Please ensure dsize_scott_binning.png is in the project root.")
    
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


def _mark_slab_area_manual():
    st.session_state.sz_area_manual = True


def _slab_area_geometry_state():
    """Resolve whether slab area comes from width × length or a manual area entry."""
    crown_width = st.session_state.get("sz_crown_width")
    slab_length = st.session_state.get("sz_slab_length")
    area_display = st.session_state.get("sz_area_override")

    area_from_dims = (crown_width * slab_length) if crown_width and slab_length else None
    area_overridden = bool(st.session_state.get("sz_area_manual", False))

    if not area_overridden and area_display is not None and area_from_dims is not None:
        tol = max(0.5, 0.001 * area_from_dims)
        area_overridden = abs(area_display - area_from_dims) > tol

    if area_overridden:
        st.session_state.sz_area_manual = True

    geometry_mode = "area" if area_overridden else "dimensions"
    crown_width_m = None if area_overridden else (crown_width * conv_length if crown_width else None)
    slab_length_m = None if area_overridden else (slab_length * conv_length if slab_length else None)

    return {
        "area_display": area_display,
        "area_from_dims": area_from_dims,
        "area_overridden": area_overridden,
        "geometry_mode": geometry_mode,
        "crown_width_m": crown_width_m,
        "slab_length_m": slab_length_m,
    }


def _refresh_start_zone_fields(selected_data: dict) -> dict:
    """Recompute area/volume/mass from the current slab area input before saving."""
    geom = _slab_area_geometry_state()
    area_display = geom["area_display"]
    if area_display is None or area_display <= 0:
        return selected_data

    area_m2 = area_display * conv_area
    depth_m = selected_data.get("crown_depth_direct_m") or selected_data.get("depth_m")
    vol_m3 = area_m2 * depth_m if depth_m else selected_data.get("volume_m3")

    density = selected_data.get("density_kgm3") or 250.0
    swe_mm = selected_data.get("slab_swe_mm") or 0.0
    dens_mode = selected_data.get("density_mode") or ""

    if dens_mode == "SWE based density estimate" and swe_mm > 0:
        slab_m = area_m2 * (swe_mm / 1000.0)
    elif vol_m3:
        slab_m = vol_m3 * density / 1000.0
    else:
        slab_m = selected_data.get("mass_tonnes") or 0.0

    entr_m = selected_data.get("entrainment_mass") or 0.0
    total_m = slab_m + entr_m

    mid_label = calcs.mass_to_dsize(total_m)["label"] if total_m > 0 else selected_data.get("calculated_d_size")
    vol_label = calcs.volume_m3_to_dsize(vol_m3)["label"] if vol_m3 and vol_m3 > 0 else selected_data.get("dsize_volume_midpoint")

    updated = dict(selected_data)
    updated.update({
        "area_m2": area_m2,
        "volume_m3": vol_m3,
        "mass_tonnes": slab_m,
        "total_mass": total_m,
        "area_overridden": geom["area_overridden"],
        "geometry_mode": geom["geometry_mode"],
        "crown_width_m": geom["crown_width_m"],
        "slab_length_m": geom["slab_length_m"],
        "calculated_d_size": mid_label,
        "dsize_mass_original": mid_label,
        "dsize_mass_midpoint": mid_label,
        "dsize_volume_midpoint": vol_label,
        "dsize_method": "mass_midpoint",
    })
    return updated


# ====================== TABS ======================
tab_start, tab_runout, tab_log = st.tabs([
    "Start Zone Method",
    "Runout/Debris Method",
    "📋 View Saved Avalanches"
])

def render_save_section(prefix=""):
    """Render the save-to-research-DB section.
    Called only from inside the Start Zone and Runout tabs so it does not appear on the View Saved page.
    Use a prefix so keys are unique when the function is called from multiple tabs.
    """
    st.divider()
    st.subheader("💾 Save to Research Database")

    p = f"{prefix}_" if prefix else ""
    observer = st.text_input("Observer Name", value="Your Name", key=f"{p}observer")
    location = st.text_input("Avalanche Location", placeholder="e.g. Chugach Mtns, AK — crown visible from highway", key=f"{p}location")
    report_link = st.text_input("Avalanche Report Link (optional)", placeholder="https://... (e.g. public avalanche report or observation)", key=f"{p}report_link")
    field_d_size_options = ["D1", "D1.5", "D2", "D2.5", "D3", "D3.5", "D4", "D4.5", "D5"]
    field_assessed_d_size = st.selectbox("Field-Assessed D-Size (what you think it actually was)", field_d_size_options, key=f"{p}field_assessed_d_size")
    notes = st.text_area("Notes", placeholder="Optional observation notes, photos description, weather, etc.", height=100, key=f"{p}notes")

    # === SMART SAVE LOGIC - Supports Start Zone (combined) and Runout ===
    sz_data = st.session_state.get("start_zone_inputs")
    runout_data = st.session_state.get("runout_inputs")

    save_choice = None
    selected_data = None
    if sz_data or runout_data:
        options = []
        if sz_data:
            options.append(("Start Zone Method", sz_data))
        if runout_data:
            options.append(("Runout/Debris Method", runout_data))

        if len(options) > 1:
            save_choice = st.radio(
                "Which calculation do you want to save?",
                [name for name, _ in options],
                horizontal=True,
                key=f"{p}save_choice"
            )
            selected_data = next(data for name, data in options if name == save_choice)
        elif options:
            save_choice, selected_data = options[0]

    if st.button("💾 Save Avalanche to Research Database", type="primary", use_container_width=True, key=f"{p}save_button"):
        if selected_data:
            if selected_data.get("method") == "start_zone":
                selected_data = _refresh_start_zone_fields(selected_data)
            data = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "observer": observer,
                "location": location,
                "report_link": report_link,
                "schema_version": selected_data.get("schema_version"),
                "method": selected_data.get("method"),
                "geometry_mode": selected_data.get("geometry_mode"),
                "density_mode": selected_data.get("density_mode"),
                "density_profile": selected_data.get("density_profile"),
                "swe_source": selected_data.get("swe_source"),
                "area_overridden": selected_data.get("area_overridden"),
                "entrainment_method": selected_data.get("entrainment_method") or selected_data.get("entrainment_method_choice"),
                "area_m2": selected_data.get("area_m2"),
                "volume_m3": selected_data.get("volume_m3"),
                "mass_tonnes": selected_data.get("mass_tonnes"),
                "entrainment_mass": selected_data.get("entrainment_mass"),
                "total_mass": selected_data.get("total_mass") or selected_data.get("mass_tonnes"),
                "calculated_d_size": selected_data.get("calculated_d_size"),
                "dsize_method": selected_data.get("dsize_method"),
                "original_calculated_d_size": selected_data.get("original_calculated_d_size"),
                "dsize_mass_original": selected_data.get("dsize_mass_original"),
                "dsize_mass_midpoint": selected_data.get("dsize_mass_midpoint"),
                "dsize_volume_midpoint": selected_data.get("dsize_volume_midpoint"),
                "unc_low": selected_data.get("unc_low"),
                "unc_high": selected_data.get("unc_high"),
                "field_assessed_d_size": field_assessed_d_size,
                
                # Slab / Start Zone fields
                "crown_width_m": selected_data.get("crown_width_m"),
                "slab_length_m": selected_data.get("slab_length_m"),
                "depth_m": selected_data.get("depth_m"),
                "crown_depth_direct_m": selected_data.get("crown_depth_direct_m"),
                "crown_depth_derived_m": selected_data.get("crown_depth_derived_m"),
                "hardness": selected_data.get("hardness"),
                "grain": selected_data.get("grain"),
                "density_kgm3": selected_data.get("density_kgm3"),
                "use_layered_density": selected_data.get("use_layered_density"),
                
                # (Legacy detailed/SNOTEL fields - populated if present)
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
            
            try:
                db.save_avalanche(data)
            except Exception as e:
                st.error(f"Could not save to the research database: {e}")
                st.stop()
            st.success(f"✅ {save_choice} saved successfully!")
            
            # Clear all session states
            for key in ["start_zone_inputs", "runout_inputs", "quick_inputs", "detailed_inputs"]:
                st.session_state.pop(key, None)
        else:
            st.warning("Calculate at least one method before saving.")


with tab_start:
    st.subheader("Start Zone Method — Slab Volume & Mass Estimate")
    st.caption("Estimate avalanche destructive size from the starting zone. Provide slab dimensions (or a polygon area) and crown depth, then select how to determine density and mass.")

    # Load stations (for optional SNOTEL SWE) - cached to save memory
    @st.cache_data
    def load_stations():
        try:
            stations_df = pd.read_csv("SNOTEL_station_list.csv")
            stations_df["ID"] = stations_df["site_name"].str.extract(r"\((\d+)\)", expand=False).astype(str)
            stations_df["display_name"] = stations_df["site_name"] + " - " + stations_df["state"]
            stations_df = stations_df[["display_name", "ID", "state"]].dropna()
            return stations_df
        except Exception:
            return None

    stations_df = load_stations()
    if stations_df is not None:
        station_options = stations_df["display_name"].tolist()
    else:
        station_options = ["(SNOTEL list unavailable)"]

    hardness_options = ["F-", "F", "F+", "4F-", "4F", "4F+", "1F-", "1F", "1F+", "P-", "P", "P+", "K-", "K", "K+"]
    grain_options = [
        "Precipitation Particles (PP)", "Graupel (PPgp)", "Decomposing/Fragmented (DF)",
        "Rounded Grains (RG)", "Faceted Rounded (RGxf)", "Faceted Crystals (FC)",
        "Rounding Faceted (FCxr)", "Depth Hoar (DH)", "Melt-Freeze Crust (MFcr)"
    ]

    # 1. Slab dimensions
    st.markdown("### 1. Slab dimensions")
    st.caption("Enter crown width and slab length to calculate area (or override with a mapped polygon area). Record your best field estimate of average crown depth. These values are used to compute slab volume.")
    col1, col2 = st.columns([1, 1])
    with col1:
        crown_width = st.number_input(f"Crown Width ({unit_length})",
                                      value=80.0 if not use_imperial else 250.0,
                                      min_value=1.0, step=1.0, key="sz_crown_width")
        slab_length = st.number_input(f"Slab Length — crown to stauchwall ({unit_length})",
                                      value=150.0 if not use_imperial else 500.0,
                                      min_value=1.0, step=1.0, key="sz_slab_length")
        area_from_dims = crown_width * slab_length
        area_from_dims_m2 = area_from_dims * conv_area
        st.caption(
            f"Computed slab area from dimensions: **{area_from_dims:,.0f} {unit_area}** "
            f"({area_from_dims_m2:,.0f} m²) — width × length"
        )

        if not st.session_state.get("sz_area_manual", False):
            st.session_state.sz_area_override = area_from_dims

        area_display = st.number_input(
            f"Slab area ({unit_area})",
            min_value=10.0,
            step=1.0,
            key="sz_area_override",
            on_change=_mark_slab_area_manual,
            help="Defaults to width × length. Edit this field to use a mapped polygon area instead.",
        )
        if st.session_state.get("sz_area_manual"):
            if st.button("↺ Reset area to width × length", key="sz_area_reset"):
                st.session_state.sz_area_manual = False
                st.session_state.sz_area_override = area_from_dims
                st.rerun()
        crown_width_m = crown_width * conv_length
        slab_length_m = slab_length * conv_length

    with col2:
        direct_depth_display = st.number_input(f"Crown depth / slab thickness — direct field estimate ({unit_length})",
                                               value=0.8 if not use_imperial else 2.5,
                                               min_value=0.05, step=0.05, key="sz_direct_depth")
        direct_depth_m = direct_depth_display * conv_length

    with st.expander("🔧 Uncertainty for Slab Dimensions (optional)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            unc_area = st.slider("Area (length × width) uncertainty %", 0, 50, 15, key="sz_unc_area")
        with c2:
            unc_depth = st.slider("Depth uncertainty %", 0, 50, 15, key="sz_unc_depth")

    # 2. DENSITY
    st.markdown("### 2. Slab Density")
    dens_mode = st.radio(
        "Density approach",
        ["Layer density + grain type", "SWE based density estimate"],
        horizontal=True,
        key="sz_dens_mode"
    )

    if dens_mode == "Layer density + grain type":
        st.caption("Use hand hardness and grain type observed in the snowpack to estimate density. The simple option uses one value for the whole slab; the detailed option lets you define multiple layers with different properties.")
    else:
        st.caption("Use snow water equivalent (SWE) to calculate slab mass directly from area. Density is back-calculated from the crown depth you measured. Grain type is not required.")

    density = 250.0
    swe_mm = 0.0
    depth_derived_m = None
    use_layered = False
    station_name = None
    use_swe_for_mass = False
    hardness = None
    grain = None

    if dens_mode == "Layer density + grain type":
        use_swe_for_mass = False
        # Second radio: simple hand hardness + grain is default inside "Layer density + grain type"
        profile = st.radio(
            "Density from grain type / hardness",
            ["Simple (hand hardness + grain type)", "Detailed layer profile"],
            horizontal=True,
            key="sz_density_profile",
            index=0
        )

        if profile == "Simple (hand hardness + grain type)":
            use_layered = False
            hardness = st.selectbox("Hand Hardness (Slab)", hardness_options, index=4, key="sz_hard")
            grain = st.selectbox("Grain Type (Slab)", grain_options, index=3, key="sz_grain")
            density = calcs.get_density_from_hardness_grain(hardness, grain)
            st.caption(f"Computed slab density: **{density:.0f} kg/m³** (from hardness + grain)")
        else:
            # Detailed multi-layer
            use_layered = True
            st.markdown("**Define Slab Layers (top to bottom)**")
            if "sz_layers" not in st.session_state:
                st.session_state.sz_layers = [{"thickness_cm": 30, "hardness": "4F", "grain": "Rounded Grains (RG)"}]
            layers = st.session_state.sz_layers
            for i in range(len(layers)):
                ca, cb, cc, cd = st.columns([2, 2, 2, 1])
                with ca:
                    layers[i]["thickness_cm"] = st.number_input(f"Layer {i+1} Thickness (cm)",
                                                                value=layers[i]["thickness_cm"],
                                                                min_value=1, key=f"sz_thick_{i}")
                with cb:
                    layers[i]["hardness"] = st.selectbox("Hardness", hardness_options,
                                                         index=hardness_options.index(layers[i]["hardness"]),
                                                         key=f"sz_hard_{i}")
                with cc:
                    layers[i]["grain"] = st.selectbox("Grain Type", grain_options,
                                                      index=grain_options.index(layers[i]["grain"]),
                                                      key=f"sz_grain_{i}")
                with cd:
                    if st.button("🗑", key=f"sz_del_{i}"):
                        layers.pop(i)
                        st.rerun()
            if st.button("➕ Add Layer", key="sz_add_layer"):
                layers.append({"thickness_cm": 20, "hardness": "4F", "grain": "Rounded Grains (RG)"})
                st.rerun()
            # compute weighted density
            tot_th = 0.0
            w_dens = 0.0
            for ly in layers:
                d = calcs.get_density_from_hardness_grain(ly["hardness"], ly["grain"])
                w_dens += d * ly["thickness_cm"]
                tot_th += ly["thickness_cm"]
            density = w_dens / tot_th if tot_th > 0 else 250.0
            st.caption(f"Computed average slab density: **{density:.0f} kg/m³** (from layers)")
    else:
        # SWE mode
        use_swe_for_mass = True
        use_layered = False
        swe_src = st.radio("SWE source", ["Manual entry", "SNOTEL station"], horizontal=True, key="sz_swe_src")
        if swe_src == "Manual entry":
            m_swe = st.number_input(f"Slab SWE ({swe_unit})", value=30.0 if not use_imperial else 1.2,
                                    min_value=0.0, step=1.0 if not use_imperial else 0.1, key="sz_swe_manual")
            swe_mm = m_swe * (25.4 if use_imperial else 1.0)
            station_name = "Manual"
        else:
            sel = st.selectbox("SNOTEL Station (Slab)", station_options, key="sz_snotel_sel")
            w_date = st.date_input("Weak layer / start of slab accumulation", value=date(2024, 12, 1), key="sz_weak")
            r_date = st.date_input("Release date", value=date.today(), key="sz_release")
            adj = st.slider("Local adjustment to SWE (%)", -50, 50, 0, key="sz_swe_adj")
            if sel and stations_df is not None and "ID" in stations_df.columns:
                try:
                    row = stations_df[stations_df["display_name"] == sel].iloc[0]
                    triplet = f"{row['ID']}:{row['state']}:SNTL"
                    data = snotel.get_slab_swe(triplet, str(w_date), str(r_date))
                    base = data.get("slab_swe_mm", 0) if isinstance(data, dict) else 0
                    swe_mm = base * (1 + adj / 100.0)
                    station_name = sel.split(" - ")[0] if " - " in sel else sel
                except Exception as ex:
                    st.warning(f"SNOTEL fetch issue: {ex}")
                    swe_mm = 0
                    station_name = sel
            else:
                swe_mm = 0
        # With direct crown depth (Step 1) + SWE we compute mass directly + implied density.
        # (Hand hardness + grain is only for the "Layer density + grain type" path when no SWE is available.)
        density = 250.0
        depth_derived_m = None
        if swe_mm > 0 and direct_depth_m > 0:
            # density_kg/m3 = (SWE in mm / depth in m) numerically, since mass/area in t/m2 / depth_m *1000 → kg/m3
            density = swe_mm / direct_depth_m
            st.caption(f"Implied slab density (from SWE + direct depth): **{density:.0f} kg/m³**")
            st.caption(f"Volume uses the direct crown depth: **{direct_depth_m:.2f} m**")
        else:
            st.caption("Provide SWE and direct crown depth (from Step 1) to derive slab density.")

    # Density/SWE uncertainty
    with st.expander("🔧 Uncertainty for Slab Density / SWE (optional)", expanded=False):
        if dens_mode == "Layer density + grain type":
            unc_dens = st.slider("Density uncertainty %", 0, 50, 20, key="sz_unc_density")
            unc_swe_v = 0
        else:
            unc_swe_v = st.slider("SWE uncertainty %", 0, 50, 10, key="sz_unc_swe")
            unc_dens = 0

    # 3. ENTRAINMENT (optional)
    st.markdown("### 3. Entrainment (optional)")
    st.caption("Add mass from snow picked up along the avalanche path. Use dimensions + grain type or a SWE value. This is optional and can significantly increase the total mass.")
    include_entr = st.toggle("Include Entrainment Mass", value=False, key="sz_entr_toggle")
    entr_method = None
    if include_entr:
        st.markdown("**Entrainment Estimation**")
        entr_method = st.radio("Entrainment method",
                               ["Dimensions + Hardness/Grain", "SWE-based"],
                               horizontal=True, key="sz_entr_method")
        ce1, ce2 = st.columns([1, 1])
        with ce1:
            e_w = st.number_input(f"Entrainment Width ({unit_length})", value=120.0 if not use_imperial else 400.0,
                                  min_value=1.0, step=1.0, key="sz_entr_w")
            e_l = st.number_input(f"Entrainment Length ({unit_length})", value=250.0 if not use_imperial else 800.0,
                                  min_value=1.0, step=1.0, key="sz_entr_l")
            e_a = st.number_input(f"Entrainment Area ({unit_area}) — optional", value=e_w * e_l,
                                  min_value=10.0, key="sz_entr_a")
            e_d = st.number_input(f"Entrainment Depth ({unit_length})", value=0.3 if not use_imperial else 1.0,
                                  min_value=0.05, step=0.05, key="sz_entr_d")
        with ce2:
            if entr_method == "Dimensions + Hardness/Grain":
                e_h = st.selectbox("Entrainment Hardness", hardness_options, index=1, key="sz_entr_h")
                e_g = st.selectbox("Entrainment Grain Type", grain_options, index=0, key="sz_entr_g")
                e_swe = None
            else:
                e_h = None
                e_g = None
                e_swe = st.number_input(f"Entrainment SWE ({swe_unit}) — optional", value=40.0 if not use_imperial else 1.5,
                                        min_value=0.0, step=1.0, key="sz_entr_swe")
        with st.expander("🔧 Uncertainty for Entrainment (optional)", expanded=False):
            ca, cb = st.columns(2)
            with ca:
                u_ea = st.slider("Entrainment area uncertainty %", 0, 50, 15, key="sz_unc_entr_a")
                u_ed = st.slider("Entrainment depth uncertainty %", 0, 50, 15, key="sz_unc_entr_d")
            with cb:
                u_e = st.slider("Entrainment density/SWE uncertainty %", 0, 50, 25, key="sz_unc_entr")
    else:
        u_ea = u_ed = u_e = 0
        e_w = e_l = e_a = e_d = 0
        e_h = e_g = e_swe = None

    if st.button("Calculate Start Zone Estimate", type="primary", use_container_width=True, key="sz_calc_btn"):
        geom = _slab_area_geometry_state()
        area_overridden = geom["area_overridden"]
        # Geometry — use the slab area field (manual entry or width × length default)
        area_m2 = geom["area_display"] * conv_area
        crown_width_m = geom["crown_width_m"]
        slab_length_m = geom["slab_length_m"]
        # Use the direct crown depth from Step 1.
        # (In SWE mode we also derive implied density = SWE / depth; grain-based derivation is not used.)
        crown_depth_m = direct_depth_m
        depth_note = "direct field estimate"
        vol_m3 = area_m2 * crown_depth_m

        # Mass
        if use_swe_for_mass:
            slab_m = area_m2 * (swe_mm / 1000.0)
            dens_for_store = density  # the derivation one
        else:
            slab_m = vol_m3 * density / 1000.0
            dens_for_store = density

        # Entrainment
        entr_res = {"entrainment_mass": 0.0, "entr_swe_mm": 0.0, "entr_area_m2": 0.0, "source": ""}
        if include_entr:
            entr_res = calcs.calculate_entrainment(
                entr_area_display = st.session_state.get("sz_entr_a", e_a),
                entr_depth_display = st.session_state.get("sz_entr_d", e_d),
                method = st.session_state.get("sz_entr_method", entr_method or ""),
                manual_swe_display = st.session_state.get("sz_entr_swe", e_swe),
                hardness = st.session_state.get("sz_entr_h", e_h),
                grain = st.session_state.get("sz_entr_g", e_g),
                use_imperial=use_imperial,
                conv_length=conv_length,
                conv_area=conv_area,
            )
        entr_m = entr_res.get("entrainment_mass", 0.0)
        total_m = slab_m + entr_m

        # RSS unc - only relevant
        ua = st.session_state.get("sz_unc_area", 15)
        ud = st.session_state.get("sz_unc_depth", 15)
        rel = (ua / 100.0)**2 + (ud / 100.0)**2
        udens = st.session_state.get("sz_unc_density", 20) if not use_swe_for_mass else 0
        uswe = st.session_state.get("sz_unc_swe", 10) if use_swe_for_mass else 0
        if udens: rel += (udens/100.0)**2
        if uswe: rel += (uswe/100.0)**2
        if include_entr:
            rel += (st.session_state.get("sz_unc_entr_a", 15)/100.0)**2
            rel += (st.session_state.get("sz_unc_entr_d", 15)/100.0)**2
            rel += (st.session_state.get("sz_unc_entr", 25)/100.0)**2
        rel = rel ** 0.5
        lo_m = total_m / (1 + rel)
        hi_m = total_m * (1 + rel)

        mid_label, lo_label, hi_label = calcs.get_uncertainty_mass_range(
            total_m, ua, ud, udens or 0, uswe or 0
        )

        st.success(f"**Estimated D-Size: {mid_label}** (uncertainty range: **{lo_label} – {hi_label}**)")
        st.info(f"**Slab Mass**: {slab_m:,.0f} t | **Entrained Mass**: {entr_m:,.0f} t | **Total Mass**: {total_m:,.0f} t ({lo_m:,.0f} – {hi_m:,.0f} t)")

        area_used_display = geom["area_display"]
        area_note = "manual area entry" if area_overridden else "width × length"
        st.caption(
            f"Slab area used: **{area_m2:,.0f} m²** ({area_used_display:,.0f} {unit_area}) — {area_note}"
        )
        if dens_mode == "SWE based density estimate":
            st.caption(f"Crown depth used: **{crown_depth_m:.2f} m** (direct) | Implied density from SWE: **{density:.0f} kg/m³**")
        else:
            st.caption(f"Crown depth used for volume: **{crown_depth_m:.2f} m** ({depth_note})")

        if include_entr and entr_m > 0:
            src = entr_res.get("source", "")
            st.caption(f"Entrainment: {entr_res.get('entr_area_m2',0):,.0f} m² × SWE {entr_res.get('entr_swe_mm',0):,.1f} mm (source: {src})")
            for n in entr_res.get("notes", []):
                st.caption(n) if "large" not in n.lower() else st.warning(n)

        # Capture method choices for research database (to distinguish "quick" vs "detailed" approaches)
        density_profile = st.session_state.get("sz_density_profile") if dens_mode == "Layer density + grain type" else None
        swe_source = st.session_state.get("sz_swe_src") if dens_mode == "SWE based density estimate" else None
        entrainment_method_choice = st.session_state.get("sz_entr_method") if include_entr else None

        # Save to session for DB
        st.session_state.start_zone_inputs = {
            "schema_version": "2.0",
            "method": "start_zone",
            "geometry_mode": geom["geometry_mode"],
            "density_mode": dens_mode,
            "density_profile": density_profile,
            "swe_source": swe_source,
            "area_overridden": area_overridden,
            "area_m2": area_m2,
            "volume_m3": vol_m3,
            "depth_m": crown_depth_m,
            "crown_depth_direct_m": direct_depth_m,
            "crown_depth_derived_m": depth_derived_m,
            "crown_width_m": crown_width_m,
            "slab_length_m": slab_length_m,
            "mass_tonnes": slab_m,
            "entrainment_mass": entr_m,
            "total_mass": total_m,
            "density_kgm3": dens_for_store,
            "use_layered_density": use_layered,
            "hardness": hardness,
            "grain": grain,
            "slab_swe_mm": swe_mm if use_swe_for_mass else None,
            "snotel_station": station_name,
            "include_entrainment": include_entr,
            "entrainment_method": entrainment_method_choice,
            "entr_width_m": e_w * conv_length if include_entr else None,
            "entr_length_m": e_l * conv_length if include_entr else None,
            "entr_area_m2": entr_res.get("entr_area_m2"),
            "entr_depth_m": entr_res.get("entr_depth_m"),
            "entr_hardness": e_h if include_entr else None,
            "entr_grain": e_g if include_entr else None,
            "entr_swe_mm": entr_res.get("entr_swe_mm"),
            "calculated_d_size": mid_label,
            "unc_low": lo_label,
            "unc_high": hi_label,
            "low_mass": lo_m,
            "high_mass": hi_m,
            "dsize_mass_original": mid_label,
            "dsize_mass_midpoint": mid_label,
            "dsize_volume_midpoint": calcs.volume_m3_to_dsize(vol_m3)["label"] if vol_m3 > 0 else None,
            "dsize_method": "mass_midpoint",
            "unc_area_pct": ua,
            "unc_depth_pct": ud,
            "unc_density_pct": udens,
            "unc_swe_pct": uswe,
            "unc_entrainment_pct": st.session_state.get("sz_unc_entr", 25) if include_entr else 0,
        }
        fig = dsize_plot.plot_dsize_with_user_mass(total_m, lo_m, hi_m)
        st.pyplot(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)  # free memory

    render_save_section(prefix="start")


# ====================== RUNOUT / DEBRIS ESTIMATE ======================
with tab_runout:
    st.subheader("Runout/Debris Method")
    st.caption("Volume-focused estimate for the deposit. Debris density is highly variable, so D-size is calculated directly from measured deposit volume using the Jamieson 2024 bins (mass is optional if you provide a custom density).")

    # Reference volume chart (static)
    with st.expander("📊 View Deposit Volume D-Size Classification Chart (Jamieson 2024)", expanded=False):
        vol_chart_path = os.path.join(os.path.dirname(__file__), "..", "dsize_volume_reference.png")
        if os.path.exists(vol_chart_path):
            st.image(
                vol_chart_path,
                caption="Avalanche Destructive Size (D-Size) Classification — Deposit Volume Ranges and Typical Values (Log Scale)",
                use_container_width=False,
                width=900
            )
        else:
            st.warning("Volume chart image not found. Please ensure dsize_volume_reference.png is in the project root.")

    st.caption("Measure or estimate the deposit length, width, and average thickness. Volume is the primary input for D-size.")

    col1, col2 = st.columns([1, 1])
    
    with col1:
        deposit_length = st.number_input(f"Deposit Length (along path) ({unit_length})", 
                                         value=400.0 if use_imperial else 120.0, 
                                         min_value=10.0, max_value=5000.0 if not use_imperial else 15000.0, step=1.0, key="runout_length")
        deposit_width = st.number_input(f"Deposit Width ({unit_length})", 
                                        value=150.0 if use_imperial else 45.0, 
                                        min_value=10.0, max_value=3000.0 if not use_imperial else 10000.0, step=1.0, key="runout_width")
        
        # Provide custom density moved to bottom of column 1
        use_custom_density = st.toggle(
            "Provide custom debris density for optional mass reference",
            value=False,
            help="Disable for pure volume-based estimate (recommended). Enable to manually enter a density for mass reference only.",
            key="runout_custom_density"
        )
        
        density = 500
        if use_custom_density:
            density = st.number_input(
                "Custom Deposit Density (kg/m³)",
                value=500,
                min_value=100,
                max_value=1200,
                step=10,
                help="Optional. Used only to show a mass reference. D-size and main estimate are always volume-based.",
                key="runout_density"
            )
    
    with col2:
        # Deposit area at top of column 2
        area_runout = st.number_input(
            f"Deposit Area ({unit_area}) — optional",
            value=deposit_length * deposit_width,
            min_value=50.0,
            help="Auto-calculated from Length × Width. Override if you have a polygon measurement.",
            key="runout_area"
        )
        
        # Average deposit thickness below area in column 2
        avg_thickness = st.number_input(f"Average Deposit Thickness ({unit_length})", 
                                        value=1.5 if use_imperial else 0.5, 
                                        min_value=0.1, max_value=30.0 if not use_imperial else 100.0, step=0.05, key="runout_thickness")
    
    # Advanced uncertainty moved underneath, full width of both columns (same as other modules)
    with st.expander("🔧 Advanced Uncertainty (per input) — RSS Method", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            unc_area = st.slider("Deposit Area (length × width) uncertainty %", 0, 50, 15, key="runout_unc_area")
            unc_depth = st.slider("Deposit Depth / Thickness uncertainty %", 0, 50, 15, key="runout_unc_depth")
    
    if st.button("Calculate Runout/Debris Method", type="primary", use_container_width=True, key="calculate_runout"):
        length_m = deposit_length * conv_length
        width_m = deposit_width * conv_length
        thickness_m = avg_thickness * conv_length
        area_m2 = area_runout * conv_area
        
        volume_m3 = area_m2 * thickness_m

        # === VOLUME-BASED D-SIZE (using Jamieson 2024 bins) - always ===
        vol_d = calcs.volume_m3_to_dsize(volume_m3)
        mid_d = vol_d["label"]

        # Advanced uncertainty using RSS (same as start zone modules)
        unc_area_val = st.session_state.get("runout_area", 15)
        unc_depth_val = st.session_state.get("runout_depth", 15)
        total_rel_unc = (
            (unc_area_val / 100.0)**2 +
            (unc_depth_val / 100.0)**2
        ) ** 0.5
        vol_factor = 1 + total_rel_unc
        vol_low = max(0.0, volume_m3 / vol_factor)
        vol_high = volume_m3 * vol_factor
        low_d = calcs.volume_m3_to_dsize(vol_low)["label"]
        high_d = calcs.volume_m3_to_dsize(vol_high)["label"]

        st.success(f"**Estimated D-Size (from volume): {mid_d}** (uncertainty range: **{low_d} – {high_d}**)")
        
        st.info(f"**Deposit Volume**: {volume_m3:,.0f} m³ ({vol_low:,.0f} – {vol_high:,.0f} m³)")

        if use_custom_density:
            # Optional mass reference only if user provided custom density
            vol_res = calcs.estimate_mass_from_volume(volume_m3, nominal_density=density)
            mass_nom = vol_res["mass_tonnes_nominal"]
            # Use the volume uncertainty (RSS from area + depth) for the mass range
            mass_low = vol_low * (density / 1000.0)
            mass_high = vol_high * (density / 1000.0)
            st.info(
                f"**Est. Mass (using your custom density {density} kg/m³)**: {mass_nom:,.0f} t "
                f"({mass_low:,.0f} – {mass_high:,.0f} t)"
            )
        else:
            mass_nom = 0.0
            mass_low = 0.0
            mass_high = 0.0
        
        # Removed redundant st.caption — D-size range is already shown in the green success line above
        
        # Store — volume is the basis for D-size. Mass only if custom density used.
        runout_store = {
            "schema_version": "2.0",
            "method": "runout_debris",
            "area_m2": area_m2,
            "volume_m3": volume_m3,
            "entrainment_mass": 0,
            "calculated_d_size": mid_d,
            "unc_low": low_d,
            "unc_high": high_d,
            "unc_area_pct": unc_area_val,
            "unc_depth_pct": unc_depth_val,
            "unc_runout_pct": round(total_rel_unc * 100, 1),  # combined RSS for backward compat
            "volume_focused": True,
            "volume_d_size": mid_d,
            "dsize_method": "volume_midpoint",
            "dsize_mass_original": None,
            "dsize_mass_midpoint": mass_nom if mass_nom > 0 else None,  # computed even for runout if mass was estimated
            "dsize_volume_midpoint": mid_d,
        }
        if use_custom_density:
            runout_store.update({
                "mass_tonnes": mass_nom,
                "mass_tonnes_low": mass_low,
                "mass_tonnes_high": mass_high,
                "total_mass": mass_nom,
                "density_kgm3": density,
            })
        else:
            runout_store.update({
                "mass_tonnes": 0,
                "mass_tonnes_low": 0,
                "mass_tonnes_high": 0,
                "total_mass": 0,
            })
        
        st.session_state.runout_inputs = runout_store
        
        # Plot the volume-based D-Size with user's volume + uncertainty bands (no mass needed)
        fig = dsize_plot.plot_dsize_volume_with_user_value(volume_m3, vol_low, vol_high)
        st.pyplot(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)  # free memory
        
        if use_custom_density:
            st.caption(f"D-Size from volume only. Optional mass reference shown using your custom density of {density} kg/m³.")
        else:
            st.caption("D-Size and estimate based purely on deposit volume (no density used).")
            
    render_save_section(prefix="runout")


# ====================== VIEW LOG TAB ======================
with tab_log:
    st.subheader("📋 Research Database — Saved Avalanches")
    @st.cache_data(ttl=300)  # cache 5 min to reduce DB load/memory
    def _load_log():
        return db.load_avalanche_log()
    log_df = _load_log()
   
    if log_df.empty:
        st.info("No avalanches saved yet. Calculate and save some entries to build the database!")
    else:
        # Show migration status
        if 'dsize_method' in log_df.columns:
            unmigrated = log_df[log_df['dsize_method'].isna() | (log_df['dsize_method'] == '')].shape[0]
            if unmigrated > 0:
                st.info(f"{unmigrated} records have not been migrated yet to the midpoint columns. Run 'Calculate' + Save on old entries or restart the app to trigger migration.")
        st.caption("V2 records (schema_version='2.0') use the consistent naming scheme. The table and CSV list columns in the clean V2-preferred order. Legacy columns from earlier versions are preserved for historical data.")
        preferred_order = [
            "timestamp", "observer", "location", "report_link", "schema_version", "method",
            "geometry_mode", "density_mode", "density_profile", "swe_source",
            "area_overridden", "entrainment_method_choice", "entrainment_method",
            "calculated_d_size", "dsize_method",
            "dsize_mass_original", "dsize_mass_midpoint", "dsize_volume_midpoint",
            "field_assessed_d_size",
            "total_mass", "mass_tonnes", "entrainment_mass",
            "area_m2", "volume_m3",
            "crown_width_m", "slab_length_m", "depth_m",
            "crown_depth_direct_m", "crown_depth_derived_m",
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

        # The three explicit D-size columns are now stored.
        # We keep a simple fallback column for very old records.
        try:
            def get_fallback(row):
                mass = row.get("mass_tonnes") or row.get("total_mass")
                vol = row.get("volume_m3")
                if pd.notna(vol) and vol > 0:
                    return calcs.volume_m3_to_dsize(vol)["label"]
                return calcs.get_current_mass_dsize(mass)
            display_df["dsize_fallback"] = display_df.apply(get_fallback, axis=1)
        except Exception:
            pass  # fail gracefully
       
        if "timestamp" in display_df.columns:
            display_df["timestamp"] = pd.to_datetime(display_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")

        # Rename columns ONLY for display and CSV export.
        # This presents a clean, consistent naming scheme on the site and downloads
        # without touching the underlying database (old data and column names are preserved).
        display_rename = {
            "unc_lw_pct": "unc_slab_lw_pct",
            "unc_area_pct": "unc_slab_area_pct",
            "unc_depth_pct": "unc_slab_depth_pct",
            "unc_density_pct": "unc_slab_density_pct",
            "unc_swe_pct": "unc_slab_swe_pct",
            "density_kgm3": "slab_density_kg_m3",
            "unc_low": "dsize_unc_low",
            "unc_high": "dsize_unc_high",
            "calculated_d_size": "dsize_calculated",
            "area_m2": "slab_area_m2",
            "volume_m3": "slab_volume_m3",
            "mass_tonnes": "slab_mass_t",
            "total_mass": "total_mass_t",
            "entrainment_mass": "entrainment_mass_t",
        }
        display_df = display_df.rename(columns=display_rename)
       
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "timestamp": st.column_config.TextColumn("Date/Time"),
                "report_link": st.column_config.LinkColumn("Avalanche Report", display_text="🔗 Open Report"),
                "schema_version": st.column_config.TextColumn("Schema Version"),
                "method": st.column_config.TextColumn("Method"),
                "geometry_mode": st.column_config.TextColumn("Geometry Mode"),
                "density_mode": st.column_config.TextColumn("Density Mode"),
                "density_profile": st.column_config.TextColumn("Density Profile"),
                "swe_source": st.column_config.TextColumn("SWE Source"),
                "area_overridden": st.column_config.TextColumn("Area Overridden?"),
                "entrainment_method_choice": st.column_config.TextColumn("Entrainment Method Choice"),
                "entrainment_method": st.column_config.TextColumn("Entrainment Method"),
                "dsize_calculated": st.column_config.TextColumn("D-size (calculated)"),
                "dsize_method": st.column_config.TextColumn("D-size Method"),
                "dsize_mass_original": st.column_config.TextColumn("D-size Mass (original)"),
                "dsize_mass_midpoint": st.column_config.TextColumn("D-size Mass (midpoint)"),
                "dsize_volume_midpoint": st.column_config.TextColumn("D-size Volume (midpoint)"),
                "field_assessed_d_size": st.column_config.TextColumn("Field-Assessed D-Size"),
                "total_mass_t": st.column_config.NumberColumn("Total Mass (t)", format="%.0f"),
                "slab_mass_t": st.column_config.NumberColumn("Slab Mass (t)", format="%.0f"),
                "entrainment_mass_t": st.column_config.NumberColumn("Entrained Mass (t)", format="%.0f"),
                "slab_area_m2": st.column_config.NumberColumn("Slab Area (m²)", format="%.0f"),
                "slab_volume_m3": st.column_config.NumberColumn("Slab Volume (m³)", format="%.0f"),
                "crown_width_m": st.column_config.NumberColumn("Crown Width (m)", format="%.0f"),
                "slab_length_m": st.column_config.NumberColumn("Slab Length (m)", format="%.0f"),
                "slab_depth_m": st.column_config.NumberColumn("Slab Depth (m)", format="%.2f"),
                "crown_depth_direct_m": st.column_config.NumberColumn("Crown Depth Direct (m)", format="%.2f"),
                "crown_depth_derived_m": st.column_config.NumberColumn("Crown Depth Derived (m)", format="%.2f"),
                "hardness": st.column_config.TextColumn("Hardness"),
                "grain": st.column_config.TextColumn("Grain Type"),
                "slab_density_kg_m3": st.column_config.NumberColumn("Slab Density (kg/m³)", format="%.0f"),
                "debris_type": st.column_config.TextColumn("Debris Type"),
                "use_layered_density": st.column_config.TextColumn("Use Layered Density"),
                "include_entrainment": st.column_config.TextColumn("Include Entrainment"),
                "entr_width_m": st.column_config.NumberColumn("Entrainment Width (m)", format="%.0f"),
                "entr_length_m": st.column_config.NumberColumn("Entrainment Length (m)", format="%.0f"),
                "entr_area_m2": st.column_config.NumberColumn("Entrainment Area (m²)", format="%.0f"),
                "entr_depth_m": st.column_config.NumberColumn("Entrainment Depth (m)", format="%.2f"),
                "entr_hardness": st.column_config.TextColumn("Entrainment Hardness"),
                "entr_grain": st.column_config.TextColumn("Entrainment Grain"),
                "entr_swe_mm": st.column_config.NumberColumn("Entrainment SWE (mm)", format="%.1f"),
                "snotel_station": st.column_config.TextColumn("SNOTEL Station"),
                "weak_layer_date": st.column_config.TextColumn("Weak Layer Date"),
                "release_date": st.column_config.TextColumn("Release Date"),
                "slab_swe_mm": st.column_config.NumberColumn("Slab SWE (mm)", format="%.1f"),
                "adjusted_swe_mm": st.column_config.NumberColumn("Adjusted SWE (mm)", format="%.1f"),
                "unc_slab_lw_pct": st.column_config.NumberColumn("Unc. Slab LW %", format="%.1f"),
                "unc_slab_area_pct": st.column_config.NumberColumn("Unc. Slab Area %", format="%.1f"),
                "unc_slab_depth_pct": st.column_config.NumberColumn("Unc. Slab Depth %", format="%.1f"),
                "unc_slab_density_pct": st.column_config.NumberColumn("Unc. Slab Density %", format="%.1f"),
                "unc_slab_swe_pct": st.column_config.NumberColumn("Unc. Slab SWE %", format="%.1f"),
                "unc_entrainment_pct": st.column_config.NumberColumn("Unc. Entrainment %", format="%.1f"),
                "unc_runout_pct": st.column_config.NumberColumn("Unc. Runout %", format="%.1f"),
                "dsize_unc_low": st.column_config.TextColumn("D-size Uncertainty Low"),
                "dsize_unc_high": st.column_config.TextColumn("D-size Uncertainty High"),
                "notes": st.column_config.TextColumn("Notes"),
                "dsize_fallback": st.column_config.TextColumn("D-size (fallback)"),
            }
        )
       
        st.download_button(
            label="📥 Download Research Database as CSV (using clean column order)",
            data=display_df.to_csv(index=False).encode(),
            file_name="avalanche_research_log.csv",
            mime="text/csv"
        )
        st.caption("Columns shown and exported use clean, consistent names for V2 (and mapped legacy). The actual database columns are unchanged to preserve all historical data.")
       
        st.caption(f"Total records in database: **{len(log_df)}**")

# ====================== FOOTER ======================
st.divider()
st.caption("💡 V2.0: Single unified Start Zone method (dimensions + optional area override) + density (grain/hardness simple or layers, or SWE). Runout is volume-first. See schema_version='2.0' in DB.")
st.caption("Uncertainty sliders (per-section) use RSS for relevant params only. Toggle Imperial units above for U.S. field use.")