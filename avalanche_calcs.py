import pandas as pd
from typing import Optional, Any

# ==================== D-SIZE CLASSIFICATION - Scott Log-Midpoint Binning (updated) ====================
# Data from Scott's table (logarithmic midpoint binning).
# D1 extended to 17.8 t so bins are contiguous.
# Only full D-sizes (D1/D2/D3/D4/D5) have typical values for display.
# IMPORTANT: 
# - 'mass_midpoint' = current log-midpoint binning for mass-based calculations (Quick/Detailed)
# - 'volume_midpoint' = log-midpoint binning for volume-based (Runout/Debris)
# - 'mass_midpoint_historical' = special marker for old records with incomplete data (e.g. missing entrainment inputs)
# Existing DB records have 'calculated_d_size' from previous binning.
# We preserve history in 'original_calculated_d_size' and mark the method used.
D_SIZE_RANGES = [
    {"label": "D1",   "min_t": 1.0,   "typical_t": 10,    "max_t": 17.8},
    {"label": "D1.5", "min_t": 17.8,  "typical_t": None,  "max_t": 56.2},
    {"label": "D2",   "min_t": 56.2,  "typical_t": 100,   "max_t": 177.8},
    {"label": "D2.5", "min_t": 177.8, "typical_t": None,  "max_t": 562},
    {"label": "D3",   "min_t": 562,   "typical_t": 1000,  "max_t": 1778},
    {"label": "D3.5", "min_t": 1778,  "typical_t": None,  "max_t": 5623},
    {"label": "D4",   "min_t": 5623,  "typical_t": 10000, "max_t": 17783},
    {"label": "D4.5", "min_t": 17783, "typical_t": None,  "max_t": 56234},
    {"label": "D5",   "min_t": 56234, "typical_t": 100000,"max_t": 177828},
]

# Legacy ranges kept for reference / potential back-comparison of old records
D_SIZE_RANGES_LEGACY = [
    {"label": "D1",     "min_t": 0,      "typical_t": 5,    "max_t": 10},
    {"label": "D1.5",   "min_t": 10,     "typical_t": 55,   "max_t": 75},
    {"label": "D2",     "min_t": 75,     "typical_t": 100,  "max_t": 250},
    {"label": "D2.5",   "min_t": 250,    "typical_t": 550,  "max_t": 750},
    {"label": "D3",     "min_t": 750,    "typical_t": 1000, "max_t": 2500},
    {"label": "D3.5",   "min_t": 2500,   "typical_t": 5500, "max_t": 7500},
    {"label": "D4",     "min_t": 7500,   "typical_t": 10000,"max_t": 25000},
    {"label": "D4.5",   "min_t": 25000,  "typical_t": 55000,"max_t": 75000},
    {"label": "D5",     "min_t": 75000,  "typical_t": 100000,"max_t": float("inf")},
]

def mass_to_dsize(mass_tonnes: float) -> dict:
    if mass_tonnes < 0:
        mass_tonnes = 0
    for d in D_SIZE_RANGES:
        if mass_tonnes < d["max_t"]:
            return d
    return D_SIZE_RANGES[-1]

def get_dsize_range_string(mass_tonnes: float) -> str:
    info = mass_to_dsize(mass_tonnes)
    if info["max_t"] == float("inf"):
        return f"{info['label']} (> {info['min_t']:,.0f} t)"
    return f"{info['label']} ({info['min_t']:,.0f} – {info['max_t']:,.0f} t)"

def get_uncertainty_mass_range(
    mass_tonnes: float,
    unc_length_width_pct: float = 15,
    unc_depth_pct: float = 15,
    unc_density_pct: float = 20,
    unc_swe_pct: float = 10
) -> tuple[str, str, str]:
    """Returns (mid_dsize_label, low_label, high_label) using Root-Sum-Square"""
    
    # Root-Sum-Square for independent uncertainties (much more realistic)
    total_rel_unc = (
        (unc_length_width_pct / 100.0)**2 +
        (unc_depth_pct / 100.0)**2 +
        (unc_density_pct / 100.0)**2 +
        (unc_swe_pct / 100.0)**2
    ) ** 0.5
    
    factor = 1 + total_rel_unc
    
    mass_low = mass_tonnes / factor
    mass_high = mass_tonnes * factor
    
    mid_label = mass_to_dsize(mass_tonnes)["label"]
    low_label = mass_to_dsize(mass_low)["label"]
    high_label = mass_to_dsize(mass_high)["label"]
    
    return mid_label, low_label, high_label

# ==================== REFINED DENSITY LOOKUP (Kim & Jamieson 2014 Table 3) ====================
DENSITY_LOOKUP = {
    "Precipitation Particles (PP)": {"F-":68,"F":82,"F+":95,"4F-":109,"4F":122,"4F+":135,"1F-":149,"1F":162,"1F+":176,"P-":189,"P":203,"P+":None,"K-":None,"K":None,"K+":None},
    "Graupel (PPgp)": {"F-":93,"F":108,"F+":124,"4F-":139,"4F":155,"4F+":170,"1F-":186,"1F":201,"1F+":216,"P-":232,"P":247,"P+":None,"K-":None,"K":None,"K+":None},
    "Decomposing/Fragmented (DF)": {"F-":88,"F":100,"F+":112,"4F-":125,"4F":137,"4F+":150,"1F-":162,"1F":175,"1F+":187,"P-":200,"P":212,"P+":224,"K-":237,"K":None,"K+":None},
    "Rounded Grains (RG)": {"F-":110,"F":120,"F+":132,"4F-":144,"4F":158,"4F+":172,"1F-":189,"1F":206,"1F+":226,"P-":247,"P":270,"P+":296,"K-":324,"K":354,"K+":387},
    "Faceted Rounded (RGxf)": {"F-":116,"F":131,"F+":147,"4F-":162,"4F":178,"4F+":193,"1F-":209,"1F":224,"1F+":239,"P-":255,"P":270,"P+":285,"K-":None,"K":None,"K+":None},
    "Faceted Crystals (FC)": {"F-":137,"F":154,"F+":170,"4F-":188,"4F":204,"4F+":221,"1F-":238,"1F":255,"1F+":271,"P-":289,"P":305,"P+":322,"K-":339,"K":356,"K+":None},
    "Rounding Faceted (FCxr)": {"F-":108,"F":127,"F+":147,"4F-":167,"4F":186,"4F+":205,"1F-":225,"1F":245,"1F+":264,"P-":284,"P":303,"P+":323,"K-":343,"K":362,"K+":381},
    "Depth Hoar (DH)": {"F-":None,"F":233,"F+":239,"4F-":246,"4F":252,"4F+":258,"1F-":265,"1F":271,"1F+":277,"P-":284,"P":290,"P+":296,"K-":303,"K":309,"K+":None},
    "Melt-Freeze Crust (MFcr)": {"F-":None,"F":None,"F+":None,"4F-":None,"4F":265,"4F+":270,"1F-":275,"1F":280,"1F+":285,"P-":290,"P":295,"P+":300,"K-":306,"K":311,"K+":315},
}

def get_density_from_hardness_grain(hardness: str, grain_type: str) -> float:
    """Returns density (kg/m³) from Kim & Jamieson 2014 Table 3."""
    # Normalize grain input (full name or abbrev)
    grain_map = {
        "PP": "Precipitation Particles (PP)", "Precipitation Particles": "Precipitation Particles (PP)",
        "PPgp": "Graupel (PPgp)", "Graupel": "Graupel (PPgp)",
        "DF": "Decomposing/Fragmented (DF)", "Decomposing/Fragmented": "Decomposing/Fragmented (DF)",
        "RG": "Rounded Grains (RG)", "Rounded Grains": "Rounded Grains (RG)",
        "RGxf": "Faceted Rounded (RGxf)", "Faceted Rounded": "Faceted Rounded (RGxf)",
        "FC": "Faceted Crystals (FC)", "Faceted Crystals": "Faceted Crystals (FC)",
        "FCxr": "Rounding Faceted (FCxr)", "Rounding Faceted": "Rounding Faceted (FCxr)",
        "DH": "Depth Hoar (DH)", "Depth Hoar": "Depth Hoar (DH)",
        "MFcr": "Melt-Freeze Crust (MFcr)", "Melt-Freeze Crust": "Melt-Freeze Crust (MFcr)",
    }
    grain_key = grain_map.get(grain_type.strip(), grain_type)

    if grain_key not in DENSITY_LOOKUP:
        return 250.0  # safe fallback

    h = hardness.upper().strip()
    # Try exact match
    if h in DENSITY_LOOKUP[grain_key] and DENSITY_LOOKUP[grain_key][h] is not None:
        return DENSITY_LOOKUP[grain_key][h]

    # Fallback to main class
    main_map = {"F": "F", "4F": "4F", "1F": "1F", "P": "P", "K": "K"}
    main_h = h[0] if h[0] in "F41PK" else h
    if main_h == "4":
        main_h = "4F"
    elif main_h == "1":
        main_h = "1F"
    if main_h in DENSITY_LOOKUP[grain_key] and DENSITY_LOOKUP[grain_key][main_h] is not None:
        return DENSITY_LOOKUP[grain_key][main_h]

    return 250.0  # final fallback


# ==================== ENTRAINMENT HELPERS (stabilized logic) ====================

def entrainment_mass_from_swe(area_m2: float, swe_mm: float) -> float:
    """Core physics: 1 mm SWE over 1 m² ≈ 0.001 tonnes of water/ice equivalent."""
    if area_m2 <= 0 or swe_mm <= 0:
        return 0.0
    return area_m2 * (swe_mm / 1000.0)


def calculate_entrainment(
    entr_area_display: float,
    entr_depth_display: float,
    method: str,
    manual_swe_display: Optional[float],
    hardness: Optional[str] = None,
    grain: Optional[str] = None,
    use_imperial: bool = False,
    conv_length: float = 1.0,
    conv_area: float = 1.0,
    snotel_swe_mm: Optional[float] = None,
) -> dict:
    """Centralized, testable entrainment calculator.

    Returns a dict with:
      entrainment_mass, entr_swe_mm, entr_area_m2, entr_depth_m, source, notes
    All internal math is in metric.
    """
    entr_area_m2 = max(0.0, entr_area_display * conv_area)
    entr_depth_m = max(0.0, entr_depth_display * conv_length)

    swe_mm = 0.0
    source = ""
    notes = []

    if method == "SWE-based":
        if manual_swe_display is not None and manual_swe_display > 0:
            factor = 25.4 if use_imperial else 1.0
            swe_mm = manual_swe_display * factor
            source = "Manual SWE"
        elif snotel_swe_mm is not None and snotel_swe_mm > 0:
            swe_mm = snotel_swe_mm
            source = "SNOTEL delta"
        else:
            source = "SWE-based (no value provided)"
            notes.append("No manual SWE or SNOTEL delta available")
    else:
        # Dimensions + Hardness/Grain path
        if hardness and grain:
            density = get_density_from_hardness_grain(hardness, grain)
            swe_mm = density * entr_depth_m  # kg/m² == mm water equivalent
            source = f"Density × depth ({density:.0f} kg/m³)"
        else:
            source = "Dimensions (missing hardness/grain)"
            notes.append("Hardness or grain type missing")

    entrainment_mass = entrainment_mass_from_swe(entr_area_m2, swe_mm)

    # Add warnings for suspiciously large values
    if entrainment_mass > 5000:
        notes.append("Very large entrainment mass — verify inputs/dates")
    if entr_area_m2 > 50000:
        notes.append("Extremely large entrainment area")

    return {
        "entrainment_mass": round(entrainment_mass, 1),
        "entr_swe_mm": round(swe_mm, 1),
        "entr_area_m2": round(entr_area_m2, 1),
        "entr_depth_m": round(entr_depth_m, 2),
        "source": source,
        "notes": notes,
    }


def is_suspicious_entrainment(mass_tonnes: float, swe_mm: float, area_m2: float) -> list[str]:
    """Return list of warning strings for extreme values."""
    warnings = []
    if mass_tonnes > 10000:
        warnings.append(f"Entrainment mass > 10,000 t ({mass_tonnes:,.0f} t)")
    if swe_mm > 800:
        warnings.append(f"SWE used for entrainment is very high ({swe_mm:,.0f} mm)")
    if area_m2 > 30000:
        warnings.append(f"Entrainment area is very large ({area_m2:,.0f} m²)")
    return warnings


# ==================== D-SIZE BY DEPOSIT VOLUME (Jamieson 2024 + log-midpoint binning) ====================
# These ranges come from the provided volume classification chart for debris/runout deposits.
# Units: cubic meters (m³). Typical values are the black reference lines in the chart.

VOLUME_D_SIZE_RANGES = [
    {"label": "D1",   "min_m3": 1,     "typical_m3": 100,     "max_m3": 177.8},  # starts at 1 to avoid low-end exaggeration on log plot
    {"label": "D1.5", "min_m3": 177.8,   "typical_m3": 562.3,   "max_m3": 562.3},
    {"label": "D2",   "min_m3": 562.3,   "typical_m3": 1000,    "max_m3": 1778},
    {"label": "D2.5", "min_m3": 1778,    "typical_m3": 5623,    "max_m3": 5623},
    {"label": "D3",   "min_m3": 5623,    "typical_m3": 10000,   "max_m3": 17782},
    {"label": "D3.5", "min_m3": 17782,   "typical_m3": 56234,   "max_m3": 56234},
    {"label": "D4",   "min_m3": 56234,   "typical_m3": 100000,  "max_m3": 177827},
    {"label": "D4.5", "min_m3": 177827,  "typical_m3": 281170,  "max_m3": 281170},
    {"label": "D5",   "min_m3": 281170,  "typical_m3": 500000,  "max_m3": float("inf")},
]


def volume_m3_to_dsize(volume_m3: float) -> dict:
    """Classify avalanche by deposit volume (m³) using the Jamieson 2024 volume bins."""
    if volume_m3 < 0:
        volume_m3 = 0.0
    for d in VOLUME_D_SIZE_RANGES:
        if volume_m3 < d["max_m3"]:
            return d
    return VOLUME_D_SIZE_RANGES[-1]


def get_dsize_from_volume_string(volume_m3: float) -> str:
    info = volume_m3_to_dsize(volume_m3)
    if info["max_m3"] == float("inf"):
        return f"{info['label']} (> {info['min_m3']:,.0f} m³)"
    return f"{info['label']} ({info['min_m3']:,.0f} – {info['max_m3']:,.0f} m³)"


# ==================== DB SAFETY HELPERS ====================
def get_current_mass_dsize(mass_tonnes: float) -> str:
    """Return the D-size label using the current Scott midpoint binning.
    Safe to use for new calculations or on-the-fly display.
    Does NOT modify any database records.
    """
    if mass_tonnes is None or mass_tonnes <= 0:
        return None
    return mass_to_dsize(mass_tonnes)["label"]


def get_legacy_mass_dsize(mass_tonnes: float) -> str:
    """Return what the D-size label would have been under the previous binning.
    Useful for comparison with historical 'calculated_d_size' in the DB.
    """
    if mass_tonnes is None or mass_tonnes <= 0:
        return None
    for d in D_SIZE_RANGES_LEGACY:
        if mass_tonnes < d["max_t"]:
            return d["label"]
    return D_SIZE_RANGES_LEGACY[-1]["label"]


# ==================== RUNOUT / VOLUME-BASED (density less reliable for debris) ====================

DEBRIS_DENSITY_RANGE = {
    "nominal": 500,   # kg/m³
    "low": 350,
    "high": 650,
}

def estimate_mass_from_volume(volume_m3: float, nominal_density: float = None) -> dict:
    """Return mass estimates using a range of plausible debris densities.
    Primary output is volume; mass is secondary estimate.
    """
    if volume_m3 <= 0:
        volume_m3 = 0.0
    nom = nominal_density or DEBRIS_DENSITY_RANGE["nominal"]
    low_d = DEBRIS_DENSITY_RANGE["low"]
    high_d = DEBRIS_DENSITY_RANGE["high"]

    mass_nom = volume_m3 * nom / 1000.0
    mass_low = volume_m3 * low_d / 1000.0
    mass_high = volume_m3 * high_d / 1000.0

    # For D-size we will primarily use the nominal mass
    return {
        "volume_m3": round(volume_m3, 0),
        "mass_tonnes_nominal": round(mass_nom, 0),
        "mass_tonnes_low": round(mass_low, 0),
        "mass_tonnes_high": round(mass_high, 0),
        "density_nominal": nom,
        "density_range": (low_d, high_d),
    }


def volume_to_dsize_nominal(volume_m3: float) -> dict:
    """Rough volume-guided D-size (still ultimately uses mass for classification).
    This acknowledges that for debris we prefer to report volume primarily.
    """
    res = estimate_mass_from_volume(volume_m3)
    d_info = mass_to_dsize(res["mass_tonnes_nominal"])
    return {
        "d_label": d_info["label"],
        "volume_m3": res["volume_m3"],
        "est_mass_range_t": (res["mass_tonnes_low"], res["mass_tonnes_high"]),
        "notes": "D-size estimated from nominal debris mass (density 350-650 kg/m³ range for deposits)"
    }