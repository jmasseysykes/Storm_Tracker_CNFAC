import pandas as pd

# ==================== D-SIZE CLASSIFICATION (your requested half-sizes) ====================
D_SIZE_RANGES = [
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
    unc_length_width_pct: float = 15,   # combined for L×W or area
    unc_depth_pct: float = 15,
    unc_density_pct: float = 20,
    unc_swe_pct: float = 10             # only used in detailed method
) -> tuple[str, str, str]:
    """Returns (mid_dsize_label, low_label, high_label)"""
    # Convert % to multiplicative factors
    f_lw = 1 + unc_length_width_pct / 100.0
    f_d  = 1 + unc_depth_pct / 100.0
    f_den = 1 + unc_density_pct / 100.0
    f_swe = 1 + unc_swe_pct / 100.0

    # Quick method: mass = L × W × D × density
    # Detailed method: mass = Area × SWE  (area already includes L×W)
    mass_low  = mass_tonnes / (f_lw * f_d * f_den * f_swe)
    mass_high = mass_tonnes * (f_lw * f_d * f_den * f_swe)

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