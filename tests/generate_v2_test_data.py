#!/usr/bin/env python3
"""
Generate a robust synthetic test dataset for the V2.0 D-Size Estimator.
Covers all major input method combinations in Start Zone and Runout/Debris.

Usage:
  python tests/generate_v2_test_data.py
  # outputs tests/v2_test_data.csv

The CSV can be used to:
- Manually import / test save flow
- Populate a test DB
- Verify display, calculations, field distinctions
"""

import pandas as pd
import sys
from datetime import datetime, date

# Make sure we can import calcs
sys.path.append(".")
import avalanche_calcs as calcs

def generate_test_records():
    records = []
    now = datetime.now().isoformat()

    # ========== Case 1: Start Zone - Simple grain, small, no entrainment ==========
    vol = 1200.0
    mass = 300.0
    dens = 250.0
    mass_d = calcs.mass_to_dsize(mass)
    vol_d = calcs.volume_m3_to_dsize(vol)
    rec = {
        "timestamp": now,
        "observer": "Test User 1",
        "location": "Test Peak - Small Slab",
        "report_link": "",
        "schema_version": "2.0",
        "method": "start_zone",
        "geometry_mode": "dimensions",
        "density_mode": "Layer density + grain type",
        "density_profile": "Simple (hand hardness + grain type)",
        "swe_source": None,
        "area_overridden": False,
        "area_m2": 1500.0,
        "volume_m3": vol,
        "depth_m": 0.8,
        "crown_depth_direct_m": 0.8,
        "crown_depth_derived_m": None,
        "crown_width_m": 50.0,
        "slab_length_m": 30.0,
        "mass_tonnes": mass,
        "entrainment_mass": 0.0,
        "total_mass": mass,
        "density_kgm3": dens,
        "use_layered_density": False,
        "hardness": "4F",
        "grain": "Rounded Grains (RG)",
        "slab_swe_mm": None,
        "snotel_station": None,
        "include_entrainment": False,
        "entrainment_method": None,
        "entr_width_m": None,
        "entr_length_m": None,
        "entr_area_m2": None,
        "entr_depth_m": None,
        "entr_hardness": None,
        "entr_grain": None,
        "entr_swe_mm": None,
        "calculated_d_size": mass_d["label"],
        "dsize_method": "mass_midpoint",
        "dsize_mass_original": mass_d["label"],
        "dsize_mass_midpoint": mass_d["label"],
        "dsize_volume_midpoint": vol_d["label"],
        "unc_low": mass_d["label"],
        "unc_high": mass_d["label"],
        "low_mass": mass * 0.8,
        "high_mass": mass * 1.2,
        "field_assessed_d_size": "D2",
        "notes": "Small start zone simple grain test case - no entrainment.",
        "debris_type": None,
        "weak_layer_date": None,
        "release_date": None,
        "slab_swe_mm": None,
        "adjusted_swe_mm": None,
        "burial_depth_ref_m": None,
        "unc_lw_pct": 15,
        "unc_depth_pct": 15,
        "unc_density_pct": 20,
        "unc_area_pct": 15,
        "unc_swe_pct": None,
        "unc_entrainment_pct": 0,
        "unc_runout_pct": None,
    }
    records.append(rec)

    # ========== Case 2: Start Zone - Detailed layers, medium, no entrainment ==========
    vol = 8000.0
    mass = 2200.0
    dens = 275.0
    mass_d = calcs.mass_to_dsize(mass)
    vol_d = calcs.volume_m3_to_dsize(vol)
    rec = {
        "timestamp": now,
        "observer": "Test User 2",
        "location": "Test Ridge - Layered Slab",
        "report_link": "https://example.com/report/2",
        "schema_version": "2.0",
        "method": "start_zone",
        "geometry_mode": "dimensions",
        "density_mode": "Layer density + grain type",
        "density_profile": "Detailed layer profile",
        "swe_source": None,
        "area_overridden": False,
        "area_m2": 10000.0,
        "volume_m3": vol,
        "depth_m": 0.8,
        "crown_depth_direct_m": 0.8,
        "crown_depth_derived_m": None,
        "crown_width_m": 100.0,
        "slab_length_m": 100.0,
        "mass_tonnes": mass,
        "entrainment_mass": 0.0,
        "total_mass": mass,
        "density_kgm3": dens,
        "use_layered_density": True,
        "hardness": None,
        "grain": None,
        "slab_swe_mm": None,
        "snotel_station": None,
        "include_entrainment": False,
        "entrainment_method": None,
        "entr_width_m": None,
        "entr_length_m": None,
        "entr_area_m2": None,
        "entr_depth_m": None,
        "entr_hardness": None,
        "entr_grain": None,
        "entr_swe_mm": None,
        "calculated_d_size": mass_d["label"],
        "dsize_method": "mass_midpoint",
        "dsize_mass_original": mass_d["label"],
        "dsize_mass_midpoint": mass_d["label"],
        "dsize_volume_midpoint": vol_d["label"],
        "unc_low": mass_d["label"],
        "unc_high": mass_d["label"],
        "low_mass": mass * 0.85,
        "high_mass": mass * 1.15,
        "field_assessed_d_size": "D3",
        "notes": "Medium start zone with detailed layers test case.",
        "debris_type": None,
        "weak_layer_date": None,
        "release_date": None,
        "slab_swe_mm": None,
        "adjusted_swe_mm": None,
        "burial_depth_ref_m": None,
        "unc_lw_pct": 12,
        "unc_depth_pct": 18,
        "unc_density_pct": 25,
        "unc_area_pct": 12,
        "unc_swe_pct": None,
        "unc_entrainment_pct": 0,
        "unc_runout_pct": None,
    }
    records.append(rec)

    # ========== Case 3: Start Zone - SWE manual, large, no entrainment ==========
    vol = 45000.0
    swe_mm = 450.0
    area = 50000.0
    depth = 0.9
    mass = area * (swe_mm / 1000.0)   # ~22500 t
    dens = swe_mm / depth   # implied ~500 kg/m3
    mass_d = calcs.mass_to_dsize(mass)
    vol_d = calcs.volume_m3_to_dsize(vol)
    rec = {
        "timestamp": now,
        "observer": "Test User 3",
        "location": "Big Bowl - SWE Test",
        "report_link": "",
        "schema_version": "2.0",
        "method": "start_zone",
        "geometry_mode": "dimensions",
        "density_mode": "SWE based density estimate",
        "density_profile": None,
        "swe_source": "Manual entry",
        "area_overridden": True,
        "area_m2": area,
        "volume_m3": vol,
        "depth_m": depth,
        "crown_depth_direct_m": depth,
        "crown_depth_derived_m": None,
        "crown_width_m": 200.0,
        "slab_length_m": 250.0,
        "mass_tonnes": mass,
        "entrainment_mass": 0.0,
        "total_mass": mass,
        "density_kgm3": dens,
        "use_layered_density": False,
        "hardness": None,
        "grain": None,
        "slab_swe_mm": swe_mm,
        "snotel_station": None,
        "include_entrainment": False,
        "entrainment_method": None,
        "entr_width_m": None,
        "entr_length_m": None,
        "entr_area_m2": None,
        "entr_depth_m": None,
        "entr_hardness": None,
        "entr_grain": None,
        "entr_swe_mm": None,
        "calculated_d_size": mass_d["label"],
        "dsize_method": "mass_midpoint",
        "dsize_mass_original": mass_d["label"],
        "dsize_mass_midpoint": mass_d["label"],
        "dsize_volume_midpoint": vol_d["label"],
        "unc_low": mass_d["label"],
        "unc_high": mass_d["label"],
        "low_mass": mass * 0.9,
        "high_mass": mass * 1.1,
        "field_assessed_d_size": "D4",
        "notes": "Large start zone using manual SWE. Area was overridden from polygon.",
        "debris_type": None,
        "weak_layer_date": None,
        "release_date": None,
        "slab_swe_mm": swe_mm,
        "adjusted_swe_mm": 0.0,
        "burial_depth_ref_m": None,
        "unc_lw_pct": 10,
        "unc_depth_pct": 20,
        "unc_density_pct": None,
        "unc_area_pct": 10,
        "unc_swe_pct": 12,
        "unc_entrainment_pct": 0,
        "unc_runout_pct": None,
    }
    records.append(rec)

    # ========== Case 4: Start Zone - Simple grain + entrainment (grain) ==========
    vol = 5000.0
    slab_mass = 1250.0
    entr_mass = 800.0
    total = slab_mass + entr_mass
    mass_d = calcs.mass_to_dsize(total)
    vol_d = calcs.volume_m3_to_dsize(vol)
    rec = {
        "timestamp": now,
        "observer": "Test User 4",
        "location": "Test Couloir - With Entrainment",
        "report_link": "",
        "schema_version": "2.0",
        "method": "start_zone",
        "geometry_mode": "dimensions",
        "density_mode": "Layer density + grain type",
        "density_profile": "Simple (hand hardness + grain type)",
        "swe_source": None,
        "area_overridden": False,
        "area_m2": 6250.0,
        "volume_m3": vol,
        "depth_m": 0.8,
        "crown_depth_direct_m": 0.8,
        "crown_depth_derived_m": None,
        "crown_width_m": 75.0,
        "slab_length_m": 83.33,
        "mass_tonnes": slab_mass,
        "entrainment_mass": entr_mass,
        "total_mass": total,
        "density_kgm3": 250.0,
        "use_layered_density": False,
        "hardness": "1F-",
        "grain": "Faceted Crystals (FC)",
        "slab_swe_mm": None,
        "snotel_station": None,
        "include_entrainment": True,
        "entrainment_method": "Dimensions + Hardness/Grain",
        "entr_width_m": 100.0,
        "entr_length_m": 150.0,
        "entr_area_m2": 15000.0,
        "entr_depth_m": 0.6,
        "entr_hardness": "F+",
        "entr_grain": "Precipitation Particles (PP)",
        "entr_swe_mm": None,
        "calculated_d_size": mass_d["label"],
        "dsize_method": "mass_midpoint",
        "dsize_mass_original": mass_d["label"],
        "dsize_mass_midpoint": mass_d["label"],
        "dsize_volume_midpoint": vol_d["label"],
        "unc_low": mass_d["label"],
        "unc_high": mass_d["label"],
        "low_mass": total * 0.8,
        "high_mass": total * 1.25,
        "field_assessed_d_size": "D3",
        "notes": "Start zone simple + entrainment via grain method.",
        "debris_type": None,
        "weak_layer_date": None,
        "release_date": None,
        "slab_swe_mm": None,
        "adjusted_swe_mm": None,
        "burial_depth_ref_m": None,
        "unc_lw_pct": 15,
        "unc_depth_pct": 15,
        "unc_density_pct": 20,
        "unc_area_pct": 15,
        "unc_swe_pct": None,
        "unc_entrainment_pct": 25,
        "unc_runout_pct": None,
    }
    records.append(rec)

    # ========== Case 5: Start Zone - SWE + entrainment SWE ==========
    vol = 12000.0
    slab_mass = 3600.0   # from swe
    entr_mass = 1200.0
    total = slab_mass + entr_mass
    mass_d = calcs.mass_to_dsize(total)
    vol_d = calcs.volume_m3_to_dsize(vol)
    rec = {
        "timestamp": now,
        "observer": "Test User 5",
        "location": "SWE Bowl with Entr",
        "report_link": "",
        "schema_version": "2.0",
        "method": "start_zone",
        "geometry_mode": "dimensions",
        "density_mode": "SWE based density estimate",
        "density_profile": None,
        "swe_source": "Manual entry",
        "area_overridden": False,
        "area_m2": 15000.0,
        "volume_m3": vol,
        "depth_m": 0.8,
        "crown_depth_direct_m": 0.8,
        "crown_depth_derived_m": None,
        "crown_width_m": 120.0,
        "slab_length_m": 125.0,
        "mass_tonnes": slab_mass,
        "entrainment_mass": entr_mass,
        "total_mass": total,
        "density_kgm3": 300.0,
        "use_layered_density": False,
        "hardness": None,
        "grain": None,
        "slab_swe_mm": 240.0,
        "snotel_station": None,
        "include_entrainment": True,
        "entrainment_method": "SWE-based",
        "entr_width_m": 80.0,
        "entr_length_m": 200.0,
        "entr_area_m2": 16000.0,
        "entr_depth_m": 0.5,
        "entr_hardness": None,
        "entr_grain": None,
        "entr_swe_mm": 75.0,
        "calculated_d_size": mass_d["label"],
        "dsize_method": "mass_midpoint",
        "dsize_mass_original": mass_d["label"],
        "dsize_mass_midpoint": mass_d["label"],
        "dsize_volume_midpoint": vol_d["label"],
        "unc_low": mass_d["label"],
        "unc_high": mass_d["label"],
        "low_mass": total * 0.82,
        "high_mass": total * 1.22,
        "field_assessed_d_size": "D3.5",
        "notes": "SWE based start zone + SWE entrainment.",
        "debris_type": None,
        "weak_layer_date": None,
        "release_date": None,
        "slab_swe_mm": 240.0,
        "adjusted_swe_mm": 0.0,
        "burial_depth_ref_m": None,
        "unc_lw_pct": 10,
        "unc_depth_pct": 15,
        "unc_density_pct": None,
        "unc_area_pct": 10,
        "unc_swe_pct": 8,
        "unc_entrainment_pct": 20,
        "unc_runout_pct": None,
    }
    records.append(rec)

    # ========== Case 6: Runout/Debris - small, no custom density ==========
    vol = 850.0
    vol_d = calcs.volume_m3_to_dsize(vol)
    rec = {
        "timestamp": now,
        "observer": "Test User 6",
        "location": "Small Debris Fan",
        "report_link": "",
        "schema_version": "2.0",
        "method": "runout_debris",
        "geometry_mode": None,
        "density_mode": None,
        "density_profile": None,
        "swe_source": None,
        "area_overridden": None,
        "area_m2": 1200.0,
        "volume_m3": vol,
        "depth_m": 0.7,
        "crown_depth_direct_m": None,
        "crown_depth_derived_m": None,
        "crown_width_m": None,
        "slab_length_m": None,
        "mass_tonnes": 0,
        "entrainment_mass": 0.0,
        "total_mass": 0,
        "density_kgm3": None,
        "use_layered_density": None,
        "hardness": None,
        "grain": None,
        "slab_swe_mm": None,
        "snotel_station": None,
        "include_entrainment": False,
        "entrainment_method": None,
        "entr_width_m": None,
        "entr_length_m": None,
        "entr_area_m2": None,
        "entr_depth_m": None,
        "entr_hardness": None,
        "entr_grain": None,
        "entr_swe_mm": None,
        "calculated_d_size": vol_d["label"],
        "dsize_method": "volume_midpoint",
        "dsize_mass_original": None,
        "dsize_mass_midpoint": None,
        "dsize_volume_midpoint": vol_d["label"],
        "unc_low": vol_d["label"],
        "unc_high": vol_d["label"],
        "low_mass": None,
        "high_mass": None,
        "field_assessed_d_size": "D2",
        "notes": "Small runout debris test - volume only.",
        "debris_type": "Dry",
        "weak_layer_date": None,
        "release_date": None,
        "slab_swe_mm": None,
        "adjusted_swe_mm": None,
        "burial_depth_ref_m": None,
        "unc_lw_pct": None,
        "unc_depth_pct": 15,
        "unc_density_pct": None,
        "unc_area_pct": 12,
        "unc_swe_pct": None,
        "unc_entrainment_pct": None,
        "unc_runout_pct": 18,
        "volume_focused": True,
        "volume_d_size": vol_d["label"],
        "mass_tonnes_low": 0,
        "mass_tonnes_high": 0,
    }
    records.append(rec)

    # ========== Case 7: Runout/Debris - large + custom density ==========
    vol = 85000.0
    custom_dens = 550.0
    mass_nom = vol * custom_dens / 1000.0
    vol_d = calcs.volume_m3_to_dsize(vol)
    mass_d = calcs.mass_to_dsize(mass_nom)
    rec = {
        "timestamp": now,
        "observer": "Test User 7",
        "location": "Large Debris Test",
        "report_link": "",
        "schema_version": "2.0",
        "method": "runout_debris",
        "geometry_mode": None,
        "density_mode": None,
        "density_profile": None,
        "swe_source": None,
        "area_overridden": None,
        "area_m2": 45000.0,
        "volume_m3": vol,
        "depth_m": 1.9,
        "crown_depth_direct_m": None,
        "crown_depth_derived_m": None,
        "crown_width_m": None,
        "slab_length_m": None,
        "mass_tonnes": mass_nom,
        "entrainment_mass": 0.0,
        "total_mass": mass_nom,
        "density_kgm3": custom_dens,
        "use_layered_density": None,
        "hardness": None,
        "grain": None,
        "slab_swe_mm": None,
        "snotel_station": None,
        "include_entrainment": False,
        "entrainment_method": None,
        "entr_width_m": None,
        "entr_length_m": None,
        "entr_area_m2": None,
        "entr_depth_m": None,
        "entr_hardness": None,
        "entr_grain": None,
        "entr_swe_mm": None,
        "calculated_d_size": vol_d["label"],
        "dsize_method": "volume_midpoint",
        "dsize_mass_original": None,
        "dsize_mass_midpoint": mass_d["label"],
        "dsize_volume_midpoint": vol_d["label"],
        "unc_low": vol_d["label"],
        "unc_high": vol_d["label"],
        "low_mass": mass_nom * 0.7,
        "high_mass": mass_nom * 1.3,
        "field_assessed_d_size": "D4",
        "notes": "Large runout with custom density for mass reference.",
        "debris_type": "Wet",
        "weak_layer_date": None,
        "release_date": None,
        "slab_swe_mm": None,
        "adjusted_swe_mm": None,
        "burial_depth_ref_m": None,
        "unc_lw_pct": None,
        "unc_depth_pct": 18,
        "unc_density_pct": None,
        "unc_area_pct": 15,
        "unc_swe_pct": None,
        "unc_entrainment_pct": None,
        "unc_runout_pct": 22,
        "volume_focused": True,
        "volume_d_size": vol_d["label"],
        "mass_tonnes_low": mass_nom * 0.7,
        "mass_tonnes_high": mass_nom * 1.3,
    }
    records.append(rec)

    return pd.DataFrame(records)

if __name__ == "__main__":
    df = generate_test_records()
    out_path = "tests/v2_test_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} test records -> {out_path}")
    print("\nMethods covered:")
    print(df["method"].value_counts())
    print("\nD-size distribution (calculated_d_size):")
    print(df["calculated_d_size"].value_counts().sort_index())
    print("\nSample columns for first record:")
    print(df.iloc[0][["method", "density_mode", "density_profile", "include_entrainment", "calculated_d_size", "schema_version"]].to_dict())
