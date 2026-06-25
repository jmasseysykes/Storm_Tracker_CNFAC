import pytest
import avalanche_calcs as calcs

def test_mass_to_dsize_basic():
    assert calcs.mass_to_dsize(5)["label"] == "D1"
    assert calcs.mass_to_dsize(50)["label"] == "D1.5"
    assert calcs.mass_to_dsize(100)["label"] == "D2"
    assert calcs.mass_to_dsize(1000)["label"] == "D3"
    assert calcs.mass_to_dsize(10000)["label"] == "D4"
    assert calcs.mass_to_dsize(100000)["label"] == "D5"
    assert calcs.mass_to_dsize(200000)["label"] == "D5"

def test_mass_to_dsize_edges():
    assert calcs.mass_to_dsize(17.7)["label"] == "D1"
    assert calcs.mass_to_dsize(17.8)["label"] == "D1.5"
    assert calcs.mass_to_dsize(0)["label"] == "D1"
    assert calcs.mass_to_dsize(-1)["label"] == "D1"

def test_get_dsize_range_string():
    assert calcs.get_dsize_range_string(10) == "D1 (1 – 18 t)"
    assert "D5" in calcs.get_dsize_range_string(100000)

def test_volume_m3_to_dsize():
    assert calcs.volume_m3_to_dsize(50)["label"] == "D1"
    assert calcs.volume_m3_to_dsize(200)["label"] == "D1.5"
    assert calcs.volume_m3_to_dsize(1000)["label"] == "D2"
    assert calcs.volume_m3_to_dsize(20000)["label"] == "D3.5"
    assert calcs.volume_m3_to_dsize(300000)["label"] == "D5"

def test_get_dsize_from_volume_string():
    s = calcs.get_dsize_from_volume_string(100)
    assert "D1" in s
    assert "m³" in s

def test_uncertainty_mass_range():
    mid, low, high = calcs.get_uncertainty_mass_range(1000, 10, 10, 10, 10)
    assert mid == "D3"
    assert low in ["D2.5", "D3"]
    assert high in ["D3", "D3.5"]

def test_density_lookup():
    d = calcs.get_density_from_hardness_grain("4F", "Rounded Grains (RG)")
    assert 150 < d < 170

def test_entrainment_mass_from_swe():
    assert calcs.entrainment_mass_from_swe(100, 100) == 10.0
    assert calcs.entrainment_mass_from_swe(0, 100) == 0.0

def test_calculate_entrainment():
    res = calcs.calculate_entrainment(100, 0.5, "Dimensions + Hardness/Grain", None, "4F", "Rounded Grains (RG)")
    assert res["entrainment_mass"] > 0
    assert "Density" in res["source"]

def test_estimate_mass_from_volume():
    res = calcs.estimate_mass_from_volume(10000, 500)
    assert res["volume_m3"] == 10000
    assert res["mass_tonnes_nominal"] == 5000
    assert res["mass_tonnes_low"] < res["mass_tonnes_nominal"]

def test_volume_to_dsize_nominal():
    res = calcs.volume_to_dsize_nominal(10000)
    assert "d_label" in res
    assert res["volume_m3"] == 10000
