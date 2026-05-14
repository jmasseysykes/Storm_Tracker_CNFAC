import requests
import pandas as pd
from datetime import datetime

def get_snotel_data_at_date(station_triplet: str, target_date: str) -> dict:
    """Fetch SWE for a specific date — exact same logic as Storm Tracker."""
    if not station_triplet:
        return {"SWE_mm": 0.0, "error": "No station triplet provided"}

    try:
        url = (
            f"https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/"
            f"customSingleStationReport/daily/"
            f"{station_triplet}%7Cid=%22%22%7Cname/"
            f"POR_BEGIN,POR_END/"
            f"WTEQ::value,PREC::value,TMAX::value,TMIN::value,TAVG::value,PRCP::value"
        )

        response = requests.get(url, verify=False, timeout=15)
        response.raise_for_status()

        df = pd.read_csv(
            pd.io.common.StringIO(response.text),
            comment='#',
            skip_blank_lines=True
        )
        df.columns = ['Date', 'SWE', 'PREC', 'TMAX', 'TMIN', 'TAVG', 'PRCP']

        df['Date'] = pd.to_datetime(df['Date'])
        target_dt = pd.to_datetime(target_date)

        closest_row = df.iloc[(df['Date'] - target_dt).abs().argsort()[:1]]

        swe_inches = float(closest_row['SWE'].values[0])

        return {
            "SWE_mm": swe_inches * 25.4,          # ← CRITICAL CONVERSION
            "SWE_inches": swe_inches
        }

    except Exception as e:
        return {"SWE_mm": 0.0, "error": str(e)}


def get_slab_swe(station_triplet: str, weak_layer_date: str, release_date: str):
    """Calculate slab SWE between two dates."""
    wl_data = get_snotel_data_at_date(station_triplet, weak_layer_date)
    rel_data = get_snotel_data_at_date(station_triplet, release_date)

    if "error" in wl_data:
        return {"error": wl_data["error"]}
    if "error" in rel_data:
        return {"error": rel_data["error"]}

    delta_swe_mm = rel_data["SWE_mm"] - wl_data["SWE_mm"]
    if delta_swe_mm < 0:
        delta_swe_mm = 0

    return {
        "slab_swe_mm": max(0, delta_swe_mm),
        "swe_weak_mm": wl_data["SWE_mm"],
        "swe_release_mm": rel_data["SWE_mm"],
        "swe_weak_inches": wl_data["SWE_inches"],
        "swe_release_inches": rel_data["SWE_inches"]
    }