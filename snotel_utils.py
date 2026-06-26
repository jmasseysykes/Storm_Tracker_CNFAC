import requests
import pandas as pd
from typing import Optional

_SNOTEL_CACHE: dict[str, pd.DataFrame] = {}


def _fetch_snotel_csv(station_triplet: str) -> pd.DataFrame:
    """Fetch and process daily SNOTEL SWE using the same logic as Storm Tracker."""
    if not station_triplet:
        raise ValueError("No station triplet provided")

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
        comment="#",
        skip_blank_lines=True,
    )
    df.columns = ["Date", "SWE", "PREC", "TMAX", "TMIN", "TAVG", "PRCP"]
    df["SWE"] = pd.to_numeric(df["SWE"], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    df = df.set_index("Date")
    df = df["SWE"].resample("D").asfreq().to_frame(name="SWE")
    df["delta_SWE"] = (df["SWE"] - df["SWE"].shift(1)).clip(lower=0)
    df["3-day"] = (df["SWE"] - df["SWE"].shift(3)).clip(lower=0)
    df["7-day"] = (df["SWE"] - df["SWE"].shift(7)).clip(lower=0)
    df["10-day"] = (df["SWE"] - df["SWE"].shift(10)).clip(lower=0)
    df = df.fillna(0)

    for window, col in zip([1, 3, 7, 10], ["delta_SWE", "3-day", "7-day", "10-day"]):
        gap_mask = df["SWE"].rolling(window).count() < window
        df.loc[gap_mask, col] = 0

    return df.reset_index()


def load_snotel_series(station_triplet: str, use_cache: bool = True) -> pd.DataFrame:
    """Load processed daily SNOTEL series (cached per station for the session)."""
    if use_cache and station_triplet in _SNOTEL_CACHE:
        return _SNOTEL_CACHE[station_triplet].copy()

    df = _fetch_snotel_csv(station_triplet)
    if use_cache:
        _SNOTEL_CACHE[station_triplet] = df.copy()
    return df


def _row_on_or_before(df: pd.DataFrame, target_date: str) -> Optional[pd.Series]:
    """Match Storm Tracker: use the last available record on or before the target date."""
    target_dt = pd.to_datetime(target_date)
    historical = df[df["Date"] <= target_dt]
    if historical.empty:
        return None
    return historical.iloc[-1]


def get_snotel_data_at_date(station_triplet: str, target_date: str) -> dict:
    """Fetch SWE for a specific date — same date-matching logic as Storm Tracker."""
    if not station_triplet:
        return {"SWE_mm": 0.0, "error": "No station triplet provided"}

    try:
        df = load_snotel_series(station_triplet)
        row = _row_on_or_before(df, target_date)
        if row is None:
            return {"SWE_mm": 0.0, "error": f"No SNOTEL data on or before {target_date}"}

        swe_inches = float(row["SWE"])
        return {
            "SWE_mm": swe_inches * 25.4,
            "SWE_inches": swe_inches,
            "matched_date": row["Date"].strftime("%Y-%m-%d"),
        }
    except Exception as e:
        return {"SWE_mm": 0.0, "error": str(e)}


def get_storm_totals_at_date(station_triplet: str, target_date: str) -> dict:
    """Return 1/3/7/10-day storm totals at a reference date (Storm Tracker convention)."""
    try:
        df = load_snotel_series(station_triplet)
        row = _row_on_or_before(df, target_date)
        if row is None:
            return {"error": f"No SNOTEL data on or before {target_date}"}

        return {
            "matched_date": row["Date"].strftime("%Y-%m-%d"),
            "1-day_in": float(row["delta_SWE"]),
            "3-day_in": float(row["3-day"]),
            "7-day_in": float(row["7-day"]),
            "10-day_in": float(row["10-day"]),
        }
    except Exception as e:
        return {"error": str(e)}


def get_slab_swe(station_triplet: str, weak_layer_date: str, release_date: str):
    """Calculate slab SWE between two dates (release minus weak-layer baseline)."""
    wl_data = get_snotel_data_at_date(station_triplet, weak_layer_date)
    rel_data = get_snotel_data_at_date(station_triplet, release_date)

    if "error" in wl_data:
        return {"error": wl_data["error"]}
    if "error" in rel_data:
        return {"error": rel_data["error"]}

    swe_weak_mm = wl_data["SWE_mm"]
    swe_release_mm = rel_data["SWE_mm"]
    delta_swe_mm = swe_release_mm - swe_weak_mm
    storm_totals = get_storm_totals_at_date(station_triplet, release_date)

    warnings = []
    if pd.to_datetime(release_date) < pd.to_datetime(weak_layer_date):
        warnings.append(
            "Release date is before the weak-layer date. "
            "Set release date to the avalanche date or the end of slab accumulation."
        )
    elif delta_swe_mm < 0:
        warnings.append(
            f"Release SWE ({swe_release_mm:.1f} mm on {rel_data.get('matched_date', release_date)}) "
            f"is lower than weak-layer SWE ({swe_weak_mm:.1f} mm on "
            f"{wl_data.get('matched_date', weak_layer_date)}). "
            "The snowpack SWE decreased between these dates (melt/settlement). "
            "Use the avalanche release date — not today's date — for past events."
        )
    elif delta_swe_mm == 0 and swe_weak_mm > 0:
        warnings.append(
            f"No SWE gain between {weak_layer_date} and {release_date}. "
            "Confirm the release date is the avalanche date (not today's date)."
        )

    period_days = (pd.to_datetime(release_date) - pd.to_datetime(weak_layer_date)).days
    result = {
        "slab_swe_mm": max(0.0, delta_swe_mm),
        "slab_swe_in": max(0.0, delta_swe_mm) / 25.4,
        "period_days": period_days,
        "swe_weak_mm": swe_weak_mm,
        "swe_release_mm": swe_release_mm,
        "swe_weak_inches": wl_data["SWE_inches"],
        "swe_release_inches": rel_data["SWE_inches"],
        "weak_layer_matched_date": wl_data.get("matched_date"),
        "release_matched_date": rel_data.get("matched_date"),
        "warnings": warnings,
    }
    if "error" not in storm_totals:
        result["storm_totals"] = storm_totals
    return result