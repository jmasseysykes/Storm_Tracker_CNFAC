import os
import psycopg2
import sqlite3
import pandas as pd
import streamlit as st

import avalanche_calcs as calcs  # for new D-size calculations during migration

USE_SUPABASE = bool(os.getenv("DB_HOST"))

if USE_SUPABASE:
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "postgres")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
else:
    DB_PATH = "avalanche_log.db"

# Columns accepted by INSERT (id is auto-generated).
AVALANCHE_COLUMNS = frozenset([
    "timestamp", "observer", "location", "method", "area_m2", "volume_m3",
    "mass_tonnes", "entrainment_mass", "total_mass", "calculated_d_size",
    "original_calculated_d_size", "dsize_method", "dsize_mass_original",
    "dsize_mass_midpoint", "dsize_volume_midpoint", "unc_low", "unc_high",
    "field_assessed_d_size", "crown_width_m", "slab_length_m", "depth_m",
    "hardness", "grain", "density_kgm3", "use_layered_density",
    "include_entrainment", "entr_width_m", "entr_length_m", "entr_area_m2",
    "entr_depth_m", "entr_hardness", "entr_grain", "entr_swe_mm", "debris_type",
    "weak_layer_date", "release_date", "snotel_station", "slab_swe_mm",
    "adjusted_swe_mm", "burial_depth_ref_m", "unc_lw_pct", "unc_depth_pct",
    "unc_density_pct", "unc_area_pct", "unc_swe_pct", "unc_entrainment_pct",
    "unc_runout_pct", "notes", "report_link", "crown_depth_direct_m",
    "crown_depth_derived_m", "geometry_mode", "density_mode", "density_profile",
    "swe_source", "entrainment_method_choice", "area_overridden", "schema_version",
    "entrainment_method",
])

# Stored as INTEGER (0/1) in Postgres and SQLite — UI often produces booleans.
INTEGER_FLAG_COLUMNS = frozenset(["area_overridden"])


def _ensure_connection_ready(conn):
    """Clear any aborted PostgreSQL transaction before running new SQL."""
    try:
        conn.rollback()
    except Exception:
        pass


def _sanitize_value(value, column=None):
    """Convert numpy/pandas scalars to native Python types for psycopg2/sqlite3."""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if column in INTEGER_FLAG_COLUMNS:
        if isinstance(value, bool):
            return int(value)
        if value in (0, 1):
            return int(value)
        return None
    if isinstance(value, bool):
        return bool(value)
    return value


def _add_column_if_missing(cur, conn, col_name, col_type):
    """Add a column, rolling back on failure so later statements are not blocked (PostgreSQL)."""
    try:
        if USE_SUPABASE:
            cur.execute(
                f"ALTER TABLE avalanches ADD COLUMN IF NOT EXISTS {col_name} {col_type}"
            )
        else:
            cur.execute(f"ALTER TABLE avalanches ADD COLUMN {col_name} {col_type}")
        return True
    except Exception as e:
        err = str(e).lower()
        if "duplicate column" in err or "already exists" in err:
            _ensure_connection_ready(conn)
            return False
        _ensure_connection_ready(conn)
        print(f"[init_db] Could not add column {col_name}: {e}")
        return False


@st.cache_resource
def get_connection():
    """Returns a long-lived cached database connection.

    IMPORTANT:
    - Do NOT call conn.close() on the object returned by this function.
    - Only close cursors (cur.close()).
    - Closing the cached connection causes "psycopg2.InterfaceError: connection already closed"
      on subsequent uses (very common with Supabase on Render).

    The connection lives for the lifetime of the Streamlit server process.
    """
    if USE_SUPABASE:
        # keepalives help prevent the server from closing idle connections (common on Supabase/Render)
        return psycopg2.connect(
            host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
            user=DB_USER, password=DB_PASSWORD, sslmode="require",
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=5
        )
    else:
        return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = None
    cur = None
    try:
        conn = get_connection()
        _ensure_connection_ready(conn)
        cur = conn.cursor()
        
        if USE_SUPABASE:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS avalanches (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    observer TEXT,
                    location TEXT,
                    method TEXT,
                    area_m2 REAL,
                    volume_m3 REAL,
                    mass_tonnes REAL,
                    entrainment_mass REAL,
                    total_mass REAL,
                    calculated_d_size TEXT,
                    original_calculated_d_size TEXT,
                    dsize_method TEXT,
                    dsize_mass_original TEXT,
                    dsize_mass_midpoint TEXT,
                    dsize_volume_midpoint TEXT,
                    unc_low TEXT,
                    unc_high TEXT,
                    field_assessed_d_size TEXT,
                    crown_width_m REAL,
                    slab_length_m REAL,
                    depth_m REAL,
                    hardness TEXT,
                    grain TEXT,
                    density_kgm3 REAL,
                    use_layered_density BOOLEAN,
                    include_entrainment BOOLEAN,
                    entr_width_m REAL,
                    entr_length_m REAL,
                    entr_area_m2 REAL,
                    entr_depth_m REAL,
                    entr_hardness TEXT,
                    entr_grain TEXT,
                    entr_swe_mm REAL,
                    debris_type TEXT,
                    weak_layer_date TEXT,
                    release_date TEXT,
                    snotel_station TEXT,
                    slab_swe_mm REAL,
                    adjusted_swe_mm REAL,
                    burial_depth_ref_m REAL,
                    unc_lw_pct REAL,
                    unc_depth_pct REAL,
                    unc_density_pct REAL,
                    unc_area_pct REAL,
                    unc_swe_pct REAL,
                    unc_entrainment_pct REAL,
                    notes TEXT,
                    report_link TEXT,
                    crown_depth_direct_m REAL,
                    crown_depth_derived_m REAL,
                    geometry_mode TEXT,
                    density_mode TEXT,
                    density_profile TEXT,
                    swe_source TEXT,
                    entrainment_method_choice TEXT,
                    area_overridden INTEGER,
                    schema_version TEXT,
                    entrainment_method TEXT
                )
            """)
        else:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS avalanches (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    observer TEXT,
                    location TEXT,
                    method TEXT,
                    area_m2 REAL,
                    volume_m3 REAL,
                    mass_tonnes REAL,
                    entrainment_mass REAL,
                    total_mass REAL,
                    calculated_d_size TEXT,
                    original_calculated_d_size TEXT,
                    dsize_method TEXT,
                    dsize_mass_original TEXT,
                    dsize_mass_midpoint TEXT,
                    dsize_volume_midpoint TEXT,
                    unc_low TEXT,
                    unc_high TEXT,
                    field_assessed_d_size TEXT,
                    crown_width_m REAL,
                    slab_length_m REAL,
                    depth_m REAL,
                    hardness TEXT,
                    grain TEXT,
                    density_kgm3 REAL,
                    use_layered_density INTEGER,
                    include_entrainment INTEGER,
                    entr_width_m REAL,
                    entr_length_m REAL,
                    entr_area_m2 REAL,
                    entr_depth_m REAL,
                    entr_hardness TEXT,
                    entr_grain TEXT,
                    entr_swe_mm REAL,
                    debris_type TEXT,
                    weak_layer_date TEXT,
                    release_date TEXT,
                    snotel_station TEXT,
                    slab_swe_mm REAL,
                    adjusted_swe_mm REAL,
                    burial_depth_ref_m REAL,
                    unc_lw_pct REAL,
                    unc_depth_pct REAL,
                    unc_density_pct REAL,
                    unc_area_pct REAL,
                    unc_swe_pct REAL,
                    unc_entrainment_pct REAL,
                    notes TEXT,
                    report_link TEXT,
                    crown_depth_direct_m REAL,
                    crown_depth_derived_m REAL,
                    geometry_mode TEXT,
                    density_mode TEXT,
                    density_profile TEXT,
                    swe_source TEXT,
                    entrainment_method_choice TEXT,
                    area_overridden INTEGER,
                    schema_version TEXT,
                    entrainment_method TEXT
                )
            """)
        
        # Safe migration for all new columns
        new_columns = [
            ("slab_length_m", "REAL"),
            ("entrainment_mass", "REAL"),
            ("total_mass", "REAL"),
            ("use_layered_density", "BOOLEAN" if USE_SUPABASE else "INTEGER"),
            ("include_entrainment", "BOOLEAN" if USE_SUPABASE else "INTEGER"),
            ("entr_width_m", "REAL"),
            ("entr_length_m", "REAL"),
            ("entr_area_m2", "REAL"),
            ("entr_depth_m", "REAL"),
            ("entr_hardness", "TEXT"),
            ("entr_grain", "TEXT"),
            ("entr_swe_mm", "REAL"),
            ("debris_type", "TEXT"),
            ("unc_lw_pct", "REAL"),
            ("unc_depth_pct", "REAL"),
            ("unc_density_pct", "REAL"),
            ("unc_area_pct", "REAL"),
            ("unc_swe_pct", "REAL"),
            ("unc_entrainment_pct", "REAL"),
            ("unc_runout_pct", "REAL"),
            ("original_calculated_d_size", "TEXT"),
            ("dsize_method", "TEXT"),
            ("report_link", "TEXT"),
            ("dsize_mass_original", "TEXT"),
            ("dsize_mass_midpoint", "TEXT"),
            ("dsize_volume_midpoint", "TEXT"),
            ("crown_depth_direct_m", "REAL"),
            ("crown_depth_derived_m", "REAL"),
            ("geometry_mode", "TEXT"),
            ("density_mode", "TEXT"),
            ("density_profile", "TEXT"),
            ("swe_source", "TEXT"),
            ("entrainment_method_choice", "TEXT"),
            ("area_overridden", "INTEGER"),
            ("schema_version", "TEXT"),
            ("entrainment_method", "TEXT"),
        ]
        
        for col_name, col_type in new_columns:
            _add_column_if_missing(cur, conn, col_name, col_type)

        conn.commit()
    except Exception as e:
        print(f"[init_db] Error: {e}")
        if conn is not None:
            _ensure_connection_ready(conn)
    finally:
        # Only close the cursor we created for schema work.
        # Never close conn here — it comes from @st.cache_resource and is meant to be long-lived.
        if cur is not None:
            try:
                cur.close()
            except:
                pass

    # Run D-size binning migration only once per process
    if not getattr(get_connection, "_migration_done", False):
        if migrate_dsize_calculations():
            get_connection._migration_done = True

def save_avalanche(data: dict):
    init_db()
    conn = get_connection()
    _ensure_connection_ready(conn)
    cur = conn.cursor()

    filtered = {
        k: _sanitize_value(v, column=k)
        for k, v in data.items()
        if k in AVALANCHE_COLUMNS
    }
    unknown = set(data.keys()) - AVALANCHE_COLUMNS
    if unknown:
        print(f"[save_avalanche] Ignoring unknown columns: {sorted(unknown)}")

    columns = ", ".join(filtered.keys())
    placeholders = ", ".join(["%s" if USE_SUPABASE else "?" for _ in filtered])
    query = f"INSERT INTO avalanches ({columns}) VALUES ({placeholders})"

    try:
        cur.execute(query, list(filtered.values()))
        conn.commit()
    except Exception as e:
        _ensure_connection_ready(conn)
        raise RuntimeError(f"Failed to save avalanche record: {e}") from e
    finally:
        cur.close()
    # Do NOT close conn here — it is cached via @st.cache_resource
    # Closing it causes "connection already closed" on next use (especially with Supabase)

def load_avalanche_log() -> pd.DataFrame:
    init_db()
    conn = get_connection()
    _ensure_connection_ready(conn)
    df = pd.read_sql("SELECT * FROM avalanches ORDER BY timestamp DESC", conn)
    # Do NOT close conn here — it is cached via @st.cache_resource.
    # Explicit close breaks the cached connection on subsequent calls (psycopg2).
    return df


def migrate_dsize_calculations():
    """Recalculate and store the three D-size views for all records.

    New columns (added if missing):
      - dsize_mass_original   : historical value that was originally stored
      - dsize_mass_midpoint   : always recalculated with current mass midpoint bins
      - dsize_volume_midpoint : recalculated with current volume midpoint bins (if volume > 0)

    We also:
      - Backfill report_link by extracting any URL from the notes field
      - Set calculated_d_size + dsize_method to the recommended current value
        (mass_midpoint for Quick/Detailed, volume_midpoint for Runout)

    Returns True on success, False on failure.
    """
    import re
    url_pattern = re.compile(r'https?://[^\s<>"\)\]]+')

    conn = None
    cur = None
    try:
        conn = get_connection()
        _ensure_connection_ready(conn)
        cur = conn.cursor()

        # Ensure the columns exist
        for col, typ in [
            ("dsize_mass_original", "TEXT"),
            ("dsize_mass_midpoint", "TEXT"),
            ("dsize_volume_midpoint", "TEXT"),
            ("report_link", "TEXT"),
            ("dsize_method", "TEXT"),
            ("crown_depth_direct_m", "REAL"),
            ("crown_depth_derived_m", "REAL"),
            ("geometry_mode", "TEXT"),
            ("density_mode", "TEXT"),
            ("density_profile", "TEXT"),
            ("swe_source", "TEXT"),
            ("entrainment_method_choice", "TEXT"),
            ("area_overridden", "INTEGER"),
            ("schema_version", "TEXT"),
            ("entrainment_method", "TEXT"),
        ]:
            _add_column_if_missing(cur, conn, col, typ)
        conn.commit()

        # Fetch all records
        if USE_SUPABASE:
            cur.execute("SELECT id, calculated_d_size, mass_tonnes, total_mass, volume_m3, method, notes, report_link FROM avalanches")
        else:
            cur.execute("SELECT id, calculated_d_size, mass_tonnes, total_mass, volume_m3, method, notes, report_link FROM avalanches")
        rows = cur.fetchall()

        updated = 0
        links_backfilled = 0

        for row in rows:
            if USE_SUPABASE:
                id_, old_calc, mass_t, total_m, vol, meth, notes, rpt = row
            else:
                id_, old_calc, mass_t, total_m, vol, meth, notes, rpt = row

            mass = (total_m or mass_t or 0)
            volume = (vol or 0)
            method_str = str(meth or '').lower()
            notes = str(notes or '')

            # Compute fresh values using current bins
            mass_mid = calcs.mass_to_dsize(mass)['label'] if mass > 0 else None
            vol_mid = calcs.volume_m3_to_dsize(volume)['label'] if volume > 0 else None

            is_runout = any(k in method_str for k in ['runout', 'debris', 'volume'])

            # 1. Store historical value into dsize_mass_original (only if we have an old value)
            if old_calc:
                if USE_SUPABASE:
                    cur.execute("UPDATE avalanches SET dsize_mass_original = %s WHERE id = %s", (old_calc, id_))
                else:
                    cur.execute("UPDATE avalanches SET dsize_mass_original = ? WHERE id = ?", (old_calc, id_))

            # 2. Write the recalculated midpoint values
            if mass_mid:
                if USE_SUPABASE:
                    cur.execute("UPDATE avalanches SET dsize_mass_midpoint = %s WHERE id = %s", (mass_mid, id_))
                else:
                    cur.execute("UPDATE avalanches SET dsize_mass_midpoint = ? WHERE id = ?", (mass_mid, id_))

            if vol_mid:
                if USE_SUPABASE:
                    cur.execute("UPDATE avalanches SET dsize_volume_midpoint = %s WHERE id = %s", (vol_mid, id_))
                else:
                    cur.execute("UPDATE avalanches SET dsize_volume_midpoint = ? WHERE id = ?", (vol_mid, id_))

            # 3. Update the "current" calculated_d_size + method
            if is_runout and vol_mid:
                if USE_SUPABASE:
                    cur.execute("UPDATE avalanches SET calculated_d_size = %s, dsize_method = %s WHERE id = %s",
                                (vol_mid, "volume_midpoint", id_))
                else:
                    cur.execute("UPDATE avalanches SET calculated_d_size = ?, dsize_method = ? WHERE id = ?",
                                (vol_mid, "volume_midpoint", id_))
            elif mass_mid:
                if USE_SUPABASE:
                    cur.execute("UPDATE avalanches SET calculated_d_size = %s, dsize_method = %s WHERE id = %s",
                                (mass_mid, "mass_midpoint", id_))
                else:
                    cur.execute("UPDATE avalanches SET calculated_d_size = ?, dsize_method = ? WHERE id = ?",
                                (mass_mid, "mass_midpoint", id_))

            # 4. Backfill report_link from notes
            if (not rpt or str(rpt).strip() == '') and notes:
                urls = url_pattern.findall(notes)
                if urls:
                    first = urls[0]
                    if USE_SUPABASE:
                        cur.execute("UPDATE avalanches SET report_link = %s WHERE id = %s", (first, id_))
                    else:
                        cur.execute("UPDATE avalanches SET report_link = ? WHERE id = ?", (first, id_))
                    links_backfilled += 1

            updated += 1

        conn.commit()
        print(f"[D-Size Migration] Processed {updated} records. Backfilled {links_backfilled} report_link(s) from notes.")
        return True
    except Exception as e:
        print(f"[D-Size Migration] Error: {e}")
        if conn is not None:
            _ensure_connection_ready(conn)
        return False
    finally:
        # Only close cursor. Do not close conn — it is the cached long-lived connection.
        if cur is not None:
            try:
                cur.close()
            except Exception:
                pass
        # Do NOT close conn here. See comments in get_connection() and load/save.