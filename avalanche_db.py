import os
import psycopg2
import sqlite3
import pandas as pd

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

def get_connection():
    if USE_SUPABASE:
        return psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
                                user=DB_USER, password=DB_PASSWORD, sslmode="require")
    else:
        return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection()
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
        try:
            if USE_SUPABASE:
                cur.execute(f"ALTER TABLE avalanches ADD COLUMN IF NOT EXISTS {col_name} {col_type}")
            else:
                cur.execute(f"ALTER TABLE avalanches ADD COLUMN {col_name} {col_type}")
        except:
            pass  # Column already exists
    
    conn.commit()
    cur.close()
    conn.close()

    # Run D-size binning migration (idempotent, logs if changes made)
    migrate_dsize_calculations()

def save_avalanche(data: dict):
    init_db()
    conn = get_connection()
    cur = conn.cursor()
    
    columns = ", ".join(data.keys())
    placeholders = ", ".join(["%s" if USE_SUPABASE else "?" for _ in data])
    query = f"INSERT INTO avalanches ({columns}) VALUES ({placeholders})"
    
    cur.execute(query, list(data.values()))
    conn.commit()
    cur.close()
    conn.close()

def load_avalanche_log() -> pd.DataFrame:
    init_db()
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM avalanches ORDER BY timestamp DESC", conn)
    conn.close()
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
    """
    import re
    url_pattern = re.compile(r'https?://[^\s<>"\)\]]+')

    conn = get_connection()
    try:
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
            try:
                if USE_SUPABASE:
                    cur.execute(f"ALTER TABLE avalanches ADD COLUMN IF NOT EXISTS {col} {typ}")
                else:
                    cur.execute(f"ALTER TABLE avalanches ADD COLUMN {col} {typ}")
            except:
                pass
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
    except Exception as e:
        print(f"[D-Size Migration] Error: {e}")
    finally:
        cur.close()
        conn.close()