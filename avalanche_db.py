import os
import psycopg2
import sqlite3
import pandas as pd

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
                notes TEXT
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
                notes TEXT
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
        ("unc_runout_pct", "REAL")
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