import os
import psycopg2
import pandas as pd

# Supabase connection using individual environment variables
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode="require"
    )

def init_db():
    conn = get_connection()
    cur = conn.cursor()
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
            calculated_d_size TEXT,
            unc_low TEXT,
            unc_high TEXT,
            field_assessed_d_size TEXT,
            crown_width_m REAL,
            slab_height_m REAL,
            depth_m REAL,
            hardness TEXT,
            grain TEXT,
            density_kgm3 REAL,
            weak_layer_date TEXT,
            release_date TEXT,
            snotel_station TEXT,
            snotel_triplet TEXT,
            slab_swe_mm REAL,
            adjusted_swe_mm REAL,
            burial_depth_ref_m REAL,
            notes TEXT
        )
    """)
    conn.commit()
    cur.close()
    conn.close()

def save_avalanche(data: dict):
    init_db()
    conn = get_connection()
    cur = conn.cursor()
    
    columns = ", ".join(data.keys())
    values = ", ".join(["%s"] * len(data))
    query = f"INSERT INTO avalanches ({columns}) VALUES ({values})"
    
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