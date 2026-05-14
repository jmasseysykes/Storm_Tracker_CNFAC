import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "avalanche_log.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS avalanches (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            observer TEXT,
            location TEXT,
            method TEXT,
            area_m2 REAL,
            mass_tonnes REAL,
            d_size TEXT,
            unc_low TEXT,
            unc_high TEXT,
            notes TEXT
        )
    """)
    # Auto-migrate old tables that don't have the new uncertainty columns
    try:
        conn.execute("ALTER TABLE avalanches ADD COLUMN unc_low TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists
    try:
        conn.execute("ALTER TABLE avalanches ADD COLUMN unc_high TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists
    conn.close()

def save_avalanche(data: dict):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.DataFrame([data])
    df.to_sql("avalanches", conn, if_exists="append", index=False)
    conn.close()

def load_avalanche_log() -> pd.DataFrame:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM avalanches ORDER BY timestamp DESC", conn)
    conn.close()
    return df