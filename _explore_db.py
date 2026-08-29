"""Explore runs.db schema and data."""
import sqlite3, json

conn = sqlite3.connect("runs.db")
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("Tables:", tables)

for t in [r[0] for r in tables]:
    cols = conn.execute(f"PRAGMA table_info({t})").fetchall()
    print(f"\n--- {t} ---")
    for c in cols:
        print(f"  {c[1]} ({c[2]})")
    rows = conn.execute(f"SELECT * FROM {t} LIMIT 5").fetchall()
    count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"  Total rows: {count}")
    if rows:
        for r in rows:
            print(f"  Row: {r}")