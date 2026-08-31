# scripts/explore_db.py
# --------------------------------------------------------------------------
# One-time tooling: dumps runs.db schema, row counts, and a sample row.
# Used during the gold-set construction workflow to understand the database
# structure before building the labeling worksheet (export_for_labeling.py).
# Kept for reproducibility: traces the provenance of tests/gold/gold_labels.jsonl
# and the κ = 0.612 result.
# --------------------------------------------------------------------------
"""Explore runs.db schema and probe data for labeling worksheet."""
import sqlite3, json

conn = sqlite3.connect('runs.db')

# Tables
cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cursor]
print("=== TABLES ===")
for t in tables:
    print(t)

# Schema for each
for t in tables:
    cursor = conn.execute(f"PRAGMA table_info({t})")
    cols = cursor.fetchall()
    print(f"\n=== {t} ===")
    for c in cols:
        print(f"  {c[1]} ({c[2]})")

# Probe count
for t in tables:
    cursor = conn.execute(f"SELECT COUNT(*) FROM {t}")
    cnt = cursor.fetchone()[0]
    print(f"\n{t}: {cnt} rows")
    if cnt > 0:
        cursor = conn.execute(f"SELECT * FROM {t} LIMIT 1")
        row = cursor.fetchone()
        print(f"  Sample: {row}")
