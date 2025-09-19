# pipeline/duckdb_loader.py
"""
Load monthly Parquet files into a DuckDB database.
Optimized for structured queries on ARGO data.
"""

import argparse
from pathlib import Path
import duckdb
import json

# ----------------- Config -----------------
DUCKDB_FILE = "argo.duckdb"  # Will be created at project root


def load_month(year: str, month: str, parquet_root: Path, conn: duckdb.DuckDBPyConnection):
    """
    Load all Parquet files for a given year-month into DuckDB.
    """
    month_dir = parquet_root / year / month
    if not month_dir.exists():
        print(f"SKIP: {month_dir} not found")
        return

    parquet_files = list(month_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"SKIP: No parquet files in {month_dir}")
        return

    print(f"\n=== Loading {year}-{month} into DuckDB ===")
    for pq in parquet_files:
        table_name = pq.stem.lower()

        # Create or replace table from parquet
        conn.execute(f"""
            CREATE OR REPLACE TABLE {table_name}_{year}_{month} AS
            SELECT * FROM parquet_scan('{pq.as_posix()}');
        """)

        # Validation: count rows
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}_{year}_{month}").fetchone()[0]
        print(f"  ✔ {table_name} → {row_count:,} rows")


def main(year: str, parquet_root: Path, db_file: Path):
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_file))

    months = [f"{m:02d}" for m in range(1, 13)]
    for m in months:
        load_month(year, m, parquet_root, conn)

    # Save database changes
    conn.close()
    print(f"\n=== DuckDB load complete: {db_file} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load ARGO Parquet into DuckDB database.")
    parser.add_argument("-y", "--year", required=True, help="Year to process (e.g. 2025)")
    parser.add_argument("-p", "--parquet", default="../parquet", help="Root parquet folder (default: ../parquet)")
    parser.add_argument("-d", "--db", default="../argo.duckdb", help="DuckDB database file (default: ../argo.duckdb)")
    args = parser.parse_args()

    main(args.year, Path(args.parquet), Path(args.db))
