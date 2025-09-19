# pipeline/data_validator.py
"""
Data validation utilities to ensure DuckDB / Parquet contain the requested data.
Functions:
 - table_rowcounts()
 - region_coverage()
 - variable_null_rate()
 - qc_flag_distribution()
 - profile_exists(platform, profile_index)
 - time_range_coverage()
"""

import duckdb
from pathlib import Path
import pandas as pd
from typing import Dict, Any, Optional

DUCKDB_PATH = Path(__file__).resolve().parents[1] / "argo.duckdb"

def open_con():
    con = duckdb.connect(database=str(DUCKDB_PATH), read_only=False)
    return con

def table_rowcounts(db_path: str = "argo.duckdb"):
    con = duckdb.connect(db_path)
    tables = con.execute("PRAGMA show_tables").fetchall()
    results = []

    for (table,) in tables:
        try:
            row_count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            results.append({"table_name": table, "row_count": row_count})
        except Exception as e:
            results.append({"table_name": table, "row_count": None, "error": str(e)})

    return pd.DataFrame(results)

def time_range_coverage(table_name: str):
    con = open_con()
    q = f"SELECT MIN(juld) AS min_juld, MAX(juld) AS max_juld FROM {table_name} WHERE juld IS NOT NULL"
    try:
        df = con.execute(q).fetchdf()
    except Exception as e:
        con.close()
        raise
    con.close()
    return df.to_dict(orient="records")[0] if not df.empty else {"min_juld": None, "max_juld": None}

def region_coverage(lat_min: float, lat_max: float, lon_min: float, lon_max: float, tables: Optional[list] = None, sample_limit: int = 10):
    """
    Returns per-table counts and a small sample of platform/profile.
    """
    con = open_con()
    if not tables:
        tables = [r[0] for r in con.execute("SELECT table_name FROM INFORMATION_SCHEMA.TABLES WHERE table_schema='main'").fetchall()]
    results = []
    for t in tables:
        if "time_location" not in t and "core_measurements" not in t:
            continue
        sql = f"""
            SELECT COUNT(*) as cnt FROM {t}
            WHERE LATITUDE IS NOT NULL AND LONGITUDE IS NOT NULL
              AND LATITUDE >= {lat_min} AND LATITUDE <= {lat_max}
              AND LONGITUDE >= {lon_min} AND LONGITUDE <= {lon_max}
        """
        try:
            cnt = con.execute(sql).fetchone()[0]
        except Exception as e:
            cnt = None
        sample = []
        if cnt and cnt > 0:
            try:
                samp = con.execute(f"SELECT * FROM {t} WHERE LATITUDE >= {lat_min} AND LATITUDE <= {lat_max} AND LONGITUDE >= {lon_min} AND LONGITUDE <= {lon_max} LIMIT {sample_limit}").fetchdf()
                sample = samp.to_dict(orient="records")
            except Exception:
                sample = []
        results.append({"table": t, "count": int(cnt) if cnt is not None else None, "sample": sample})
    con.close()
    return results

def variable_null_rate(table_name: str, variable: str):
    con = open_con()
    try:
        total = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        non_null = con.execute(f"SELECT COUNT({variable}) FROM {table_name} WHERE {variable} IS NOT NULL").fetchone()[0]
    except Exception as e:
        con.close()
        raise
    con.close()
    return {"table": table_name, "variable": variable, "total": int(total), "non_null": int(non_null), "null_rate": 1.0 - (non_null / total) if total>0 else None}

def qc_flag_distribution(table_name: str, qc_col: str = "PSAL_QC"):
    con = open_con()
    try:
        df = con.execute(f"SELECT {qc_col}, COUNT(*) as cnt FROM {table_name} GROUP BY {qc_col} ORDER BY cnt DESC").fetchdf()
    except Exception as e:
        con.close()
        raise
    con.close()
    return df.to_dict(orient="records")

def profile_exists(platform_number: int, profile_index: Optional[int] = None):
    con = open_con()
    try:
        if profile_index is None:
            q = f"SELECT COUNT(DISTINCT profile_index) FROM core_measurements_2025_01 WHERE platform_number={platform_number}"
            cnt = con.execute(q).fetchone()[0]
            con.close()
            return int(cnt) > 0
        else:
            # search across core tables for the pair
            tables = [r[0] for r in con.execute("SELECT table_name FROM INFORMATION_SCHEMA.TABLES WHERE table_schema='main'").fetchall()]
            found = False
            for t in tables:
                if not t.startswith("core_measurements_"):
                    continue
                q = f"SELECT COUNT(*) FROM {t} WHERE platform_number={platform_number} AND profile_index={profile_index}"
                try:
                    cnt = con.execute(q).fetchone()[0]
                    if cnt and cnt > 0:
                        found = True
                        break
                except Exception:
                    continue
            con.close()
            return found
    except Exception:
        con.close()
        raise

if __name__ == "__main__":
    print("Row counts (sample):")
    print(table_rowcounts().head(10))
    print("\nTime coverage core_measurements_2025_01:")
    print(time_range_coverage("core_measurements_2025_01"))
    print("\nSample region coverage around equator  -10..10, 70..90")
    print(region_coverage(-10,10,70,90, sample_limit=3))
