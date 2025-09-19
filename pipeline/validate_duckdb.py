import duckdb

def validate_duckdb(db_path="argo.duckdb"):
    con = duckdb.connect(db_path)

    # 1. Show tables
    tables = con.execute("SHOW TABLES").fetchall()
    print(f"Total tables: {len(tables)}")
    
    # 2. Row counts for each table
    for (table_name,) in tables:
        count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"{table_name}: {count} rows")

    # 3. Schema check for a few critical tables
    for critical in ["core_measurements_2025_01", "metadata_clean_2025_01", "time_location_2025_01"]:
        if (critical,) in tables:
            schema = con.execute(f"PRAGMA table_info({critical})").fetchall()
            print(f"\nSchema for {critical}:")
            for col in schema:
                print(f" - {col[1]} ({col[2]})")

    con.close()

if __name__ == "__main__":
    validate_duckdb()