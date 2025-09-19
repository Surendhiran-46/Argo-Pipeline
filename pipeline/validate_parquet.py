# validate_parquet.py
import argparse
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
import json
import os

# ---- copy of the layer patterns used earlier ----
LAYER_PATTERNS = {
    "core_measurements": ["argo_core_measurements"],
    "qc_per_level": ["argo_qc_per_level"],
    "qc_profile": ["argo_qc_profile"],
    "metadata_clean": ["argo_metadata_clean"],
    "metadata_full": ["argo_metadata_full"],
    "time_location": ["argo_time_location"],
    "calibration": ["argo_calibration"],
    "history": ["argo_history"],
}

# ---- helpers copied/robust ----
def is_gzip(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"\x1f\x8b"
    except Exception:
        return False

def safe_read_csv_rows(csv_path: Path, chunksize=200_000):
    """Yield row counts per chunk; robust to mislabelled .gz"""
    compression = None
    if csv_path.suffix == ".gz":
        compression = "gzip" if is_gzip(csv_path) else None
    try:
        for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False, compression=compression):
            yield len(chunk)
    except Exception as e:
        print(f"[ERROR] reading {csv_path}: {e}")
        # return 0 for this file but record the error by returning None via raising sentinel
        raise

def safe_read_sample(csv_path: Path, nrows=5):
    compression = None
    if csv_path.suffix == ".gz":
        compression = "gzip" if is_gzip(csv_path) else None
    try:
        return pd.read_csv(csv_path, nrows=nrows, low_memory=False, compression=compression)
    except Exception as e:
        print(f"[ERROR] sample read {csv_path}: {e}")
        return None

def parquet_row_count(parquet_path: Path):
    pf = pq.ParquetFile(str(parquet_path))
    return pf.metadata.num_rows

def parquet_columns(parquet_path: Path):
    pf = pq.ParquetFile(str(parquet_path))
    return pf.schema.to_arrow_schema().names

# ---- critical columns per layer for null-checks ----
CRITICAL_COLS = {
    "core_measurements": ["profile_index","PRES","TEMP","PSAL"],
    "qc_per_level": ["profile_index","PRES","TEMP","PSAL"],
    "time_location": ["profile_index","LATITUDE","LONGITUDE","JULD"],
    "metadata_clean": ["PLATFORM_NUMBER"],
    "qc_profile": ["profile_index","DATA_MODE"],
    "calibration": ["profile_index","variable","value"],
    "history": ["profile_index","history_var","value"],
    "metadata_full": ["PLATFORM_NUMBER"],
}

# ---- single layer validation ----
def validate_layer(year, month, output_root: Path, parquet_root: Path, layer_key: str):
    month_dir = Path(output_root) / year / month
    out_dir = Path(parquet_root) / year / month
    result = {"layer": layer_key, "csv_files": [], "csv_total_rows": 0, "parquet_file": None,
              "parquet_rows": None, "matches": False, "issues": [], "null_summary": {}}
    if not month_dir.exists():
        result["issues"].append(f"month_dir_missing: {month_dir}")
        return result

    # find matched CSVs
    patterns = LAYER_PATTERNS.get(layer_key, [])
    csvs = []
    for pat in patterns:
        csvs.extend([p for p in month_dir.rglob(f"*{pat}*.csv*") if "process_log" not in p.name.lower() and ".ipynb_checkpoints" not in str(p)])
    csvs = sorted(set(csvs))
    result["csv_files"] = [str(p) for p in csvs]

    # count total rows from CSVs (chunked)
    csv_total = 0
    for p in csvs:
        try:
            for n in safe_read_csv_rows(p):
                csv_total += n
        except Exception as e:
            result["issues"].append(f"read_error:{p}:{e}")
    result["csv_total_rows"] = int(csv_total)

    # parquet file path
    parquet_file = out_dir / f"{layer_key}.parquet"
    result["parquet_file"] = str(parquet_file)
    if not parquet_file.exists():
        result["issues"].append("parquet_missing")
        return result

    # parquet rows (fast metadata)
    try:
        pq_rows = parquet_row_count(parquet_file)
        result["parquet_rows"] = int(pq_rows)
    except Exception as e:
        result["issues"].append(f"parquet_meta_error:{e}")
        return result

    # simple match check
    result["matches"] = (result["csv_total_rows"] == result["parquet_rows"])

    # compute null-summary for critical columns (read parquet minimally)
    crit = CRITICAL_COLS.get(layer_key, [])[:]
    # if parquet small-ish, read all; else read by columns
    try:
        dfp = pd.read_parquet(parquet_file, columns=[c for c in crit if c in parquet_columns(parquet_file)])
        for c in crit:
            if c in dfp.columns:
                nn = int(dfp[c].notnull().sum())
                result["null_summary"][c] = {"non_null": nn, "total": int(len(dfp))}
            else:
                result["null_summary"][c] = {"non_null": 0, "total": 0}
    except Exception as e:
        result["issues"].append(f"parquet_read_error:{e}")

    # capture sample rows (first CSV row and first parquet row) for quick human inspection
    if csvs:
        sample_csv = safe_read_sample(csvs[0], nrows=3)
        if sample_csv is not None:
            result["sample_csv_head"] = sample_csv.head(3).to_dict(orient="list")
        else:
            result["sample_csv_head"] = None
    try:
        import pyarrow as pa
        pf = pq.ParquetFile(str(parquet_file))
        # read first row-group or first few rows
        table = pf.read_row_groups([0]) if pf.num_row_groups>0 else pf.read() 
        # convert first min(3, rows) to pandas
        df_sample = table.to_pandas().head(3)
        result["sample_parquet_head"] = df_sample.to_dict(orient="list")
    except Exception:
        try:
            dfp_all = pd.read_parquet(parquet_file, nrows=3)
            result["sample_parquet_head"] = dfp_all.head(3).to_dict(orient="list")
        except Exception as e:
            result["sample_parquet_head"] = None
            result["issues"].append(f"sample_parquet_read_fail:{e}")

    # write a manifest for easy auditing
    manifest = out_dir / f"manifest-{layer_key}.json"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(manifest, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, default=str)
        result["manifest"] = str(manifest)
    except Exception as e:
        result["issues"].append(f"manifest_write_error:{e}")

    return result

# ---- month validation ----
def validate_month(year, month, output_root: Path, parquet_root: Path):
    print(f"\nValidating {year}-{month} ...")
    report = {}
    for layer in LAYER_PATTERNS.keys():
        r = validate_layer(year, month, output_root, parquet_root, layer)
        report[layer] = r
        # print summary
        print(f" - {layer}: csv_rows={r.get('csv_total_rows')} parquet_rows={r.get('parquet_rows')} match={r.get('matches')} issues={r.get('issues')}")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year","-y", required=True)
    parser.add_argument("--month","-m", required=False)
    parser.add_argument("--output","-o", default="../output")
    parser.add_argument("--parquet","-p", default="../parquet")
    args = parser.parse_args()
    year = args.year
    if args.month:
        months = [args.month]
    else:
        months = [f"{m:02d}" for m in range(1,13)]

    out_root = Path(args.output)
    pq_root = Path(args.parquet)
    overall = {}
    for m in months:
        overall[m] = validate_month(year, m, out_root, pq_root)
    # write overall summary file
    with open(pq_root / year / "validation_summary.json", "w", encoding="utf-8") as fh:
        json.dump(overall, fh, indent=2, default=str)
    print("\nValidation complete. Summary written to:", pq_root / year / "validation_summary.json")
