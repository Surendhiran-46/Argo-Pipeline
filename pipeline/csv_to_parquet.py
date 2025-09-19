# pipeline/csv_to_parquet.py
"""
Convert monthly ARGO CSV outputs into Parquet files (robust + layer-aware).
Writes one parquet file per layer per month:
  parquet/<year>/<month>/core_measurements.parquet
  parquet/<year>/<month>/qc_per_level.parquet
  parquet/<year>/<month>/metadata_clean.parquet
  ...
This avoids schema-union problems (e.g. mixing process_log rows with data rows).
"""

import argparse
from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import os
import sys

# ----------------- Helpers -----------------
def is_gzip(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"\x1f\x8b"
    except Exception:
        return False

def safe_read_csv(csv_file: Path, chunksize=200_000):
    """Read CSV robustly, even if mislabelled .gz"""
    compression = None
    if csv_file.suffix == ".gz":
        compression = "gzip" if is_gzip(csv_file) else None

    try:
        for chunk in pd.read_csv(csv_file, chunksize=chunksize, low_memory=False, compression=compression):
            yield chunk
    except Exception as e:
        print(f"[ERROR] Failed to read {csv_file}: {e}")
        return

def normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert problematic columns before saving to Parquet.
    - Parse datetime-like columns safely.
    - Keep identifier/object columns as strings.
    - Fix integer-like floats back to Int64.
    """
    if df.empty:
        return df

    # Keywords to detect datetime-like columns
    dt_keywords = ("JULD", "DATE", "TIME", "REFERENCE", "HISTORY_DATE", "TIMESTAMP")

    for col in df.columns:
        col_up = col.upper()

        # --- Handle datetime-like columns ---
        if any(k in col_up for k in dt_keywords):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
            except Exception:
                df[col] = df[col].astype(str)
            continue

        # --- Handle object/string columns ---
        if df[col].dtype == "object":
            # Always store identifiers and mixed values as string
            df[col] = df[col].astype(str).str.strip()
            continue

        # --- Handle float columns that are actually IDs ---
        if pd.api.types.is_float_dtype(df[col]):
            non_na = df[col].dropna()
            if len(non_na) > 0 and non_na.apply(float.is_integer).all():
                df[col] = df[col].astype("Int64")  # nullable integer

    return df

def atomic_write_parquet(df: pd.DataFrame, out_path: Path):
    """Write parquet atomically using to_parquet (pyarrow) - tmp then replace."""
    tmp = str(out_path) + ".tmp"
    # ensure parent dir exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # use pyarrow engine
    df.to_parquet(tmp, engine="pyarrow", index=False, compression="snappy")
    os.replace(tmp, out_path)

# ----------------- Layer definitions -----------------
# map layer_key -> list of filename patterns (case-insensitive substrings)
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

# ----------------- Convert single layer for the month -----------------
def convert_layer_month(year: str, month: str, output_root: Path, parquet_root: Path, layer_key: str):
    month_dir = output_root / year / month
    if not month_dir.exists():
        print(f"SKIP: {month_dir} not found")
        return None

    patterns = LAYER_PATTERNS.get(layer_key, [])
    # find matching CSV files (recursively) by pattern
    matched_files = []
    for pat in patterns:
        for p in month_dir.rglob(f"*{pat}*.csv*"):
            # skip process logs explicitly
            if p.name.lower().startswith("process_log"):
                continue
            # skip ipynb checkpoints
            if ".ipynb_checkpoints" in str(p):
                continue
            matched_files.append(p)
    matched_files = sorted(set(matched_files))

    if not matched_files:
        print(f"  [SKIP LAYER] No files for layer '{layer_key}' in {month_dir}")
        return None

    print(f"  [LAYER] {layer_key} -> {len(matched_files)} files")

    # read chunks and accumulate (we will concat per-layer)
    chunks = []
    total_rows = 0
    for f in matched_files:
        for chunk in safe_read_csv(f):
            chunks.append(chunk)
            total_rows += len(chunk)

    if not chunks:
        print(f"  [SKIP LAYER] {layer_key} loaded 0 rows")
        return None

    df = pd.concat(chunks, ignore_index=True)
    # normalize dtypes (smart heuristic)
    df = normalize_dtypes(df)

    # write per-layer parquet inside month folder
    out_dir = parquet_root / year / month
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{layer_key}.parquet"

    atomic_write_parquet(df, out_file)

    print(f"  SAVED: {out_file} ({len(df):,} rows, {len(df.columns)} cols)")
    return str(out_file)

# ----------------- Convert all layers for a month -----------------
def convert_month(year: str, month: str, output_root: Path, parquet_root: Path):
    print(f"\n=== Converting {year}-{month} ===")
    results = {}
    for layer in LAYER_PATTERNS.keys():
        try:
            r = convert_layer_month(year, month, output_root, parquet_root, layer)
            results[layer] = r
        except Exception as e:
            print(f"  [ERROR] layer {layer}: {e}")
            results[layer] = None
    return results

# ----------------- Main Runner -----------------
def main(year: str, output_root: Path, parquet_root: Path, parallel: bool):
    months = [f"{m:02d}" for m in range(1, 13)]
    tasks = [(year, m, output_root, parquet_root) for m in months]

    if parallel:
        with ProcessPoolExecutor() as ex:
            # map convert_month across months in parallel
            futures = [ex.submit(convert_month, *t) for t in tasks]
            results = [f.result() for f in futures]
    else:
        results = []
        for t in tasks:
            results.append(convert_month(*t))

    print("\n=== Conversion complete ===")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ARGO CSV outputs into per-layer Parquet per month.")
    parser.add_argument("-y", "--year", required=True, help="Year to process (e.g. 2025)")
    parser.add_argument("-o", "--output", default="../output", help="Root output folder (default: ../output)")
    parser.add_argument("-p", "--parquet", default="../parquet", help="Root parquet folder (default: ../parquet)")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing of months")
    args = parser.parse_args()

    main(args.year, Path(args.output), Path(args.parquet), args.parallel)