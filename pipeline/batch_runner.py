# batch_runner.py
from pathlib import Path
import argparse
import sys
import time
import csv

# ensure local pipeline folder is importable
here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))

from extract_metadata import extract_metadata
from extract_location import extract_location
from extract_core import extract_core
from extract_qc import extract_qc
from extract_history import extract_history
from utils import ensure_dir

def process_one(nc_file: Path, out_root: Path):
    day_out = out_root / nc_file.stem
    ensure_dir(day_out)
    print(f"[{time.strftime('%H:%M:%S')}] PROCESS: {nc_file.name} -> {day_out}")
    try:
        extract_metadata(nc_file, day_out)
        extract_location(nc_file, day_out)
        extract_core(nc_file, day_out)
        extract_qc(nc_file, day_out)
        extract_history(nc_file, day_out)
        return True, ""
    except Exception as e:
        return False, str(e)

def main(input_dir, output_dir, resume_log):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    processed = set()
    if resume_log.exists():
        with resume_log.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                processed.add(r["filename"])

    nc_files = sorted(input_dir.glob("*.nc"))
    with resume_log.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["timestamp","filename","status","message"])
        if fh.tell() == 0:
            writer.writeheader()
        for nc in nc_files:
            if nc.name in processed:
                print(f"SKIP (already processed): {nc.name}")
                continue
            ok, msg = process_one(nc, output_dir)
            writer.writerow({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "filename": nc.name, "status": "OK" if ok else "ERROR", "message": msg})
            fh.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process Argo .nc files into CSV layers.")
    parser.add_argument("--input", "-i", required=True, help="Input folder with .nc files (e.g. ../data/2025/09)")
    parser.add_argument("--output", "-o", required=True, help="Output root for daily folders (e.g. ../output/2025/09)")
    parser.add_argument("--log", default="process_log.csv", help="Processing resume log")
    args = parser.parse_args()
    main(args.input, args.output, Path(args.log))
