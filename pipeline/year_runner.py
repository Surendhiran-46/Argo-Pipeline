# year_runner.py
from pathlib import Path
import argparse
import subprocess
import sys

def main(year, data_root, output_root):
    for month in range(1, 6):
        month_str = f"{month:02d}"
        in_dir = Path(data_root) / str(year) / month_str
        out_dir = Path(output_root) / str(year) / month_str
        log_file = out_dir / "process_log.csv"

        if not in_dir.exists():
            print(f"SKIP: {in_dir} does not exist")
            continue

        print(f"\n=== Processing {year}-{month_str} ===")
        cmd = [
            sys.executable,  # same Python used to launch this script
            "batch_runner.py",
            "-i", str(in_dir),
            "-o", str(out_dir),
            "--log", str(log_file)
        ]
        print("RUN:", " ".join(cmd))
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Argo pipeline for a full year.")
    parser.add_argument("-y", "--year", required=True, help="Year to process (e.g. 2025)")
    parser.add_argument("-d", "--data", default="../data", help="Root data folder (default: ../data)")
    parser.add_argument("-o", "--output", default="../output", help="Root output folder (default: ../output)")
    args = parser.parse_args()

    main(args.year, Path(args.data), Path(args.output))
