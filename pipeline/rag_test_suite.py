# pipeline/rag_test_suite.py
"""
Small test harness to run hero queries and assert some invariants.
Each test returns a small report; store all reports in pipeline_tests/<timestamp>.json
"""

import json
from datetime import datetime, timezone
datetime.now(timezone.utc)
from pathlib import Path
from rag_guard import guard_and_run

TESTS = [
    {
        "name": "equator_march_2025",
        "query": "Show me salinity profiles near the equator in March 2025",
        "year": "2025",
        "months": ["03"],
        "date_from": "2025-03-01",
        "date_to": "2025-03-31"
    },
    {
        "name": "arabian_6mo",
        "query": "Compare salinity trends in the Arabian Sea last 6 months",
        "year": "2025",
        "months": ["04","05","06","07","08","09"],
        "date_from": "2025-04-01",
        "date_to": "2025-09-01"
    },
    {
        "name": "nearest_floats",
        "query": "What are the nearest ARGO floats to 10.5, 75.3?",
        "year": "2025",
        "months": ["01","02","03","04","05","06","07","08","09"]
    }
]

OUT_DIR = Path(__file__).resolve().parents[1] / "pipeline_tests"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def run_all():
    reports = []
    for t in TESTS:
        print("Running test:", t["name"])
        res = guard_and_run(query=t["query"], year=t["year"], months=t.get("months"), top_k=30, date_from=t.get("date_from"), date_to=t.get("date_to"), out_json=str(OUT_DIR / f"{t['name']}.json"))
        reports.append({"test": t["name"], "result": res})
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    summary = OUT_DIR / f"summary_{ts}.json"
    summary.write_text(json.dumps(reports, default=str, indent=2))
    print("Test summary saved to", summary)

if __name__ == "__main__":
    run_all()
