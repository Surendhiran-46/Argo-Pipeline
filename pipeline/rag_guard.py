# pipeline/rag_guard.py
"""
RAG Guard orchestrator:
 - Validate user query intent and required fields
 - Validate data coverage in DuckDB / Parquet
 - If sufficient, call existing rag_pipeline.run_rag_query (which performs retrieval + LLM)
 - If insufficient, return structured response 'INSUFFICIENT_DATA' with hints for clarifying questions.
 - Always write an audit JSON with provenance: retrieval ids, SQL executed, row counts, LLM output (if any).
"""

import json
import uuid
import time
from pathlib import Path
from typing import Any, Dict
from datetime import datetime, timezone
datetime.now(timezone.utc)

from query_validator import analyze_query
from data_validator import region_coverage, profile_exists, variable_null_rate
from rag_pipeline import run_rag_query  # your existing script's function

# output folder for audits
AUDIT_DIR = Path(__file__).resolve().parents[1] / "pipeline_audits"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

def write_audit(audit: Dict[str,Any], run_id: str):
    p = AUDIT_DIR / f"audit_{run_id}.json"
    p.write_text(json.dumps(audit, default=str, indent=2))
    print("Audit written:", str(p))

def guard_and_run(query: str, year: str="2025", months: list=None, top_k: int=30, date_from: str=None, date_to: str=None, out_json: str=None):
    run_id = uuid.uuid4().hex
    t0 = time.time()
    qinfo = analyze_query(query)
    audit = {"run_id": run_id, "timestamp": datetime.utcnow().isoformat(), "query": query, "qinfo": qinfo, "checks": {}, "result": None}
    # ------------- Pre-checks based on intent -------------
    intent = qinfo["intent"]
    # Example pre-checks:
    if intent in ("measurement","compare","nearest","qc_query"):
        # If query has bbox, validate region coverage
        b = qinfo.get("bbox")
        if b:
            rc = region_coverage(b["lat_min"], b["lat_max"], b["lon_min"], b["lon_max"], sample_limit=3)
            audit["checks"]["region_coverage"] = rc
            total_hits = sum([r["count"] or 0 for r in rc])
            if total_hits == 0:
                audit["checks"]["decision"] = "INSUFFICIENT_DATA_REGION"
                write_audit(audit, run_id)
                return {"status":"INSUFFICIENT_DATA", "reason":"No data in requested region", "audit": str(AUDIT_DIR / f"audit_{run_id}.json")}
        # If platform specified, check profile exists
        if qinfo.get("platform_number"):
            exists = profile_exists(qinfo["platform_number"], qinfo.get("profile_index"))
            audit["checks"]["profile_exists"] = exists
            if not exists:
                audit["checks"]["decision"] = "INSUFFICIENT_DATA_PLATFORM"
                write_audit(audit, run_id)
                return {"status":"INSUFFICIENT_DATA", "reason":"Platform/profile not found", "audit": str(AUDIT_DIR / f"audit_{run_id}.json")}
    elif intent == "float_detail" or intent == "project_info":
        # these are metadata queries; ensure Chroma contains matches via run_rag_query with top_k small
        pass  # we'll rely on existing retrieval
    # ------------- Data-quality checks (example: TEMP null rate) -------------
    # If variables requested, check null rate in sample table core_measurements_2025_01
    for var in qinfo.get("variables", []):
        try:
            v = variable_null_rate("core_measurements_2025_01", var)
            audit["checks"][f"null_rate_{var}"] = v
            if v["null_rate"] is not None and v["null_rate"] > 0.8:
                audit["checks"]["decision"] = f"INSUFFICIENT_DATA_NULLRATE_{var}"
                write_audit(audit, run_id)
                return {"status":"INSUFFICIENT_DATA", "reason":f"High null rate for {var}", "audit": str(AUDIT_DIR / f"audit_{run_id}.json")}
        except Exception as e:
            # continue: if the var column isn't present, we'll let the RAG pipeline fail gracefully
            audit["checks"][f"null_rate_{var}"] = f"error:{e}"
    # ------------- All pre-checks passed: call rag pipeline -------------
    audit["checks"]["decision"] = "PROCEED_RAG"
    result = run_rag_query(query=query, year=year, months=months, top_k=top_k, date_from=date_from, date_to=date_to, output_json=None)
    audit["result_summary"] = {"retrieved_count": result.get("retrieved_count"), "candidates_count": result.get("candidates_count"), "numeric_rows": result.get("numeric_rows")}
    audit["llm_raw"] = result.get("llm_raw", {}) if isinstance(result.get("llm_raw"), dict) else None
    audit["answer_snippet"] = (result.get("answer") or "")[:800]
    t1 = time.time()
    audit["timing_seconds"] = t1 - t0
    write_audit(audit, run_id)
    # Save final result (optional)
    outp = None
    if out_json:
        p = Path(out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"run_id": run_id, "result": result}, default=str, indent=2))
        outp = str(p)
    return {"status":"OK","run_id":run_id,"audit": str(AUDIT_DIR / f"audit_{run_id}.json"), "output_json": outp}
