# server/api.py
"""
FastAPI wrapper around the existing pipeline.run_rag_query(...) function.

Features:
- POST /api/run  => enqueue a RAG job (background)
- GET  /api/status/{job_id} => job status (queued/running/done/error)
- GET  /api/result/{job_id} => job result JSON (when finished)
- GET  /api/list => list recent jobs
- POST /api/run_sync => run synchronously (not recommended for heavy jobs)

Implementation notes:
- Jobs persisted to server_outputs/jobs/{job_id}.json
- Results persisted to server_outputs/results/{job_id}.json
- Uses BackgroundTasks + a semaphore to limit concurrent runs (MAX_CONCURRENT_JOBS env var)
- Expects run_rag_query available at pipeline.rag_pipeline.run_rag_query
- Loads environment from .env at startup
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional, List, Any, Dict
import uuid
import time
import json
import os
import sys
import logging
from datetime import datetime, timezone
import threading

# Ensure project root is importable (so pipeline.* modules import)
# Add project root to sys.path
ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv(str(ROOT / ".env"))

# Import the pipeline function (user provided)
try:
    from pipeline.rag_pipeline import run_rag_query  # type: ignore
except Exception as e:
    # Import error will be surfaced later at runtime; keep running server for debugging
    run_rag_query = None  # type: ignore

# Config
SERVER_OUT = ROOT / "server_outputs"
JOBS_DIR = SERVER_OUT / "jobs"
RESULTS_DIR = SERVER_OUT / "results"
AUDITS_DIR = ROOT / "pipeline_audits"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
AUDITS_DIR.mkdir(parents=True, exist_ok=True)

# Concurrency control
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))
JOB_SEMAPHORE = threading.Semaphore(MAX_CONCURRENT_JOBS)

# Logger
logging.basicConfig(level=os.getenv("SERVER_LOG_LEVEL", "INFO"))
logger = logging.getLogger("floatchat.server")

app = FastAPI(title="FloatChat RAG Server",
              description="API to run run_rag_query jobs and fetch results.",
              version="0.1.0")

# Allow CORS for local frontend during prototype (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Request models ----
class RunRequest(BaseModel):
    query: str = Field(..., example="What is the minimum pressure recorded by floats in 2025?")
    year: str = Field("2025", example="2025")
    months: Optional[List[str]] = Field(None, example=["01","02"])
    top_k: int = Field(490, example=200)
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    # run sync (blocking) - only for tiny debug queries
    sync: bool = Field(False, description="When true, server runs request synchronously (not recommended).")
    # optional custom output filename (will be ignored for job-running)
    out_name: Optional[str] = None

# ---- Utility helpers ----
def _now_iso():
    return datetime.now(timezone.utc).isoformat()

def _write_json_atomic(path: Path, obj: Any):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, default=str, indent=2))
    tmp.replace(path)

def _job_meta_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"

def _result_path(job_id: str) -> Path:
    return RESULTS_DIR / f"{job_id}.json"

def _make_job_meta(job_id: str, params: Dict[str, Any]):
    meta = {
        "job_id": job_id,
        "created_at": _now_iso(),
        "status": "queued",
        "params": params,
        "updated_at": _now_iso(),
        "notes": None,
        "result_path": None,
        "error": None
    }
    _write_json_atomic(_job_meta_path(job_id), meta)
    return meta

def _update_job_meta(job_id: str, **kwargs):
    p = _job_meta_path(job_id)
    if not p.exists():
        raise FileNotFoundError(f"Job meta not found: {job_id}")
    meta = json.loads(p.read_text())
    meta.update(kwargs)
    meta["updated_at"] = _now_iso()
    _write_json_atomic(p, meta)
    return meta

def _read_job_meta(job_id: str) -> Dict[str,Any]:
    p = _job_meta_path(job_id)
    if not p.exists():
        raise FileNotFoundError(f"Job not found: {job_id}")
    return json.loads(p.read_text())

# ---- Background job runner ----
def _run_job(job_id: str, params: dict):
    """
    Background worker that runs run_rag_query and persists result.
    Uses a semaphore to limit concurrent executions.
    """
    job_meta_path = _job_meta_path(job_id)
    result_file = _result_path(job_id)

    # mark running
    try:
        _update_job_meta(job_id, status="running")
    except Exception as e:
        logger.exception("Failed to set job running: %s", e)

    # try to acquire run slot
    acquired = JOB_SEMAPHORE.acquire(timeout=None)
    logger.info("Job %s acquired semaphore: %s", job_id, acquired)
    try:
        # Ensure run_rag_query imported
        global run_rag_query
        if run_rag_query is None:
            raise RuntimeError("Pipeline function run_rag_query not importable. Check pipeline.rag_pipeline.")

        # Call run_rag_query with safe args and direct result capture
        out_path = result_file
        try:
            # Note: run_rag_query returns dict and also writes output_json if provided.
            result = run_rag_query(
                query=params.get("query"),
                year=params.get("year"),
                months=params.get("months"),
                top_k=int(params.get("top_k", 490)),
                date_from=params.get("date_from"),
                date_to=params.get("date_to"),
                output_json=out_path
            )
        except Exception as e:
            # ensure we capture exceptions from the pipeline
            logger.exception("Job %s failed during run_rag_query: %s", job_id, e)
            _update_job_meta(job_id, status="error", error=str(e), notes="pipeline_exception")
            # write a minimal result file with error info
            _write_json_atomic(result_file, {"status":"error","error":str(e)})
            return

        # successful: ensure result is persisted (run_rag_query should have written it too)
        if isinstance(result, dict):
            # augment with job_id & timestamp
            result.setdefault("job_id", job_id)
            result.setdefault("completed_at", _now_iso())
            _write_json_atomic(result_file, result)
            _update_job_meta(job_id, status="done", result_path=str(result_file))
            logger.info("Job %s finished and persisted result.", job_id)
        else:
            # unexpected return type
            _write_json_atomic(result_file, {"status":"error", "error":"pipeline returned non-dict"})
            _update_job_meta(job_id, status="error", error="pipeline returned non-dict")
    except Exception as e:
        logger.exception("Job runner caught unexpected error: %s", e)
        try:
            _update_job_meta(job_id, status="error", error=str(e))
        except Exception:
            pass
    finally:
        JOB_SEMAPHORE.release()
        logger.info("Job %s released semaphore", job_id)

# ---- API endpoints ----

@app.get("/api/health")
def health():
    return {"status":"ok", "time": _now_iso(), "max_concurrent_jobs": MAX_CONCURRENT_JOBS}

@app.post("/api/run", status_code=202)
def enqueue_run(req: RunRequest, background_tasks: BackgroundTasks):
    """
    Enqueue a run as a background job and return job_id.
    """
    params = req.dict()
    # basic validation / normalization
    if not params.get("query") or not isinstance(params.get("query"), str):
        raise HTTPException(status_code=400, detail="Missing 'query' string")

    job_id = uuid.uuid4().hex
    _make_job_meta(job_id, params)

    # add background job that will run _run_job(job_id, params)
    background_tasks.add_task(_run_job, job_id, params)

    return {"job_id": job_id, "status": "queued", "job_meta": str(_job_meta_path(job_id))}

@app.post("/api/run_sync")
def run_sync(req: RunRequest):
    """
    Run synchronously (blocking) and return result. Use only for small debug queries.
    """
    params = req.dict()
    if run_rag_query is None:
        raise HTTPException(status_code=500, detail="Pipeline run function not available on server.")
    # run directly
    try:
        out = run_rag_query(
            query=params.get("query"),
            year=params.get("year"),
            months=params.get("months"),
            top_k=int(params.get("top_k", 490)),
            date_from=params.get("date_from"),
            date_to=params.get("date_to"),
            output_json=None
        )
    except Exception as e:
        logger.exception("Sync run failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return out

@app.get("/api/status/{job_id}")
def job_status(job_id: str):
    try:
        meta = _read_job_meta(job_id)
        # If done and results exist, attach small info
        if meta.get("status") == "done" and meta.get("result_path"):
            rp = Path(meta["result_path"])
            meta["result_exists"] = rp.exists()
            try:
                meta["result_size_bytes"] = rp.stat().st_size if rp.exists() else None
            except Exception:
                meta["result_size_bytes"] = None
        return meta
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")

@app.get("/api/result/{job_id}")
def job_result(job_id: str):
    rp = _result_path(job_id)
    if not rp.exists():
        raise HTTPException(status_code=404, detail="Result not available yet")
    # read and return JSON
    try:
        content = json.loads(rp.read_text())
        return content
    except Exception as e:
        logger.exception("Failed to read result file %s: %s", rp, e)
        raise HTTPException(status_code=500, detail="Failed to read result file")

@app.get("/api/list")
def list_jobs(limit: int = 50):
    """
    List the most recent job metas (sorted by created_at desc).
    """
    metas = []
    for p in sorted(JOBS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        try:
            metas.append(json.loads(p.read_text()))
        except Exception:
            continue
    return {"count": len(metas), "jobs": metas}

# Run with uvicorn server.api:app --reload (see README instructions)
