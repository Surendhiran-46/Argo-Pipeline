# pipeline/rag_pipeline.py
"""
RAG prototype for Argo data:
 - semantic retrieval from Chroma (profile-level metadata)
 - numeric retrieval from DuckDB (core_measurements / qc tables)
 - context assembly + LLM call (Azure OpenAI Chat Completion endpoint)
 - outputs human-friendly answer and saved data samples

Usage (examples in README section below).
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import duckdb
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv

# vector DB
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ------------------- CONFIG & HELPERS -------------------

load_dotenv()  # loads .env at repo root

# ENV variables (must be set in .env)
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  # full chat completions endpoint URL
# Optional: deployment name (if using azure-style deployments). Not strictly required when endpoint includes deployment.
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")

# Embedding model choices (local sentence-transformers by default)
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Paths (relative to pipeline/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DUCKDB_PATH = PROJECT_ROOT / "argo.duckdb"
CHROMA_DIR = PROJECT_ROOT / "chroma_store"
CHROMA_COLLECTION = "argo_metadata"

# DuckDB: which tables are the monthly core_measurements (pattern)
# We'll query all tables that start with 'core_measurements_' if present.
CORE_TABLE_PREFIX = "core_measurements_"

# Timeouts & limits
REQUEST_TIMEOUT = 60  # seconds for LLM requests

# ---------- Utility functions ----------

def assert_env():
    missing = []
    if not AZURE_OPENAI_KEY:
        missing.append("AZURE_OPENAI_KEY")
    if not AZURE_OPENAI_ENDPOINT:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {missing}. Put them in .env at repo root.")

def init_chroma(client_path: Path = CHROMA_DIR):
    # Use persistent client pointing to the same chroma_store you used earlier
    client = chromadb.PersistentClient(path=str(client_path))
    coll = client.get_collection(CHROMA_COLLECTION)
    return client, coll

def init_embed_model(model_name: str = EMBED_MODEL):
    model = SentenceTransformer(model_name)
    return model

def embed_texts(model, texts: List[str]) -> np.ndarray:
    return np.asarray(model.encode(texts, show_progress_bar=False))

def semantic_retrieve(
    coll,
    embed_model,
    query: str,
    top_k: int = 20,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Perform semantic retrieval from Chroma.
    Returns a list of dicts: {'id': ..., 'score': ..., 'metadata': {...}, 'document': ...}
    If metadata_filter provided, we perform local filtering on returned results (Chroma 'where' may be limited).
    """
    q_vec = embed_texts(embed_model, [query])[0].tolist()
    # Query by embedding to get candidate docs
    # include metadatas, documents, ids
    res = coll.query(
    query_embeddings=[q_vec],
    n_results=top_k,
    include=["metadatas", "documents", "distances"]  # removed "ids"
)
    docs = []
    if not res or "ids" not in res or len(res["ids"]) == 0:
        return docs
    # res fields are lists per query
    ids = res["ids"][0]
    metadatas = res["metadatas"][0]
    documents = res["documents"][0]
    distances = res.get("distances", [[]])[0] if res.get("distances") else [None] * len(ids)
    for i, _id in enumerate(ids):
        md = metadatas[i] if i < len(metadatas) else {}
        doc = documents[i] if i < len(documents) else ""
        dist = distances[i] if i < len(distances) else None
        if metadata_filter:
            # simple conjunctive filter: all keys must match (exact)
            ok = True
            for k, v in metadata_filter.items():
                if k not in md or md[k] != v:
                    ok = False
                    break
            if not ok:
                continue
        docs.append({"id": _id, "metadata": md, "document": doc, "distance": dist})
    return docs

def open_duckdb(db_path: Path = DUCKDB_PATH):
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB database not found: {db_path}")
    con = duckdb.connect(database=str(db_path), read_only=False)
    return con

def build_platform_profile_set(retrieved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    From Chroma metadata, extract platform/profile pairs.
    Metadata expected to contain 'PLATFORM_NUMBER' and optionally 'profile_index' or 'PROFILE_INDEX'.
    Returns list of dicts with keys: platform_number (int) and profile_index (int|None) and optional juld.
    """
    out = []
    for r in retrieved:
        md = r.get("metadata", {}) or {}
        # try different keys
        platform = md.get("PLATFORM_NUMBER") or md.get("platform_number") or md.get("platform")
        if platform is None:
            # skip if no platform number
            continue
        # convert to int if possible
        try:
            platform = int(platform)
        except Exception:
            try:
                platform = int(str(platform).strip())
            except Exception:
                continue
        profile = md.get("profile_index") or md.get("PROFILE_INDEX") or md.get("profile")
        try:
            profile = int(profile) if profile is not None else None
        except Exception:
            profile = None
        juld = md.get("JULD") or md.get("juld")
        out.append({"platform_number": platform, "profile_index": profile, "juld": juld, "metadata": md})
    # deduplicate by platform+profile
    seen = set()
    unique = []
    for e in out:
        key = (e["platform_number"], e["profile_index"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(e)
    return unique

def fetch_metadata_for_candidates(
    con: duckdb.DuckDBPyConnection,
    candidates: List[Dict[str, Any]],
    year: str
) -> pd.DataFrame:
    """
    Fetch project/researcher metadata from DuckDB for candidate platforms.
    Looks across metadata_full tables for the given year.
    """
    if not candidates:
        return pd.DataFrame()

    # Collect unique platform numbers
    platforms = sorted({int(c["platform_number"]) for c in candidates if c.get("platform_number") is not None})
    if not platforms:
        return pd.DataFrame()

    # Discover all metadata_full tables for this year
    tables = [t[0] for t in con.execute(
        "SELECT table_name FROM INFORMATION_SCHEMA.TABLES WHERE table_schema='main'"
    ).fetchall()]
    meta_tables = [t for t in tables if t.startswith("metadata_full_") and year in t]

    if not meta_tables:
        raise RuntimeError(f"No metadata_full tables found for year={year}")

    plat_list = ",".join(map(str, platforms))
    rows = []
    for tbl in meta_tables:
        sql = f"""
        SELECT platform_number, project_name, pi_name, wmo_inst_type, float_serial_no
        FROM {tbl}
        WHERE platform_number IN ({plat_list})
        """
        try:
            df = con.execute(sql).fetchdf()
            if not df.empty:
                rows.append(df)
        except Exception as e:
            print(f"[WARN] metadata query failed on {tbl}: {e}")
            continue

    if rows:
        return pd.concat(rows, ignore_index=True)
    else:
        return pd.DataFrame(columns=["platform_number","project_name","pi_name","wmo_inst_type","float_serial_no"])


def fetch_measurements_for_candidates(
    con: duckdb.DuckDBPyConnection,
    candidates: List[Dict[str, Any]],
    months: Optional[List[str]] = None,
    date_range: Optional[List[str]] = None,
    limit_per_profile: int = 200
) -> pd.DataFrame:
    """
    Fetch numeric records from DuckDB for candidate platform/profile pairs.
    - months: list of 'MM' strings (e.g. ['01','02']). If None, query all core_measurements_* tables.
    - date_range: [start_iso, end_iso] to filter juld (ISO strings)
    - limit_per_profile: safety cap to avoid huge fetches.
    """
    if not candidates:
        return pd.DataFrame()

    # determine tables to search
    # If months provided, build list of table names like core_measurements_YYYY_MM
    year_tables = []
    if months:
        for m in months:
            name = f"{CORE_TABLE_PREFIX}{con.execute('SELECT 1').fetch_df().columns}"  # dummy keep variable
    # Simpler approach: query all tables matching prefix by reading duckdb catalog
    tables = [t[0] for t in con.execute("SELECT table_name FROM INFORMATION_SCHEMA.TABLES WHERE table_schema='main'").fetchall()]
    core_tables = [t for t in tables if t.startswith(CORE_TABLE_PREFIX)]
    if months:
        # filter by ending _YYYY_MM or containing the months given
        core_tables = [t for t in core_tables if any(t.endswith(f"_{m}") or f"_{m}" in t for m in months)]

    if not core_tables:
        raise RuntimeError("No core_measurements tables found in DuckDB.")

    # Build a safe list of platform numbers
    platforms = sorted({int(c["platform_number"]) for c in candidates if c.get("platform_number") is not None})
    # If profiles present, build mapping
    profiles_map = {}
    for c in candidates:
        pnum = c.get("platform_number")
        pidx = c.get("profile_index")
        if pnum is None:
            continue
        profiles_map.setdefault(pnum, set())
        if pidx is not None:
            profiles_map[pnum].add(int(pidx))

    rows = []
    # Query per table to keep results manageable
    for tbl in core_tables:
        # Build platform filter
        plat_list = ",".join(map(str, platforms))
        if len(plat_list) == 0:
            continue
        sql = f"SELECT platform_number, profile_index, juld, PRES, TEMP, PSAL FROM {tbl} WHERE platform_number IN ({plat_list})"
        if date_range and len(date_range) == 2:
            # DuckDB accepts ISO timestamps
            start, end = date_range
            sql += f" AND juld >= TIMESTAMP '{start}' AND juld <= TIMESTAMP '{end}'"
        # small safety cap
        sql += f" LIMIT {max(100000, limit_per_profile * len(platforms))}"
        try:
            df = con.execute(sql).fetchdf()
            if not df.empty:
                rows.append(df)
        except Exception as e:
            print(f"[WARN] query failed on {tbl}: {e}")
            continue

    if rows:
        df_all = pd.concat(rows, ignore_index=True)
    else:
        df_all = pd.DataFrame(columns=["platform_number","profile_index","juld","PRES","TEMP","PSAL"])
    return df_all

def assemble_context(retrieved_meta: List[Dict[str, Any]], df_samples: pd.DataFrame,
                     df_metadata: pd.DataFrame, query: str, max_rows: int = 20) -> str:
    """
    Build a context string for the LLM: top metadata entries + sample numeric + stats.
    """
    parts = []
    parts.append("User query:\n" + query + "\n")

    # Metadata (from Chroma)
    parts.append("Top retrieved profiles (metadata from Chroma):")
    for i, r in enumerate(retrieved_meta[:10]):
        md = r.get("metadata", {})
        summary_items = []
        for k in ("PLATFORM_NUMBER","PROJECT_NAME","PI_NAME","WMO_INST_TYPE","FLOAT_SERIAL_NO","JULD"):
            if k in md and md[k] is not None:
                summary_items.append(f"{k}={md[k]}")
        if summary_items:
            parts.append(f"  {i+1}. " + "; ".join(summary_items))

    # Metadata (direct from DuckDB)
    if not df_metadata.empty:
        parts.append("\nAdditional metadata from DuckDB:")
        parts.append(df_metadata.head(max_rows).to_csv(index=False))

    # Numeric rows
    parts.append("\nSample numeric rows (first rows):")
    if not df_samples.empty:
        parts.append(df_samples.head(max_rows).to_csv(index=False))
    else:
        parts.append("  (no numeric data retrieved)")

    # Summary stats
    if not df_samples.empty:
        stats = df_samples[["TEMP","PSAL","PRES"]].agg(["mean","min","max"]).to_dict()
        parts.append("\nSummary stats (TEMP/PSAL/PRES):")
        parts.append(json.dumps(stats, default=str, indent=2))

    return "\n".join(parts)


def call_azure_openai_chat(system_prompt: str, user_prompt: str, endpoint: str = AZURE_OPENAI_ENDPOINT, api_key: str = AZURE_OPENAI_KEY, max_tokens: int = 512, temperature: float = 0.0):
    """
    Call Azure OpenAI chat completions via direct POST to the endpoint (endpoint should be full chat completions URL).
    Uses header 'api-key' for authentication as Azure requires.
    """
    assert api_key, "AZURE_OPENAI_KEY not set in environment"
    headers = {"Content-Type": "application/json", "api-key": api_key}
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    if resp.status_code != 200:
        raise RuntimeError(f"LLM request failed ({resp.status_code}): {resp.text}")
    data = resp.json()
    # Azure returns choices -> message -> content
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError("Unexpected LLM response structure: " + json.dumps(data)[:1000])
    return content, data

# --------------- Orchestration / CLI ---------------

def run_rag_query(
    query: str,
    year: str,
    months: Optional[List[str]],
    top_k: int,
    date_from: Optional[str],
    date_to: Optional[str],
    output_json: Optional[Path] = None
):
    """
    High-level function: semantic retrieve -> fetch numeric -> assemble -> call LLM
    """
    print("[1/6] Initializing services ...")
    client, coll = init_chroma(CHROMA_DIR)
    embed_model = init_embed_model()
    con = open_duckdb(DUCKDB_PATH)

    print("[2/6] Semantic retrieval from Chroma ...")
    metadata_filter = None  # reserved for exact filter usage
    retrieved = semantic_retrieve(coll, embed_model, query, top_k=top_k, metadata_filter=metadata_filter)
    print(f"  Retrieved {len(retrieved)} semantic candidates")

    print("[3/6] Extract platform/profile candidates ...")
    candidates = build_platform_profile_set(retrieved)
    print(f"  Extracted {len(candidates)} unique platform/profile candidates")

    if len(candidates) == 0:
        print("[WARN] No platform/profile candidates found. Returning retrieved docs to LLM for best-effort answer.")
        # Provide retrieved docs as context without numeric query
        metadata_text = "\n".join([r.get("document","") for r in retrieved[:10]])
        system_prompt = "You are an expert oceanography assistant. Use the provided context strictly to answer."
        user_prompt = f"User question:\n{query}\n\nContext documents:\n{metadata_text}"
        answer, raw = call_azure_openai_chat(system_prompt, user_prompt)
        result = {"answer": answer, "raw_llm": raw, "retrieved": retrieved, "dataframe_rows": 0}
        if output_json:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(result, default=str, indent=2))
        return result

    print("[4/6] Fetch numeric + metadata from DuckDB ...")
    date_range = None
    if date_from and date_to:
        date_range = [date_from, date_to]

    df_numeric = fetch_measurements_for_candidates(con, candidates, months=months, date_range=date_range, limit_per_profile=500)
    print(f"  Retrieved {len(df_numeric)} numeric rows from DuckDB")

    df_metadata = fetch_metadata_for_candidates(con, candidates, year)
    print(f"  Retrieved {len(df_metadata)} metadata rows from DuckDB")


    print("[5/6] Assemble context for LLM ...")
    context = assemble_context(retrieved, df_numeric, df_metadata, query, max_rows=20)


    system_prompt = (
            "You are an expert oceanography assistant with access to *only* the provided context documents and numeric data. Do NOT invent facts. Strict rules:"
            "1) Only use the data passages and numeric rows supplied in the 'Context' section below."
            """2) When citing data, include explicit provenance tags in square brackets:
               - For Chroma docs: [DOC:<id>]
               - For SQL results: [SQL:<table_name>:<row_count>] """
            "   3) If the answer requires data not in context, respond exactly with: 'INSUFFICIENT_CONTEXT' and list what is missing (e.g., 'missing: JULD range' or 'missing: variable PSAL in dataset')."
            "4) Provide a concise summary (3 sentences max), then list 1-3 recommended visualizations with explicit columns (e.g., 'Plot 1: Depth vs TEMP — x=PRES, y=TEMP, color=platform_number')."
            "5) If you use numeric aggregates (mean/min/max), show the exact SQL-derived numbers or CSV snippet used, and tag provenance."
    )
    user_prompt = f"Answer concisely. Query: {query}\n\nContext:\n{context}\n\nReturn a short plain-English summary and an optional list of recommended charts (title and x/y columns)."

    print("[6/6] Calling LLM (Azure OpenAI) ...")
    answer_text, raw = call_azure_openai_chat(system_prompt, user_prompt)
    print("LLM responded (truncated):\n", answer_text[:1000])

    # Build output payload
    out = {
        "query": query,
        "retrieved_count": len(retrieved),
        "candidates_count": len(candidates),
        "numeric_rows": len(df_numeric),
        "metadata_rows": len(df_metadata),
        "answer": answer_text,
        "llm_raw": raw,
    }
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        # Also store a small CSV sample if numeric rows present
        sample_csv_path = None
        if not df_numeric.empty:
            sample_csv_path = str(output_json.with_suffix(".sample.csv"))
            df_numeric.head(200).to_csv(sample_csv_path, index=False)
            out["numeric_sample_csv"] = sample_csv_path
        output_json.write_text(json.dumps(out, default=str, indent=2))

    # close resources
    con.close()
    return out

# --------------- CLI ---------------

def parse_args():
    parser = argparse.ArgumentParser(description="RAG pipeline: semantic retrieval (Chroma) -> DuckDB -> Azure OpenAI")
    parser.add_argument("--query", "-q", required=True, help="User natural language query")
    parser.add_argument("--year", "-y", default="2025", help="Year to search (used to narrow months/tables)")
    parser.add_argument("--months", "-m", nargs="+", help="Optional list of month numbers as two-digit strings, e.g. 01 02 03")
    parser.add_argument("--top_k", type=int, default=490, help="Number of semantic candidates to retrieve from Chroma")
    parser.add_argument("--date_from", type=str, default=None, help="Start date in ISO format (e.g. 2025-03-01)")
    parser.add_argument("--date_to", type=str, default=None, help="End date in ISO format")
    parser.add_argument("--out", type=str, default="./pipeline_outputs/rag_result.json", help="Output JSON path (relative to pipeline/)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    out_path = Path(__file__).resolve().parents[1] / args.out
    try:
        assert_env()
        result = run_rag_query(
            query=args.query,
            year=args.year,
            months=args.months,
            top_k=args.top_k,
            date_from=args.date_from,
            date_to=args.date_to,
            output_json=out_path
        )
        print("\n=== RAG result saved to:", out_path)
        print("Summary:", {k: result.get(k) for k in ('retrieved_count','candidates_count','numeric_rows')})
    except Exception as e:
        print("ERROR:", e)
        raise