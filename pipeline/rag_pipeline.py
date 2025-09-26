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
from typing import List, Dict, Any, Optional, Tuple

import duckdb
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv

# vector DB
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from pipeline.canvas_builder import decide_and_build
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
    Perform semantic retrieval from Chroma and return a cleaned list of candidates.

    Returns a list of dicts with keys:
      - id: document id (string or None)
      - metadata: sanitized metadata dict (keys as returned by Chroma)
      - document: text of the document
      - distance: raw distance value returned by Chroma (float) or None
      - score: computed similarity-like score (float) where higher is better (1/(1+distance)) or None
      - platform_number: int or None (normalized)
      - profile_index: int or None (normalized)

    Behavior:
      - Tries server-side filtering with `where=metadata_filter` if provided (some Chroma versions accept it).
      - If that fails or returns broader results, applies a robust local filter (case-insensitive, numeric-aware).
      - Normalizes common metadata fields (handles bytes, arrays, numpy scalars).
      - Deduplicates by (platform_number, profile_index), keeping candidate with highest score.
      - Returns up to `top_k` final candidates sorted by score desc (None scores last).
    """
    import logging
    logger = logging.getLogger(__name__)

    # --- embed query vector ---
    try:
        q_vec = embed_texts(embed_model, [query])[0].tolist()
    except Exception as e:
        logger.exception("Embedding failed for query: %s", e)
        return []

    # Try server-side query with optional 'where' filter. Fallback if 'where' unsupported.
    try:
        if metadata_filter:
            # some chroma versions accept 'where'; if not, it will raise TypeError
            res = coll.query(
                query_embeddings=[q_vec],
                n_results=top_k,
                include=["metadatas", "documents", "distances"],
                where=metadata_filter,
            )
        else:
            res = coll.query(
                query_embeddings=[q_vec],
                n_results=top_k,
                include=["metadatas", "documents", "distances"],
            )
    except TypeError:
        # client doesn't accept 'where' param signature -> retry without where
        try:
            res = coll.query(
                query_embeddings=[q_vec],
                n_results=top_k,
                include=["metadatas", "documents", "distances"],
            )
        except Exception as e:
            logger.exception("Chroma query failed (no-where retry): %s", e)
            return []
    except Exception as e:
        logger.exception("Chroma query failed: %s", e)
        return []

    # --- Validate response shape (Chroma responses can be nested lists for multi-query) ---
    if not res:
        return []

    # Chroma may return lists of lists (one per query)
    # Normalize to lists for the first (and only) query we made
    try:
        ids = res.get("ids", [[]])[0] if isinstance(res.get("ids"), list) else []
        metadatas = res.get("metadatas", [[]])[0] if isinstance(res.get("metadatas"), list) else []
        documents = res.get("documents", [[]])[0] if isinstance(res.get("documents"), list) else []
        distances = res.get("distances", [[]])[0] if isinstance(res.get("distances"), list) else []
    except Exception:
        # Fallback: try to read non-nested values
        ids = res.get("ids") or []
        metadatas = res.get("metadatas") or []
        documents = res.get("documents") or []
        distances = res.get("distances") or []

    # Helper: sanitize metadata values into plain python types/strings
    def _sanitize_value(v):
        try:
            # bytes -> str
            if isinstance(v, (bytes, bytearray)):
                return v.decode("utf-8", "ignore").strip()
            # numpy scalar
            import numpy as _np
            if isinstance(v, _np.generic):
                return _np.asscalar(v) if hasattr(_np, "asscalar") else v.item()
        except Exception:
            pass
        # lists/tuples -> comma-join small items
        try:
            if isinstance(v, (list, tuple)):
                # convert nested bytes/numpy entries to string
                return ",".join([str(_sanitize_value(x)) for x in v])
        except Exception:
            pass
        # default
        return v

    # Helper: find key case-insensitively
    def _get_key_case_insensitive(d: Dict[str, Any], candidates: List[str]):
        if not isinstance(d, dict):
            return None, None
        lowered = {k.lower(): k for k in d.keys()}
        for cand in candidates:
            if cand.lower() in lowered:
                real_key = lowered[cand.lower()]
                return real_key, d[real_key]
        return None, None

    # Validate and collect items
    items = []
    n = max(len(ids), len(metadatas), len(documents), len(distances))
    for i in range(n):
        try:
            _id = ids[i] if i < len(ids) else None
            raw_md = metadatas[i] if i < len(metadatas) else {}
            doc = documents[i] if i < len(documents) else ""
            dist = distances[i] if i < len(distances) else None
        except Exception:
            # skip malformed entry
            continue

        # sanitize metadata dict
        sanitized = {}
        if isinstance(raw_md, dict):
            for k, v in raw_md.items():
                sanitized[k] = _sanitize_value(v)
        else:
            # rare case: metadata is not dict
            sanitized = {"_raw": _sanitize_value(raw_md)}

        # If metadata_filter provided, apply robust local filtering as fallback
        if metadata_filter:
            match_ok = True
            for fk, fv in metadata_filter.items():
                # find the matching metadata field in sanitized (case-insensitive)
                found_key = None
                for sk in sanitized.keys():
                    if sk.lower() == str(fk).lower():
                        found_key = sk
                        break
                # if not found, treat as non-match
                if found_key is None:
                    match_ok = False
                    break
                val = sanitized.get(found_key)
                # numeric-aware comparison
                try:
                    if val is None or (isinstance(val, str) and str(val).strip() == ""):
                        match_ok = False
                        break
                    if isinstance(fv, (int, float)):
                        # try cast val to float
                        vnum = float(val)
                        if vnum != float(fv):
                            match_ok = False
                            break
                    else:
                        # string compare case-insensitive
                        if str(val).strip().lower() != str(fv).strip().lower():
                            match_ok = False
                            break
                except Exception:
                    # fallback to case-insensitive string compare
                    if str(val).strip().lower() != str(fv).strip().lower():
                        match_ok = False
                        break
            if not match_ok:
                continue  # filtered out

        # extract platform_number/profile_index (robust keys)
        platform_candidate_key, platform_val = _get_key_case_insensitive(sanitized, ["PLATFORM_NUMBER", "platform_number", "platform", "wmo"])
        platform_number = None
        if platform_candidate_key is not None:
            try:
                platform_number = int(platform_val)
            except Exception:
                try:
                    platform_number = int(float(platform_val))
                except Exception:
                    # leave as None if unparseable
                    platform_number = None

        profile_candidate_key, profile_val = _get_key_case_insensitive(sanitized, ["PROFILE_INDEX", "profile_index", "profile"])
        profile_index = None
        if profile_candidate_key is not None:
            try:
                profile_index = int(profile_val)
            except Exception:
                try:
                    profile_index = int(float(profile_val))
                except Exception:
                    profile_index = None

        # compute similarity score from distance when possible
        score = None
        try:
            if dist is not None:
                dnum = float(dist)
                # monotonic transform: higher score => better match
                if dnum >= 0:
                    score = 1.0 / (1.0 + dnum)
                else:
                    score = None
        except Exception:
            score = None

        items.append({
            "id": _id,
            "metadata": sanitized,
            "document": doc,
            "distance": float(dist) if dist is not None else None,
            "score": score,
            "platform_number": platform_number,
            "profile_index": profile_index,
        })

    # Deduplicate: keep best (highest score) per (platform, profile) pair.
    dedup_map: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    for it in items:
        key = (it["platform_number"], it["profile_index"])
        if key in dedup_map:
            existing = dedup_map[key]
            existing_score = existing.get("score") if existing.get("score") is not None else -1.0
            new_score = it.get("score") if it.get("score") is not None else -1.0
            if new_score > existing_score:
                dedup_map[key] = it
        else:
            dedup_map[key] = it

    deduped = list(dedup_map.values())

    # final sort: score desc (None last), tie-breaker by smaller distance
    def _sort_key(x):
        sc = x.get("score")
        if sc is None:
            sc_sort = -999999.0
        else:
            sc_sort = sc
        dist = x.get("distance")
        if dist is None:
            dist_sort = float("inf")
        else:
            dist_sort = dist
        return (-sc_sort, dist_sort)

    deduped.sort(key=_sort_key)

    # trim to top_k
    out = deduped[:top_k]
    return out

def open_duckdb(db_path: Path = DUCKDB_PATH):
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB database not found: {db_path}")
    con = duckdb.connect(database=str(db_path), read_only=False)
    return con

def build_platform_profile_set(retrieved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    From Chroma retrieval items, extract normalized platform/profile candidates.

    Each returned dict contains:
      - platform_number: int if parseable, else None
      - platform_number_raw: original raw value from metadata or doc (string or None)
      - float_serial_no: string if present in metadata (FLOAT_SERIAL_NO)
      - profile_index: int if parseable, else None
      - profile_index_raw: original raw profile string if any
      - juld: ISO-8601 string (UTC) if parseable, else None
      - metadata: original sanitized metadata dict (as returned by semantic_retrieve)
      - document: document string (as returned by semantic_retrieve)
      - id: source document id from Chroma (if available)
      - score: numeric score if available (float) else None

    Heuristics:
      - Accepts many key-name variants (case-insensitive).
      - Falls back to regex parsing in the document text for missing platform/profile/juld.
      - Deduplicates by (platform_number_or_raw, profile_index_or_raw), keeping highest-score item.
    """
    import re
    import logging
    logger = logging.getLogger(__name__)
    import pandas as pd

    out_candidates = []
    for r in retrieved:
        md = (r.get("metadata") or {}) if isinstance(r.get("metadata"), dict) else {}
        doc = r.get("document") or ""
        doc = doc if isinstance(doc, str) else str(doc)
        doc_id = r.get("id")
        score = r.get("score")

        # Helper to fetch metadata value case-insensitively
        def _md_get(keys):
            for k in keys:
                if k in md:
                    return md[k]
            # fallback case-insensitive
            for k in md.keys():
                if k.lower() in [kk.lower() for kk in keys]:
                    return md[k]
            return None

        # extract platform_number (numeric if possible) and raw
        pn_raw = _md_get(["PLATFORM_NUMBER", "platform_number", "platform", "WMO", "wmo", "wmo_inst_type"])
        pn_int = None
        if pn_raw is not None:
            try:
                # some values may be lists or numpy scalars
                if isinstance(pn_raw, (list, tuple)):
                    v = pn_raw[0]
                else:
                    v = pn_raw
                # bytes -> str
                if isinstance(v, (bytes, bytearray)):
                    v = v.decode("utf-8", "ignore")
                # castable to int?
                pn_int = int(v)
            except Exception:
                try:
                    pn_int = int(float(str(v).strip()))
                except Exception:
                    pn_int = None
            try:
                pn_raw = str(pn_raw)
            except Exception:
                pn_raw = None

        # If not found in metadata, try doc text regex for platform/wmo
        if pn_int is None and not pn_raw:
            m = re.search(r'\b(?:platform(?:_)?number|platform|wmo)\s*[:=]?\s*([A-Za-z0-9\-\_]+)\b', doc, flags=re.IGNORECASE)
            if m:
                candidate = m.group(1)
                pn_raw = candidate
                try:
                    pn_int = int(candidate)
                except Exception:
                    try:
                        pn_int = int(float(candidate))
                    except Exception:
                        pn_int = None

        # extract float serial number (useful to map to platform_number if needed)
        float_serial = _md_get(["FLOAT_SERIAL_NO", "float_serial_no", "float_serial", "serial_no"])
        if float_serial is not None:
            try:
                if isinstance(float_serial, (list, tuple)):
                    float_serial = str(float_serial[0])
                else:
                    float_serial = str(float_serial)
            except Exception:
                float_serial = None

        # extract profile index
        prof_raw = _md_get(["PROFILE_INDEX", "profile_index", "profile", "cycle_number", "CYCLE_NUMBER"])
        prof_int = None
        if prof_raw is not None:
            try:
                if isinstance(prof_raw, (list, tuple)):
                    v = prof_raw[0]
                else:
                    v = prof_raw
                if isinstance(v, (bytes, bytearray)):
                    v = v.decode("utf-8", "ignore")
                prof_int = int(v)
            except Exception:
                try:
                    prof_int = int(float(str(v).strip()))
                except Exception:
                    prof_int = None
            try:
                prof_raw = str(prof_raw)
            except Exception:
                prof_raw = None

        # fallback: parse profile index from document like "profile 12" or "cycle 12"
        if prof_int is None:
            m = re.search(r'\b(?:profile|profile_index|cycle|cycle_number)\s*[:=]?\s*(\d+)\b', doc, flags=re.IGNORECASE)
            if m:
                try:
                    prof_int = int(m.group(1))
                    prof_raw = m.group(1)
                except Exception:
                    prof_int = None

        # extract juld (timestamp) - try common metadata keys and doc text fallback
        juld_val = None
        juld_raw = _md_get(["JULD", "juld", "JULD_LOCATION", "REFERENCE_DATE_TIME", "reference_date_time"])
        if juld_raw is not None:
            # flatten
            try:
                if isinstance(juld_raw, (list, tuple)):
                    jr = juld_raw[0]
                else:
                    jr = juld_raw
                # bytes -> str
                if isinstance(jr, (bytes, bytearray)):
                    jr = jr.decode("utf-8", "ignore")
                # try to parse with pandas
                ts = pd.to_datetime(jr, errors="coerce", utc=True)
                if pd.notnull(ts):
                    juld_val = ts.isoformat()
                else:
                    juld_val = None
            except Exception:
                juld_val = None

        # doc-text fallback for ISO-like timestamp (e.g., 2025-01-31T23:56:21 or 2025-01-31 23:56:21)
        if juld_val is None:
            m = re.search(r'(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+\-]\d{2}:?\d{2})?)', doc)
            if not m:
                m = re.search(r'(\d{4}-\d{2}-\d{2})', doc)
            if m:
                try:
                    ts = pd.to_datetime(m.group(1), errors="coerce", utc=True)
                    if pd.notnull(ts):
                        juld_val = ts.isoformat()
                except Exception:
                    juld_val = None

        # Build candidate dictionary
        candidate = {
            "platform_number": pn_int,
            "platform_number_raw": pn_raw,
            "float_serial_no": float_serial,
            "profile_index": prof_int,
            "profile_index_raw": prof_raw,
            "juld": juld_val,
            "metadata": md,
            "document": doc,
            "id": doc_id,
            "score": r.get("score"),
        }

        # If neither platform nor float_serial present, skip (no way to join to DB)
        if candidate["platform_number"] is None and not candidate["float_serial_no"] and not candidate["platform_number_raw"]:
            # still keep candidates that have profile_index + juld? no — skip as orphan
            logger.debug("Skipping candidate with no platform/serial: id=%s", doc_id)
            continue

        out_candidates.append(candidate)

    # Deduplicate by (platform_key, profile_key) where platform_key is platform_number if present else platform_number_raw
    dedup = {}
    for c in out_candidates:
        pkey = c["platform_number"] if c["platform_number"] is not None else c["platform_number_raw"]
        prof_key = c["profile_index"] if c["profile_index"] is not None else c["profile_index_raw"]
        dedup_key = (str(pkey), str(prof_key))
        existing = dedup.get(dedup_key)
        if existing is None:
            dedup[dedup_key] = c
        else:
            # keep the one with higher numeric score (None treated as -inf)
            existing_score = existing.get("score") if existing.get("score") is not None else -9999.0
            new_score = c.get("score") if c.get("score") is not None else -9999.0
            if new_score > existing_score:
                dedup[dedup_key] = c

    results = list(dedup.values())
    # sort by score desc (None scores go last)
    results.sort(key=lambda x: (-(x.get("score") or -9999.0), x.get("juld") or ""))

    return results

def fetch_metadata_for_candidates(
    con: "duckdb.DuckDBPyConnection",
    candidates: list,
    year: str
) -> "pd.DataFrame":
    """
    Fetch project/researcher metadata from DuckDB for candidate platforms.
    - Looks for monthly metadata tables for the given year first (metadata_full_YYYY_MM),
      falls back to metadata_clean_YYYY_MM and then to any metadata_full_* / metadata_clean_* tables.
    - Handles case-insensitive / variant column names and returns a canonical dataframe.

    Returns a pandas.DataFrame with canonical columns (if available):
      ['platform_number','project_name','pi_name','wmo_inst_type','float_serial_no',
       'format_version','handbook_version','firmware_version','config_mission_number','reference_date_time']

    Defensive: returns empty DataFrame if no matches found. Does not raise on missing columns.
    """
    import pandas as pd
    # fast exit
    if not candidates:
        return pd.DataFrame()

    # Build unique platform list (safe: upstream cast to int is expected)
    platforms = sorted({int(c["platform_number"]) for c in candidates if c.get("platform_number") is not None})
    if not platforms:
        return pd.DataFrame()

    # Query available table names in DuckDB (main schema)
    try:
        tables = [t[0] for t in con.execute("SELECT table_name FROM INFORMATION_SCHEMA.TABLES WHERE table_schema='main'").fetchall()]
    except Exception:
        tables = []

    # Prefer metadata_full for the given year, then metadata_clean, then any metadata tables
    def choose_meta_tables(year):
        candidates_list = []
        # exact year-month prefix e.g. metadata_full_2025_
        candidates_list += [t for t in tables if t.startswith(f"metadata_full_{year}_") or t == f"metadata_full_{year}"]
        if not candidates_list:
            candidates_list += [t for t in tables if t.startswith(f"metadata_clean_{year}_") or t == f"metadata_clean_{year}"]
        # fallback to any year's metadata_full
        if not candidates_list:
            candidates_list += [t for t in tables if t.startswith("metadata_full_")]
        # fallback to any metadata_clean
        if not candidates_list:
            candidates_list += [t for t in tables if t.startswith("metadata_clean_")]
        return sorted(set(candidates_list))

    meta_tables = choose_meta_tables(year)

    if not meta_tables:
        # nothing to read
        return pd.DataFrame(columns=[
            "platform_number","project_name","pi_name","wmo_inst_type","float_serial_no",
            "format_version","handbook_version","firmware_version","config_mission_number","reference_date_time"
        ])

    # Helper: map available columns in a table to canonical names
    def map_columns_for_table(tbl_name):
        # read PRAGMA table_info to get actual column names
        try:
            cols_info = con.execute(f"PRAGMA table_info('{tbl_name}')").fetchall()
            existing_cols = [c[1] for c in cols_info]  # PRAGMA returns (cid,name,type,notnull,default,pk)
        except Exception:
            existing_cols = []

        # lower map for lookup
        existing_lower = {c.lower(): c for c in existing_cols}

        # candidate keys to search for each desired canonical field (order = priority)
        lookup = {
            "platform_number": ["platform_number", "platform", "PLATFORM_NUMBER", "PLATFORM"],
            "project_name": ["project_name", "project", "PROJECT_NAME"],
            "pi_name": ["pi_name", "pi", "PI_NAME", "investigator"],
            "wmo_inst_type": ["wmo_inst_type", "wmo", "WMO_INST_TYPE"],
            "float_serial_no": ["float_serial_no", "float_serial", "FLOAT_SERIAL_NO", "serial_number"],
            "format_version": ["format_version", "FORMAT_VERSION"],
            "handbook_version": ["handbook_version", "HANDBOOK_VERSION"],
            "firmware_version": ["firmware_version", "FIRMWARE_VERSION"],
            "config_mission_number": ["config_mission_number", "CONFIG_MISSION_NUMBER"],
            "reference_date_time": ["reference_date_time", "REFERENCE_DATE_TIME", "ref_date"]
        }

        mapped = {}
        for canon, candidates_keys in lookup.items():
            found = None
            for k in candidates_keys:
                if k.lower() in existing_lower:
                    found = existing_lower[k.lower()]
                    break
            mapped[canon] = found  # may be None
        return mapped, existing_cols

    # Build a safe platform list string (we already converted to int)
    plat_list = ",".join(map(str, platforms))

    dfs = []
    for tbl in meta_tables:
        mapped_cols, existing_cols = map_columns_for_table(tbl)
        # if platform column does not exist in this table, skip it
        plat_col = mapped_cols.get("platform_number")
        if plat_col is None:
            # skip this table (can't filter without platform column)
            continue

        # Build SELECT clause: use existing column AS canonical_name, else NULL AS canonical_name
        select_snippets = []
        for canon in ["platform_number","project_name","pi_name","wmo_inst_type","float_serial_no",
                      "format_version","handbook_version","firmware_version","config_mission_number","reference_date_time"]:
            src = mapped_cols.get(canon)
            if src:
                # alias to canonical name (lowercase canonical column)
                select_snippets.append(f"{src} AS {canon}")
            else:
                select_snippets.append(f"NULL AS {canon}")

        sql = f"SELECT {', '.join(select_snippets)} FROM {tbl} WHERE {plat_col} IN ({plat_list})"
        try:
            df = con.execute(sql).fetchdf()
            if df is None or df.empty:
                continue
            # ensure canonical columns exist (duckdb may return uppercase/lowercase as-is)
            df.columns = [c.lower() for c in df.columns]
            # rename to canonical lowercase names we used
            # coerces types
            # convert platform_number to integer-like (nullable)
            if "platform_number" in df.columns:
                df["platform_number"] = pd.to_numeric(df["platform_number"], errors="coerce").astype("Int64")
            # parse reference_date_time to datetime if present
            if "reference_date_time" in df.columns:
                try:
                    df["reference_date_time"] = pd.to_datetime(df["reference_date_time"], errors="coerce", utc=True)
                except Exception:
                    df["reference_date_time"] = df["reference_date_time"].astype(str)
            dfs.append(df)
        except Exception as e:
            # print warning and continue
            print(f"[WARN] metadata query failed on {tbl}: {e}")
            continue

    if not dfs:
        # nothing matched
        return pd.DataFrame(columns=[
            "platform_number","project_name","pi_name","wmo_inst_type","float_serial_no",
            "format_version","handbook_version","firmware_version","config_mission_number","reference_date_time"
        ])

    # concat and deduplicate by platform_number keeping the first observed row
    df_all = pd.concat(dfs, ignore_index=True, sort=False)

    # normalize column order & names (lowercase canonical)
    desired_cols = ["platform_number","project_name","pi_name","wmo_inst_type","float_serial_no",
                    "format_version","handbook_version","firmware_version","config_mission_number","reference_date_time"]
    for c in desired_cols:
        if c not in df_all.columns:
            df_all[c] = pd.NA

    # keep only desired columns (in canonical order)
    df_all = df_all[desired_cols]

    # drop duplicates (prefer earliest encountered row)
    df_all = df_all.drop_duplicates(subset=["platform_number"], keep="first").reset_index(drop=True)

    # final tidy: cast numeric-like columns where possible
    for c in ["format_version","handbook_version","config_mission_number","wmo_inst_type"]:
        if c in df_all.columns:
            df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

    return df_all


def fetch_measurements_for_candidates(
    con: "duckdb.DuckDBPyConnection",
    candidates: List[Dict[str, Any]],
    months: Optional[List[str]] = None,
    date_range: Optional[List[str]] = None,
    limit_per_profile: int = 200
) -> pd.DataFrame:
    """
    Fetch numeric records from DuckDB for candidate platform/profile pairs.

    - con: duckdb connection
    - candidates: list of dicts from build_platform_profile_set(...)
    - months: optional list of 'MM' or 'YYYY_MM' strings to restrict which core_measurements tables are queried.
    - date_range: [start_iso, end_iso] or None; uses DuckDB TIMESTAMP comparisons.
    - limit_per_profile: max number of rows returned per (platform_number, profile_index) partition.

    Returns: pandas.DataFrame with available numeric columns merged from matching core tables.
    """
    import logging, math, time
    logger = logging.getLogger(__name__)
    start_time = time.time()

    # quick guard
    if not candidates:
        # consistent empty dataframe with common expected columns
        cols = ["platform_number", "profile_index", "juld", "level_index", "PRES", "TEMP", "PSAL",
                "PRES_QC", "TEMP_QC", "PSAL_QC", "PRES_ADJUSTED", "TEMP_ADJUSTED", "PSAL_ADJUSTED"]
        return pd.DataFrame(columns=cols)

    # --- Discover core_measurements tables in DuckDB catalog ---
    try:
        all_tables = [t[0] for t in con.execute(
            "SELECT table_name FROM INFORMATION_SCHEMA.TABLES WHERE table_schema='main'"
        ).fetchall()]
    except Exception as e:
        logger.exception("Failed to read DuckDB catalog: %s", e)
        raise

    # default prefix (safe)
    CORE_PREFIX = globals().get("CORE_TABLE_PREFIX", "core_measurements_")
    core_tables = [t for t in all_tables if t.startswith(CORE_PREFIX)]
    if not core_tables:
        raise RuntimeError("No core_measurements tables found in DuckDB (prefix '%s')." % CORE_PREFIX)

    # If months provided, filter tables conservatively.
    if months:
        months_normalized = []
        for m in months:
            m = str(m).strip()
            # Accept 'MM' or 'YYYY_MM'
            if len(m) == 2 and m.isdigit():
                months_normalized.append(m)
            else:
                months_normalized.append(m)
        filtered = []
        for t in core_tables:
            for m in months_normalized:
                if m in t:
                    filtered.append(t)
                    break
        core_tables = sorted(set(filtered))
        if not core_tables:
            logger.warning("No core tables matched months filter; falling back to all core tables.")
            # keep core_tables as originally discovered

    # --- Build platform_number list and profile_map from candidates ---
    platforms = set()
    profiles_map = {}  # platform -> set(profile_index)
    serial_lookup_needed = set()
    for c in candidates:
        pn = c.get("platform_number")
        if pn is not None:
            try:
                pn_int = int(pn)
                platforms.add(pn_int)
            except Exception:
                # sometimes platform_number_raw exists
                pass
        else:
            # try to resolve later via float_serial_no
            fs = c.get("float_serial_no")
            if fs:
                serial_lookup_needed.add(str(fs))

        pidx = c.get("profile_index")
        if pn is not None and pidx is not None:
            try:
                profiles_map.setdefault(int(pn), set()).add(int(pidx))
            except Exception:
                pass

    # --- If some candidates lack numeric platform_number but have float_serial_no, try mapping via metadata tables ---
    if serial_lookup_needed:
        # find metadata tables
        metadata_tables = [t for t in all_tables if t.startswith("metadata_full_") or t.startswith("metadata_clean_")]
        if metadata_tables:
            # We'll build a single union query across metadata tables to resolve serial -> platform_number
            union_sql_parts = []
            for mt in metadata_tables:
                union_sql_parts.append(f"SELECT PLATFORM_NUMBER, FLOAT_SERIAL_NO FROM {mt}")
            union_sql = " UNION ALL ".join(union_sql_parts)
            # sanitize serials for SQL
            serials = list(set(serial_lookup_needed))
            escaped = [s.replace("'", "''") for s in serials]
            in_list = ",".join(f"'{s}'" for s in escaped)
            lookup_sql = f"SELECT DISTINCT PLATFORM_NUMBER, FLOAT_SERIAL_NO FROM ({union_sql}) WHERE FLOAT_SERIAL_NO IN ({in_list})"
            try:
                df_map = con.execute(lookup_sql).fetchdf()
                # populate platforms set
                if not df_map.empty:
                    for _, r in df_map.iterrows():
                        try:
                            mapped = int(r["PLATFORM_NUMBER"])
                            platforms.add(mapped)
                        except Exception:
                            pass
            except Exception as e:
                logger.exception("Serial->platform lookup failed: %s", e)
                # continue without mapping

    # After attempts, if still no platforms, nothing to query
    if not platforms:
        logger.debug("No numeric platform_numbers resolved from candidates.")
        return pd.DataFrame(columns=["platform_number", "profile_index", "juld", "level_index", "PRES", "TEMP", "PSAL"])

    # sanitize platforms -> sorted list
    try:
        platform_list = sorted({int(p) for p in platforms})
    except Exception:
        platform_list = sorted([int(p) for p in platforms if isinstance(p, (int, float))])

    if not platform_list:
        return pd.DataFrame(columns=["platform_number", "profile_index", "juld", "level_index", "PRES", "TEMP", "PSAL"])

    # helper: build safe comma list
    plat_list_sql = ",".join(str(int(x)) for x in platform_list)

    # date filter SQL fragment
    date_clause = ""
    if date_range and len(date_range) == 2:
        start = str(date_range[0]).replace("'", "''")
        end = str(date_range[1]).replace("'", "''")
        # Use TIMESTAMP literal - DuckDB will parse ISO strings.
        date_clause = f" AND juld >= TIMESTAMP '{start}' AND juld <= TIMESTAMP '{end}' "

    # desired columns (prefer to include QC and adjusted columns if present)
    desired_cols = ["platform_number", "profile_index", "juld", "level_index",
                    "PRES", "TEMP", "PSAL", "PRES_QC", "TEMP_QC", "PSAL_QC",
                    "PRES_ADJUSTED", "TEMP_ADJUSTED", "PSAL_ADJUSTED"]

    dfs = []
    # For each table, detect available columns and query using a window function to limit per profile
    for tbl in core_tables:
        try:
            # Get table columns (PRAGMA table_info)
            try:
                tbl_info = con.execute(f"PRAGMA table_info('{tbl}')").fetchdf()
                if "name" in tbl_info.columns:
                    available_cols = [str(x) for x in tbl_info["name"].tolist()]
                elif "column_name" in tbl_info.columns:
                    available_cols = [str(x) for x in tbl_info["column_name"].tolist()]
                else:
                    # fallback: fetchall tuples (first element is column name)
                    available_cols = [r[0] for r in con.execute(f"PRAGMA table_info('{tbl}')").fetchall()]
            except Exception:
                # if PRAGMA fails, assume typical columns (best-effort)
                available_cols = []

            proj_cols = [c for c in desired_cols if c in available_cols]
            # always include platform_number/profile_index/juld if present
            for c in ("platform_number", "profile_index", "juld"):
                if c not in proj_cols and c in available_cols:
                    proj_cols.insert(0, c)

            if not proj_cols:
                # nothing to select from this table (odd), skip
                continue

            proj_sql = ", ".join(proj_cols)

            # Build windowed SQL to cap rows per (platform_number, profile_index)
            sql = f"""
                SELECT {proj_sql} FROM (
                    SELECT {proj_sql},
                           ROW_NUMBER() OVER (PARTITION BY platform_number, profile_index ORDER BY juld DESC NULLS LAST) as __rn
                    FROM {tbl}
                    WHERE platform_number IN ({plat_list_sql})
                    {date_clause}
                ) sub
                WHERE __rn <= {int(limit_per_profile)}
            """

            # Execute and fetch as DataFrame
            df = con.execute(sql).fetchdf()
            if df is None or df.empty:
                continue

            # Ensure consistent column names / types
            # Convert juld to pandas datetime (DuckDB often returns timezone-aware)
            if "juld" in df.columns:
                try:
                    df["juld"] = pd.to_datetime(df["juld"], errors="coerce", utc=True)
                except Exception:
                    pass

            dfs.append(df)
        except Exception as exc:
            logger.warning("Query failed on table %s: %s", tbl, exc)
            # continue with other tables

    # finalize results
    if not dfs:
        total_time = time.time() - start_time
        logger.debug("No numeric rows fetched. elapsed=%.2fs", total_time)
        return pd.DataFrame(columns=["platform_number", "profile_index", "juld", "level_index", "PRES", "TEMP", "PSAL"])

    result_df = pd.concat(dfs, ignore_index=True, sort=False)

    # Post-processing: ensure platform_number/profile_index ints where possible
    if "platform_number" in result_df.columns:
        try:
            result_df["platform_number"] = result_df["platform_number"].astype("Int64")
        except Exception:
            pass
    if "profile_index" in result_df.columns:
        try:
            result_df["profile_index"] = result_df["profile_index"].astype("Int64")
        except Exception:
            pass

    total_time = time.time() - start_time
    logger.debug("Fetched %d rows from %d tables in %.2fs", len(result_df), len(dfs), total_time)
    return result_df

def assemble_context(retrieved_meta: List[Dict[str, Any]], df_samples: "pd.DataFrame",
                     df_metadata: "pd.DataFrame", query: str, max_rows: int = 20) -> str:
    """
    Build a context string for the LLM: top metadata entries + sample numeric + stats.
    - retrieved_meta: list of items from semantic_retrieve (contains metadata, id, score, etc.)
    - df_samples: pandas.DataFrame of numeric rows fetched from DuckDB (core measurements)
    - df_metadata: pandas.DataFrame of metadata rows fetched from DuckDB (if available)
    - query: original user query string
    - max_rows: desired upper bound on numeric/sample rows included in context
    Returns a single string (the context) to be sent to the LLM.
    """
    import pandas as pd
    import numpy as np
    import json
    from textwrap import shorten

    # Defensive defaults
    if retrieved_meta is None:
        retrieved_meta = []
    if df_samples is None:
        df_samples = pd.DataFrame()
    if df_metadata is None:
        df_metadata = pd.DataFrame()

    # Columns we prefer to show from numeric table (ordered)
    preferred_cols = [
        "platform_number", "profile_index", "juld", "level_index",
        "PRES", "TEMP", "PSAL",
        "PRES_ADJUSTED", "TEMP_ADJUSTED", "PSAL_ADJUSTED",
        "PRES_QC", "TEMP_QC", "PSAL_QC"
    ]

    # Short helper for safe head->csv export
    def _df_to_csv_snippet(df: pd.DataFrame, n: int = 10, cols: Optional[list] = None) -> str:
        if df.empty:
            return "(empty)\n"
        use = list(cols) if cols else list(df.columns)
        use = [c for c in use if c in df.columns]
        if not use:
            # nothing to show
            return "(no displayable columns)\n"
        snippet = df[use].head(n).copy()
        # Convert datetimes to ISO to avoid large reprs
        from pandas.api.types import is_datetime64_any_dtype
        for c in snippet.columns:
            if is_datetime64_any_dtype(snippet[c]):
                snippet[c] = snippet[c].astype(str)
        return snippet.to_csv(index=False)

    # --- SYSTEM INSTRUCTION (anti-hallucination) ---
    instruction = (
        "IMPORTANT: Use *only* the data provided below to answer. "
        "Do not invent numbers, dates, or facts not present here. "
        "If the requested information cannot be determined from the provided data, reply exactly with: INSUFFICIENT_CONTEXT\n"
        "When reporting numeric results, include brief provenance tags such as [SOURCE:DUCKDB rows=<n>] or [DOC:id=<id>].\n"
    )

    parts = []
    parts.append("=== DATA CONTEXT FOR QUERY ===\n")
    parts.append("SYSTEM_INSTRUCTION:\n" + instruction + "\n")
    parts.append("User query:\n" + query.strip() + "\n")

    # --- Retrieval summary ---
    n_meta = len(retrieved_meta)
    n_numeric = int(len(df_samples)) if hasattr(df_samples, "__len__") else 0
    unique_platforms = set()
    unique_profiles = set()
    sample_juld_min = None
    sample_juld_max = None
    lat_min = lon_min = None
    lat_max = lon_max = None

    # extract quick stats from retrieved_meta
    for r in retrieved_meta:
        md = r.get("metadata", {}) or {}
        pn = md.get("PLATFORM_NUMBER") or md.get("platform_number") or md.get("platform")
        try:
            if pn is not None:
                unique_platforms.add(int(pn))
        except Exception:
            unique_platforms.add(str(pn))
        # profile
        pi = md.get("PROFILE_INDEX") or md.get("profile_index") or md.get("profile")
        if pi is not None:
            try:
                unique_profiles.add(int(pi))
            except Exception:
                unique_profiles.add(str(pi))

    # from df_samples, date & geo coverage (if columns present)
    if n_numeric > 0:
        if "juld" in df_samples.columns:
            try:
                jmin = pd.to_datetime(df_samples["juld"], errors="coerce").min()
                jmax = pd.to_datetime(df_samples["juld"], errors="coerce").max()
                sample_juld_min = jmin.isoformat() if pd.notnull(jmin) else None
                sample_juld_max = jmax.isoformat() if pd.notnull(jmax) else None
            except Exception:
                pass
        for c in ("LATITUDE", "latitude", "LAT", "lat"):
            if c in df_samples.columns:
                lat_min = float(df_samples[c].min())
                lat_max = float(df_samples[c].max())
                break
        for c in ("LONGITUDE", "longitude", "LON", "lon"):
            if c in df_samples.columns:
                lon_min = float(df_samples[c].min())
                lon_max = float(df_samples[c].max())
                break
        # expand unique platform/profile from actual numeric rows
        if "platform_number" in df_samples.columns:
            unique_platforms.update(df_samples["platform_number"].dropna().unique().tolist())
        if "profile_index" in df_samples.columns:
            unique_profiles.update(df_samples["profile_index"].dropna().unique().tolist())

    parts.append("Retrieval summary:\n")
    parts.append(f"  - Chroma candidates returned: {n_meta}\n")
    parts.append(f"  - Numeric rows retrieved from DuckDB: {n_numeric}\n")
    parts.append(f"  - Unique platforms in context (sample): {len(unique_platforms)}\n")
    parts.append(f"  - Unique profiles in context (sample): {len(unique_profiles)}\n")
    if sample_juld_min or sample_juld_max:
        parts.append(f"  - Time coverage (sample rows): {sample_juld_min} -> {sample_juld_max}\n")
    if lat_min is not None and lon_min is not None:
        parts.append(f"  - Geo bbox (sample rows): lat {lat_min:.4f}..{lat_max:.4f}, lon {lon_min:.4f}..{lon_max:.4f}\n")

    parts.append("\nTop Chroma metadata hits (up to 10):")
    # --- top metadata items ---
    def _fmt_md_short(md):
        fields = []
        if not isinstance(md, dict):
            return str(md)
        for k in ("PLATFORM_NUMBER", "platform_number", "PROJECT_NAME", "PI_NAME", "FLOAT_SERIAL_NO", "JULD", "LATITUDE", "LONGITUDE"):
            # case-insensitive key find
            val = None
            for kk in md.keys():
                if kk.lower() == k.lower():
                    val = md[kk]
                    break
            if val is not None:
                fields.append(f"{k}={str(val)}")
        return "; ".join(fields) if fields else "(no typical fields)"

    for idx, r in enumerate(retrieved_meta[:10], start=1):
        md = r.get("metadata", {}) or {}
        docid = r.get("id")
        score = r.get("score")
        short = _fmt_md_short(md)
        parts.append(f"  {idx}. doc_id={docid} score={score} => {short}")

    # --- Additional metadata from DuckDB (if available) ---
    if not df_metadata.empty:
        parts.append("\nAdditional metadata rows (from DuckDB, top rows):")
        # convert datetimes to strings to keep CSV stable
        md_cols = list(df_metadata.columns)
        md_preview = df_metadata.head(min(max_rows, 20)).copy()
        from pandas.api.types import is_datetime64_any_dtype
        for c in md_preview.columns:
            if is_datetime64_any_dtype(md_preview[c]):
                md_preview[c] = md_preview[c].astype(str)
        parts.append(_df_to_csv_snippet(md_preview, n=min(max_rows, 20), cols=md_cols))

    # --- Numeric sample selection strategy ---
    parts.append("\nRepresentative numeric samples (strategy: surface/median/deep per recent profiles):")
    sample_text = ""
    if df_samples.empty:
        sample_text = "  (no numeric data retrieved)\n"
    else:
        # Ensure juld is datetime for ordering
        df = df_samples.copy()
        if "juld" in df.columns:
            try:
                df["juld"] = pd.to_datetime(df["juld"], errors="coerce", utc=True)
            except Exception:
                pass

        # compute per-profile summaries and order by most recent juld
        group_by = []
        if "platform_number" in df.columns and "profile_index" in df.columns:
            group_by = ["platform_number", "profile_index"]
            grp = df.groupby(group_by, dropna=True, observed=True)
            profile_info = grp.agg({'juld': 'max', 'level_index': 'count'}).reset_index().rename(columns={'juld': 'latest_juld', 'level_index': 'n_levels'})
            profile_info = profile_info.sort_values(by='latest_juld', ascending=False).reset_index(drop=True)
        else:
            # fallback group by profile_index alone, or treat whole df as single profile
            if "profile_index" in df.columns:
                group_by = ["profile_index"]
                grp = df.groupby(group_by, dropna=True, observed=True)
                profile_info = grp.agg({'juld': 'max', 'level_index': 'count'}).reset_index().sort_values(by='juld', ascending=False)
            else:
                # single group
                profile_info = pd.DataFrame([{"_single": True}])

        # Decide profile sampling size
        total_profiles = len(profile_info)
        if total_profiles == 0:
            sample_text = "  (no profiles found in numeric rows)\n"
        else:
            # choose up to min(total_profiles, max_profiles) profiles
            max_profiles = min(total_profiles, max(10, max_rows))  # cap selection
            # target rows per profile; try to collect surface/median/deep (up to 3)
            per_profile_target = 3
            # ensure we don't exceed max_rows
            est_profiles = min(total_profiles, max(1, max_rows // per_profile_target))
            est_profiles = min(est_profiles, max_profiles)
            selected_profiles = profile_info.head(est_profiles)

            chosen_rows = []
            for _, prow in selected_profiles.iterrows():
                # build mask for this profile
                if set(["platform_number", "profile_index"]).issubset(profile_info.columns) or set(["platform_number", "profile_index"]).issubset(df.columns):
                    if "platform_number" in prow and "profile_index" in prow:
                        mask = (df["platform_number"] == prow["platform_number"]) & (df["profile_index"] == prow["profile_index"])
                    else:
                        # fallback: by profile_index only
                        mask = (df["profile_index"] == prow.get("profile_index"))
                else:
                    mask = pd.Series([True]*len(df), index=df.index)

                dfp = df[mask]
                if dfp.empty:
                    continue

                # pick surface (min PRES) if PRES exists
                if "PRES" in dfp.columns and dfp["PRES"].notna().any():
                    try:
                        min_row = dfp.loc[dfp["PRES"].idxmin()]
                        max_row = dfp.loc[dfp["PRES"].idxmax()]
                        # median by PRES
                        median_val = dfp["PRES"].quantile(0.5)
                        # find row closest to median
                        median_idx = (dfp["PRES"] - median_val).abs().idxmin()
                        median_row = dfp.loc[median_idx]
                        candidates_rows = [min_row, median_row, max_row]
                    except Exception:
                        # fallback to first/last rows
                        candidates_rows = [dfp.iloc[0], dfp.iloc[-1]]
                else:
                    # fallback by level_index ordering if present
                    if "level_index" in dfp.columns:
                        candidates_rows = [dfp.sort_values("level_index").iloc[0], dfp.sort_values("level_index").iloc[-1]]
                    else:
                        candidates_rows = [dfp.iloc[0]]

                # append unique rows (by index)
                for rr in candidates_rows:
                    if rr.name not in [r.name for r in chosen_rows]:
                        chosen_rows.append(rr)
                    # stop early if exceeded
                    if len(chosen_rows) >= max_rows:
                        break
                if len(chosen_rows) >= max_rows:
                    break

            # Build a DataFrame for chosen rows
            if chosen_rows:
                sample_df = pd.DataFrame(chosen_rows)
                # ensure consistent columns and datetimes
                from pandas.api.types import is_datetime64_any_dtype
                for c in sample_df.columns:
                    if is_datetime64_any_dtype(sample_df[c]):
                        sample_df[c] = sample_df[c].astype(str)
                # pick display columns
                display_cols = [c for c in preferred_cols if c in sample_df.columns]
                sample_text = _df_to_csv_snippet(sample_df, n=len(sample_df), cols=display_cols)
            else:
                sample_text = "  (no representative rows could be constructed)\n"

    # Attach numeric sample text
    parts.append(sample_text)

    # --- Summary statistics for numeric vars ---
    parts.append("\nAggregated summary statistics (per variable):")
    stats_text = ""
    if not df_samples.empty:
        numeric_vars = [c for c in ["TEMP", "PSAL", "PRES", "TEMP_ADJUSTED", "PSAL_ADJUSTED", "PRES_ADJUSTED"] if c in df_samples.columns]
        if numeric_vars:
            try:
                agg = df_samples[numeric_vars].agg(["count", "mean", "median", "min", "max", "std"]).to_dict()
                # sanitize and reduce precision to reasonable digits
                short_agg = {}
                for var, metrics in agg.items():
                    short_agg[var] = {k: (None if pd.isna(v) else (float(np.round(v, 6)) if isinstance(v, (int, float, np.number)) else v)) for k, v in metrics.items()}
                stats_text = json.dumps(short_agg, indent=2)
            except Exception as e:
                stats_text = "(error computing stats)\n"
        else:
            stats_text = "(no numeric variables present)"
    else:
        stats_text = "(no numeric rows)"

    parts.append(stats_text)

    # --- Provenance & next-step checks ---
    parts.append("\nProvenance & checks:")
    parts.append(f"  - Number of Chroma docs: {n_meta}")
    parts.append(f"  - Number of numeric rows: {n_numeric}")
    parts.append("  - NOTE: This context only contains the sample rows and aggregated stats above. "
                 "If you need to verify exact SQL used, consult the pipeline audit file written by the system (pipeline_audits/...).")

    # Final assembly & token-budget guard (approximate)
    # Estimate tokens roughly as characters / 4 (conservative)
    context = "\n".join(parts)
    def _estimate_tokens(s: str) -> int:
        return max(1, int(len(s) / 4))

    token_limit = 12000  # safe default. adjust in pipeline if needed.
    est = _estimate_tokens(context)
    # if too large, shrink numeric sample iteratively
    current_max = max_rows
    while est > token_limit and current_max > 2:
        # reduce sample size by half and rebuild
        current_max = max(2, current_max // 2)
        # recursive-ish: call self with decreased max_rows to regenerate smaller sample block
        # to avoid recursion, replicate sampling logic by re-calling assemble_context limited to smaller max_rows is okay only if function external, but here we rebuild minimal parts:
        # Simpler: remove df_metadata preview and reduce sample rows text
        # remove metadata preview
        # re-build context but with smaller chosen max_rows; we will just truncate numeric sample_text
        if isinstance(sample_text, str) and sample_text.strip():
            # naive truncation: keep first N lines
            lines = sample_text.splitlines()
            truncated = "\n".join(lines[:max(10, current_max)])
            parts = parts[:0]  # reconstruct minimal context
            parts.append("=== DATA CONTEXT FOR QUERY ===\n")
            parts.append("SYSTEM_INSTRUCTION:\n" + instruction + "\n")
            parts.append("User query:\n" + query.strip() + "\n")
            parts.append("Retrieval summary:\n")
            parts.append(f"  - Chroma candidates returned: {n_meta}\n")
            parts.append(f"  - Numeric rows retrieved from DuckDB: {n_numeric}\n")
            parts.append("\nRepresentative numeric samples (TRUNCATED):\n")
            parts.append(truncated)
            parts.append("\nAggregated summary statistics (per variable):")
            parts.append(stats_text)
            context = "\n".join(parts)
            est = _estimate_tokens(context)
        else:
            # nothing to truncate -> break
            break

    # Final trim: if still huge, shorten the user query and metadata sections
    if _estimate_tokens(context) > token_limit:
        # last resort: return a compact context with only summary stats + top metadata ids
        compact = [
            "=== COMPACT DATA CONTEXT ===",
            "SYSTEM_INSTRUCTION:",
            instruction,
            "User query:",
            shorten(query.strip(), width=200, placeholder="..."),
            f"Chroma candidates: {n_meta}",
            f"Numeric rows: {n_numeric}",
            "Top metadata doc ids and scores:",
        ]
        for r in retrieved_meta[:5]:
            compact.append(f"  id={r.get('id')} score={r.get('score')}")
        compact.append("Aggregated stats (abbreviated):")
        compact.append(stats_text if stats_text else "(no stats)")
        compact.append("\nPROVENANCE: see pipeline_audits for executed SQL.")
        context = "\n".join(compact)

    return context


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

import re, json

# ------------ Extraction + Validation helpers -------------------

def extract_structured_json(answer_text: str):
    """
    Try to extract structured JSON from answer_text.
    1) Prefer marker extraction between ===STRUCTURED_JSON_START=== and ===STRUCTURED_JSON_END===
    2) Next, try ```json ... ``` block
    3) Fallback: attempt to find top-level JSON object (first { ... last }).
    Returns (payload_dict_or_None, error_str_or_None)
    """
    if not isinstance(answer_text, str):
        return None, "answer_not_string"

    # 1) marker extraction
    start_marker = "===STRUCTURED_JSON_START==="
    end_marker = "===STRUCTURED_JSON_END==="
    if start_marker in answer_text and end_marker in answer_text:
        try:
            inner = answer_text.split(start_marker, 1)[1].split(end_marker, 1)[0].strip()
            payload = json.loads(inner)
            return payload, None
        except Exception as e:
            return None, f"marker_json_parse_error: {e}"

    # 2) fenced code block (```json ... ```)
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", answer_text, flags=re.S | re.I)
    if m:
        txt = m.group(1)
        try:
            payload = json.loads(txt)
            return payload, None
        except Exception as e:
            return None, f"fenced_json_parse_error: {e}"

    # 3) best-effort first {...} ... } match
    try:
        first = answer_text.find("{")
        last = answer_text.rfind("}")
        if first != -1 and last != -1 and last > first:
            snippet = answer_text[first:last+1]
            payload = json.loads(snippet)
            return payload, None
    except Exception as e:
        return None, f"fallback_json_parse_error: {e}"

    return None, "no_json_found"

def validate_structured_payload(payload: dict):
    """
    Lightweight schema validator for types supported by frontend.
    Returns (True, None) or (False, error_string)
    """
    if not isinstance(payload, dict):
        return False, "not_object"

    t = payload.get("type")
    if t == "table":
        if not isinstance(payload.get("headers"), list):
            return False, "table_missing_headers"
        if not isinstance(payload.get("rows"), list):
            return False, "table_missing_rows"
    elif t == "chart":
        if payload.get("chartType") not in ("line","bar","area"):
            return False, "chart_invalid_chartType"
        if not payload.get("xKey") or not payload.get("yKey"):
            return False, "chart_missing_keys"
        if not isinstance(payload.get("data"), list):
            return False, "chart_missing_data"
    elif t == "map":
        if not isinstance(payload.get("points"), list):
            return False, "map_missing_points"
    elif t == "text":
        if "content" not in payload:
            return False, "text_missing_content"
    elif t == "multi":
        if not isinstance(payload.get("visuals"), list):
            return False, "multi_missing_visuals"
    else:
        return False, f"unknown_type:{t}"

    return True, None

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
    Orchestrate one end-to-end RAG run:
      1. initialize services (Chroma, embedder, DuckDB)
      2. semantic retrieval
      3. build platform/profile candidates
      4. fetch numeric rows & metadata from DuckDB
      5. assemble LLM context
      6. call LLM (with retries)
      7. write audit + optional outputs, return structured dict

    Returns a dict with keys:
      - status: 'ok'|'error'
      - query, retrieved_count, candidates_count, numeric_rows, metadata_rows
      - answer, llm_raw
      - audit_path, output_json (if written)
      - error (if any)
    """
    import time
    import uuid
    import json
    import traceback
    from pathlib import Path
    from datetime import datetime, timezone
    import logging

    logger = logging.getLogger(__name__)
    start_time = time.time()
    audit = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "query": query,
        "year": year,
        "months": months,
        "date_from": date_from,
        "date_to": date_to,
        "top_k": top_k,
        "steps": {},
        "errors": None,
    }

    # Ensure env / keys present (assert_env should raise if critical env missing)
    try:
        if "assert_env" in globals():
            assert_env()
    except Exception as e:
        audit["errors"] = {"stage": "assert_env", "error": str(e)}
        # early return with structured error
        return {"status": "error", "error": str(e), "audit": audit}

    # create audit folder
    audits_dir = Path("pipeline_audits")
    audits_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audits_dir / f"audit_{audit['id']}.json"

    # Step 1: init services
    try:
        t0 = time.time()
        client, coll = init_chroma(CHROMA_DIR)
        embed_model = init_embed_model()
        con = open_duckdb(DUCKDB_PATH)
        audit["steps"]["init"] = {"duration_s": time.time() - t0}
    except Exception as e:
        audit["errors"] = {"stage": "init_services", "error": str(e), "trace": traceback.format_exc()}
        audit_path.write_text(json.dumps(audit, default=str, indent=2))
        return {"status": "error", "error": "Failed to initialize services: " + str(e), "audit": audit}

    try:
        # Step 2: semantic retrieval (allow a light metadata_filter detection for explicit platform queries)
        t0 = time.time()
        metadata_filter = None
        # simple heuristic: detect "platform 123456" in query and set metadata_filter
        import re
        m = re.search(r'\bplatform\s*(number\s*)?[:=]?\s*(\d{3,10})\b', query, flags=re.IGNORECASE)
        if m:
            try:
                metadata_filter = {"PLATFORM_NUMBER": int(m.group(2))}
                logger.debug("Detected explicit platform filter from query: %s", metadata_filter)
            except Exception:
                metadata_filter = None

        retrieved = semantic_retrieve(coll, embed_model, query, top_k=top_k, metadata_filter=metadata_filter)
        audit["steps"]["semantic_retrieve"] = {"duration_s": time.time() - t0, "returned": len(retrieved)}
    except Exception as e:
        audit["errors"] = {"stage": "semantic_retrieve", "error": str(e), "trace": traceback.format_exc()}
        audit_path.write_text(json.dumps(audit, default=str, indent=2))
        con.close()
        return {"status": "error", "error": "Semantic retrieval failed: " + str(e), "audit": audit}

    # Step 3: build platform/profile candidates
    try:
        t0 = time.time()
        candidates = build_platform_profile_set(retrieved)
        audit["steps"]["build_candidates"] = {"duration_s": time.time() - t0, "candidates_count": len(candidates)}
    except Exception as e:
        audit["errors"] = {"stage": "build_platform_profile_set", "error": str(e), "trace": traceback.format_exc()}
        audit_path.write_text(json.dumps(audit, default=str, indent=2))
        con.close()
        return {"status": "error", "error": "Candidate extraction failed: " + str(e), "audit": audit}

    # If no candidates, fall back to metadata-only best-effort answer
    if not candidates:
        try:
            # assemble a minimal metadata-only context
            metadata_text = ""
            for r in retrieved[:10]:
                md = r.get("metadata", {}) or {}
                docid = r.get("id")
                score = r.get("score")
                metadata_text += f"DOC_ID={docid} SCORE={score} METADATA={json.dumps(md, default=str)}\n"

            system_prompt = (
                "You are an expert oceanography assistant. Use *only* the context provided below. "
                "If the answer cannot be determined from the context, return EXACTLY: INSUFFICIENT_CONTEXT"
            )
            user_prompt = f"User question:\n{query}\n\nContext documents:\n{metadata_text}"

            # try LLM once for best-effort (no heavy retries here)
            answer_text, raw = call_azure_openai_chat(system_prompt, user_prompt)
            out = {
                "status": "ok",
                "query": query,
                "retrieved_count": len(retrieved),
                "candidates_count": 0,
                "numeric_rows": 0,
                "metadata_rows": 0,
                "answer": answer_text,
                "llm_raw": raw,
            }
            # write audit
            audit["result_summary"] = {"retrieved": len(retrieved), "candidates": 0, "numeric_rows": 0}
            audit["llm_call"] = {"note": "metadata-only fallback"}
            audit_path.write_text(json.dumps(audit, default=str, indent=2))
            con.close()
            # also write output_json if requested
            if output_json:
                output_json.parent.mkdir(parents=True, exist_ok=True)
                output_json.write_text(json.dumps(out, default=str, indent=2))
            return out
        except Exception as e:
            audit["errors"] = {"stage": "fallback_llm", "error": str(e), "trace": traceback.format_exc()}
            audit_path.write_text(json.dumps(audit, default=str, indent=2))
            con.close()
            return {"status": "error", "error": "Fallback LLM failed: " + str(e), "audit": audit}

    # Step 4: fetch numeric + metadata rows from DuckDB
    try:
        t0 = time.time()
        date_range = None
        if date_from and date_to:
            date_range = [date_from, date_to]

        # Limit_per_profile chosen conservatively; increase via param if needed
        df_numeric = fetch_measurements_for_candidates(con, candidates, months=months, date_range=date_range, limit_per_profile=500)
        t1 = time.time()
        df_metadata = fetch_metadata_for_candidates(con, candidates, year)
        t2 = time.time()
        audit["steps"]["fetch_duckdb"] = {
            "duration_numeric_s": t1 - t0,
            "numeric_rows": int(len(df_numeric)) if hasattr(df_numeric, "__len__") else 0,
            "duration_metadata_s": t2 - t1,
            "metadata_rows": int(len(df_metadata)) if hasattr(df_metadata, "__len__") else 0
        }
    except Exception as e:
        audit["errors"] = {"stage": "fetch_from_duckdb", "error": str(e), "trace": traceback.format_exc()}
        audit_path.write_text(json.dumps(audit, default=str, indent=2))
        con.close()
        return {"status": "error", "error": "DuckDB fetch failed: " + str(e), "audit": audit}

    # Step 5: assemble context for LLM (respect token budgets)
    try:
        t0 = time.time()
        # compute context size target: if many numeric rows, reduce max_rows for assemble_context
        numeric_count = int(len(df_numeric))
        if numeric_count <= 50:
            ctx_max_rows = 50
        elif numeric_count <= 1000:
            ctx_max_rows = 200
        elif numeric_count <= 10000:
            ctx_max_rows = 500
        else:
            ctx_max_rows = 200  # keep context manageable for very large fetches

        context = assemble_context(retrieved, df_numeric, df_metadata, query, max_rows=ctx_max_rows)
        audit["steps"]["assemble_context"] = {"duration_s": time.time() - t0, "ctx_max_rows": ctx_max_rows, "context_chars": len(context)}
    except Exception as e:
        audit["errors"] = {"stage": "assemble_context", "error": str(e), "trace": traceback.format_exc()}
        audit_path.write_text(json.dumps(audit, default=str, indent=2))
        con.close()
        return {"status": "error", "error": "Context assembly failed: " + str(e), "audit": audit}

    # Save small numeric sample if not empty (this path will be referenced in visuals)
    sample_csv_path = None
    if not df_numeric.empty:
        sample_csv_path = str(output_json.with_suffix(".sample.csv")) if output_json else None
        if sample_csv_path:
            df_numeric.head(2000).to_csv(sample_csv_path, index=False)

    # Build deterministic canvas payloads (no LLM)
    try:
        canvas = decide_and_build(query, df_numeric, df_metadata, sample_csv=sample_csv_path)
    except Exception as e:
        # Canvas builder should never crash the pipeline — swallow errors and continue
        canvas = {"visuals": [], "rationale": f"canvas_builder_error: {str(e)}"}

    # Step 6: prepare system/user prompts and call LLM with retry/backoff
    system_prompt = (
        "You are an expert oceanography assistant with access ONLY to the provided context. Strict rules:\n"
        "1) Use only the data in the Context section below. Do NOT invent facts.\n"
        "2) If requested data is not present, respond EXACTLY: INSUFFICIENT_CONTEXT.\n"
        "3) Cite provenance for numeric claims using [SQL:<note>] or [DOC:<id>].\n"
        "4) Produce TWO parts in your response:\n"
        "   A) A short human-friendly answer (3 sentences max) with citations and recommended visualizations.\n"
        "   B) A machine-readable JSON block named 'structured' enclosed between the markers\n"
        "      ===STRUCTURED_JSON_START=== and ===STRUCTURED_JSON_END===\n"
        "      The JSON must be valid UTF-8 JSON (no trailing text inside the markers) and follow one of these shapes:\n"
        "        TABLE:  {\"type\":\"table\",\"title\":str,\"headers\":[...],\"rows\":[[...],...]}\n"
        "        CHART:  {\"type\":\"chart\",\"title\":str,\"chartType\":\"line|bar|area\",\"xKey\":str,\"yKey\":str,\"data\":[{x:...,y:...},...]}\n"
        "        MAP:    {\"type\":\"map\",\"title\":str,\"points\":[{\"lat\":num,\"lon\":num,\"label\":str},...]}\n"
        "        TEXT:   {\"type\":\"text\",\"content\":str}\n"
        "5) If multiple visuals are useful, 'structured' may have {\"type\":\"multi\",\"visuals\":[...]}.\n"
        "6) IMPORTANT: Put ONLY the valid JSON between the markers. The JSON block will be parsed programmatically.\n"
        "7) If you cannot produce structured JSON with confidence, still give the human answer and set structured to {\"type\":\"text\",\"content\":\"INSUFFICIENT_CONTEXT\"} between the markers.\n"
    )


    user_prompt = f"Query: {query}\n\nContext:\n{context}\n\nAnswer concisely."

    # perform the LLM call with simple retry/backoff
    answer_text = None
    raw = None
    llm_error = None
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            t0 = time.time()
            answer_text, raw = call_azure_openai_chat(system_prompt, user_prompt, max_tokens=1200, temperature=0.0)
            audit["steps"].setdefault("llm_calls", []).append({"attempt": attempt, "duration_s": time.time() - t0})
            break
        except Exception as e:
            llm_error = str(e)
            audit["steps"].setdefault("llm_calls", []).append({"attempt": attempt, "error": str(e)})
            backoff = 1.5 ** attempt
            time.sleep(backoff)
    if answer_text is None:
        audit["errors"] = {"stage": "llm_call", "error": llm_error, "trace": traceback.format_exc()}
        # still return a structured error (no exception)
        out = {"status": "error", "error": "LLM call failed: " + str(llm_error), "audit": audit}
        audit_path.write_text(json.dumps(audit, default=str, indent=2))
        con.close()
        return out
    
    # After LLM call:
    structured_payload, struct_err = extract_structured_json(answer_text)
    structured_valid = False
    structured_validation_error = None
    if structured_payload is not None:
        ok, v_err = validate_structured_payload(structured_payload)
        if ok:
            structured_valid = True
        else:
            structured_validation_error = v_err
    else:
        structured_validation_error = struct_err

    # Choose final 'structured' to include in out
    if structured_valid:
        final_structured = structured_payload
    else:
        # Fallback deterministic canvas you already built earlier
        final_structured = canvas if canvas is not None else {"type":"text","content":"INSUFFICIENT_CONTEXT"}


    # Step 7: finalize output and write files
    try:
        out = {
            "status": "ok",
            "query": query,
            "retrieved_count": len(retrieved),
            "candidates_count": len(candidates),
            "numeric_rows": int(len(df_numeric)),
            "metadata_rows": int(len(df_metadata)),
            "answer": answer_text,
            "llm_raw": raw,
            "canvas": canvas
        }
        out["structured"] = final_structured
        out["structured_valid"] = bool(structured_valid)
        if structured_valid is False:
            out["structured_error"] = structured_validation_error
            out["structured_raw_attempt"] = (structured_payload or None)

        # write optional numeric sample CSV and output JSON
        if output_json:
            output_json = Path(output_json)
            output_json.parent.mkdir(parents=True, exist_ok=True)
            if not df_numeric.empty:
                sample_csv_path = output_json.with_suffix(".sample.csv")
                # write top 200 rows safely
                df_numeric.head(200).to_csv(sample_csv_path, index=False)
                out["numeric_sample_csv"] = str(sample_csv_path)
            output_json.write_text(json.dumps(out, default=str, indent=2))
            out["output_json"] = str(output_json)

        # write audit with timings + small provenance
        audit["result_summary"] = {
            "retrieved_count": len(retrieved),
            "candidates_count": len(candidates),
            "numeric_rows": int(len(df_numeric)),
            "metadata_rows": int(len(df_metadata)),
            "answer_present": True
        }
        audit["runtime_s"] = time.time() - start_time
        audit["created_output_json"] = str(output_json) if output_json else None
        audit_path.write_text(json.dumps(audit, default=str, indent=2))
        out["audit_path"] = str(audit_path)
    except Exception as e:
        audit["errors"] = {"stage": "final_write", "error": str(e), "trace": traceback.format_exc()}
        audit_path.write_text(json.dumps(audit, default=str, indent=2))
        con.close()
        return {"status": "error", "error": "Final output write failed: " + str(e), "audit": audit}

    # close DB connection gracefully
    try:
        con.close()
    except Exception:
        pass

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