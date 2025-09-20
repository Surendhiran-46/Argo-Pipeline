# pipeline/canvas_builder.py
"""
Deterministic Canvas payload builder for Floatchat.
Produces JSON payloads the frontend can render:
 - type: "table"
 - type: "chart" with chartType in {"line","bar","area"}
 - type: "map" (point list)
This module intentionally avoids any LLM content generation.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import re
import math
import json

# Configuration - tune as needed
MAX_TABLE_ROWS = 200            # max rows to embed directly in JSON
MAX_CHART_POINTS = 2000         # max points for a chart (frontend will downsample)
SAMPLE_CSV_SUFFIX = ".sample.csv"  # if you produce a sample csv during run_rag_query

# Heuristics (simple, explicit)
_TIME_KEYWORDS = {"time", "trend", "over time", "change", "temporal", "date", "juld", "time series"}
_PROFILE_KEYWORDS = {"profile", "depth", "vertical", "pres", "pressure"}
_AGG_KEYWORDS = {"average","mean","min","max","median","count","how many","number of"}
_MAP_KEYWORDS = {"near", "nearest", "map", "location", "lat", "lon", "latitude", "longitude"}

# Helpers
def _query_tokens(q: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", (q or "").lower())

def _contains_any(q: str, words):
    qlow = q.lower() if q else ""
    return any(w in qlow for w in words)

def _safe_head(df: pd.DataFrame, n: int = MAX_TABLE_ROWS):
    if df is None or df.empty:
        return df.head(0)
    return df.head(n)


def build_table_payload(title: str, df: pd.DataFrame, sample_csv_path: Optional[str]=None, max_rows:int=MAX_TABLE_ROWS) -> Dict[str,Any]:
    """Return a 'table' JSON payload (headers + rows)."""
    df2 = _safe_head(df, max_rows)
    headers = list(df2.columns.astype(str))
    # convert values to simple types (str/number)
    rows = []
    for _, r in df2.iterrows():
        row = []
        for c in headers:
            v = r.get(c, None)
            if pd.isna(v):
                row.append(None)
            else:
                # convert timestamps to iso strings
                if isinstance(v, (pd.Timestamp,)):
                    row.append(v.isoformat())
                elif isinstance(v, (np.integer, np.floating, int, float)):
                    # force python scalar
                    if isinstance(v, (np.integer, np.floating)):
                        v = v.item()
                    row.append(v)
                else:
                    row.append(str(v))
        rows.append(row)

    payload = {
        "type": "table",
        "title": title,
        "headers": headers,
        "rows": rows,
        "row_count": len(df),
        "sample_csv": sample_csv_path or None,
    }
    return payload

def _downsample_df_for_chart(df: pd.DataFrame, max_points: int = MAX_CHART_POINTS) -> pd.DataFrame:
    """Downsample by uniform sampling across index to limit points"""
    if df is None or df.empty:
        return df
    n = len(df)
    if n <= max_points:
        return df
    idx = np.linspace(0, n-1, max_points, dtype=int)
    return df.iloc[idx]

def build_chart_payload(title: str, df: pd.DataFrame, x: str, y: str,
                        chart_type: str = "line", color: Optional[str] = None,
                        sample_csv_path: Optional[str] = None) -> Dict[str,Any]:
    """
    Build a chart payload. df should include x and y columns.
    chart_type: 'line'|'bar'|'area'
    """
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        return {"type":"visualization_unavailable", "reason": f"Missing columns for chart: need {x} and {y}."}

    # Downsample if needed
    df2 = _downsample_df_for_chart(df[[x,y] + ([color] if color and color in df.columns else [])].copy(), max_points=MAX_CHART_POINTS)
    # convert timestamps to iso strings for JSON friendliness
    data = []
    for _, r in df2.iterrows():
        xv = r[x]
        if isinstance(xv, pd.Timestamp):
            xv = xv.isoformat()
        yv = r[y]
        if isinstance(yv, pd.Timestamp):
            yv = yv.isoformat()
        rec = {x: xv, y: (None if pd.isna(yv) else (yv.item() if isinstance(yv, (np.integer,np.floating)) else yv))}
        if color and color in df2.columns:
            cv = r[color]
            rec[color] = None if pd.isna(cv) else (cv.item() if isinstance(cv,(np.integer,np.floating)) else str(cv))
        data.append(rec)

    payload = {
        "type":"chart",
        "title": title,
        "chartType": chart_type,
        "xKey": x,
        "yKey": y,
        "data": data,
        "row_count": len(df),
        "sample_csv": sample_csv_path or None
    }
    if color and color in df.columns:
        payload["color"] = color
    return payload

def build_map_payload(title: str, df: pd.DataFrame, lat_col="LATITUDE", lon_col="LONGITUDE", label_col:Optional[str]=None, sample_csv_path:Optional[str]=None):
    """Return a map points payload (list of points)."""
    if df is None or df.empty or lat_col not in df.columns or lon_col not in df.columns:
        return {"type":"visualization_unavailable", "reason": "Missing lat/lon columns."}
    df2 = _safe_head(df[[lat_col,lon_col] + ([label_col] if label_col and label_col in df.columns else [])], MAX_TABLE_ROWS)
    points = []
    for _, r in df2.iterrows():
        lat = r[lat_col]
        lon = r[lon_col]
        if pd.isna(lat) or pd.isna(lon):
            continue
        pt = {"lat": float(lat), "lon": float(lon)}
        if label_col and label_col in df2.columns:
            lab = r[label_col]
            pt["label"] = None if pd.isna(lab) else str(lab)
        points.append(pt)
    return {"type":"map", "title":title, "points": points, "row_count": len(df), "sample_csv": sample_csv_path or None}

def decide_and_build(query: str,
                     df_numeric: pd.DataFrame,
                     df_metadata: pd.DataFrame,
                     sample_csv: Optional[str] = None) -> Dict[str,Any]:
    """
    Decide the best canvas payload(s) for the query and available data.
    Returns a dict with keys:
      - visuals: list[ payloads ]
      - rationale: string explanation (useful for frontend debug)
    """
    visuals = []
    rationale = []

    q = (query or "").lower()

    # If user asked for maps / "nearest", prefer map
    if _contains_any(q, _MAP_KEYWORDS) and (df_metadata is not None and not df_metadata.empty):
        rationale.append("Detected geographic intent -> build map from time_location/metadata.")
        visuals.append(build_map_payload("Float locations (sample)", df_metadata, lat_col="LATITUDE", lon_col="LONGITUDE", label_col="PLATFORM_NUMBER", sample_csv_path=sample_csv))
        return {"visuals": visuals, "rationale": "; ".join(rationale)}

    # If numeric time-series intent and juld present -> try timeseries (x=juld, y=some numeric)
    if _contains_any(q, _TIME_KEYWORDS) and df_numeric is not None and not df_numeric.empty and "juld" in [c.lower() for c in df_numeric.columns]:
        # choose a sensible y: prefer TEMP/PSAL/PRES (if named exactly)
        cols = [c for c in df_numeric.columns if c.upper() in ("TEMP","PSAL","PRES","PSAL_ADJUSTED","TEMP_ADJUSTED")]
        if cols:
            ycol = cols[0]
            # convert df_numeric column names to lower mapping
            colmap = {c.lower(): c for c in df_numeric.columns}
            xcol = colmap.get("juld","juld")
            rationale.append(f"Detected time-series intent; using {ycol} vs {xcol}.")
            visuals.append(build_chart_payload(f"{ycol} over time", df_numeric.sort_values(by=xcol), x=xcol, y=ycol, chart_type="line", sample_csv_path=sample_csv))
            return {"visuals": visuals, "rationale": "; ".join(rationale)}

    # If profile / depth intent -> PRES vs TEMP (profile plot)
    if _contains_any(q, _PROFILE_KEYWORDS) and df_numeric is not None and not df_numeric.empty and "PRES" in [c.upper() for c in df_numeric.columns]:
        # common profile: PRES (x=pressure) vs TEMP (y=temp) or PSAL
        # use column case-insensitive
        colmap = {c.upper(): c for c in df_numeric.columns}
        pres_col = colmap.get("PRES")
        temp_col = colmap.get("TEMP") or colmap.get("TEMP_ADJUSTED")
        psal_col = colmap.get("PSAL") or colmap.get("PSAL_ADJUSTED")
        # if we have temp, make vertical line chart (PRES is depth; frontend may invert axis)
        if pres_col and temp_col:
            # sample profile: choose latest profile_index if available
            visuals.append(build_chart_payload(f"Temperature profile (sample)", df_numeric.sort_values(by=pres_col), x=pres_col, y=temp_col, chart_type="line", sample_csv_path=sample_csv))
            rationale.append("Detected profile intent -> PRES vs TEMP.")
            return {"visuals": visuals, "rationale": "; ".join(rationale)}
        if pres_col and psal_col:
            visuals.append(build_chart_payload(f"Salinity profile (sample)", df_numeric.sort_values(by=pres_col), x=pres_col, y=psal_col, chart_type="line", sample_csv_path=sample_csv))
            rationale.append("Detected profile intent -> PRES vs PSAL.")
            return {"visuals": visuals, "rationale": "; ".join(rationale)}

    # If aggregation intent (average/min/max) -> produce a small bar/histogram or summary table
    if _contains_any(q, _AGG_KEYWORDS) and df_numeric is not None and not df_numeric.empty:
        # pick first numeric col that is meaningful
        numeric_cols = [c for c in df_numeric.columns if pd.api.types.is_numeric_dtype(df_numeric[c])]
        if numeric_cols:
            # compute aggregates
            col = numeric_cols[0]
            agg = df_numeric[col].agg(["count","mean","min","max"]).to_dict()
            # small table for aggregates
            agg_df = pd.DataFrame([agg])
            visuals.append(build_table_payload(f"Aggregates for {col}", agg_df))
            rationale.append(f"Detected aggregation intent -> aggregates for {col}")
            return {"visuals": visuals, "rationale": "; ".join(rationale)}

    # Otherwise fallback: if metadata present show metadata table, else numeric sample table
    if df_metadata is not None and not df_metadata.empty:
        rationale.append("Fallback -> show metadata sample table.")
        visuals.append(build_table_payload("Metadata sample", df_metadata, sample_csv_path=sample_csv))
        return {"visuals": visuals, "rationale": "; ".join(rationale)}
    if df_numeric is not None and not df_numeric.empty:
        rationale.append("Fallback -> show numeric sample table.")
        visuals.append(build_table_payload("Numeric sample", df_numeric, sample_csv_path=sample_csv))
        return {"visuals": visuals, "rationale": "; ".join(rationale)}

    return {"visuals": [], "rationale": "No suitable data for visualization."}
