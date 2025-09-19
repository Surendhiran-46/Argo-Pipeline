# pipeline/query_validator.py
"""
Query validator / intent classifier for FloatChat RAG pipeline.
Rule-based, deterministic heuristics to map a user query into:
 - intent: one of ['measurement', 'compare', 'nearest', 'float_detail', 'project_info', 'qc_query', 'history', 'unknown']
 - required_filters: variables (TEMP/PSAL/PRES), bbox/time/etc.
 - missing_fields: which fields are missing that are required for this intent.
"""

import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

VAR_KEYWORDS = {
    "TEMP": ["temp", "temperature", "°c", "deg c", "°C"],
    "PSAL": ["salinity", "psal", "psu"],
    "PRES": ["pressure", "pres", "depth", "db", "dbar"],
    "DOXY": ["doxy", "oxygen", "o2"],
    "CHLA": ["chl", "chla", "chlorophyll"],
    "NITRATE": ["nitrate", "no3"],
}

INTENT_PATTERNS = {
    "compare": [
        r"\bcompare\b",
        r"\bvs\b",
        r"\bversus\b",
        r"\bcompare.*(trend|profile|profiles|salinity|temperature|nitrate)\b",
    ],
    "measurement": [
        r"\bshow me\b",
        r"\bplot\b",
        r"\bdisplay\b",
        r"\bprofile\b",
        r"\btime series\b",
    ],
    "nearest": [r"\bnearest\b", r"\bclosest\b", r"\bnear this location\b"],
    "float_detail": [r"\bfloat\b", r"\bserial\b", r"\bplatform\b", r"\bfloat number\b"],
    "project_info": [r"\bproject\b", r"\bPI\b", r"\bprincipal investigator\b", r"\bresearcher\b"],
    "qc_query": [r"\bQC\b", r"\bquality flag\b", r"\bquality flags\b", r"\bprofile quality\b"],
    "history": [r"\bhistory\b", r"\bprocessed\b", r"\badjusted\b", r"\bHISTORY\b"],
}

@dataclass
class QueryIntent:
    intent: str
    variables: List[str]
    bbox: Optional[Dict[str,float]]     # {lat_min, lat_max, lon_min, lon_max}
    date_from: Optional[str]           # ISO
    date_to: Optional[str]
    platform_number: Optional[int]
    profile_index: Optional[int]
    text: str
    missing: List[str]

def _find_vars(text: str) -> List[str]:
    found = set()
    t = text.lower()
    for k,vlist in VAR_KEYWORDS.items():
        for token in vlist:
            if token in t:
                found.add(k)
                break
    return sorted(found)

def _match_intent(text: str) -> str:
    t = text.lower()
    for intent, patterns in INTENT_PATTERNS.items():
        for p in patterns:
            if re.search(p, t):
                return intent
    # fallback heuristics
    if "where" in t or "lat" in t or "lon" in t or "nearest" in t:
        return "nearest"
    if any(k in t for k in ["temp", "salin", "pres", "profile"]):
        return "measurement"
    return "unknown"

def _extract_bbox(text: str):
    # Look for "lat X to Y lon A to B" or "near lat,lon +/- radius" patterns
    # Very conservative rule-based extraction; return None if ambiguous
    # Try "lat X,Y lon U,V" or "lat X to Y lon U to V"
    t = text
    # format: "lat -10 to 10 lon 70 to 80"
    m = re.search(r"lat(?:itude)?\s*([-\d\.]+)\s*(?:to|-)\s*([-\d\.]+).*?lon(?:gitude)?\s*([-\d\.]+)\s*(?:to|-)\s*([-\d\.]+)", t, flags=re.I)
    if m:
        lat1, lat2, lon1, lon2 = map(float, m.groups())
        return {"lat_min": min(lat1,lat2), "lat_max": max(lat1,lat2), "lon_min": min(lon1,lon2), "lon_max": max(lon1,lon2)}
    # simple "near X,Y" pattern
    m2 = re.search(r"near\s*([-\d\.]+)\s*,\s*([-\d\.]+)", t, flags=re.I)
    if m2:
        lat, lon = map(float, m2.groups())
        # return a small bbox of +/- 1 degree by default
        return {"lat_min": lat-1.0, "lat_max": lat+1.0, "lon_min": lon-1.0, "lon_max": lon+1.0}
    return None

def _extract_date_range(text: str):
    # Very simple ISO date or "last 6 months" detection
    m = re.search(r"(\d{4}-\d{2}-\d{2})\s*(?:to|-)\s*(\d{4}-\d{2}-\d{2})", text)
    if m:
        return m.group(1), m.group(2)
    # "March 2025" -> convert to 2025-03-01..2025-03-31 (best-effort)
    m2 = re.search(r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})", text, flags=re.I)
    if m2:
        month_name, year = m2.groups()
        import calendar
        mn = list(calendar.month_name).index(month_name.capitalize())
        from datetime import datetime, timezone
        datetime.now(timezone.utc)
        start = f"{year}-{mn:02d}-01"
        # end = last day
        import calendar as _cal
        last = _cal.monthrange(int(year), mn)[1]
        end = f"{year}-{mn:02d}-{last:02d}"
        return start, end
    # "last 6 months"
    m3 = re.search(r"last\s+(\d+)\s+months?", text, flags=re.I)
    if m3:
        n = int(m3.group(1))
        from datetime import datetime, timedelta, timezone
        datetime.now(timezone.utc)
        end = datetime.utcnow().date()
        import dateutil.relativedelta as rd
        start = (end - rd.relativedelta(months=n)).isoformat()
        return start, end.isoformat()
    return None, None

def _extract_platform_profile(text: str):
    # Look for platform numbers (large integers) or pattern "platform 5906527" or "float 5906527"
    m = re.search(r"(?:platform|float|wmo)\s*(?:number)?\s*[:#]?\s*(\d{4,8})", text, flags=re.I)
    if m:
        try:
            return int(m.group(1)), None
        except:
            return None, None
    # profile index pattern "profile 3"
    m2 = re.search(r"profile\s*(\d+)", text, flags=re.I)
    if m2:
        try:
            return None, int(m2.group(1))
        except:
            pass
    return None, None

def analyze_query(text: str) -> Dict[str, Any]:
    """
    Returns a dictionary with:
    - intent
    - variables (list)
    - bbox (if any)
    - date_from/date_to
    - platform_number/profile_index
    - missing (list of missing required fields)
    """
    intent = _match_intent(text)
    variables = _find_vars(text)
    bbox = _extract_bbox(text)
    date_from, date_to = _extract_date_range(text)
    platform_number, profile_index = _extract_platform_profile(text)

    # Define required fields per intent
    required = []
    if intent == "measurement":
        # need at least one variable and a timeframe or location
        required = []
        if not variables:
            required.append("variable (e.g., TEMP or PSAL)")
        if not (bbox or date_from or platform_number):
            required.append("spatial or temporal constraint (lat/lon or date range or platform)")
    elif intent == "compare":
        if not variables:
            required.append("variables to compare (e.g., TEMP, PSAL)")
        if not (date_from or bbox or platform_number):
            required.append("time range or region")
    elif intent == "nearest":
        if not bbox and "near" not in text.lower():
            required.append("a location (lat,lon) or bounding box")
    elif intent == "float_detail":
        if not platform_number:
            required.append("platform number or float identifier")
    elif intent == "project_info":
        # OK with no fields; we can search metadata
        required = []
    elif intent == "qc_query":
        if not platform_number and not bbox:
            required.append("platform number or region")
    else:
        # unknown: no strict reqs
        required = []

    missing = required
    return {
        "intent": intent,
        "variables": variables,
        "bbox": bbox,
        "date_from": date_from,
        "date_to": date_to,
        "platform_number": platform_number,
        "profile_index": profile_index,
        "missing": missing,
        "raw": text,
    }

if __name__ == "__main__":
    # quick manual test
    examples = [
        "Show me salinity profiles near the equator in March 2025",
        "Compare temperature vs salinity in the Arabian Sea for the last 6 months",
        "What are the nearest ARGO floats to 10.5, 75.3?",
        "Give float details for platform 5906527",
    ]
    for q in examples:
        print("Q:", q)
        print(analyze_query(q))
        print("----")
