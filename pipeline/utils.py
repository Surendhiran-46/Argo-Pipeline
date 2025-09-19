# utils.py -- shared helpers for extraction pipeline
from datetime import datetime
import os
import numpy as np
import pandas as pd

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def atomic_write_df(df, out_path, **to_csv_kwargs):
    """Write DataFrame atomically (temp -> move). Supports CSV/Parquet by extension."""
    tmp = str(out_path) + ".tmp"
    # choose output method
    if str(out_path).lower().endswith(".parquet"):
        df.to_parquet(tmp, index=False)
    else:
        df.to_csv(tmp, index=to_csv_kwargs.pop("index", False), **to_csv_kwargs)
    os.replace(tmp, out_path)

def parse_reference_datetime(ref):
    if ref is None:
        return None
    if isinstance(ref, (bytes, bytearray)):
        ref = ref.decode("utf-8", "ignore")
    s = str(ref).strip()
    if s == "":
        return None
    for fmt in ("%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y%m%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    # fallback to pandas
    pdt = pd.to_datetime(s, errors="coerce")
    if pd.notnull(pdt):
        return pdt.to_pydatetime()
    return None

def elem_to_string(elem):
    """Robust conversion to readable string."""
    if elem is None:
        return ""
    if isinstance(elem, (bytes, bytearray)):
        return elem.decode("utf-8", "ignore").strip()
    if np.isscalar(elem):
        return str(elem).strip()
    arr = np.asarray(elem)
    if arr.size == 0:
        return ""
    pieces = []
    for x in arr.ravel():
        if isinstance(x, (bytes, bytearray)):
            pieces.append(x.decode("utf-8", "ignore"))
        elif isinstance(x, np.ndarray):
            pieces.append("".join([(b.decode("utf-8","ignore") if isinstance(b,(bytes,bytearray)) else str(b)) for b in x.ravel()]))
        else:
            pieces.append("" if x is None else str(x))
    return "".join(pieces).replace("\x00", "").strip()

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

def elem_to_number(elem):
    """Get a numeric (float) out of various representations."""
    if elem is None:
        return float("nan")
    if np.isscalar(elem):
        try:
            return float(elem)
        except:
            return float("nan")
    arr = np.asarray(elem)
    if arr.size == 0:
        return float("nan")
    if arr.dtype.kind in ("f","i","u"):
        try:
            return float(arr.ravel()[0])
        except:
            return float("nan")
    # try to join char pieces into a number
    s = elem_to_string(arr)
    try:
        return float(s)
    except:
        return float("nan")

def is_metadata_var(varname, ds, n_profs):
    """
    Return True if ds[varname] looks like metadata (one value per profile).
    Criteria: scalar OR first dimension == N_PROF and (1D OR small 2D char array).
    """
    if varname not in ds.variables:
        return False
    arr = ds[varname].values
    # scalar
    try:
        if np.isscalar(arr) or getattr(arr, "shape", ()) == ():
            return True
    except Exception:
        pass
    # must have first dim equal to n_profs
    try:
        shape = np.asarray(arr).shape
        if len(shape) >= 1 and shape[0] == n_profs:
            # 1D metadata ok
            if len(shape) == 1:
                return True
            # 2D: allow if second dim small and dtype string-like
            if len(shape) == 2:
                if shape[1] <= 512 and np.asarray(arr).dtype.kind in ("S","U","O","b"):
                    return True
            # small numeric arrays per-profile acceptable (rare)
            if len(shape) == 2 and shape[1] <= 8 and np.asarray(arr).dtype.kind in ("f","i","u"):
                return True
    except Exception:
        pass
    return False

def decode_juld_variable(var, ds):
    """
    Convert JULD-like variable to pandas.DatetimeIndex.
    Handles numeric 'days since' with units or REFERENCE_DATE_TIME fallback.
    """
    arr = var.values
    # numeric branch
    if np.issubdtype(np.asarray(arr).dtype, np.number):
        units_attr = str(var.attrs.get("units") or "")
        if "since" in units_attr:
            try:
                base_str = units_attr.split("since",1)[1].strip()
                base = pd.to_datetime(base_str, errors="coerce")
                if pd.notnull(base):
                    if "day" in units_attr.lower():
                        return base + pd.to_timedelta(arr, unit="D")
                    if "sec" in units_attr.lower():
                        return base + pd.to_timedelta(arr, unit="s")
            except Exception:
                pass
        # fallback: try reference var or dataset attr
        ref = None
        if "REFERENCE_DATE_TIME" in ds.variables:
            r = ds["REFERENCE_DATE_TIME"].values
            ref = r if np.isscalar(r) else (r.ravel()[0] if r.size>0 else None)
        if ref is None:
            ref = ds.attrs.get("REFERENCE_DATE_TIME") or var.attrs.get("REFERENCE_DATE_TIME")
        ref_dt = parse_reference_datetime(ref) if ref is not None else None
        if ref_dt is not None:
            arrf = np.asarray(arr).astype(float)
            # heuristic: extremely large values may be epoch seconds
            if np.nanmean(np.abs(arrf)) > 1e6:
                return pd.to_datetime(arrf, unit="s", origin="unix", errors="coerce")
            return pd.to_datetime(ref_dt) + pd.to_timedelta(arrf, unit="D")
        # last resort: days since 1950-01-01
        arrf = np.asarray(arr).astype(float)
        return pd.to_datetime("1950-01-01") + pd.to_timedelta(arrf, unit="D")

    # string/char branch - parse per element
    out = []
    for v in arr:
        if isinstance(v, (bytes, bytearray)):
            s = v.decode("utf-8", "ignore").strip()
            out.append(pd.to_datetime(s, errors="coerce"))
            continue
        vv = np.asarray(v)
        if vv.dtype.kind in ("S","U","O"):
            s = "".join([(x.decode("utf-8","ignore") if isinstance(x, (bytes, bytearray)) else str(x)) for x in vv.ravel()])
            out.append(pd.to_datetime(s.replace("\x00","").strip(), errors="coerce"))
        else:
            out.append(pd.to_datetime(str(v), errors="coerce"))
    return pd.to_datetime(out)
