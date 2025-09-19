# pipeline/extract_history.py
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
from datetime import timezone
from utils import ensure_dir, atomic_write_df, elem_to_string, safe_float, parse_reference_datetime

def parse_history_date(elem):
    """Parse HISTORY_DATE-like values into pandas.Timestamp (UTC) or NaT."""
    s = elem_to_string(elem)
    if not s:
        return pd.NaT
    s = s.strip()
    if len(s) in (14, 12, 8) and s.isdigit():
        for fmt in ("%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y%m%d"):
            try:
                return pd.Timestamp.strptime(s, fmt).tz_localize(timezone.utc)
            except Exception:
                pass
    t = pd.to_datetime(s, errors="coerce")
    if pd.isna(t):
        return pd.NaT
    return t.tz_localize(timezone.utc) if t.tzinfo is None else t.tz_convert(timezone.utc)

def extract_history(nc_path: Path, out_dir: Path):
    out_dir = ensure_dir(out_dir)
    out_file = out_dir / "argo_history.csv"

    with xr.open_dataset(nc_path, decode_times=False) as ds:
        n_prof = int(ds.sizes.get("N_PROF", 0))
        n_history = int(ds.sizes.get("N_HISTORY", 0)) if "N_HISTORY" in ds.dims else 0

        wanted_fields = [
            "HISTORY_DATE","HISTORY_ACTION","HISTORY_INSTITUTION","HISTORY_SOFTWARE",
            "HISTORY_SOFTWARE_RELEASE","HISTORY_REFERENCE","HISTORY_PARAMETER",
            "HISTORY_QCTEST","HISTORY_PREVIOUS_VALUE","HISTORY_START_PRES",
            "HISTORY_STOP_PRES","HISTORY_STEP"
        ]

        headers = ["profile_index","history_index","platform_number"] + wanted_fields

        if n_history == 0:
            df_empty = pd.DataFrame(columns=headers)
            atomic_write_df(df_empty, out_file)
            return out_file

        # PLATFORM_NUMBER for context
        platforms = []
        if "PLATFORM_NUMBER" in ds.variables:
            arr = ds["PLATFORM_NUMBER"].values
            for i in range(n_prof):
                platforms.append(elem_to_string(arr[i]))
        else:
            platforms = ["" for _ in range(n_prof)]

        rows = []
        for h in range(n_history):
            for p in range(n_prof):
                row = {"profile_index": p, "history_index": h, "platform_number": platforms[p]}
                for fld in wanted_fields:
                    if fld not in ds.variables:
                        row[fld] = ""
                        continue
                    arr = np.asarray(ds[fld].values)
                    try:
                        if arr.ndim == 3:
                            elem = arr[h, p, :]
                            row[fld] = parse_history_date(elem) if fld == "HISTORY_DATE" else elem_to_string(elem)
                        elif arr.ndim == 2:
                            elem = arr[h, p]
                            if fld == "HISTORY_DATE":
                                row[fld] = parse_history_date(elem)
                            elif fld == "HISTORY_PREVIOUS_VALUE" or arr.dtype.kind in ("f","i","u"):
                                row[fld] = safe_float(elem)
                            else:
                                row[fld] = elem_to_string(elem)
                        elif arr.ndim == 1:
                            if arr.shape[0] == n_prof:
                                elem = arr[p]
                            elif arr.shape[0] == n_history:
                                elem = arr[h]
                            else:
                                elem = arr.ravel()[0]
                            row[fld] = parse_history_date(elem) if fld == "HISTORY_DATE" else elem_to_string(elem)
                        else:
                            elem = arr.item() if arr.size == 1 else arr.ravel()[0]
                            row[fld] = parse_history_date(elem) if fld == "HISTORY_DATE" else elem_to_string(elem)
                    except Exception:
                        row[fld] = elem_to_string(arr)
                rows.append(row)

        df_hist = pd.DataFrame(rows)

        # Normalize HISTORY_DATE column
        if "HISTORY_DATE" in df_hist.columns:
            df_hist["HISTORY_DATE"] = pd.to_datetime(df_hist["HISTORY_DATE"], errors="coerce")
            df_hist["HISTORY_DATE"] = df_hist["HISTORY_DATE"].apply(
                lambda t: pd.NaT if pd.isna(t) else (t.tz_localize(timezone.utc) if t.tzinfo is None else t.tz_convert(timezone.utc))
            )

        atomic_write_df(df_hist, out_file)
        return out_file
