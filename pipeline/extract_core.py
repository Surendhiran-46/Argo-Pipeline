from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd
from utils import ensure_dir, atomic_write_df, safe_float, elem_to_string, decode_juld_variable

def extract_core(nc_path: Path, out_dir: Path):
    out_dir = ensure_dir(out_dir)
    with xr.open_dataset(nc_path, decode_times=False) as ds:
        # determine dims
        if "N_PROF" in ds.dims:
            n_profs = int(ds.sizes["N_PROF"])
        else:
            raise RuntimeError("N_PROF missing.")

        if "N_LEVELS" in ds.dims:
            n_levels = int(ds.sizes["N_LEVELS"])
        else:
            # fallback: find a 2D var with first dim == N_PROF
            n_levels = None
            for v in ("PRES","TEMP","PSAL"):
                if v in ds.variables:
                    arr = np.asarray(ds[v].values)
                    if arr.ndim == 2 and arr.shape[0] == n_profs:
                        n_levels = arr.shape[1]
                        break
            if n_levels is None:
                n_levels = 1

        def get_val(varname, i, k=None):
            if varname not in ds.variables:
                return None
            arr = np.asarray(ds[varname].values)
            try:
                if arr.ndim == 2:
                    return arr[i, k]
                if arr.ndim == 1:
                    return arr[i]
                if arr.ndim == 0:
                    return arr.item()
            except Exception:
                return None
            return None

        # ---- JULD decoding ----
        juld_values = [pd.NaT] * n_profs
        if "JULD" in ds.variables:
            try:
                juld_series = decode_juld_variable(ds["JULD"], ds)
                if len(juld_series) == n_profs:
                    juld_values = list(juld_series)
            except Exception:
                pass

        rows = []
        for i in range(n_profs):
            for k in range(n_levels):
                pres = get_val("PRES", i, k)
                temp = get_val("TEMP", i, k)
                psal = get_val("PSAL", i, k)

                # skip if all missing
                if (pres is None or np.isnan(safe_float(pres))) and \
                   (temp is None or np.isnan(safe_float(temp))) and \
                   (psal is None or np.isnan(safe_float(psal))):
                    continue

                row = {
                    "profile_index": i,
                    "platform_number": elem_to_string(ds["PLATFORM_NUMBER"].values[i]) if "PLATFORM_NUMBER" in ds.variables else "",
                    "juld": juld_values[i],
                    "level_index": k,
                    "PRES": safe_float(pres),
                    "TEMP": safe_float(temp),
                    "PSAL": safe_float(psal),
                }

                # attach adjusted and QC if available (dimension-safe)
                for v in ("PRES","TEMP","PSAL"):
                    qc = v + "_QC"
                    adj = v + "_ADJUSTED"

                    if qc in ds.variables:
                        qarr = np.asarray(ds[qc].values)
                        if qarr.ndim == 2:
                            qv = get_val(qc, i, k)
                        else:
                            qv = get_val(qc, i)
                        row[qc] = elem_to_string(qv)
                    else:
                        row[qc] = ""

                    if adj in ds.variables:
                        aarr = np.asarray(ds[adj].values)
                        if aarr.ndim == 2:
                            av = get_val(adj, i, k)
                        else:
                            av = get_val(adj, i)
                        row[adj] = safe_float(av) if av is not None else float("nan")
                    else:
                        row[adj] = float("nan")

                rows.append(row)

    df = pd.DataFrame(rows)
    # write compressed CSV
    atomic_write_df(df, out_dir / "argo_core_measurements.csv.gz", index=False)
    return out_dir / "argo_core_measurements.csv.gz"
