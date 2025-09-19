# extract_qc.py
from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd
from utils import ensure_dir, atomic_write_df, elem_to_string, safe_float

def extract_qc(nc_path: Path, out_dir: Path):
    out_dir = ensure_dir(out_dir)
    with xr.open_dataset(nc_path, decode_times=False) as ds:
        n_prof = int(ds.sizes.get("N_PROF", 0))
        perlevel_rows = []
        for p in range(n_prof):
            # determine number of levels for this profile from PRES if present
            if "PRES" in ds:
                pres_arr = np.asarray(ds["PRES"].values)
                n_levels = pres_arr.shape[1] if pres_arr.ndim == 2 else 1
            else:
                n_levels = int(ds.sizes.get("N_LEVELS", 1))
            platform = elem_to_string(ds["PLATFORM_NUMBER"].values[p]) if "PLATFORM_NUMBER" in ds else ""
            juld_val = None
            if "JULD" in ds:
                try:
                    juld_val = pd.to_datetime(ds["JULD"].values[p])
                except:
                    juld_val = None

            for lev in range(n_levels):
                row = {"profile_index": p, "platform_number": platform, "juld": juld_val, "level_index": lev}
                # core QC/adjusted/error fields
                for v in ("PRES","TEMP","PSAL"):
                    if v in ds.variables:
                        # value already present in core extraction; here we capture QC/adj/error explicitly if available
                        row[v] = safe_float(np.asarray(ds[v].values)[p,lev]) if np.asarray(ds[v].values).ndim==2 else safe_float(np.asarray(ds[v].values)[p])
                    row[v + "_QC"] = elem_to_string(np.asarray(ds[v + "_QC"].values)[p,lev]) if (v + "_QC") in ds.variables and np.asarray(ds[v + "_QC"].values).ndim==2 else (elem_to_string(np.asarray(ds[v + "_QC"].values)[p]) if (v + "_QC") in ds.variables else "")
                    row[v + "_ADJUSTED"] = safe_float(np.asarray(ds[v + "_ADJUSTED"].values)[p,lev]) if (v + "_ADJUSTED") in ds.variables and np.asarray(ds[v + "_ADJUSTED"].values).ndim==2 else (safe_float(np.asarray(ds[v + "_ADJUSTED"].values)[p]) if (v + "_ADJUSTED") in ds.variables else float("nan"))
                    row[v + "_ADJUSTED_ERROR"] = safe_float(np.asarray(ds[v + "_ADJUSTED_ERROR"].values)[p,lev]) if (v + "_ADJUSTED_ERROR") in ds.variables and np.asarray(ds[v + "_ADJUSTED_ERROR"].values).ndim==2 else (safe_float(np.asarray(ds[v + "_ADJUSTED_ERROR"].values)[p]) if (v + "_ADJUSTED_ERROR") in ds.variables else float("nan"))
                perlevel_rows.append(row)

        # profile-level QC
        profile_rows = []
        for p in range(n_prof):
            prow = {"profile_index": p}
            # dynamically detect all PROFILE_*_QC vars
            for v in ds.variables:
                if v.startswith("PROFILE_") and v.endswith("_QC"):
                    prow[v] = elem_to_string(ds[v].values[p])
            # also add extra flags
            for v in ("DATA_MODE","DATA_STATE_INDICATOR"):
                prow[v] = elem_to_string(ds[v].values[p]) if v in ds.variables else ""
            profile_rows.append(prow)


        # calibration metadata
        calib_rows = []
        for p in range(n_prof):
            for v in ("SCIENTIFIC_CALIB_EQUATION","SCIENTIFIC_CALIB_COEFFICIENT","SCIENTIFIC_CALIB_COMMENT","SCIENTIFIC_CALIB_DATE"):
                if v in ds.variables:
                    val = ds[v].values[p]
                    calib_rows.append({"profile_index": p, "variable": v, "value": elem_to_string(val)})

    perlevel_df = pd.DataFrame(perlevel_rows)
    profile_df = pd.DataFrame(profile_rows)
    calib_df = pd.DataFrame(calib_rows)

    atomic_write_df(perlevel_df, out_dir / "argo_qc_per_level.csv")
    atomic_write_df(profile_df, out_dir / "argo_qc_profile.csv")
    atomic_write_df(calib_df, out_dir / "argo_calibration.csv")
    return (out_dir / "argo_qc_per_level.csv", out_dir / "argo_qc_profile.csv", out_dir / "argo_calibration.csv")
