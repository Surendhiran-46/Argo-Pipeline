# extract_metadata.py
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
from utils import is_metadata_var, elem_to_string, ensure_dir, atomic_write_df

def extract_metadata(nc_path: Path, out_dir: Path):
    out_dir = ensure_dir(out_dir)
    with xr.open_dataset(nc_path, decode_times=False) as ds:
        # determine N_PROF
        if "N_PROF" in ds.dims:
            n_profs = int(ds.sizes["N_PROF"])
        else:
            # fallback: PLATFORM_NUMBER or first variable length
            if "PLATFORM_NUMBER" in ds.variables:
                n_profs = ds["PLATFORM_NUMBER"].values.shape[0]
            else:
                # try any 1D var
                n_profs = None
                for v in ds.variables:
                    arr = ds[v].values
                    try:
                        if getattr(arr, "shape", None) and len(arr.shape) == 1:
                            n_profs = arr.shape[0]
                            break
                    except Exception:
                        pass
                if n_profs is None:
                    raise RuntimeError("Cannot determine N_PROF for metadata extraction.")

        # collect metadata-like vars
        meta = {}
        for var in ds.variables:
            try:
                if is_metadata_var(var, ds, n_profs):
                    arr = ds[var].values
                    # build vector of strings (one per profile)
                    rows = []
                    if np.isscalar(arr) or getattr(arr, "shape", ()) == ():
                        rows = [elem_to_string(arr)] * n_profs
                    else:
                        for i in range(n_profs):
                            try:
                                rows.append(elem_to_string(arr[i]))
                            except Exception:
                                rows.append(elem_to_string(arr.ravel()[i] if i < arr.size else ""))
                    meta[var] = rows
            except Exception:
                continue

        df_full = pd.DataFrame(meta)
        # cleaned subset (presentation)
        important = [
            "PLATFORM_NUMBER","FLOAT_SERIAL_NO","PLATFORM_TYPE","WMO_INST_TYPE",
            "PROJECT_NAME","PI_NAME","DATA_CENTRE","FORMAT_VERSION","HANDBOOK_VERSION",
            "FIRMWARE_VERSION","CONFIG_MISSION_NUMBER"
        ]
        available = [v for v in important if v in df_full.columns]
        df_clean = df_full[available].copy() if available else pd.DataFrame()

        # save
        atomic_write_df(df_full, out_dir / "argo_metadata_full.csv")
        if not df_clean.empty:
            atomic_write_df(df_clean, out_dir / "argo_metadata_clean.csv")
        else:
            # still write an empty cleaned file for consistency
            df_clean.to_csv(out_dir / "argo_metadata_clean.csv", index=False)

    return out_dir / "argo_metadata_clean.csv"
