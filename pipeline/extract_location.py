from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd
from utils import elem_to_number, elem_to_string, decode_juld_variable, parse_reference_datetime, ensure_dir, atomic_write_df

def extract_location(nc_path: Path, out_dir: Path):
    out_dir = ensure_dir(out_dir)
    with xr.open_dataset(nc_path, decode_times=False) as ds:
        if "N_PROF" in ds.dims:
            n_profs = int(ds.sizes["N_PROF"])
        elif "PLATFORM_NUMBER" in ds.variables:
            n_profs = ds["PLATFORM_NUMBER"].values.shape[0]
        else:
            raise RuntimeError("Cannot determine N_PROF for location extraction.")

        output = {"profile_index": list(range(n_profs))}

        # lat/lon
        for v in ("LATITUDE", "LONGITUDE"):
            if v in ds.variables:
                arr = ds[v].values
                output[v] = [elem_to_number(arr[i]) for i in range(n_profs)]
            else:
                output[v] = [np.nan] * n_profs

        # juld and juld_location
        output["JULD"] = [pd.NaT]*n_profs
        output["JULD_LOCATION"] = [pd.NaT]*n_profs
        if "JULD" in ds.variables:
            try:
                juld_series = decode_juld_variable(ds["JULD"], ds)
                if len(juld_series) == n_profs:
                    output["JULD"] = list(pd.to_datetime(juld_series).round("s"))
            except Exception:
                pass
        if "JULD_LOCATION" in ds.variables:
            try:
                jl = decode_juld_variable(ds["JULD_LOCATION"], ds)
                if len(jl) == n_profs:
                    output["JULD_LOCATION"] = list(pd.to_datetime(jl).round("s"))
            except Exception:
                pass

        # reference
        if "REFERENCE_DATE_TIME" in ds.variables:
            r = ds["REFERENCE_DATE_TIME"].values
            ref_elem = r if np.isscalar(r) else (r.ravel()[0] if r.size > 0 else None)
        else:
            ref_elem = ds.attrs.get("REFERENCE_DATE_TIME")
        ref_dt = parse_reference_datetime(ref_elem)
        output["REFERENCE_DATE_TIME"] = [ref_dt] * n_profs

        # position qc/system/direction
        for v in ("POSITION_QC","POSITIONING_SYSTEM","DIRECTION"):
            if v in ds.variables:
                arr = ds[v].values
                output[v] = [elem_to_string(arr[i]) for i in range(n_profs)]
            else:
                output[v] = [""] * n_profs

        # attach a couple of ids (platform, project, pi) for context
        for v in ("PLATFORM_NUMBER","PROJECT_NAME","PI_NAME"):
            if v in ds.variables:
                arr = ds[v].values
                output[v] = [elem_to_string(arr[i]) for i in range(n_profs)]
            else:
                output[v] = [""] * n_profs

    df = pd.DataFrame(output)
    atomic_write_df(df, out_dir / "argo_time_location.csv")
    return out_dir / "argo_time_location.csv"
