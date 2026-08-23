"""Extract the 13-layer covariate stack at each SFI sample location and write
out the combined point table that every downstream script reads from.
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from joblib import Parallel, delayed

from config import COVARIATE_NAMES, CROPPED_DIR, DATA_PATH, FINAL_GPKG, N_JOBS_CPU, RESPONSES


def extract_at_points(raster_path, points):
    name = os.path.splitext(os.path.basename(raster_path))[0]
    with rasterio.open(raster_path) as src:
        pts = points.to_crs(src.crs)
        coords = [(geom.x, geom.y) for geom in pts.geometry]
        vals = np.array([v[0] for v in src.sample(coords)], dtype=np.float64)
        if src.nodata is not None:
            vals[np.isclose(vals, src.nodata)] = np.nan
    return name, vals


def build_covariate_table():
    data = gpd.read_file(DATA_PATH)
    raster_files = sorted(f for f in os.listdir(CROPPED_DIR) if f.lower().endswith((".tif", ".tiff")))
    raster_paths = [os.path.join(CROPPED_DIR, f) for f in raster_files]

    extracted = Parallel(n_jobs=N_JOBS_CPU)(
        delayed(extract_at_points)(p, data) for p in raster_paths
    )

    out = data.reset_index(drop=True).copy()
    for name, vals in extracted:
        out[name] = vals

    keep = [c for c in RESPONSES + COVARIATE_NAMES + ["geometry"] if c in out.columns]
    out = out[keep]
    out.to_file(FINAL_GPKG, driver="GPKG")
    return out


def covariate_summary(table):
    long = table[COVARIATE_NAMES].melt(var_name="variable", value_name="value")
    summary = long.groupby("variable")["value"].agg(min="min", max="max", mean="mean",
                                                      median="median", std="std").reset_index()
    summary["cv"] = summary["std"] / summary["mean"] * 100
    return summary


if __name__ == "__main__":
    table = build_covariate_table()
    print(covariate_summary(table))
