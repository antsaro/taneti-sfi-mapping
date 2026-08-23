"""Wall-to-wall raster prediction of each SFI depth layer using cuML's
Forest Inference Library. Tiled so memory use stays bounded regardless of
raster size, and each tile is tracked so a crashed run can be diagnosed
without redoing the whole surface.
"""

import csv
import gc
import os
import time

import cupy as cp
import cuml
import joblib
import numpy as np
import rasterio
from cuml.fil import ForestInference
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from tqdm import tqdm

from config import (COVARIATE_NAMES, CROPPED_DIR, FIL_BATCH_SIZE, GPU_ID, NODATA_OUT,
                     PREDICTION_DIR, RESPONSES, RESULTS_DIR, TILE_SIZE)

ALIGNED_DIR = "/kaggle/working/Tanety/aligned"


def build_fil_model(rf_model):
    with cp.cuda.Device(GPU_ID):
        return ForestInference.load_from_sklearn(rf_model)


def gpu_predict(fil, X_cpu):
    out = np.empty(X_cpu.shape[0], dtype=np.float32)
    with cp.cuda.Device(GPU_ID):
        for p0 in range(0, X_cpu.shape[0], FIL_BATCH_SIZE):
            p1 = min(p0 + FIL_BATCH_SIZE, X_cpu.shape[0])
            X_gpu = cp.asarray(X_cpu[p0:p1])
            raw = fil.predict(X_gpu)
            raw = cp.asarray(raw.values if hasattr(raw, "values") else raw)
            out[p0:p1] = cp.asnumpy(raw.ravel().astype(cp.float32))
            del X_gpu
            cp.get_default_memory_pool().free_all_blocks()
    return out


def align_raster(path, ref_profile):
    with rasterio.open(path) as src:
        if (src.crs == ref_profile["crs"] and src.width == ref_profile["width"]
                and src.height == ref_profile["height"] and src.transform == ref_profile["transform"]):
            return path
        os.makedirs(ALIGNED_DIR, exist_ok=True)
        dst_path = os.path.join(ALIGNED_DIR, os.path.basename(path).replace(".tif", "_aligned.tif"))
        with rasterio.open(dst_path, "w", **{**src.profile, **ref_profile, "count": src.count,
                                              "dtype": src.profile["dtype"]}) as dst:
            for i in range(1, src.count + 1):
                reproject(source=rasterio.band(src, i), destination=rasterio.band(dst, i),
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=ref_profile["transform"], dst_crs=ref_profile["crs"],
                          resampling=Resampling.bilinear)
        return dst_path


def predict_surface(response, fil, ref_path, covariate_paths):
    with rasterio.open(ref_path) as ref:
        H, W, transform, crs = ref.height, ref.width, ref.transform, ref.crs
        out_profile = ref.profile.copy()
    out_profile.update(dtype="float32", count=1, nodata=NODATA_OUT, compress="lzw",
                        predictor=2, tiled=True, blockxsize=512, blockysize=512, bigtiff="YES")

    handles = {name: rasterio.open(covariate_paths[name]) for name in COVARIATE_NAMES}
    nodata_by_name = {name: handles[name].nodata for name in COVARIATE_NAMES}

    row_starts = list(range(0, H, TILE_SIZE))
    col_starts = list(range(0, W, TILE_SIZE))
    tiles = [(ti * len(col_starts) + tj + 1, r, c)
             for ti, r in enumerate(row_starts) for tj, c in enumerate(col_starts)]

    out_path = os.path.join(PREDICTION_DIR, f"{response}_prediction_FIL.tif")
    tracking_csv = os.path.join(PREDICTION_DIR, f"tile_tracking_{response}.csv")

    with open(tracking_csv, "w", newline="") as trk_file:
        writer = csv.DictWriter(trk_file, fieldnames=["tile_idx", "row_start", "col_start",
                                                        "n_valid_pixels", "infer_s", "total_s", "status"])
        writer.writeheader()

        with rasterio.open(out_path, "w", **out_profile) as h_out:
            for tidx, rs, cs in tqdm(tiles, desc=f"{response} FIL tiles", unit="tile"):
                t0 = time.time()
                re, ce = min(rs + TILE_SIZE, H), min(cs + TILE_SIZE, W)
                th, tw = re - rs, ce - cs
                win = Window(cs, rs, tw, th)
                tile_arr = np.full((th, tw), NODATA_OUT, np.float32)
                n_valid, infer_s, status = 0, 0.0, "nodata"

                try:
                    bands, valid = {}, np.ones((th, tw), dtype=bool)
                    for name in COVARIATE_NAMES:
                        arr = handles[name].read(1, window=win).astype(np.float32)
                        nd = nodata_by_name[name]
                        mask = ~np.isfinite(arr) | (np.isclose(arr, nd) if nd is not None else False)
                        valid &= ~mask
                        bands[name] = arr
                    n_valid = int(valid.sum())
                    if n_valid > 0:
                        flat = np.where(valid.ravel())[0].astype(np.int64)
                        X = np.column_stack([bands[n].ravel()[flat] for n in COVARIATE_NAMES]).astype(np.float32)
                        t_i = time.time()
                        preds = gpu_predict(fil, X)
                        infer_s = time.time() - t_i
                        r2, c2 = np.unravel_index(flat, (th, tw))
                        tile_arr[r2, c2] = preds
                        status = "ok"
                except Exception as e:
                    status = f"error:{e}"

                h_out.write(tile_arr, 1, window=win)
                del tile_arr
                gc.collect()
                writer.writerow(dict(tile_idx=tidx, row_start=rs, col_start=cs, n_valid_pixels=n_valid,
                                      infer_s=f"{infer_s:.2f}", total_s=f"{time.time()-t0:.2f}", status=status))

    for h in handles.values():
        h.close()
    with cp.cuda.Device(GPU_ID):
        cp.get_default_memory_pool().free_all_blocks()
    return out_path


if __name__ == "__main__":
    assert GPU_ID < cp.cuda.runtime.getDeviceCount()
    cp.cuda.Device(GPU_ID).use()

    cropped_files = sorted(f for f in os.listdir(CROPPED_DIR) if f.lower().endswith((".tif", ".tiff")))
    ref_path = os.path.join(CROPPED_DIR, "chm.tif")
    with rasterio.open(ref_path) as ref:
        ref_profile = dict(crs=ref.crs, width=ref.width, height=ref.height, transform=ref.transform)

    covariate_paths = {}
    for name in COVARIATE_NAMES:
        match = next(f for f in cropped_files if f.lower().startswith(name.lower() + "."))
        covariate_paths[name] = align_raster(os.path.join(CROPPED_DIR, match), ref_profile)

    for response in RESPONSES:
        model_bundle = joblib.load(os.path.join(RESULTS_DIR, f"rf_model_{response}.pkl"))
        fil = build_fil_model(model_bundle["model"])
        out_path = predict_surface(response, fil, ref_path, covariate_paths)
        print(f"{response} -> {out_path}")
