"""Shared paths, covariate names and run settings for the Tanety SFI pipeline.

Every other script in this repo imports from here instead of redefining its
own constants, and every path below is a Kaggle path -- point these at your
own storage before running anything.
"""

import os

ROI_PATH = "/kaggle/input/datasets/antsasarobidyran/area-of-interest/AOI_dissolved.gpkg"
DATA_PATH = "/kaggle/input/datasets/antsasarobidyran/sfi-data/Data_SFI.gpkg"
CROPPED_DIR = "/kaggle/input/datasets/antsasarobidyran/tanety-covariates"

FINAL_GPKG = "/kaggle/working/Tanety/Data_covariates.gpkg"
RESULTS_DIR = "/kaggle/working/Tanety/ML_Results"
PREDICTION_DIR = "/kaggle/working/Tanety/predictions_FIL"

# 13-layer covariate stack: terrain, vegetation, climate, soil context
COVARIATE_NAMES = [
    "chm", "ELEV_ALOS", "lulc", "map_corrected", "mat_corrected",
    "NDVI", "NDWI", "NIRI", "NPP", "SLOPE_ALOS",
    "soil_type", "treecover", "TWI",
]
CATEGORICAL_COVARIATES = ["lulc", "soil_type"]

# SFI depth layers: a=0-10cm, b=10-20cm, c=20-30cm, d=30-60cm, e=60-90cm
RESPONSES = ["ISF_a", "ISF_b", "ISF_c", "ISF_d", "ISF_e"]
RESPONSE_LABELS = {
    "ISF_a": r"SFI$_{0-10}$",
    "ISF_b": r"SFI$_{10-20}$",
    "ISF_c": r"SFI$_{20-30}$",
    "ISF_d": r"SFI$_{30-60}$",
    "ISF_e": r"SFI$_{60-90}$",
}

N_JOBS_CPU = -1
N_FOLDS_OUTER = 5
N_FOLDS_INNER = 3
OPTUNA_TRIALS_CV = 50
OPTUNA_TRIALS_FINAL = 50
SEED = 123

TILE_SIZE = 2048
NODATA_OUT = -9999.0
FIL_BATCH_SIZE = 600_000
GPU_ID = 0

for d in (os.path.dirname(FINAL_GPKG), RESULTS_DIR, PREDICTION_DIR):
    os.makedirs(d, exist_ok=True)
