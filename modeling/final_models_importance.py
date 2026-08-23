"""Fit one final Random Forest per depth layer on the full dataset, and
compute permutation importance (the %IncMSE analogue from randomForest)
for each covariate. Produces the lollipop-style importance figure.
"""

import os

import geopandas as gpd
import joblib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error

from config import (COVARIATE_NAMES, FINAL_GPKG, N_FOLDS_INNER, N_JOBS_CPU, OPTUNA_TRIALS_FINAL,
                     RESPONSE_LABELS, RESPONSES, RESULTS_DIR, SEED)
from modeling.common import tune_rf

ARIAL_PATH = "/kaggle/input/datasets/hammaadali/arial-font/arial.ttf"
N_REPEATS = 20

COVARIATE_LABELS = {
    "chm": "CHM", "ELEV_ALOS": "Elevation", "lulc": "LULC",
    "map_corrected": "MAP", "mat_corrected": "MAT", "NDVI": "NDVI",
    "NDWI": "NDWI", "NIRI": "NIRI", "NPP": "NPP", "SLOPE_ALOS": "Slope",
    "soil_type": "Soil Type", "treecover": "Tree Cover", "TWI": "TWI",
}
PANEL_LABELS = ["a.", "b.", "c.", "d.", "e."]


def load_font():
    if os.path.exists(ARIAL_PATH):
        fm.fontManager.addfont(ARIAL_PATH)
        return fm.FontProperties(fname=ARIAL_PATH).get_name()
    return "sans-serif"


def spine_cleanup(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("black")
        ax.spines[sp].set_linewidth(1.8)
    ax.tick_params(axis="both", colors="black", direction="out", length=6, width=1.8)
    ax.grid(False)


def draw_lollipop(ax, feature_cols, means, stds, font_family, label_map=None):
    paired = sorted(zip(feature_cols, means, stds), key=lambda x: x[1], reverse=True)
    keys = [k for k, _, _ in paired]
    vals = np.array([v for _, v, _ in paired])
    stds = np.array([s for _, _, s in paired])
    ypos = np.arange(len(paired))[::-1]

    for y, val, std in zip(ypos, vals, stds):
        ax.plot([0, val], [y, y], linestyle="--", color="black", linewidth=1.4, alpha=0.65, zorder=1)
        ax.errorbar(val, y, xerr=std, fmt="none", color="black", capsize=4, capthick=1.2,
                    elinewidth=1.2, zorder=2, alpha=0.65)
        ax.scatter(val, y, color="black", s=160, zorder=3, edgecolors="white", linewidths=1.0)

    labels = [label_map.get(k, k) for k in keys] if label_map else keys
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=18, fontfamily=font_family)
    ax.set_xlabel("%IncMSE", fontsize=20, fontfamily=font_family)
    ax.set_xlim(left=0, right=(vals + stds).max() * 1.22)
    ax.set_ylim(-0.6, len(paired) - 0.4)
    spine_cleanup(ax)
    ax.xaxis.grid(True, linestyle=":", linewidth=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)


def fit_final_models(df):
    models, panel_data = {}, []
    for r in RESPONSES:
        X = df[COVARIATE_NAMES].values
        y = df[r].values.astype(np.float64)

        params = tune_rf(X, y, OPTUNA_TRIALS_FINAL, N_FOLDS_INNER)
        n_estimators = max(params.pop("n_estimators", 500), 500)
        model = RandomForestRegressor(**params, n_estimators=n_estimators)
        model.fit(X, y)
        models[r] = model
        joblib.dump({"model": model, "features": COVARIATE_NAMES},
                    os.path.join(RESULTS_DIR, f"rf_model_{r}.pkl"))

        baseline_mse = mean_squared_error(y, model.predict(X))
        perm = permutation_importance(model, X, y, n_repeats=N_REPEATS, random_state=SEED,
                                       scoring="neg_mean_squared_error", n_jobs=N_JOBS_CPU)
        pct_inc_mse = perm.importances_mean / baseline_mse * 100
        pct_inc_mse_std = perm.importances_std / baseline_mse * 100

        imp = pd.DataFrame({"variable": COVARIATE_NAMES, "PctIncMSE": pct_inc_mse,
                             "PctIncMSE_std": pct_inc_mse_std}).sort_values(
            "PctIncMSE", ascending=False).reset_index(drop=True)
        imp.to_csv(os.path.join(RESULTS_DIR, f"varimp_PctIncMSE_{r}.csv"), index=False)
        panel_data.append({"tag": r, "mean": pct_inc_mse, "std": pct_inc_mse_std})

    return models, panel_data


def plot_importance(panel_data, font_family):
    fig, axes = plt.subplots(2, 3, figsize=(19, 11), gridspec_kw=dict(wspace=0.65, hspace=0.35))
    axes_flat = axes.flatten()

    for idx, pdat in enumerate(panel_data):
        ax = axes_flat[idx]
        draw_lollipop(ax, COVARIATE_NAMES, pdat["mean"], pdat["std"], font_family, COVARIATE_LABELS)
        ax.set_title(RESPONSE_LABELS.get(pdat["tag"], pdat["tag"]), fontweight="bold",
                     fontsize=21, loc="left", pad=10, fontfamily=font_family)
        ax.text(-0.55, 1.07, PANEL_LABELS[idx], transform=ax.transAxes, fontsize=30,
                fontweight="bold", va="bottom", ha="left", fontfamily=font_family, color="#1a1a2e")

    for j in range(len(panel_data), len(axes_flat)):
        axes_flat[j].axis("off")

    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(RESULTS_DIR, f"rf_varimp_PctIncMSE_lollipop.{ext}"),
                    dpi=600, bbox_inches="tight", facecolor="white", pad_inches=0.05)
    plt.close(fig)


if __name__ == "__main__":
    font_family = load_font()
    plt.rcParams.update({"font.family": font_family, "font.size": 19, "pdf.fonttype": 42, "ps.fonttype": 42})

    df = gpd.read_file(FINAL_GPKG).drop(columns="geometry")
    models, panel_data = fit_final_models(df)
    plot_importance(panel_data, font_family)
