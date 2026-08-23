"""Density-colored observed vs predicted scatterplots, one per SFI depth
layer, built from the out-of-fold predictions produced by nested_cv_rf.py.
"""

import os

import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from config import RESPONSE_LABELS, RESPONSES, RESULTS_DIR

ARIAL_PATH = "/kaggle/input/datasets/hammaadali/arial-font/arial.ttf"
PANEL_LABELS = ["a.", "b.", "c.", "d.", "e."]


def load_font():
    if os.path.exists(ARIAL_PATH):
        fm.fontManager.addfont(ARIAL_PATH)
        return fm.FontProperties(fname=ARIAL_PATH).get_name()
    return "sans-serif"


def point_density(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    xy = np.vstack([x[mask], y[mask]])
    try:
        density = gaussian_kde(xy)(xy)
    except Exception:
        density = np.ones(mask.sum())
    density = (density - density.min()) / (density.max() - density.min() + 1e-12)
    return density, mask


def plot_scatter_grid(cv_results, font_family):
    panel_data = []
    for r in RESPONSES:
        obs_pred, m = cv_results[r]["predictions"], cv_results[r]["metrics"]
        panel_data.append(dict(tag=r, y_true=obs_pred["obs"].values, y_pred=obs_pred["pred"].values,
                                r2=m["R2"].mean(), rmse=m["RMSE"].mean(), rpiq=m["RPIQ"].mean(),
                                n=len(obs_pred)))

    fig = plt.figure(figsize=(20, 13), dpi=300)
    gs = gridspec.GridSpec(2, 3, figure=fig, left=0.06, right=0.97, top=0.93, bottom=0.08,
                            wspace=0.45, hspace=0.25)

    for idx, pdat in enumerate(panel_data):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])
        y_true, y_pred = pdat["y_true"], pdat["y_pred"]
        dens, mask = point_density(y_true, y_pred)
        yt, yp = y_true[mask], y_pred[mask]
        order = dens.argsort()

        sc = ax.scatter(yt[order], yp[order], c=dens[order], cmap="viridis", s=20,
                         alpha=0.70, edgecolors="none", rasterized=True)

        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.4, alpha=0.65, zorder=1)
        z = np.polyfit(yt, yp, 1)
        x_fit = np.linspace(yt.min(), yt.max(), 300)
        ax.plot(x_fit, np.poly1d(z)(x_fit), color="red", linewidth=1.8, alpha=0.85, zorder=2)

        label = RESPONSE_LABELS.get(pdat["tag"], pdat["tag"])
        ax.set_xlabel(f"Observed {label}", fontfamily=font_family)
        ax.set_ylabel(f"Predicted {label}", fontfamily=font_family)
        ax.set_title(label, fontweight="bold", loc="left", pad=10, fontfamily=font_family)
        ax.text(-0.17, 1.08, PANEL_LABELS[idx], transform=ax.transAxes, fontsize=32,
                fontweight="bold", va="bottom", ha="left", fontfamily=font_family, color="#1a1a2e")

        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        cbar = plt.colorbar(sc, ax=ax, pad=0.02, aspect=20, shrink=0.78)
        cbar.set_label("Point density", rotation=270, labelpad=22, fontfamily=font_family)

        stats_txt = f"$R^2$    = {pdat['r2']:.3f}\nRMSE = {pdat['rmse']:.2f}\nRPIQ  = {pdat['rpiq']:.2f}\n$n$      = {pdat['n']:,}"
        ax.text(0.05, 0.95, stats_txt, transform=ax.transAxes, fontsize=19, verticalalignment="top",
                fontfamily=font_family, linespacing=1.7)
        ax.set_aspect("equal", adjustable="box")

    for j in range(len(panel_data), 6):
        row, col = divmod(j, 3)
        fig.add_subplot(gs[row, col]).axis("off")

    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(RESULTS_DIR, f"rf_scatter_2x3_SFI.{ext}"), dpi=600,
                    bbox_inches="tight", facecolor="white", pad_inches=0.05)
    plt.close(fig)


if __name__ == "__main__":
    from modeling.nested_cv_rf import run_nested_cv
    import geopandas as gpd
    from config import COVARIATE_NAMES, FINAL_GPKG

    font_family = load_font()
    plt.rcParams.update({"font.family": font_family, "font.size": 22, "pdf.fonttype": 42, "ps.fonttype": 42})

    df = gpd.read_file(FINAL_GPKG).drop(columns="geometry")
    cv_results = {r: run_nested_cv(df, r, COVARIATE_NAMES) for r in RESPONSES}
    plot_scatter_grid(cv_results, font_family)
