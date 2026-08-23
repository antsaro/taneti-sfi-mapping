"""SFI prediction maps by bioclimatic region: 5 rows (depth layers) x 3
columns (CH, NW, SE), each colored on a shared, quantile-based 3-class
scale so panels stay comparable across regions and depths.

Quantile (equal-count) bins are used instead of equal-width bins because
SFI's distribution is skewed enough that equal-width classes leave most
pixels crammed into a single color -- quantile bins keep contrast even
on that kind of distribution.
"""

import glob
import os

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask

from config import NODATA_OUT, PREDICTION_DIR, RESPONSE_LABELS, RESPONSES, ROI_PATH

REGION_COL = "Region"
OUTPUT_DIR = "/kaggle/working"
DPI = 600
N_BINS = 3
P_LOW, P_HIGH = 2.0, 98.0
ROUND_TO = 0.05
DECIMALS = 2
ARIAL_PATH = "/kaggle/input/datasets/hammaadali/arial-font/arial.ttf"
BASE_FS = 12

SFI_COLORS = ["#D73027", "#FEE08B", "#1A9850"]  # low -> moderate -> high fertility


def load_font():
    matches = glob.glob(ARIAL_PATH) or glob.glob("/kaggle/input/**/*.ttf", recursive=True)
    if matches:
        fm.fontManager.addfont(matches[0])
        return fm.FontProperties(fname=matches[0]).get_name()
    return "sans-serif"


def make_cmap(hex_colors):
    cmap = mcolors.ListedColormap(hex_colors, name="sfi")
    cmap.set_bad(color="white")
    return cmap


def quantile_norm(data_masked, hex_colors, n_bins, p_low=P_LOW, p_high=P_HIGH,
                   round_to=ROUND_TO, decimals=DECIMALS):
    """Bin edges as true quantiles (equal pixel count per bin), rounded for
    readable legend labels and extended to the data's actual min/max so
    every valid pixel gets a color."""
    valid = data_masked.compressed()
    raw_bounds = np.percentile(valid, np.linspace(p_low, p_high, n_bins + 1))
    bounds = np.round(raw_bounds / round_to) * round_to
    for i in range(1, len(bounds)):
        if bounds[i] <= bounds[i - 1]:
            bounds[i] = bounds[i - 1] + round_to
    bounds[0] = min(bounds[0], np.floor(valid.min() / round_to) * round_to)
    bounds[-1] = max(bounds[-1], np.ceil(valid.max() / round_to) * round_to)
    bounds = np.round(bounds, decimals + 4)

    cmap = make_cmap(hex_colors)
    norm = mcolors.BoundaryNorm(bounds, ncolors=n_bins)

    fmt = f"{{:.{decimals}f}}"
    labels = []
    for i in range(n_bins):
        if i == 0:
            labels.append(f"< {fmt.format(bounds[1])}")
        elif i == n_bins - 1:
            labels.append(f"> {fmt.format(bounds[-2])}")
        else:
            labels.append(f"{fmt.format(bounds[i])}\u2013{fmt.format(bounds[i+1])}")
    return norm, cmap, labels


def crop_to_region(raster_path, region_gdf):
    with rasterio.open(raster_path) as src:
        geoms = [g.__geo_interface__ for g in region_gdf.to_crs(src.crs).geometry]
        out_arr, out_transform = rio_mask(src, geoms, crop=True, nodata=NODATA_OUT)
        out_arr = out_arr[0].astype(np.float32)
        nodata = src.nodata if src.nodata is not None else NODATA_OUT

    mask = ~np.isfinite(out_arr) | np.isclose(out_arr, nodata) | np.isclose(out_arr, NODATA_OUT)
    h, w = out_arr.shape
    left, top = out_transform.c, out_transform.f
    extent = [left, left + out_transform.a * w, top + out_transform.e * h, top]
    return np.ma.array(out_arr, mask=mask), extent


def draw_panel(ax, data, extent, norm, cmap, letter, title, legend_title, labels):
    ax.set_facecolor("white")
    ax.imshow(data, cmap=cmap, norm=norm, extent=extent, origin="upper", interpolation="nearest")
    ax.set_axis_off()

    orig_bbox = ax.get_position(original=True)
    cell_transform = mtransforms.BboxTransformTo(orig_bbox) + ax.figure.transFigure

    ax.text(0.01, 1.15, letter, transform=cell_transform, fontsize=BASE_FS + 9,
            fontweight="bold", color="#111111", ha="left", va="top")
    ax.text(0.12, 1.15, title, transform=cell_transform, fontsize=BASE_FS + 9,
            fontweight="bold", color="#111111", ha="left", va="top", linespacing=1.5)

    patches = [mpatches.Patch(facecolor=cmap.colors[i], edgecolor="#555555", linewidth=0.5, label=labels[i])
               for i in range(len(labels))]
    leg = ax.legend(handles=patches, title=legend_title, title_fontsize=BASE_FS + 1,
                     fontsize=BASE_FS + 1, loc="lower right", bbox_to_anchor=(1.5, 0.03),
                     bbox_transform=cell_transform, frameon=False, handlelength=1.5,
                     handleheight=1.0, borderpad=0.65, labelspacing=0.25)
    leg.get_title().set_fontweight("bold")


def build_figure(response_order, region_names, data_by_resp_region, extent_by_resp_region,
                  norms, cmaps, labels_by_resp, out_name="ISF_by_region_5x3", fig_h_per_row=5.5):
    n_rows = len(response_order)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18.0, fig_h_per_row * n_rows), facecolor="white",
                              squeeze=False, gridspec_kw={"wspace": 0.45, "hspace": 0.15})
    letters = [chr(ord("a") + i) + "." for i in range(n_rows * 3)]

    for row, r in enumerate(response_order):
        lbl = RESPONSE_LABELS[r]
        row_letters = letters[row * 3:(row + 1) * 3]
        for col, region in enumerate(region_names):
            draw_panel(axes[row, col], data_by_resp_region[r][region], extent_by_resp_region[r][region],
                       norms[r], cmaps[r], row_letters[col], f"{lbl}\n{region}",
                       f"Predicted {lbl}\n(index)", labels_by_resp[r])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"{out_name}.{ext}"), dpi=DPI,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    font_family = load_font()
    plt.rcParams.update({"font.family": font_family, "font.size": BASE_FS, "axes.linewidth": 0.8})

    aoi = gpd.read_file(ROI_PATH)
    aoi_regions = aoi.dissolve(by=REGION_COL, as_index=False)
    region_names = sorted(aoi_regions[REGION_COL].unique().tolist())

    pred_paths = {r: os.path.join(PREDICTION_DIR, f"{r}_prediction_FIL.tif") for r in RESPONSES}

    data_by_resp_region = {r: {} for r in RESPONSES}
    extent_by_resp_region = {r: {} for r in RESPONSES}
    for r in RESPONSES:
        for region in region_names:
            region_gdf = aoi_regions[aoi_regions[REGION_COL] == region]
            data, extent = crop_to_region(pred_paths[r], region_gdf)
            data_by_resp_region[r][region] = data
            extent_by_resp_region[r][region] = extent

    norms, cmaps, labels_by_resp = {}, {}, {}
    for r in RESPONSES:
        combined = np.ma.concatenate([data_by_resp_region[r][region].compressed() for region in region_names])
        combined_masked = np.ma.array(combined, mask=np.zeros_like(combined, dtype=bool))
        norms[r], cmaps[r], labels_by_resp[r] = quantile_norm(combined_masked, SFI_COLORS, N_BINS)

    build_figure(RESPONSES, region_names, data_by_resp_region, extent_by_resp_region,
                 norms, cmaps, labels_by_resp)
