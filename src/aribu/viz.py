import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from .dataset import (CLIP_HI, CLIP_LO, LABEL_LAND, LABEL_NODATA, LABEL_WATER,
                      load_chip, read_s2, valid_mask, water_indices)
from .metrics import confusion, metrics_from_counts

__all__ = ["LABEL_CMAP", "LABEL_NAMES", "ERROR_COLOURS", "INDEX_CMAPS",
           "percentile_limits", "shifted_cmap", "true_colour", "error_panel"]

LABEL_CMAP = plt.get_cmap("Blues").copy()
LABEL_CMAP.set_bad("0.85")   # masked (no-data) pixels -> light grey

LABEL_NAMES = {LABEL_NODATA: "no data", LABEL_LAND: "land", LABEL_WATER: "water"}

ERROR_COLOURS = np.array([
    [247, 251, 255],     # 0 correct land   (true negative)
    [  8,  48, 107],     # 1 correct water  (true positive)
    [253, 191,  51],     # 2 false alarm    (predicted water, was land)
    [214,  48,  38],     # 3 missed flood   (was water, predicted land)
    [217, 217, 217],     # 4 no data
]) / 255

def percentile_limits(band, low=2, high=98):
    """
    Colour limits for displaying a raster band, ignoring invalid pixels.
    Returns (vmin, vmax) to pass to imshow.
    """
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return CLIP_LO, CLIP_HI
    vmin, vmax = np.percentile(finite, [low, high])
    return float(vmin), float(vmax)

def shifted_cmap(name, midpoint, power=0.5, n=256):
    """Remap a diverging colormap so its neutral colour sits at `midpoint` in [0, 1].

    power < 1 compresses the pale middle, power > 1 expands it.
    """
    base = plt.get_cmap(name)
    x = np.linspace(0, 1, n)
    y = np.empty_like(x)
    left = x <= midpoint
    y[left] = 0.5 * (x[left] / midpoint) ** power                          # [0, midpoint] -> [0, 0.5]
    y[~left] = 0.5 + 0.5 * ((x[~left] - midpoint) / (1 - midpoint)) ** power   # -> [0.5, 1]
    return LinearSegmentedColormap.from_list(f"{name}_shifted", base(y), N=n)

INDEX_CMAPS = {"NDWI": shifted_cmap("PuOr", midpoint=0.40, power=0.80),
               "MNDWI": shifted_cmap("PiYG", midpoint=0.65, power=0.90)}

def true_colour(chip_id, low=2, high=98):
    """A displayable (H, W, 3) image taking values between 0 and 1 from Sentinel-2 in RGB = (B4, B3, B2).

    Each channel is stretched between its own percentiles.
    """
    a = read_s2(chip_id, ("B4", "B3", "B2"))
    empty = (a == 0).all(0)
    out = np.empty_like(a)
    for i, band in enumerate(a):
        lo, hi = np.percentile(band[~empty], [low, high]) if (~empty).any() else (0.0, 1.0)
        out[i] = np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.moveaxis(out, 0, -1)       # move the channel axis to the end for `imshow`
    rgb[empty] = 1.0
    return rgb

def error_panel(chip_id, predict_fn, axes, first="VH"):
    """Four panels on the given axes: first, truth, prediction, colour-coded errors.

    first is either "VH" or "MNDWI" to select the first panel.

    `axes` must be a sequence of four matplotlib axes.
    """
    if first not in ("VH", "MNDWI"):
        raise ValueError(f"unknown first panel {first!r}; expected 'VH' or 'MNDWI'")
    c = load_chip(chip_id)
    v = valid_mask(c)
    pred = np.asarray(predict_fn(c.vv, c.vh, v), bool)
    truth = c.label == LABEL_WATER

    code = np.where(pred & truth, 1, np.where(pred & ~truth, 2, np.where(~pred & truth, 3, 0)))
    code[~v] = 4
    m = metrics_from_counts(confusion(pred, truth, v))

    if first == "MNDWI":
        arr = water_indices(chip_id)[1]
        lim = np.percentile(np.abs(arr), 98)
        axes[0].imshow(arr, cmap=INDEX_CMAPS["MNDWI"], vmin=-lim, vmax=lim)
        axes[0].set_title(f"{chip_id}\nMNDWI", fontsize=9)
    else:
        vmin, vmax = percentile_limits(c.vh)
        axes[0].imshow(c.vh, cmap="gray", vmin=vmin, vmax=vmax)
        axes[0].set_title(f"{chip_id}\nVH backscatter", fontsize=9)

    for ax, arr, title in ((axes[1], truth, "truth"), (axes[2], pred, "prediction")):
        ax.imshow(np.ma.masked_where(~v, arr), cmap=LABEL_CMAP,
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=9)

    axes[3].imshow(ERROR_COLOURS[code], interpolation="nearest")
    axes[3].set_title(f"IoU {m['iou']:.3f}   P {m['precision']:.2f}  R {m['recall']:.2f}",
                      fontsize=9)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
