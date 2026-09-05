import numpy as np
import matplotlib.pyplot as plt
from .dataset import (CLIP_HI, CLIP_LO, LABEL_LAND, LABEL_NODATA, LABEL_WATER,
                      load_chip, valid_mask)
from .metrics import confusion, metrics_from_counts

__all__ = ["LABEL_CMAP", "LABEL_NAMES", "ERROR_COLOURS",
           "percentile_limits", "error_panel"]

LABEL_CMAP = plt.get_cmap("Blues").copy()
LABEL_CMAP.set_bad("0.85")   # masked (no-data) pixels -> light grey

LABEL_NAMES = {LABEL_NODATA: "no data", LABEL_LAND: "land", LABEL_WATER: "water"}

ERROR_COLOURS = np.array([
    [0.93, 0.93, 0.93],     # 0 correct land   (true negative)
    [0.16, 0.44, 0.71],     # 1 correct water  (true positive)
    [0.84, 0.19, 0.15],     # 2 false alarm    (predicted water, was land)
    [0.99, 0.75, 0.20],     # 3 missed flood   (was water, predicted land)
    [1.00, 1.00, 1.00],     # 4 no data
])

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

def error_panel(chip_id, predict_fn, axes):
    """Four panels on the given axes: VH, truth, prediction, colour-coded errors.

    predict_fn(vv, vh, valid) -> bool array

    `axes` must be a sequence of four matplotlib axes.
    """
    c = load_chip(chip_id)
    v = valid_mask(c)
    pred = np.asarray(predict_fn(c.vv, c.vh, v), bool)
    truth = c.label == LABEL_WATER

    code = np.where(pred & truth, 1, np.where(pred & ~truth, 2, np.where(~pred & truth, 3, 0)))
    code[~v] = 4
    m = metrics_from_counts(confusion(pred, truth, v))

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
