import numpy as np
import matplotlib.pyplot as plt
from .dataset import CLIP_HI, CLIP_LO

LABEL_CMAP = plt.get_cmap("Blues").copy()
LABEL_CMAP.set_bad("0.85")   # masked (no-data) pixels -> light grey

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