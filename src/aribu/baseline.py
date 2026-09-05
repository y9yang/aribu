import numpy as np
from skimage.filters import median
from skimage.morphology import footprint_rectangle

__all__ = ["median_filter_db", "make_baseline_predictor"]

def median_filter_db(band, valid, size):
    """Median speckle filter that does not let no-data leak into the result.

    Returns float32, same shape as `band`.
    """
    # `valid` is already a subset of isfinite(band), so the median OF THE VALID PIXELS
    # is the same number a nan-median of a NaN-filled copy would give, in one pass.
    fill = np.median(band[valid]) if valid.any() else np.float32(0)
    x = np.where(valid, band, fill).astype(np.float32)
    return median(x, footprint_rectangle((size, size)))

def make_baseline_predictor(cfg):
    """Build a predict_fn from a config dict.

    cfg keys: speckle_median (0 to skip), clip_db (lo, hi), threshold_db.

    Returns a function (vv, vh, valid) -> bool array.
    """
    def predict(vv, vh, valid):
        x = median_filter_db(vh, valid, cfg["speckle_median"]) if cfg["speckle_median"] else vh
        return np.clip(x, *cfg["clip_db"]) < cfg["threshold_db"]
    return predict
