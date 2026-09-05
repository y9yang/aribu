import time
import warnings
import numpy as np
import pandas as pd
from .dataset import LABEL_WATER, load_chip, valid_mask

__all__ = [
    "COUNT_NAMES", "METRIC_NAMES",
    "confusion", "metrics_from_counts",
    "evaluate", "evaluate_with_chip_id", "compare",
]

COUNT_NAMES = ("tp", "fp", "fn", "tn")
METRIC_NAMES = ("iou", "f1", "precision", "recall", "accuracy", "specificity")

def confusion(pred, truth, valid):
    """Four counts (tp, fp, fn, tn) for one chip, water as the positive class."""
    p, w = pred[valid], truth[valid]
    return np.array([np.count_nonzero(p & w), np.count_nonzero(p & ~w),
                     np.count_nonzero(~p & w), np.count_nonzero(~p & ~w)], dtype=np.int64)

def metrics_from_counts(counts):
    """Standard segmentation metrics from (tp, fp, fn, tn)."""
    tp, fp, fn, tn = (float(v) for v in counts)

    def ratio(num, den):
        return num / den if den > 0 else np.nan

    return {"iou": ratio(tp, tp + fp + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn),
            "accuracy": ratio(tp + tn, tp + fp + fn + tn),
            "specificity": ratio(tn, tn + fp)}

def _score_chips(make_predictor, chip_ids, name):
    """The measurement loop."""
    rows, total, n_valid_px = [], np.zeros(4, dtype=np.int64), 0
    t0 = time.perf_counter()
    for chip_id in chip_ids:
        c = load_chip(chip_id)
        v = valid_mask(c)
        if not v.any():
            continue          # six chips have no usable pixels
        pred = np.asarray(make_predictor(chip_id)(c.vv, c.vh, v), bool)
        counts = confusion(pred, c.label == LABEL_WATER, v)
        total += counts
        n_valid_px += int(v.sum())
        rows.append({"chip_id": chip_id, "loc": c.loc,
                     **dict(zip(COUNT_NAMES, counts)),
                     **metrics_from_counts(counts)})

    per_chip = pd.DataFrame(rows)
    assert total.sum() == n_valid_px, f"{total.sum():,} counted vs {n_valid_px:,} valid px"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        macro = {k: float(np.nanmean(per_chip[k])) for k in METRIC_NAMES}

    return {"name": name, "n_chips": len(per_chip), "counts": dict(zip(COUNT_NAMES, total.tolist())),
            "micro": metrics_from_counts(total), "macro": macro,
            "per_chip": per_chip, "seconds": time.perf_counter() - t0}

def evaluate(predict_fn, chip_ids, name="model"):
    """Score any predictor over any set of chips.

    Returns micro metrics (pool every pixel), macro metrics (average over chips), and
    the per-chip counts.
    """
    return _score_chips(lambda _chip_id: predict_fn, chip_ids, name)

def evaluate_with_chip_id(make_predictor, chip_ids, name="model"):
    """evaluate(), but the predictor depends on the chip's identity."""
    return _score_chips(make_predictor, chip_ids, name)

def compare(*results, keys=("iou", "f1", "precision", "recall", "accuracy")):
    """Results side by side, micro over macro. Returns a DataFrame."""
    out = {}
    for r in results:
        out[r["name"]] = {f"{k} (micro)": r["micro"][k] for k in keys}
        out[r["name"]].update({f"{k} (macro)": r["macro"][k] for k in ("iou", "f1")})
    return pd.DataFrame(out).T.round(4)
