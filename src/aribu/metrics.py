import time
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

def _evaluate(make_predictor, chip_ids, name):
    """The evaluation loop shared by evaluate() and evaluate_with_chip_id()."""
    rows, total = [], np.zeros(4, dtype=np.int64)
    t0 = time.perf_counter()
    for chip_id in chip_ids:
        c = load_chip(chip_id)
        v = valid_mask(c)
        if not v.any():
            continue          # six chips have no usable pixels
        pred = np.asarray(make_predictor(chip_id)(c.vv, c.vh, v), bool)
        counts = confusion(pred, c.label == LABEL_WATER, v)
        total += counts
        rows.append({"chip_id": chip_id, "loc": c.loc,
                     **dict(zip(COUNT_NAMES, counts)),
                     **metrics_from_counts(counts)})

    per_chip = pd.DataFrame(rows)
    macro = {k: float(per_chip[k].mean()) for k in METRIC_NAMES}    # mean() skips the NaN of chips with no water

    return {"name": name, "n_chips": len(per_chip), "counts": dict(zip(COUNT_NAMES, total.tolist())),
            "micro": metrics_from_counts(total), "macro": macro,
            "per_chip": per_chip, "seconds": time.perf_counter() - t0}

def evaluate(predict_fn, chip_ids, name="model"):
    """Score a Sentinel-1-only, chip_id independent predictor over any set of chips.

    `predict_fn(vv, vh, valid)` gets a chip's Sentinel-1 bands and its `valid_mask`, and returns a bool water mask
    (for a predictor that needs more, such as the Sentinel-2 image, use `evaluate_with_chip_id`).

    Returns micro metrics (pool every pixel from a flood event), macro metrics (average over chips), and
    the per-chip counts.
    """
    return _evaluate(lambda _chip_id: predict_fn, chip_ids, name)

def evaluate_with_chip_id(make_predictor, chip_ids, name="model"):
    """evaluate(), but the predictor depends on the chip's identity.

    Use it when the predictor depends on chip_id or needs more than the Sentinel-1 bands, such as the Sentinel-2 image:
    `make_predictor(chip_id)` returns that chip's `predict_fn`.
    """
    return _evaluate(make_predictor, chip_ids, name)

def compare(*results, keys=("iou", "f1", "precision", "recall", "accuracy")):
    """Put several results in one table, one row per model.

    `results` are the dicts from `evaluate()` or `evaluate_with_chip_id()`, and each row is named by the `name` assigned to the model.
    The columns are the micro metrics in `keys`, followed by macro IoU and F1.
    """
    out = {}
    for r in results:
        out[r["name"]] = {f"{k} (micro)": r["micro"][k] for k in keys}
        out[r["name"]].update({f"{k} (macro)": r["macro"][k] for k in ("iou", "f1")})
    return pd.DataFrame(out).T.round(4)
