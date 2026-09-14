import numpy as np
from aribu.metrics import METRIC_NAMES, confusion, metrics_from_counts

def test_confusion_counts_only_valid_pixels():
    pred = np.array([[True, True, False], [False, True, False]])
    truth = np.array([[True, False, True], [False, True, True]])
    valid = np.array([[True, True, True], [True, True, False]])
    counts = confusion(pred, truth, valid)
    assert counts.tolist() == [2, 1, 1, 1]
    assert counts.sum() == valid.sum()

def test_metrics_from_known_counts():
    m = metrics_from_counts((2, 1, 1, 4))
    expected = {"iou": 0.5, "f1": 2 / 3, "precision": 2 / 3, "recall": 2 / 3, "accuracy": 0.75, "specificity": 0.8}
    assert set(m) == set(METRIC_NAMES)
    for k, v in expected.items():
        assert np.isclose(m[k], v), k

def test_perfect_prediction_scores_one():
    truth = np.array([[True, False], [False, True]])
    m = metrics_from_counts(confusion(truth, truth, np.ones_like(truth)))
    assert all(np.isclose(v, 1.0) for v in m.values())

def test_no_water_gives_nan_not_error():
    m = metrics_from_counts((0, 0, 0, 5))
    assert all(np.isnan(m[k]) for k in ("iou", "f1", "precision", "recall"))
    assert m["accuracy"] == 1.0 and m["specificity"] == 1.0
