import numpy as np
import pytest
from aribu.dataset import CLIP_HI, CLIP_LO, IGNORE_INDEX, LABEL_NODATA, Chip, chip_input, cloud_mask, preprocess, read_split, valid_mask

def _chip():
    """One row of five pixels: valid water, no VV, no VH, no label, valid land."""
    vv = np.array([[-80.0, np.nan, -10.0, -12.0, 5.0]], np.float32)
    vh = np.array([[-20.0, -20.0, np.nan, -15.0, -15.0]], np.float32)
    label = np.array([[1, 0, 1, LABEL_NODATA, 0]], np.int16)
    return Chip(id="Test_1", loc="Test", vv=vv, vh=vh, label=label, crs=None, transform=None) # pyright: ignore[reportArgumentType]

def test_valid_mask_drops_missing_radar_and_labels():
    assert valid_mask(_chip()).tolist() == [[True, False, False, False, True]]

def test_preprocess_without_stats_clips_and_ignores():
    x, y, _ = preprocess(_chip())
    assert x.shape == (2, 1, 5) and x.dtype == np.float32
    assert x[0, 0, 0] == CLIP_LO and x[0, 0, 4] == CLIP_HI
    assert np.isnan(x[0, 0, 1])
    assert y.tolist() == [[1, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 0]]

def test_preprocess_with_stats_fills_missing_radar_with_zero():
    x, _, _ = preprocess(_chip(), mean=(-15.0, -18.0), std=(5.0, 4.0))
    assert np.isfinite(x).all()
    assert x[0, 0, 1] == 0 and x[1, 0, 2] == 0
    assert np.isclose(x[0, 0, 0], -7.0) and np.isclose(x[1, 0, 1], -0.5)

def test_preprocess_needs_both_stats():
    with pytest.raises(ValueError):
        preprocess(_chip(), mean=(-15.0, -18.0))

def test_cloud_mask_is_repeatable_and_covers_fraction():
    mask = cloud_mask("Test_1", 0.3)
    assert (mask == cloud_mask("Test_1", 0.3)).all()
    assert (mask != cloud_mask("Test_2", 0.3)).any()
    assert abs(mask.mean() - 0.3) < 0.01
    assert not cloud_mask("Test_1", 0).any() and cloud_mask("Test_1", 1).all()

def test_unknown_arm_and_split_are_rejected():
    with pytest.raises(ValueError):
        chip_input("Test_1", "sar", None, None)
    with pytest.raises(ValueError):
        read_split("holdout")
