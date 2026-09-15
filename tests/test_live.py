import os
from datetime import date
import numpy as np
import pytest
from rasterio.transform import array_bounds
from aribu.dataset import LABEL_NODATA, Chip, load_chip, normalised_indices, radar_input, stack_arm
from aribu.live import (CLOUD_PROBABILITY, PIXEL_SIZE, SQUARE_SIZE, fetch_square, find_scenes, init_earth_engine,
                        open_water, pick_model, square_input, square_transform)
from aribu.paths import S1_DIR

needs_earth_engine = pytest.mark.skipif(not (os.environ.get("EE_KEY_JSON") or os.environ.get("EE_KEY_FILE")),
                                        reason="needs an Earth Engine key in EE_KEY_JSON or EE_KEY_FILE")

def _chip():
    """One row of five pixels: valid water, no VV, no VH, no label, valid land."""
    vv = np.array([[-80.0, np.nan, -10.0, -12.0, 5.0]], np.float32)
    vh = np.array([[-20.0, -20.0, np.nan, -15.0, -15.0]], np.float32)
    label = np.array([[1, 0, 1, LABEL_NODATA, 0]], np.int16)
    return Chip(id="Test_1", loc="Test", vv=vv, vh=vh, label=label, crs=None, transform=None) # pyright: ignore[reportArgumentType]

def test_radar_input_and_stack_arm_give_the_chip_input():
    chip, mean, std = _chip(), (-15.0, -18.0), (5.0, 4.0)
    radar = radar_input(chip.vv, chip.vh, mean, std)
    assert np.allclose(radar[:, 0], [[-7.0, 0.0, 1.0, 0.6, 3.2], [-0.5, -0.5, 0.0, 0.75, 0.75]])
    indices = np.ones_like(radar)
    assert stack_arm(radar, None, "s1") is radar
    assert stack_arm(None, indices, "s2") is indices
    assert np.array_equal(stack_arm(radar, indices, "s1+s2"), np.concatenate([radar, indices]))

def test_normalised_indices_on_known_values():
    green, nir, swir = np.array([[3.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 3.0]]), np.array([[1.0, 0.0, 1.0]])
    ndwi, mndwi = normalised_indices(green, nir, swir)
    assert np.allclose(ndwi, [[0.5, 0.0, -0.5]]) and np.allclose(mndwi, [[0.5, 0.0, 0.0]])

def test_square_transform_spans_a_chip():
    left, bottom, right, top = array_bounds(SQUARE_SIZE, SQUARE_SIZE, square_transform(-63.5, -17.2))
    assert (left, top) == (-63.5, -17.2)
    assert np.isclose(right - left, 512 * PIXEL_SIZE) and np.isclose(top - bottom, 512 * PIXEL_SIZE)
    if S1_DIR.exists():
        t = load_chip("Bolivia_103757").transform
        assert (t.a, -t.e) == (PIXEL_SIZE, PIXEL_SIZE)

def test_pick_model_switches_at_a_quarter_cloud():
    assert [pick_model(s) for s in (0.24, 0.25, 0.9)] == ["fusion", "radar", "radar"]

def test_square_input_hides_the_optical_view_under_clouds_and_no_data():
    square = {"vv": np.full((1, 3), -10.0), "vh": np.full((1, 3), -20.0),
              "green": np.array([[3.0, 3.0, 0.0]]), "nir": np.array([[1.0, 1.0, 0.0]]), "swir": np.array([[1.0, 1.0, 0.0]]),
              "cloud": np.array([[0.0, CLOUD_PROBABILITY, 0.0]])}
    x = square_input(square, "s1+s2", (-10.0, -20.0), (1.0, 1.0))
    assert x.shape == (4, 1, 3)
    assert np.allclose(x[:2], 0.0) and np.allclose(x[2:, 0], [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])

def test_open_water_needs_a_clear_view_and_a_strong_water_index():
    # clear water, water under cloud, weak water index (MNDWI 0.1), land, no data
    square = {"green": np.array([[3.0, 3.0, 11.0, 1.0, 0.0]]), "nir": np.array([[1.0, 1.0, 1.0, 1.0, 0.0]]),
              "swir": np.array([[1.0, 1.0, 9.0, 3.0, 0.0]]), "cloud": np.array([[0.0, CLOUD_PROBABILITY, 0.0, 0.0, 0.0]])}
    assert open_water(square).tolist() == [[True, False, False, False, False]]

@needs_earth_engine
def test_earth_engine_radar_matches_a_bolivia_chip():
    if not S1_DIR.exists():
        pytest.skip("needs the Sen1Floods11 chips")
    init_earth_engine()
    chip = load_chip("Bolivia_129334")
    scenes = find_scenes(chip.transform, date(2018, 2, 15), days=0)
    assert scenes and (scenes["s1_date"], scenes["s1_pass"], scenes["s1_orbit"], scenes["s2_date"]) == \
        ("2018-02-15", "DESCENDING", 156, "2018-02-15")
    square = fetch_square(chip.transform, scenes)
    both = np.isfinite(square["vv"]) & np.isfinite(chip.vv)
    assert both.mean() > 0.9
    assert np.median(np.abs(square["vv"][both] - chip.vv[both])) < 0.1
