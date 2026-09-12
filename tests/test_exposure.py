import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import box
from aribu.exposure import districts_on_grid, pixel_area_m2, population_on_grid, tally

def test_population_total_is_preserved():
    pop = np.arange(9, dtype=np.float32).reshape(3, 3)
    fine = population_on_grid(pop, from_origin(10, 1, 0.001, 0.001), from_origin(10, 1, 0.0001, 0.0001), (30, 30))
    assert np.isclose(fine.sum(), pop.sum(), rtol=1e-4)
    assert np.allclose(fine[:10, 10:20], pop[0, 1] / 100)

def test_population_no_data_counts_as_nobody():
    pop = np.array([[5.0, -99999.0]], np.float32)
    fine = population_on_grid(pop, from_origin(0, 0.001, 0.001, 0.001), from_origin(0, 0.001, 0.0005, 0.0005), (2, 4))
    assert np.isclose(fine.sum(), 5.0)

def test_districts_split_the_grid():
    ids = districts_on_grid([box(0, 0, 2, 4), box(2, 0, 4, 4)], from_origin(0, 4, 1, 1), (4, 4))
    assert (ids[:, :2] == 1).all() and (ids[:, 2:] == 2).all()

def test_tally_sums_per_district():
    mask = np.array([[True, True], [True, False]])
    district = np.array([[1, 2], [0, 1]])
    pop = np.array([[10.0, 20.0], [30.0, 40.0]])
    people, pixels = tally(mask, district, pop, 2)
    assert people.tolist() == [30.0, 10.0, 20.0]
    assert pixels.tolist() == [1, 1, 1]

def test_pixel_area_shrinks_with_latitude():
    equator = pixel_area_m2(from_origin(0, 0.5 * 8.983e-05, 8.983e-05, 8.983e-05), 1)
    sixty = pixel_area_m2(from_origin(0, 60 + 0.5 * 8.983e-05, 8.983e-05, 8.983e-05), 1)
    assert np.isclose(equator, 100, rtol=0.01)
    assert np.isclose(sixty / equator, 0.5, rtol=0.01)
