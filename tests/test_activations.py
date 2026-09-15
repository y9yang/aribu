import numpy as np
from shapely.geometry import Polygon, box
from aribu.activations import flood_share, ghs_pop_tiles, square_grid
from aribu.live import PIXEL_SIZE, SQUARE_SIZE

SIZE = SQUARE_SIZE * PIXEL_SIZE

def test_square_ids_and_bounds_start_in_the_north_west():
    squares = dict(square_grid(box(10, 20 - 1.5 * SIZE, 10 + 2.5 * SIZE, 20)))
    assert list(squares) == ["A1", "B1", "C1", "A2", "B2", "C2"]
    assert np.allclose(squares["B2"], (10 + SIZE, 20 - 2 * SIZE, 10 + 2 * SIZE, 20 - SIZE))

def test_squares_outside_the_area_are_left_out():
    l_shape = Polygon([(0, 0), (0, 2 * SIZE), (SIZE, 2 * SIZE), (SIZE, SIZE), (2 * SIZE, SIZE), (2 * SIZE, 0)])
    assert [sid for sid, _ in square_grid(l_shape)] == ["A1", "A2", "B2"]

def test_flood_share_of_a_half_covered_square():
    assert np.isclose(flood_share(box(0, 0, SIZE / 2, SIZE), (0, 0, SIZE, SIZE)), 0.5)

def test_ghs_pop_tiles_across_a_row_boundary():
    assert ghs_pop_tiles((-76.2, 8.0, -75.6, 9.4)) == [(8, 11), (9, 11)]
    assert ghs_pop_tiles((32.1, -24.2, 32.6, -23.8)) == [(12, 22)]
