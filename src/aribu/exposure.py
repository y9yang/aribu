import numpy as np
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject

__all__ = ["EARTH_RADIUS_M", "pixel_area_m2", "population_on_grid", "districts_on_grid", "tally"]

EARTH_RADIUS_M = 6_371_008.8

def pixel_area_m2(transform, height):
    """Ground area of one pixel."""
    lat = np.radians(transform.f + transform.e * height / 2)
    return (EARTH_RADIUS_M * np.radians(abs(transform.e))) * (EARTH_RADIUS_M * np.radians(transform.a) * np.cos(lat))

def population_on_grid(pop, src_transform, dst_transform, shape, crs="EPSG:4326"):
    """People per pixel on a finer lat/lon grid, from a coarser raster of people per cell.

    Counts become densities, are copied across by nearest neighbour, and become counts again,
    so the total is preserved. Negative or non-finite cells (no data) count as nobody.
    """
    density = np.where(np.isfinite(pop) & (pop > 0), pop, 0).astype(np.float32) / abs(src_transform.a * src_transform.e)
    out = np.zeros(shape, np.float32)
    reproject(density, out, src_transform=src_transform, src_crs=crs,
              dst_transform=dst_transform, dst_crs=crs, resampling=Resampling.nearest)
    return out * abs(dst_transform.a * dst_transform.e)

def districts_on_grid(geometries, transform, shape):
    """District id per pixel: `i + 1` for the i-th geometry, 0 outside all of them."""
    return rasterize(((g, i + 1) for i, g in enumerate(geometries)),
                     out_shape=shape, transform=transform, fill=0, dtype="int32")

def tally(mask, district, pop, n_districts):
    """People and pixels inside `mask`, per district id.

    Returns (people, pixels), both of length n_districts + 1, indexed by id (0 = no district).
    """
    d = district[mask]
    return (np.bincount(d, weights=pop[mask], minlength=n_districts + 1),
            np.bincount(d, minlength=n_districts + 1))
