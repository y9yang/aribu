"""Downloads and raster windows, shared by the scripts that build the app data and by the app."""
import json
import math
import urllib.request
import rasterio
from rasterio.windows import Window

__all__ = ["GEOBOUNDARIES", "download", "geoboundaries", "population_window"]

GEOBOUNDARIES = "https://www.geoboundaries.org/api/current/gbOpen/{iso}/ADM2/"

def download(url, path):
    """Fetch `url` to `path` once."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(url, part)
        part.replace(path)
    return path

def geoboundaries(iso, cache_dir):
    """geoBoundaries ADM2 districts of one country, downloaded once. Returns (metadata, path of the GeoJSON)."""
    info = json.loads(download(GEOBOUNDARIES.format(iso=iso), cache_dir / f"geoBoundaries-{iso}-ADM2.json").read_text(encoding="utf-8"))
    return info, download(info["gjDownloadURL"], cache_dir / f"geoBoundaries-{iso}-ADM2.geojson")

def population_window(path, bounds, pad=0.002):
    """People per cell around `bounds`, padded, as (array, transform), from a latitude/longitude raster."""
    with rasterio.open(path) as src:
        left, bottom, right, top = bounds
        col0, row0 = ~src.transform * (left - pad, top + pad)
        col1, row1 = ~src.transform * (right + pad, bottom - pad)
        window = Window(math.floor(col0), math.floor(row0),          # pyright: ignore[reportCallIssue]
                        math.ceil(col1) - math.floor(col0), math.ceil(row1) - math.floor(row0))
        pop = src.read(1, window=window, boundless=True, fill_value=src.nodata)
        return pop, src.window_transform(window)
