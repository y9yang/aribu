"""Build the live page's data for a few Copernicus EMS Rapid Mapping flood activations.

For each activation, data/live/<code>/ gets the analysts' flood outline, the 5 km squares to pick from,
the water there before the flood, population and districts. Run once, locally, with Earth Engine access:

    python -m aribu.activations
"""
import json
import math
import zipfile
from datetime import datetime
from pathlib import Path
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.merge import merge
from shapely.geometry import box, mapping
from .exposure import districts_on_grid
from .live import PIXEL_SIZE, SQUARE_SIZE, fetch_water_before, init_earth_engine, square_transform
from .paths import DATA_DIR, LIVE_DIR
from .report import write_json
from .sources import download, geoboundaries

__all__ = ["ACTIVATIONS", "MIN_FLOOD_SHARE", "square_grid", "flood_share", "ghs_pop_tiles", "build_activation", "main"]

# code: (area of interest, ISO country code). Each area was mapped from a Sentinel-1 image, on a flat floodplain.
ACTIVATIONS = {"EMSR838": (2, "PAK"), "EMSR857": (5, "MOZ"), "EMSR865": (1, "COL")}
AREA_NAMES = {"Monterìa": "Montería"}      # spellings to correct in the EMS data
MIN_FLOOD_SHARE = 0.05      # squares with less of their area inside the outline are left out
CACHE_DIR = DATA_DIR / "cache"
EMS_API = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations/?code={code}"
EMS_PAGE = "https://mapping.emergency.copernicus.eu/activations/{code}/"
GHS_POP = ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E2025_GLOBE_R2023A_4326_3ss/"
           "V1-0/tiles/GHS_POP_E2025_GLOBE_R2023A_4326_3ss_V1_0_R{row}_C{col}.zip")
# The outline file only draws the map, so it is simplified to about 30 m. The analysts' map that we
# compare with comes from the full outline, as pixels per square in analysts.npz.
OUTLINE_TOLERANCE = 0.0003

def square_grid(area):
    """5 km squares over `area` (a shapely geometry in latitude/longitude), on a grid from its north-west corner.

    Returns [(id, (left, bottom, right, top))] for the squares that overlap the area, with columns A, B, ...
    from the west and rows 1, 2, ... from the north.
    """
    size = SQUARE_SIZE * PIXEL_SIZE
    left, bottom, right, top = area.bounds
    n_cols, n_rows = math.ceil((right - left) / size), math.ceil((top - bottom) / size)
    if n_cols > 26:
        raise ValueError(f"the area is {n_cols} squares wide, more than the letters A to Z")
    squares = []
    for row in range(n_rows):
        for col in range(n_cols):
            bounds = (left + col * size, top - (row + 1) * size, left + (col + 1) * size, top - row * size)
            if box(*bounds).intersection(area).area > 0:
                squares.append((f"{chr(ord('A') + col)}{row + 1}", bounds))
    return squares

def flood_share(outline, bounds):
    """Share of the square with `bounds` inside `outline`."""
    square = box(*bounds)
    return outline.intersection(square).area / square.area

def ghs_pop_tiles(bounds, pad=0.01):
    """(row, column) of the GHS-POP tiles over `bounds`: 10° tiles, row 1 from 89.1° N, column 1 from 180° W."""
    left, bottom, right, top = bounds
    rows = range(math.floor((89.1 - top - pad) / 10) + 1, math.floor((89.1 - bottom + pad) / 10) + 2)
    cols = range(math.floor((left - pad + 180) / 10) + 1, math.floor((right + pad + 180) / 10) + 2)
    return [(r, c) for r in rows for c in cols]

def _ems_member(product, ext, layer):
    """One layer, such as observedEventA, of a Copernicus EMS product zip, as a path that geopandas reads."""
    with zipfile.ZipFile(product) as z:
        member = next(n for n in z.namelist() if n.endswith(ext) and n.split("_")[-2] == layer)
    return f"zip://{product}!{member}"

def _ems_layer(product, name):
    """One vector layer, such as observedEventA, from a Copernicus EMS product zip."""
    return gpd.read_file(_ems_member(product, ".json", name)).to_crs("EPSG:4326").geometry.union_all()

def _pre_event_image(product):
    """Acquisition time of the Sentinel-2 image from before the flood, from the source table of a Copernicus EMS product zip."""
    sources = gpd.read_file(_ems_member(product, ".dbf", "source"))
    row = sources[(sources["source_nam"] == "Sentinel-2") & (sources["eventphase"] == "Pre-event")].iloc[0]
    return datetime.strptime(row["src_date"] + row["source_tm"], "%d/%m/%YT%H:%M:%SZ")

def build_activation(code, aoi_number, iso):
    """Write data/live/<code>/: outline.geojson, squares.geojson, analysts.npz, water_before.npz, population.tif, districts.geojson.

    Returns the activation's entry for activations.json.
    """
    cache, out = CACHE_DIR / "activations" / code, LIVE_DIR / code
    out.mkdir(parents=True, exist_ok=True)
    activation = json.loads(download(EMS_API.format(code=code), cache / f"{code}.json").read_text(encoding="utf-8"))["results"][0]
    aoi = next(a for a in activation["aois"] if a["number"] == aoi_number)
    product = min((p for p in aoi["products"] if p["type"] == "DEL"), key=lambda p: p["monitoringNumber"])
    zip_path = download(product["downloadPath"], cache / Path(product["downloadPath"]).name)

    # the squares with enough flood in them, the analysts' map on each, and the outline around them for the map
    area = _ems_layer(zip_path, "areaOfInterestA")
    flood = _ems_layer(zip_path, "observedEventA").intersection(area)
    squares = [(sid, b, share) for sid, b in square_grid(area) if (share := flood_share(flood, b)) >= MIN_FLOOD_SHARE]
    np.savez_compressed(out / "analysts.npz", **{
        sid: districts_on_grid([flood.intersection(box(*b))], square_transform(b[0], b[3]), (SQUARE_SIZE, SQUARE_SIZE)) > 0 # pyright: ignore[reportArgumentType, reportOptionalOperand]
        for sid, b, _ in squares})          # pyright: ignore[reportArgumentType]
    # the analysts mapped only the flood, so the page adds the water that was there before it, seen in their Sentinel-2 image
    pre_event = _pre_event_image(zip_path)
    np.savez_compressed(out / "water_before.npz", **{
        sid: fetch_water_before(square_transform(b[0], b[3]), pre_event.date()) for sid, b, _ in squares})
    write_json(out / "squares.geojson", {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": mapping(box(*b)),
         "properties": {"id": sid, "bounds": list(b), "flood_share": round(share, 3)}}
        for sid, b, share in squares]})
    corners = np.array([b for _, b, _ in squares])
    extent = (*(corners[:, :2].min(0) - 0.01), *(corners[:, 2:].max(0) + 0.01))
    outline = flood.intersection(box(*extent)).simplify(OUTLINE_TOLERANCE)
    gpd.GeoDataFrame(geometry=[outline], crs="EPSG:4326").to_file(out / "outline.geojson", driver="GeoJSON",
                                                                  COORDINATE_PRECISION=5)

    # people per cell of 3 arcseconds, cropped to the squares
    tiles = [download(GHS_POP.format(row=r, col=c), cache / Path(GHS_POP.format(row=r, col=c)).name)
             for r, c in ghs_pop_tiles(extent)]
    pop, transform = merge([f"zip://{t}!{t.stem}.tif" for t in tiles], bounds=extent, dtype="float32")
    with rasterio.open(out / "population.tif", "w", driver="GTiff", height=pop.shape[1], width=pop.shape[2], count=1,
                       dtype="float32", crs="EPSG:4326", transform=transform, compress="deflate", predictor=3) as dst:
        dst.write(pop)

    # districts, clipped to the squares
    info, districts_path = geoboundaries(iso, CACHE_DIR)
    districts = gpd.read_file(districts_path)
    districts = districts[districts.intersects(box(*extent))]
    gpd.GeoDataFrame({"id": np.arange(1, len(districts) + 1),
                      "name": districts["shapeName"].map(lambda s: s.title() if s.isupper() else s).to_numpy()},  # pyright: ignore[reportAttributeAccessIssue]
                     geometry=districts.geometry.intersection(box(*extent)).simplify(0.0005, preserve_topology=True).to_numpy(),
                     crs=districts.crs).to_file(out / "districts.geojson", driver="GeoJSON", COORDINATE_PRECISION=5)

    image = product["images"][0]
    flooded_km2 = gpd.GeoSeries([flood.intersection(box(*extent))], crs="EPSG:4326").to_crs("EPSG:6933").area.iloc[0] / 1e6
    print(f"{code} {aoi['name']}: {len(squares)} squares, {flooded_km2:,.0f} km² of flood around them, "
          f"{np.clip(pop, 0, None).sum():,.0f} people in the crop, {len(districts)} districts")
    return {"name": " ".join(activation["name"].split()), "country": activation["countries"][0]["name"],
            "area": AREA_NAMES.get(aoi["name"], aoi["name"]),
            "event_date": activation["eventTime"][:10], "image": {"sensor": image["sensorName"], "time": image["acquisitionTime"][:16]},
            "pre_event_image": {"sensor": "Sentinel-2", "time": pre_event.isoformat(timespec="minutes")},
            "link": EMS_PAGE.format(code=code), "boundaries_license": info["boundaryLicense"],
            "extent": [round(v, 6) for v in extent]}

def main(activations=ACTIVATIONS):
    init_earth_engine()
    entries = {code: build_activation(code, aoi, iso) for code, (aoi, iso) in activations.items()}
    write_json(LIVE_DIR / "activations.json", entries)
    print(f"wrote {LIVE_DIR}")

if __name__ == "__main__":
    main()
