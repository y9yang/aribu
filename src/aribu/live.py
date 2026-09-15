"""Satellite images for one 5 km square of a recent flood, from Google Earth Engine.

The images are prepared as in Sen1Floods11, which was exported from Earth Engine: Sentinel-1 GRD in dB and
Sentinel-2 Level-1C, on a 10 m grid in latitude/longitude.
"""
import json
import os
from datetime import datetime, time, timezone
from pathlib import Path
import ee
import numpy as np
from rasterio.transform import Affine, array_bounds
from .dataset import normalised_indices, radar_input, stack_arm

__all__ = ["COLLECTIONS", "CLOUD_PROBABILITY", "MNDWI_WATER", "RADAR_ONLY_FROM", "PIXEL_SIZE", "SQUARE_SIZE",
           "init_earth_engine", "square_transform", "find_scenes", "fetch_square", "fetch_water_before",
           "no_optical_data", "hidden_optical", "open_water", "pick_model", "square_input"]

COLLECTIONS = {"s1": "COPERNICUS/S1_GRD", "s2": "COPERNICUS/S2_HARMONIZED",
               "clouds": "COPERNICUS/S2_CLOUD_PROBABILITY", "water": "JRC/GSW1_4/GlobalSurfaceWater"}
CLOUD_PROBABILITY = 50      # a pixel counts as cloud from this probability (0 to 100)
MNDWI_WATER = 0.2           # open water above this MNDWI: on EMSR865, pixels from 0 to 0.2 were mostly dry in the flood-day radar
RADAR_ONLY_FROM = 0.25      # cloud share from which we pick radar only (Bolivia cloud sweep)
PIXEL_SIZE = 8.983152841195215e-05      # degrees, transform.a and -transform.e of every S1Hand chip
SQUARE_SIZE = 512           # pixels per side, as a chip
S2_BANDS = ("B2", "B3", "B4", "B8", "B11")
NODATA = -9999              # marks radar pixels outside the swath in a download

def init_earth_engine(project=None):
    """Log in with the service account key from EE_KEY_JSON (its text) or EE_KEY_FILE (a path), else as yourself.

    `project` defaults to the Cloud project of the service account.
    """
    key_json, key_file = os.environ.get("EE_KEY_JSON"), os.environ.get("EE_KEY_FILE")
    if not (key_json or key_file):
        ee.Initialize(project=project)
        return
    key = key_json or Path(key_file).read_text()     # pyright: ignore[reportArgumentType]
    info = json.loads(key)
    ee.Initialize(ee.ServiceAccountCredentials(info["client_email"], key_data=key), project=project or info["project_id"])

def square_transform(left, top):
    """Transform of a SQUARE_SIZE × SQUARE_SIZE square with the Sen1Floods11 pixel size, top-left corner at (left, top)."""
    return Affine(PIXEL_SIZE, 0, left, 0, -PIXEL_SIZE, top)

def _region(transform):
    return ee.Geometry.Rectangle(list(array_bounds(SQUARE_SIZE, SQUARE_SIZE, transform)), "EPSG:4326", False)

def _grid(transform):
    """An Earth Engine pixel grid identical to a rasterio transform."""
    return {"dimensions": {"width": SQUARE_SIZE, "height": SQUARE_SIZE},
            "affineTransform": {"scaleX": transform.a, "shearX": transform.b, "translateX": transform.c,
                                "shearY": transform.d, "scaleY": transform.e, "translateY": transform.f},
            "crsCode": "EPSG:4326"}

def _pixels(image, transform):
    """The bands of `image` on the square's grid, as a NumPy structured array of float32."""
    return ee.data.computePixels({"expression": image.toFloat(), "fileFormat": "NUMPY_NDARRAY", "grid": _grid(transform)})

def _radar(region, start, end):
    """Sentinel-1 scenes in IW mode with VV and VH over `region`, from `start` to `end` (ee.Date)."""
    return (ee.ImageCollection(COLLECTIONS["s1"]).filterBounds(region).filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))

def _optical(region, start, end):
    """Sentinel-2 scenes over `region`, from `start` to `end`, with their cloud probability as a band."""
    s2 = ee.ImageCollection(COLLECTIONS["s2"]).filterBounds(region).filterDate(start, end)
    clouds = ee.ImageCollection(COLLECTIONS["clouds"]).filterBounds(region).filterDate(start, end)
    joined = ee.Join.saveFirst("clouds").apply(s2, clouds, ee.Filter.equals(leftField="system:index", rightField="system:index"))
    return ee.ImageCollection(joined).map(lambda im: ee.Image(im).addBands(ee.Image(im.get("clouds")).select("probability")))

def _optical_day(transform, day):
    """The Sentinel-2 bands and cloud probability over the square on `day` (an ISO date), 0 without data."""
    start = ee.Date(day)
    return _optical(_region(transform), start, start.advance(1, "day")).mosaic().select([*S2_BANDS, "probability"]).unmask(0)

def _optical_bands(arr):
    """green, nir, swir, rgb, cloud and perm from a download, as in `fetch_square`."""
    def band(name):
        return np.asarray(arr[name], np.float32)

    return {"green": band("B3"), "nir": band("B8"), "swir": band("B11"), "rgb": np.stack([band("B4"), band("B3"), band("B2")]),
            "cloud": band("probability"), "perm": band("perm") == 1}

def _day(ms):
    """The UTC day of a timestamp in milliseconds."""
    return datetime.fromtimestamp(ms / 1000, timezone.utc).date()

def _permanent_water():
    """Band perm: 1 on permanent water, else 0."""
    # water all 12 months of the year: closer to the chips' JRCWaterHand (IoU 0.89) than transition == 1 (0.87)
    return ee.Image(COLLECTIONS["water"]).select("seasonality").eq(12).rename("perm").unmask(0)

def find_scenes(transform, day, days=3):
    """The Sentinel-1 scene nearest to `day` (a datetime.date) over the square, and the Sentinel-2 scene nearest to that one.

    Both at most `days` away. Returns None without a Sentinel-1 scene, else a dict: s1_date, s1_pass and s1_orbit;
    s2_date (None without a Sentinel-2 scene); and cloud_share, the share of the square without a clear optical view.
    """
    region, start = _region(transform), ee.Date(day.isoformat())
    found = ee.Dictionary({
        "s1": _radar(region, start.advance(-days, "day"), start.advance(days + 1, "day")).reduceColumns(
            ee.Reducer.toList(3), ["system:time_start", "orbitProperties_pass", "relativeOrbitNumber_start"]).get("list"),
        "s2": _optical(region, start.advance(-2 * days, "day"), start.advance(2 * days + 1, "day")).aggregate_array("system:time_start"),
    }).getInfo()
    if not found["s1"]:
        return None
    noon = datetime.combine(day, time(12), timezone.utc).timestamp() * 1000
    t1, s1_pass, s1_orbit = min(found["s1"], key=lambda s: abs(s[0] - noon))
    s1_day = _day(t1)
    scenes = {"s1_date": s1_day.isoformat(), "s1_pass": s1_pass, "s1_orbit": int(s1_orbit), "s2_date": None, "cloud_share": 1.0}
    near = [t for t in found["s2"] if abs((_day(t) - s1_day).days) <= days]
    if near:
        scenes["s2_date"] = _day(min(near, key=lambda t: abs(t - t1))).isoformat()
        s2_day = ee.Date(scenes["s2_date"])
        hidden = (_optical(region, s2_day, s2_day.advance(1, "day")).mosaic().select("probability")
                  .gte(CLOUD_PROBABILITY).unmask(1))
        scenes["cloud_share"] = hidden.reduceRegion(ee.Reducer.mean(), region, crs="EPSG:4326",
                                                    crsTransform=list(transform)[:6]).get("probability").getInfo()
    return scenes

def fetch_square(transform, scenes):
    """The square's pixels on its SQUARE_SIZE × SQUARE_SIZE grid, for the scenes from `find_scenes`.

    Returns a dict of float32 arrays (H, W): vv and vh in dB, NaN outside the radar swath; green, nir and swir
    (B3, B8, B11) and rgb (3, H, W), in Sentinel-2 digital numbers, 0 without data; cloud, the cloud probability
    from 0 to 100; and perm, True on permanent water.
    """
    s1_day = ee.Date(scenes["s1_date"])
    radar = (_radar(_region(transform), s1_day, s1_day.advance(1, "day"))
             .filter(ee.Filter.eq("orbitProperties_pass", scenes["s1_pass"]))
             .filter(ee.Filter.eq("relativeOrbitNumber_start", scenes["s1_orbit"]))
             .mosaic()                  # a square can straddle two scenes of the same pass
             .select(["VV", "VH"]).unmask(NODATA))
    if scenes["s2_date"]:
        optical = _optical_day(transform, scenes["s2_date"])
    else:
        optical = ee.Image.constant([0] * (len(S2_BANDS) + 1)).rename([*S2_BANDS, "probability"])
    arr = _pixels(ee.Image.cat([radar, optical, _permanent_water()]), transform)
    vv, vh = (np.asarray(arr[b], np.float32) for b in ("VV", "VH"))
    return {"vv": np.where(vv == NODATA, np.nan, vv), "vh": np.where(vh == NODATA, np.nan, vh), **_optical_bands(arr)}

def fetch_water_before(transform, day):
    """Water in the square before a flood, as a bool array (H, W), from the Sentinel-2 image of `day` (a datetime.date).

    Open water in that image, and permanent water where the image gives no clear view.
    """
    square = _optical_bands(_pixels(ee.Image.cat([_optical_day(transform, day.isoformat()), _permanent_water()]), transform))
    return open_water(square) | (square["perm"] & hidden_optical(square))

def no_optical_data(square):
    """True where Sentinel-2 has no data over the square."""
    return (square["green"] == 0) & (square["nir"] == 0) & (square["swir"] == 0)

def hidden_optical(square):
    """True where Sentinel-2 gives no clear view of the square: no data, or cloud."""
    return no_optical_data(square) | (square["cloud"] >= CLOUD_PROBABILITY)

def open_water(square):
    """True where the square's Sentinel-2 image shows open water: MNDWI above MNDWI_WATER, in a clear view."""
    mndwi = normalised_indices(square["green"], square["nir"], square["swir"])[1]
    return (mndwi > MNDWI_WATER) & ~hidden_optical(square)

def pick_model(cloud_share):
    """"radar" from RADAR_ONLY_FROM onward, else "fusion"."""
    return "radar" if cloud_share >= RADAR_ONLY_FROM else "fusion"

def square_input(square, arm, mean, std):
    """Model input (C, H, W) for `arm` from `fetch_square`, prepared as for the chips.

    Both water indices are 0 where the optical view is hidden, as in our cloud sweep.
    """
    indices = normalised_indices(square["green"], square["nir"], square["swir"])
    indices[:, hidden_optical(square)] = 0.0
    return stack_arm(radar_input(square["vv"], square["vh"], mean, std), indices, arm)
