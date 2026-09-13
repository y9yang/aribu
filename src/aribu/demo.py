"""Build the Streamlit demo data.

Both models' water probabilities on held-out chips, with population and districts on the same
10 m grid. Run once, locally, on the GPU:

    python -m aribu.demo
"""
import json
import math
import urllib.request
import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import array_bounds
from rasterio.windows import Window
from tqdm import tqdm
from .dataset import LABEL_WATER, load_chip, read_split, valid_mask
from .exposure import districts_on_grid, pixel_area_m2, population_on_grid
from .metrics import confusion, metrics_from_counts
from .model import chip_prob, load_checkpoint
from .paths import DATA_DIR, DEMO_DIR, METADATA_PATH, MODELS_DIR, RAW_DIR, RESULTS_DIR
from .report import write_json
from .viz import percentile_limits, true_colour

__all__ = ["EVENTS", "MODELS", "CACHE_DIR", "event_chips", "build_event", "main"]

EVENTS = ("Nigeria", "Sri-Lanka", "Bolivia", "Somalia", "Pakistan")
MODELS = {"radar": ("Radar only", "radar-only.pt", "radar-only (s1)"),
          "fusion": ("Radar + optical", "fusion.pt", "fusion (s1+s2)")}   # key: (label, checkpoint, unet.json name)
CACHE_DIR = DATA_DIR / "cache"
WORLDPOP = "https://data.worldpop.org/GIS/Population/Global_2000_2020/{year}/{iso}/{iso_lower}_ppp_{year}.tif"
GEOBOUNDARIES = "https://www.geoboundaries.org/api/current/gbOpen/{iso}/ADM2/"

def _download(url, path):
    """Fetch `url` to `path` once."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(url, part)
        part.replace(path)
    return path

def event_chips(event):
    """Held-out chip ids for one event: the Bolivia split, or the event's test chips."""
    split = "bolivia" if event == "Bolivia" else "test"
    return [c for c in read_split(split) if c.split("_")[0] == event]

def _population_window(path, bounds, pad=0.002):
    """WorldPop counts around `bounds`, padded, as (array, transform)."""
    with rasterio.open(path) as src:
        left, bottom, right, top = bounds
        col0, row0 = ~src.transform * (left - pad, top + pad)
        col1, row1 = ~src.transform * (right + pad, bottom - pad)
        window = Window(math.floor(col0), math.floor(row0),          # pyright: ignore[reportCallIssue]
                        math.ceil(col1) - math.floor(col0), math.ceil(row1) - math.floor(row0))
        pop = src.read(1, window=window, boundless=True, fill_value=src.nodata)
        return pop, src.window_transform(window)

def _display_vh(vh):
    """VH backscatter as greyscale uint8, no-data light grey."""
    lo, hi = percentile_limits(vh)
    grey = np.clip((vh - lo) / (hi - lo), 0, 1) * 255
    return np.where(np.isfinite(vh), grey, 217).astype(np.uint8)

def build_event(event, props, models):
    """Write data/demo/<event>/: chips.npz, districts.geojson and two JPEGs per chip.

    Chips are stored best fusion IoU first. Returns the event's manifest entry.
    """
    year, iso = props["s1_date"][:4], props["ISO_CC"]
    pop_path = _download(WORLDPOP.format(year=year, iso=iso, iso_lower=iso.lower()),
                         CACHE_DIR / f"{iso.lower()}_ppp_{year}.tif")
    info = json.loads(_download(GEOBOUNDARIES.format(iso=iso), CACHE_DIR / f"geoBoundaries-{iso}-ADM2.json").read_text())
    districts = gpd.read_file(_download(info["gjDownloadURL"], CACHE_DIR / f"geoBoundaries-{iso}-ADM2.geojson"))

    out = DEMO_DIR / event
    out.mkdir(parents=True, exist_ok=True)
    keys = ("ids", "bounds", "px_area", "label", "valid", "perm", "pop", "district", *(f"prob_{m}" for m in models))
    arrays = {k: [] for k in keys}

    for chip_id in tqdm(event_chips(event), desc=event):
        c = load_chip(chip_id)
        valid = valid_mask(c)
        if not valid.any():
            continue
        shape = c.label.shape
        with rasterio.open(RAW_DIR / "JRCWaterHand" / f"{chip_id}_JRCWaterHand.tif") as src:
            arrays["perm"].append(src.read(1) == 1)
        b = array_bounds(*shape, c.transform)           # pyright: ignore[reportCallIssue]
        pop, pop_transform = _population_window(pop_path, b)
        arrays["pop"].append(population_on_grid(pop, pop_transform, c.transform, shape))
        arrays["district"].append(districts_on_grid(districts.geometry, c.transform, shape)
                                  .astype(np.int16)) # pyright: ignore[reportOptionalMemberAccess]
        arrays["label"].append(c.label.astype(np.int8))
        arrays["valid"].append(valid)
        for name, (model, ckpt) in models.items():
            norm = ckpt["normalisation"]
            p = chip_prob(model, ckpt["arm"], chip_id, norm["mean"], norm["std"])
            arrays[f"prob_{name}"].append(np.round(p * 255).astype(np.uint8))

        Image.fromarray(_display_vh(c.vh)).save(out / f"{chip_id}_vh.jpg", quality=90)
        Image.fromarray((true_colour(chip_id) * 255).astype(np.uint8)).save(out / f"{chip_id}_rgb.jpg", quality=90)
        arrays["ids"].append(chip_id)
        arrays["bounds"].append(b)
        arrays["px_area"].append(pixel_area_m2(c.transform, shape[0]))

    stacked = {k: np.stack(v) for k, v in arrays.items()}
    iou = [metrics_from_counts(confusion(prob > models["fusion"][1]["threshold"] * 255, label == LABEL_WATER, valid))["iou"]
           for prob, label, valid in zip(stacked["prob_fusion"], stacked["label"], stacked["valid"])]
    order = np.argsort(-np.array(iou))          # NaN (no water in label or prediction) sorts last
    np.savez_compressed(out / "chips.npz", **{k: v[order] for k, v in stacked.items()})  # pyright: ignore[reportArgumentType]

    present = np.unique(stacked["district"])
    present = present[present > 0]
    shown = districts.iloc[present - 1]
    gpd.GeoDataFrame({"id": present, "name": shown["shapeName"]
                      .map(lambda s: s.title() if s.isupper() else s).to_numpy(),  # pyright: ignore[reportAttributeAccessIssue]
                      "area_km2": (shown.to_crs("EPSG:6933").area / 1e6).to_numpy()},
                     geometry=shown.geometry.simplify(0.0005, preserve_topology=True).to_numpy(),
                     crs=districts.crs).to_file(out / "districts.geojson", driver="GeoJSON")

    return {"country": event.replace("-", " "), "s1_date": props["s1_date"].replace("/", "-"),
            "split": "bolivia" if event == "Bolivia" else "test",
            "worldpop_year": int(year), "boundaries_license": info["boundaryLicense"]}

def main(events=EVENTS):
    features = json.loads(METADATA_PATH.read_text())["features"]
    meta = {f["properties"]["location"]: f["properties"] for f in features}
    models = {name: load_checkpoint(MODELS_DIR / ckpt) for name, (_, ckpt, _) in MODELS.items()}
    unet = json.loads((RESULTS_DIR / "unet.json").read_text())

    manifest = {
        "models": {name: {"label": label, "threshold": models[name][1]["threshold"],
                          "micro_iou": {s: unet["splits"][s][key]["micro"]["iou"] for s in ("test", "bolivia")}}
                   for name, (label, _, key) in MODELS.items()},
        "events": {e: build_event(e, meta[e], models) for e in events},
    }
    write_json(DEMO_DIR / "manifest.json", manifest)
    print(f"wrote {DEMO_DIR}")

if __name__ == "__main__":
    main()
