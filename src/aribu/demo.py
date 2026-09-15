"""Build the Streamlit demo data.

Both models' water probabilities on every chip of each event, with population and districts on the same
10 m grid. Run once, locally, on the GPU:

    python -m aribu.demo
"""
import json
import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import array_bounds
from tqdm import tqdm
from .dataset import LABEL_WATER, SPLIT_ORDER, load_chip, read_split, valid_mask, water_indices
from .exposure import districts_on_grid, pixel_area_m2, population_on_grid
from .metrics import confusion, metrics_from_counts
from .model import chip_prob, load_checkpoint
from .paths import DATA_DIR, DEMO_DIR, METADATA_PATH, MODELS_DIR, RAW_DIR, RESULTS_DIR
from .report import write_json
from .sources import download, geoboundaries, population_window
from .viz import mndwi_image, radar_image, true_colour

__all__ = ["EVENTS", "MODELS", "CACHE_DIR", "SWAPS", "event_chips", "build_event", "main"]

EVENTS = ("Nigeria", "Sri-Lanka", "Bolivia", "Somalia", "Pakistan")
MODELS = {"radar": ("Radar only", "radar-only.pt", "radar-only (s1)"),
          "fusion": ("Radar + optical", "fusion.pt", "fusion (s1+s2)")}   # key: (label, checkpoint, unet.json name)
CACHE_DIR = DATA_DIR / "cache"
WORLDPOP = "https://data.worldpop.org/GIS/Population/Global_2000_2020/{year}/{iso}/{iso_lower}_ppp_{year}.tif"
# test chip -> validation chip, dropping big no-data patches, low fusion IoU, or a chip that is all water
SWAPS = {"Nigeria_417184": "Nigeria_1095404", "Somalia_166342": "Somalia_12849",
         "Pakistan_664885": "Pakistan_94095", "Pakistan_528249": "Pakistan_210595",
         "Sri-Lanka_534068": "Sri-Lanka_612594", "Sri-Lanka_1049830": "Sri-Lanka_321316",
         "Sri-Lanka_117737": "Sri-Lanka_236030"}

def event_chips(event):
    """Held-out chip ids for one event: the Bolivia split, or the event's test chips with SWAPS applied."""
    split = "bolivia" if event == "Bolivia" else "test"
    return [SWAPS.get(c, c) for c in read_split(split) if c.split("_")[0] == event]

def build_event(event, props, models):
    """Write data/demo/<event>/: chips.npz, districts.geojson and three JPEGs per held-out chip.

    chips.npz holds the event's chips from all splits, for the exposure. `held_out` marks the ones
    the chip viewer shows, stored first, best fusion IoU first. Returns the event's manifest entry.
    """
    year, iso = props["s1_date"][:4], props["ISO_CC"]
    pop_path = download(WORLDPOP.format(year=year, iso=iso, iso_lower=iso.lower()),
                        CACHE_DIR / f"{iso.lower()}_ppp_{year}.tif")
    info, districts_path = geoboundaries(iso, CACHE_DIR)
    districts = gpd.read_file(districts_path)

    out = DEMO_DIR / event
    out.mkdir(parents=True, exist_ok=True)
    keys = ("ids", "held_out", "bounds", "px_area", "label", "valid", "perm", "pop", "district", *(f"prob_{m}" for m in models))
    arrays = {k: [] for k in keys}
    held_out = set(event_chips(event))

    for chip_id in tqdm([c for s in SPLIT_ORDER for c in read_split(s) if c.split("_")[0] == event], desc=event):
        c = load_chip(chip_id)
        valid = valid_mask(c)
        if not valid.any():
            continue
        shape = c.label.shape
        with rasterio.open(RAW_DIR / "JRCWaterHand" / f"{chip_id}_JRCWaterHand.tif") as src:
            arrays["perm"].append(src.read(1) == 1)
        b = array_bounds(*shape, c.transform)           # pyright: ignore[reportCallIssue]
        pop, pop_transform = population_window(pop_path, b)
        arrays["pop"].append(population_on_grid(pop, pop_transform, c.transform, shape))
        arrays["district"].append(districts_on_grid(districts.geometry, c.transform, shape)
                                  .astype(np.int16)) # pyright: ignore[reportOptionalMemberAccess]
        arrays["label"].append(c.label.astype(np.int8))
        arrays["valid"].append(valid)
        for name, (model, ckpt) in models.items():
            norm = ckpt["normalisation"]
            p = chip_prob(model, ckpt["arm"], chip_id, norm["mean"], norm["std"])
            arrays[f"prob_{name}"].append(np.round(p * 255).astype(np.uint8))

        if chip_id in held_out:
            Image.fromarray(radar_image(c.vh)).save(out / f"{chip_id}_vh.jpg", quality=90)
            Image.fromarray(mndwi_image(water_indices(chip_id)[1])).save(out / f"{chip_id}_mndwi.jpg", quality=90)
            Image.fromarray((true_colour(chip_id) * 255).astype(np.uint8)).save(out / f"{chip_id}_rgb.jpg", quality=90)
        arrays["ids"].append(chip_id)
        arrays["held_out"].append(chip_id in held_out)
        arrays["bounds"].append(b)
        arrays["px_area"].append(pixel_area_m2(c.transform, shape[0]))

    stacked = {k: np.stack(v) for k, v in arrays.items()}
    iou = [metrics_from_counts(confusion(prob > models["fusion"][1]["threshold"] * 255, label == LABEL_WATER, valid))["iou"]
           for prob, label, valid in zip(stacked["prob_fusion"], stacked["label"], stacked["valid"])]
    order = np.lexsort((-np.array(iou), ~stacked["held_out"]))  # held-out first; NaN (no water in label or prediction) sorts last
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
