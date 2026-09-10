from typing import NamedTuple
from zlib import crc32
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from scipy.ndimage import gaussian_filter
from .paths import S1_DIR, S2_DIR, LABEL_DIR, SPLIT_DIR

__all__ = [
    "LABEL_NODATA", "LABEL_LAND", "LABEL_WATER",
    "CLIP_LO", "CLIP_HI", "VV_BAND", "VH_BAND", "IGNORE_INDEX",
    "S2_BANDS", "SPLIT_FILES", "SPLIT_ORDER", "ARMS",
    "Chip", "load_chip", "valid_mask", "preprocess", "read_split", "load_splits",
    "read_s2", "water_indices", "chip_input", "cloud_mask", "occluded_input",
]

LABEL_NODATA, LABEL_LAND, LABEL_WATER = -1, 0, 1
CLIP_LO, CLIP_HI = -50.0, 1.0
VV_BAND, VH_BAND = 1, 2
IGNORE_INDEX = 255
# 1-based band index
S2_BANDS = {name: i + 1 for i, name in enumerate(
    ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"))}
SPLIT_FILES = {
    "train": "flood_train_data.csv",
    "val": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
    "bolivia": "flood_bolivia_data.csv",
}
SPLIT_ORDER = tuple(SPLIT_FILES)
ARMS = ("s1", "s1+s2", "s2")     # using data from which sensors to train and evaluate a model

class Chip(NamedTuple):
    """the data contained in a chip."""
    id: str
    loc: str            # the location/country name, parsed from the chip id with exception: Mekong -> Cambodia
    vv: np.ndarray          # float32 (512, 512), dB, NaN where no measurement
    vh: np.ndarray          # float32 (512, 512), dB, NaN where no measurement
    label: np.ndarray           # int16   (512, 512), -1 / 0 / 1
    crs: CRS           # the CRS of the radar bands
    transform: Affine            # the transform of the radar bands

def load_chip(chip_id):
    """Read one chip's data into a Chip object."""
    with rasterio.open(S1_DIR / f"{chip_id}_S1Hand.tif") as src:
        vv = src.read(VV_BAND)
        vh = src.read(VH_BAND)
        crs = src.crs
        transform = src.transform
    with rasterio.open(LABEL_DIR / f"{chip_id}_LabelHand.tif") as src:
        label = src.read(1)
    loc = chip_id.split("_")[0]
    if loc == "Mekong":
        loc = "Cambodia"
    return Chip(id=chip_id, loc=loc, vv=vv, vh=vh, label=label, crs=crs, transform=transform)

def valid_mask(chip):
    """
    Pixels on a chip where a prediction can be both made and scored.
    A pixel is valid if and only if it has a valid VV and VH measurement and a valid label.

    Returns a bool array.
    """
    return np.isfinite(chip.vv) & np.isfinite(chip.vh) & (chip.label != LABEL_NODATA)

def preprocess(chip, mean=None, std=None):
    """Turn a Chip into model input, target, and mask.

    Pass both mean and std, or neither.

    With stats: x is clipped, standardised, and invalid pixels are filled
    with 0, which after standardising *is* the dataset mean.

    Without stats: x stays in decibels and invalid pixels stay NaN. Select with `valid` before using x.

    Returns
    x: float32 (2, H, W) -- channels (VV, VH), clipped to [CLIP_LO, CLIP_HI]
    y: uint8 (H, W) -- 0 land, 1 water, IGNORE_INDEX where unusable
    valid: bool (H, W) -- the validity mask
    """
    if (mean is None) != (std is None):
        raise ValueError("pass both mean and std, or neither")
    valid = valid_mask(chip)
    x = np.stack([chip.vv, chip.vh]).astype(np.float32)
    x = np.clip(x, CLIP_LO, CLIP_HI)
    if mean is not None:
        m = np.asarray(mean, np.float32)[:, None, None]
        s = np.asarray(std, np.float32)[:, None, None]
        x = (x - m) / s
        x = np.nan_to_num(x, nan=0.0)
    y = np.where(chip.label == LABEL_WATER, 1, 0).astype(np.uint8)
    y[~valid] = IGNORE_INDEX
    return x, y, valid

def read_s2(chip_id, bands):
    """Read Sentinel-2 chips at given bands as float32 (len(bands), H, W). 0 marks no data."""
    with rasterio.open(S2_DIR / f"{chip_id}_S2Hand.tif") as src:
        return src.read([S2_BANDS[b] for b in bands]).astype(np.float32)

def water_indices(chip_id):
    """NDWI and MNDWI as float32 (2, H, W), -1 <= values <= 1.

    NDWI  = (green - NIR)   / (green + NIR)
    MNDWI = (green - SWIR1) / (green + SWIR1), which holds up better over built-up ground
    """
    green, nir, swir = read_s2(chip_id, ("B3", "B8", "B11"))
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.stack([(green - nir) / (green + nir), (green - swir) / (green + swir)])
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)    # S2 no-data is 0, so 0/0 off-swath

def chip_input(chip_id, arm, mean, std):
    """Model input (C, H, W) and target labels (H, W) for one chip.

    `arm` from `ARMS`; `C = 4 if arm == "s1+s2" else 2`.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    x, y, _ = preprocess(load_chip(chip_id), mean, std)
    if arm == "s1":
        return x, y
    s2 = water_indices(chip_id)
    return (s2 if arm == "s2" else np.concatenate([x, s2])), y

def cloud_mask(chip_id, fraction, shape=(512, 512), sigma=24):
    """Simulated cloud occlusion over `fraction` of the chip. `True` for covered pixels.
    
    Seeded according to `chip_id` so the same chip always gets the same mask.
    """
    if fraction <= 0:
        return np.zeros(shape, bool)
    if fraction >= 1:
        return np.ones(shape, bool)
    rng = np.random.default_rng(crc32(chip_id.encode()))
    field = gaussian_filter(rng.standard_normal(shape), sigma)
    return field < np.quantile(field, fraction)

def occluded_input(chip_id, arm, fraction, mean, std):
    """chip_input, with the optical channels (NDWI, MNDWI) hidden under clouds."""
    x, y = chip_input(chip_id, arm, mean, std)
    if arm != "s1" and fraction > 0:
        x[-2:, cloud_mask(chip_id, fraction)] = 0.0     # the last two channels are always S2
    return x, y

def read_split(splitname):
    """Chip IDs listed in one Sen1Floods11 split CSV."""
    if splitname not in SPLIT_FILES:
        raise ValueError(f"Unknown split {splitname!r}; expected one of {SPLIT_ORDER}")
    filename = SPLIT_FILES[splitname]
    df = pd.read_csv(SPLIT_DIR / filename, header=None, names=["s1", "label"])
    stems = df["s1"].str.replace("_S1Hand.tif", "", regex=False)
    if not (stems == df["label"].str.replace("_LabelHand.tif", "", regex=False)).all():
        raise ValueError(f"{filename}: S1 and label columns disagree")
    return stems.tolist()

def load_splits():
    """Returns {split name: list of chip IDs}, in SPLIT_ORDER."""
    splits = {name: read_split(name) for name in SPLIT_ORDER}
    for a in SPLIT_ORDER:
        for b in SPLIT_ORDER:
            if a < b:
                overlap = set(splits[a]) & set(splits[b])
                if overlap:
                    raise ValueError(f"leakage: {a} and {b} share {len(overlap)} chips: "
                                     f"{sorted(overlap)[:3]}")
    listed = {c for chips in splits.values() for c in chips}
    on_disk = {p.name.replace("_S1Hand.tif", "") for p in S1_DIR.glob("*_S1Hand.tif")}
    if listed != on_disk:
        raise ValueError(f"split/disk mismatch: {len(listed ^ on_disk)} chips differ")
    return splits
