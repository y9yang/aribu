from typing import NamedTuple
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from .paths import S1_DIR, LABEL_DIR, SPLIT_DIR

LABEL_NODATA, LABEL_LAND, LABEL_WATER = -1, 0, 1
CLIP_LO, CLIP_HI = -50.0, 1.0
VV_BAND, VH_BAND = 1, 2
IGNORE_INDEX = 255
SPLIT_FILES = {
    "train": "flood_train_data.csv",
    "val": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
    "bolivia": "flood_bolivia_data.csv",
}
SPLIT_ORDER = tuple(SPLIT_FILES)

class Chip(NamedTuple):
    """the data contained in a chip."""
    chip_id: str
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
    return Chip(chip_id=chip_id, loc=loc, vv=vv, vh=vh, label=label, crs=crs, transform=transform)

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