import os, sys
from pathlib import Path

for _var, _sub in (("GDAL_DATA", "Library/share/gdal"), ("PROJ_DATA", "Library/share/proj")):
    _path = Path(sys.prefix) / _sub
    if _var not in os.environ and _path.is_dir():
        os.environ[_var] = str(_path)

__all__ = [
    "REPO_ROOT", "DATA_DIR", "RAW_DIR", "SPLIT_DIR",
    "RESULTS_DIR", "DEMO_DIR", "EXTERNAL_DIR", "MODELS_DIR",
    "S1_DIR", "LABEL_DIR", "OTSU_DIR", "METADATA_PATH",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
if not (REPO_ROOT / "pyproject.toml").exists():
    raise RuntimeError(f"REPO_ROOT looks wrong: {REPO_ROOT}")

DATA_DIR     = REPO_ROOT / "data"
RAW_DIR      = DATA_DIR / "raw" / "HandLabeled"
SPLIT_DIR    = DATA_DIR / "splits"
RESULTS_DIR  = DATA_DIR / "results"
DEMO_DIR     = DATA_DIR / "demo"
EXTERNAL_DIR = DATA_DIR / "external"
MODELS_DIR   = REPO_ROOT / "models"

S1_DIR    = RAW_DIR / "S1Hand"
LABEL_DIR = RAW_DIR / "LabelHand"
OTSU_DIR  = RAW_DIR / "S1OtsuLabelHand"

METADATA_PATH = EXTERNAL_DIR / "Sen1Floods11_Metadata.geojson"