import json
from pathlib import Path
import numpy as np

__all__ = ["json_safe", "write_json"]

def json_safe(obj):
    """Recursively convert NaN to None and numpy scalars to Python types."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else float(obj)
    return obj

def _strict(name):
    raise ValueError(f"{name} is not valid JSON")

def write_json(path, obj):
    """Write `obj` to `path` as JSON, then prove a strict parser can read it back.

    Returns the reparsed content.
    """
    path = Path(path)
    path.write_text(json.dumps(json_safe(obj), indent=2))
    return json.loads(path.read_text(), parse_constant=_strict)
