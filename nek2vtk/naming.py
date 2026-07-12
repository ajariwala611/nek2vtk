"""Assign human-readable names to boundary regions.

Names are persisted to a small JSON file (``<case>_boundaries.json``) keyed by
a stable region signature so that:

* the first run prompts interactively (once), and
* every later run - including non-interactive / MPI batch runs - reuses the
  saved names automatically.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from .regions import RegionInfo, format_region_table  # noqa: F401

# Default name suggestions from the Nek boundary-condition code.
_CODE_DEFAULTS = {
    "W": "wall",
    "v": "inlet",
    "V": "inlet",
    "o": "outlet",
    "O": "outlet",
    "P": "periodic",
    "SYM": "symmetry",
    "sym": "symmetry",
    "mv": "moving_wall",
}


def suggest_name(info: RegionInfo, used: set) -> str:
    """Suggest a unique default name for a region."""
    base = _CODE_DEFAULTS.get(info.code.strip(), info.code.strip() or "bc")
    name = base
    i = 2
    while name in used:
        name = f"{base}_{i}"
        i += 1
    return name


def sanitize(name: str) -> str:
    """Make a name safe for use in file names."""
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9_.\-]", "", name)
    return name or "bc"


def config_path(case_dir: Path, casename: str) -> Path:
    return Path(case_dir) / f"{casename}_boundaries.json"


def load_names(path: Path, infos: List[RegionInfo]) -> Optional[Dict[int, str]]:
    """Load a name mapping if a config file exists and matches the mesh.

    Returns a ``{region_id: name}`` dict, or ``None`` if no usable config.
    Matching is by region signature so it is robust to region-id reordering.
    """
    path = Path(path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    by_sig = {entry["signature"]: entry["name"] for entry in data.get("regions", [])}
    mapping: Dict[int, str] = {}
    for info in infos:
        sig = info.signature()
        if sig not in by_sig:
            return None  # config is stale / doesn't match this mesh
        mapping[info.region_id] = by_sig[sig]
    return mapping


def save_names(path: Path, casename: str, infos: List[RegionInfo],
               names: Dict[int, str]) -> None:
    """Write the name mapping to a JSON config file."""
    entries = []
    for info in infos:
        entries.append(
            {
                "region_id": info.region_id,
                "name": names[info.region_id],
                "code": info.code,
                "nfaces": info.nfaces,
                "signature": info.signature(),
                "centroid": [float(v) for v in info.centroid],
                "bbox_min": [float(v) for v in info.bbox_min],
                "bbox_max": [float(v) for v in info.bbox_max],
            }
        )
    doc = {"case": casename, "regions": entries}
    Path(path).write_text(json.dumps(doc, indent=2))


def prompt_names(infos: List[RegionInfo]) -> Dict[int, str]:
    """Interactively ask the user to name each region (defaults offered)."""
    print("\nDetected boundary regions:")
    print(format_region_table(infos))
    print("\nName each region (press Enter to accept the [default]).")
    names: Dict[int, str] = {}
    used: set = set()
    for info in infos:
        default = suggest_name(info, used)
        try:
            raw = input(f"  region {info.region_id} (code '{info.code}', "
                        f"{info.nfaces} faces) name [{default}]: ")
        except EOFError:
            raw = ""
        name = sanitize(raw) if raw.strip() else default
        while name in used:
            name = f"{name}_2"
        used.add(name)
        names[info.region_id] = name
    return names


def resolve_names(infos: List[RegionInfo], cfg_path: Path, casename: str,
                  interactive: bool, reconfigure: bool) -> Dict[int, str]:
    """Return the final ``{region_id: name}`` mapping.

    Order of precedence: an existing (matching) config file, unless
    ``reconfigure`` is set; then interactive prompting; then auto-generated
    defaults.
    """
    if not reconfigure:
        existing = load_names(cfg_path, infos)
        if existing is not None:
            print(f"Using boundary names from {cfg_path}")
            return existing

    if interactive:
        names = prompt_names(infos)
    else:
        used: set = set()
        names = {}
        for info in infos:
            n = suggest_name(info, used)
            used.add(n)
            names[info.region_id] = n
        print("Non-interactive: using default boundary names "
              "(edit the JSON and re-run to change).")

    save_names(cfg_path, casename, infos, names)
    print(f"Saved boundary names to {cfg_path}")
    return names
