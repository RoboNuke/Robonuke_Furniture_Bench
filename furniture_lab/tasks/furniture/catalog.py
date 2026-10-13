"""The asset catalog: which file in ``assets/`` each FurnitureBench part loads, and the
mesh frame that file uses.

The repo's ``assets/`` folder follows UWLab's layout (``SquareLeg/square_leg.usd`` and a
``metadata.yaml`` beside it), so a part name is mapped to a folder here; a new furniture is
a new row. Pure python + yaml, no Isaac Lab -- the catalog is read in tests too.

Mesh frames: UWLab's USDs put a part's long axis (leg) or thickness (table top) on ``z``,
where FurnitureBench's OBJs have ``y``, with the same ranges; that is the ``uwlab`` frame of
:data:`.square_table.MESH_FRAMES`. Their ``metadata.yaml`` corroborates the FurnitureBench
assembled pose in that frame to 0.35 mm (``tests/test_catalog.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

__all__ = ["ASSETS_ROOT", "AssetEntry", "CATALOG", "entry", "usd_path", "metadata", "mesh_frame_of"]

#: ``<repo>/assets``
ASSETS_ROOT = Path(__file__).resolve().parents[3] / "assets"


@dataclass(frozen=True)
class AssetEntry:
    #: folder under ``assets/``
    folder: str
    #: USD file inside it
    usd: str
    #: key of :data:`.square_table.MESH_FRAMES` the file's mesh frame corresponds to
    frame: str = "uwlab"


CATALOG: Dict[str, AssetEntry] = {
    "square_table_top": AssetEntry("SquareTableTop", "square_table_top.usd"),
    # square_leg_phase0.usd is UWLab's square_leg.usd with its bolt prims turned +26 deg
    # about the leg axis (tools/fix_leg_phase.py), so the thread seats with the leg square
    # to the top -- as the real part does and as FurnitureBench's assembled rule assumes.
    # In the original the seat lands at 26 deg of twist (measured by scripts/smoke.py).
    "square_table_leg1": AssetEntry("SquareLeg", "square_leg_phase0.usd"),
    "square_table_leg2": AssetEntry("SquareLeg", "square_leg_phase0.usd"),
    "square_table_leg3": AssetEntry("SquareLeg", "square_leg_phase0.usd"),
    "square_table_leg4": AssetEntry("SquareLeg", "square_leg_phase0.usd"),
    # the drawer parts are in assets/ too; they get rows when the drawer task is ported
}


def entry(part: str) -> AssetEntry:
    try:
        return CATALOG[part]
    except KeyError:
        raise KeyError(f"no asset catalog row for part {part!r}; add one to {__name__}.CATALOG") from None


def usd_path(part: str) -> str:
    """Absolute path of the part's USD; raises if the file is missing from the checkout."""
    e = entry(part)
    path = ASSETS_ROOT / e.folder / e.usd
    if not path.is_file():
        raise FileNotFoundError(f"{path} (part {part!r}) is not in the checkout")
    return str(path)


def metadata(part: str) -> Optional[Dict[str, Any]]:
    """The folder's ``metadata.yaml`` as a dict, or None when there is none."""
    path = ASSETS_ROOT / entry(part).folder / "metadata.yaml"
    if not path.is_file():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def mesh_frame_of(part: str) -> str:
    return entry(part).frame
