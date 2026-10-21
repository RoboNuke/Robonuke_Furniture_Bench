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

__all__ = ["ASSETS_ROOT", "AssetEntry", "CATALOG", "CATALOG_FURNITUREBENCH", "ASSET_SETS", "entry", "usd_path", "metadata", "mesh_frame_of"]

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


#: FurnitureBench's own OBJs (ResiP's fork) converted by ``tools/convert_assets.py`` into
#: ``assets/FurnitureBench``: one collision mesh per part, an SDF at 512 on the top and leg 4 and a
#: convex hull on legs 1-3, as ResiP's URDFs give Isaac Gym; the OBJ mesh frame
CATALOG_FURNITUREBENCH: Dict[str, AssetEntry] = {
    name: AssetEntry("FurnitureBench", f"{name}.usd", frame="obj")
    for name in ("square_table_top", "square_table_leg1", "square_table_leg2", "square_table_leg3", "square_table_leg4")
}
#: ``CATALOG_FURNITUREBENCH`` with the top's SDF at 1024 (0.16 mm cells over its 162.5 mm, about the
#: leg's 0.17 at 512; 512 gives 0.32 mm, about the thread's 0.38 mm crest-to-root clearance)
CATALOG_FURNITUREBENCH_TOP1024: Dict[str, AssetEntry] = {
    **CATALOG_FURNITUREBENCH,
    "square_table_top": AssetEntry("FurnitureBench", "square_table_top_sdf1024.usd", frame="obj"),
}
#: ``furniture.part_assets`` -> its catalog
ASSET_SETS: Dict[str, Dict[str, AssetEntry]] = {
    "uwlab": CATALOG,
    "furniturebench": CATALOG_FURNITUREBENCH,
    "furniturebench_top1024": CATALOG_FURNITUREBENCH_TOP1024,
}


def entry(part: str, assets: str = "uwlab") -> AssetEntry:
    if assets not in ASSET_SETS:
        raise ValueError(f"unknown part asset set {assets!r}; one of {sorted(ASSET_SETS)}")
    try:
        return ASSET_SETS[assets][part]
    except KeyError:
        raise KeyError(f"no {assets!r} asset catalog row for part {part!r}; add one to {__name__}") from None


def usd_path(part: str, assets: str = "uwlab") -> str:
    """Absolute path of the part's USD; raises if the file is missing from the checkout."""
    e = entry(part, assets)
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


def mesh_frame_of(part: str, assets: str = "uwlab") -> str:
    return entry(part, assets).frame
