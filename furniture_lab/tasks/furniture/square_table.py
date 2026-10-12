"""The square table, as FurnitureBench defines it (``furniture/square_table.py``,
``config["furniture"]["square_table"]``, ``sim_config``).

Every number here is copied from upstream in upstream's own frame (the AprilTag frame, see
:mod:`.frames`) so that a line-by-line diff against FurnitureBench stays possible. The
world-frame poses the env spawns are derived once, in :func:`reset_pose_world`.

``one_leg`` is the square table with one assembly pair to make: ``(0, 4)``, the table top and
leg 4. All five parts are still spawned, exactly as upstream does.

Part frames are the **mesh frames** of the upstream OBJs (the URDF link origins): the table
top's thickness runs along its mesh ``y`` (31.25 mm, +-15.625), its footprint is 162.5 mm in
``x``/``z``; a leg's long axis is its mesh ``y`` (-56.25 .. +31.25 mm, the thread at the
negative end). The assembled relative poses below are expressed in those frames, so a USD
whose root prim is not at the mesh origin breaks the assembly check -- ``tools/check_usd.py``
prints the bounds to compare.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .frames import rot_mat, tag_to_world

__all__ = [
    "PartSpec",
    "PARTS",
    "PART_NAMES",
    "ASSEMBLED_REL_POSES",
    "ONE_LEG_PAIRS",
    "SQUARE_TABLE_PAIRS",
    "ASSEMBLED_POS_THRESHOLD",
    "ORI_BOUND",
    "PART_FRICTION",
    "TABLE_FRICTION",
    "MESH_FRAMES",
    "MESH_FRAME_VAR",
    "mesh_frame",
    "reset_pose_world",
    "rel_pose_targets",
    "pairs_for",
]


@dataclass(frozen=True)
class PartSpec:
    name: str
    part_idx: int
    #: ``reset_pos[0]`` / ``reset_ori[0]`` -- the from-scratch (skill 0) reset, tag frame
    reset_pos_tag: Tuple[float, float, float]
    reset_rot_tag: np.ndarray
    #: kg, from the ``sim_config`` comments ("Mass: 151g", "Mass: 23.1g")
    mass: float
    #: radius of a disc that covers the part's footprint lying flat, for the reset-time
    #: collision rejection (upstream rejects overlapping axis-aligned boxes; a disc is the
    #: conservative batched stand-in)
    footprint_radius: float


_LEG_ROT = rot_mat((0.0, -np.pi / 2, 0.0))
_TOP_ROT = rot_mat((-np.pi / 2, 0.0, np.pi))

#: table-top footprint: 0.1625 square -> half-diagonal; leg: 0.0875 long, 0.03 wide
_TOP_RADIUS = 0.1625 * np.sqrt(2) / 2
_LEG_RADIUS = 0.0875 / 2 + 0.005

PARTS: List[PartSpec] = [
    PartSpec("square_table_top", 0, (0.0, 0.24, -0.015625), _TOP_ROT, 0.151, _TOP_RADIUS),
    PartSpec("square_table_leg1", 1, (-0.20, 0.07, -0.015), _LEG_ROT, 0.0231, _LEG_RADIUS),
    PartSpec("square_table_leg2", 2, (-0.12, 0.07, -0.015), _LEG_ROT, 0.0231, _LEG_RADIUS),
    PartSpec("square_table_leg3", 3, (0.12, 0.07, -0.015), _LEG_ROT, 0.0231, _LEG_RADIUS),
    PartSpec("square_table_leg4", 4, (0.20, 0.07, -0.015), _LEG_ROT, 0.0231, _LEG_RADIUS),
]
PART_NAMES: List[str] = [p.name for p in PARTS]

#: ``assembled_rel_poses[(0, j)]``: where leg ``j`` sits in the table top's frame when
#: screwed in -- any of the four holes, with the leg's axes aligned to the top's
_HOLES = np.array(
    [
        [-0.05625, 0.046875, -0.05625],
        [0.05625, 0.046875, -0.05625],
        [-0.05625, 0.046875, 0.05625],
        [0.05625, 0.046875, 0.05625],
    ]
)
ASSEMBLED_REL_POSES: Dict[Tuple[int, int], np.ndarray] = {(0, j): _HOLES for j in (1, 2, 3, 4)}

#: ``should_be_assembled``
ONE_LEG_PAIRS: List[Tuple[int, int]] = [(0, 4)]
SQUARE_TABLE_PAIRS: List[Tuple[int, int]] = [(0, 4), (0, 3), (0, 1), (0, 2)]

#: ``config["furniture"]["assembled_pos_threshold"]`` (per axis of the top's frame) and
#: ``Furniture.ori_bound``: each column of the relative rotation must have cosine >= 0.94
#: (about 20 deg) with the identity's
ASSEMBLED_POS_THRESHOLD = (0.010, 0.005, 0.010)
ORI_BOUND = 0.94

#: ``sim_config["parts"]["friction"]`` / ``sim_config["table"]["friction"]``
PART_FRICTION = 0.15
TABLE_FRICTION = 0.10

# ------------------------------------------------------------------------ mesh frames
#: How a part USD's frame relates to upstream's OBJ frame: ``p_usd = R @ p_obj``.
#:
#: ``obj``   -- the USD is a straight conversion of the OBJ (``tools/convert_assets.py``).
#: ``uwlab`` -- UWLab's ``square_leg.usd`` / ``square_table_top.usd``: the leg's long axis
#:              and the top's thickness run along USD ``z`` where the OBJ has ``y``, with the
#:              same ranges (bolt at -z, plate at -z), i.e. ``Rx(+90 deg)``. The rotation
#:              about that axis cannot be read off the bounds; it is assumed to be the same
#:              for both parts, which the seated-leg orientation check relies on. Verify by
#:              teleporting a leg to its assembled pose before trusting a training run.
MESH_FRAMES: Dict[str, np.ndarray] = {
    "obj": np.eye(3),
    "uwlab": rot_mat((np.pi / 2, 0.0, 0.0)),
}
MESH_FRAME_VAR = "FURNITURE_MESH_FRAME"


def mesh_frame() -> str:
    """``$FURNITURE_MESH_FRAME`` (default ``obj``), validated."""
    name = os.environ.get(MESH_FRAME_VAR, "obj")
    if name not in MESH_FRAMES:
        raise ValueError(f"{MESH_FRAME_VAR}={name!r}; expected one of {sorted(MESH_FRAMES)}")
    return name


def reset_pose_world(
    part: PartSpec, frame: Optional[str] = None
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """The part's from-scratch reset pose in the robot-base frame: (pos xyz, quat wxyz).

    ``frame`` picks the USD's mesh frame (see :data:`MESH_FRAMES`); the pose returned is that
    of the USD prim: ``R_world_usd = R_world_obj @ R^T``.
    """
    R = MESH_FRAMES[frame or mesh_frame()]
    rot_tag = np.asarray(part.reset_rot_tag)[:3, :3] @ R.T
    pos, quat = tag_to_world(part.reset_pos_tag, rot_tag)
    return tuple(float(v) for v in pos), tuple(float(v) for v in quat)


def pairs_for(task: str) -> List[Tuple[int, int]]:
    if task == "one_leg":
        return list(ONE_LEG_PAIRS)
    if task == "square_table":
        return list(SQUARE_TABLE_PAIRS)
    raise ValueError(f"unknown furniture task {task!r}; expected 'one_leg' or 'square_table'")


def rel_pose_targets(pairs: Sequence[Tuple[int, int]], frame: Optional[str] = None) -> np.ndarray:
    """``(num_pairs, K, 3)`` assembled positions, one row of K candidates per pair, in the
    USD frame: with both parts' frames rotated by the same ``R``, the relative rotation stays
    the identity and the relative translation becomes ``R @ t``."""
    R = MESH_FRAMES[frame or mesh_frame()]
    return np.stack([ASSEMBLED_REL_POSES[pair] @ R.T for pair in pairs])
