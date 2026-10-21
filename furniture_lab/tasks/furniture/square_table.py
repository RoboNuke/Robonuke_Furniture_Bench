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

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import catalog
from .frames import rot_mat, tag_to_world

__all__ = [
    "PartSpec",
    "PARTS",
    "PART_NAMES",
    "ASSEMBLED_REL_POSES",
    "ONE_LEG_PAIRS",
    "SQUARE_TABLE_PAIRS",
    "ASSEMBLED_POS_THRESHOLD",
    "pos_threshold_in",
    "ORI_BOUND",
    "PART_FRICTION",
    "TABLE_FRICTION",
    "MESH_FRAMES",
    "mesh_frame",
    "reset_pose_world",
    "tag_pose_world",
    "ISAAC_GYM_MASS",
    "RANDOMIZE_INIT_POSE",
    "RANDOMIZE_MAX_TRIAL",
    "PARTS_POS_LIM",
    "RESET_LEN",
    "COLLISION_MARGIN",
    "HIGH_RAND_RESET",
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
    #: half-extents (x, y) of the part's footprint lying flat at its nominal reset yaw, in the
    #: robot-base frame -- the box upstream's reset-time collision rejection uses
    footprint: Tuple[float, float]


_LEG_ROT = rot_mat((0.0, -np.pi / 2, 0.0))
_TOP_ROT = rot_mat((-np.pi / 2, 0.0, np.pi))

#: table-top: 0.1625 square; leg: 0.0875 long along the robot's x at reset, 0.03 wide
_TOP_FOOT = (0.1625 / 2, 0.1625 / 2)
_LEG_FOOT = (0.0875 / 2, 0.03 / 2)

PARTS: List[PartSpec] = [
    PartSpec("square_table_top", 0, (0.0, 0.24, -0.015625), _TOP_ROT, 0.151, _TOP_FOOT),
    PartSpec("square_table_leg1", 1, (-0.20, 0.07, -0.015), _LEG_ROT, 0.0231, _LEG_FOOT),
    PartSpec("square_table_leg2", 2, (-0.12, 0.07, -0.015), _LEG_ROT, 0.0231, _LEG_FOOT),
    PartSpec("square_table_leg3", 3, (0.12, 0.07, -0.015), _LEG_ROT, 0.0231, _LEG_FOOT),
    PartSpec("square_table_leg4", 4, (0.20, 0.07, -0.015), _LEG_ROT, 0.0231, _LEG_FOOT),
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

#: ``config["furniture"]["assembled_pos_threshold"]`` (per axis of the top's **OBJ** frame,
#: whose y is the thickness / seat axis: 5 mm of seat depth, 10 mm in the plane) and
#: ``Furniture.ori_bound``: each column of the relative rotation must have cosine >= 0.94
#: (about 20 deg) with the identity's. Use :func:`pos_threshold_in` for another mesh frame.
ASSEMBLED_POS_THRESHOLD = (0.010, 0.005, 0.010)
ORI_BOUND = 0.94

#: the masses FurnitureSim actually simulates (kg): Isaac Gym gives a URDF link with no
#: ``<inertial>`` density x the volume of its collision mesh's convex hull, so the top weighs
#: 0.411511 kg and the legs 0.026361 / 0.026522 / 0.026360 / 0.029137 (read back from their sim
#: with their URDFs and asset options), not the ``sim_config`` comments' 151 g / 23.1 g
#: (``PartSpec.mass``). Rounded to 0.1 g: the values ResiP's checkpoint scored 16% with in the port.
#: ``furniture.part_masses`` defaults to these.
ISAAC_GYM_MASS: Dict[str, float] = {
    "square_table_top": 0.4115,
    "square_table_leg1": 0.0264,
    "square_table_leg2": 0.0265,
    "square_table_leg3": 0.0264,
    "square_table_leg4": 0.0291,
}

#: ``sim_config["parts"]["friction"]`` / ``sim_config["table"]["friction"]``
PART_FRICTION = 0.15
TABLE_FRICTION = 0.10

# --------------------------------------------------- FurnitureBench's reset randomization
#: ``FurnitureSimEnv.reset_env`` / ``Furniture.randomize_init_pose``: each part's tag-frame x, y
#: offset uniform in +-pos (m), its yaw about the tag z uniform in +-yaw (deg); every part redrawn
#: together until no two collide and all lie inside :data:`PARTS_POS_LIM` (``max_trial`` 300000)
RANDOMIZE_INIT_POSE: Dict[str, Tuple[float, float]] = {"low": (0.015, 15.0), "med": (0.05, 45.0)}
RANDOMIZE_MAX_TRIAL = 300000
#: ``config["furniture"]["position_limits"]`` (``Furniture.parts_pos_lim``): tag-frame x, y of
#: every part's reset position (``Part.in_boundary``)
PARTS_POS_LIM = ((-0.21, 0.21), (0.07, 0.37))
#: ``Part.reset_x_len`` / ``reset_y_len``: each part's tag-frame footprint (m) for the reset
#: collision check (the leg's x includes upstream's 0.02 margin), and ``Part.collision_margin``
RESET_LEN: Dict[str, Tuple[float, float]] = {
    "square_table_top": (0.1625, 0.1625),
    **{f"square_table_leg{i}": (0.03 + 0.02, 0.0875) for i in range(1, 5)},
}
COLLISION_MARGIN = 0.01
#: ``high_rand_reset_pos`` / ``high_rand_reset_ori`` (``Part.randomize_init_pose_high``): the
#: three fixed layouts of ``high`` randomness, per part ``(pos, rot)`` in the tag frame (recorded
#: poses: the rotations are near, not exactly, orthonormal), copied from ``config.py``
HIGH_RAND_RESET: Dict[str, List[Tuple[Tuple[float, float, float], np.ndarray]]] = {
    "square_table_top": [
        ((0.00165764, 0.23166147, -0.01571113), np.array([(-0.22164249, -0.00542635, -0.97511292), (0.97511977, -0.00533661, -0.22161436), (-0.00400126, -0.99997103, 0.00647417)])),
        ((-0.02074748, 0.20895012, -0.01565138), np.array([(0.18759322, -0.01144737, -0.98218012), (0.98217165, -0.01018527, 0.18771034), (-0.01215252, -0.99988264, 0.00933257)])),
        ((-0.07365019, 0.13859308, -0.01718625), np.array([(-0.34981984, -0.00092953, -0.93681651), (0.93678069, -0.00914782, -0.34979743), (-0.00824469, -0.99995774, 0.00407088)])),
    ],
    "square_table_leg1": [
        ((0.12411077, 0.172485, -0.01282854), np.array([(-0.99050552, 0.13746983, -0.00065276), (0.13745576, 0.9904536, 0.01037287), (0.00207249, 0.01018466, -0.99994576)])),
        ((0.11414307, 0.16866885, -0.01511348), np.array([(0.99296767, -0.11772603, 0.01248006), (0.11803517, 0.99261737, -0.02790101), (-0.00910325, 0.02917789, 0.99953276)])),
        ((0.07065351, 0.16127343, -0.0162864), np.array([(0.00333085, -0.52690697, -0.8499164), (-0.01105705, 0.84984976, -0.52690899), (0.9999333, 0.01115263, -0.00299532)])),
    ],
    "square_table_leg2": [
        ((0.14908047, 0.14851391, -0.02330433), np.array([(0.98727685, 0.15897338, 0.0034444), (-0.15894894, 0.98726553, -0.00647893), (-0.00443052, 0.00584901, 0.99997306)])),
        ((-0.04624121, 0.20801648, -0.01895902), np.array([(0.9999948, 0.0032234, 9.178e-05), (-0.00322341, 0.9999948, 6.646e-05), (-9.156e-05, -6.676e-05, 1.0)])),
        ((-0.07811823, 0.31678003, -0.01311489), np.array([(-0.00119546, -0.40191379, -0.91567671), (-0.00777176, 0.91565347, -0.40189344), (0.99996912, 0.00663596, -0.0042182)])),
    ],
    "square_table_leg3": [
        ((-0.11946367, 0.18800808, -0.01739212), np.array([(-0.01415714, -0.15679362, 0.98752993), (0.02789432, 0.98718256, 0.15713838), (-0.99951059, 0.0297711, -0.00960203)])),
        ((0.21110627, 0.09755837, -0.01618727), np.array([(0.00594478, -0.18838574, -0.98207718), (-0.01028048, 0.98203105, -0.18843913), (0.99992949, 0.01121645, 0.00390126)])),
        ((0.1526014, 0.10060443, -0.0157905), np.array([(-0.98098826, -0.19385707, -0.00902952), (-0.19405723, 0.98034805, 0.03548874), (0.00197233, 0.03656628, -0.99932933)])),
    ],
    "square_table_leg4": [
        ((-0.15307285, 0.12734982, -0.01829645), np.array([(-0.79311496, -0.6090616, -0.0035405), (-0.60903323, 0.79298568, 0.01588211), (-0.00686562, 0.01475262, -0.99986762)])),
        ((-0.14806452, 0.25308317, -0.01665781), np.array([(-0.88222593, -0.47080287, -0.00470781), (-0.470696, 0.88170207, 0.03235343), (-0.0110812, 0.03075899, -0.99946541)])),
        ((0.08496851, 0.28050846, -0.01651655), np.array([(0.01009589, -0.99462211, -0.10307747), (-0.01416063, 0.10293016, -0.99458778), (0.99984878, 0.01150092, -0.0130454)])),
    ],
}

# ------------------------------------------------------------------------ mesh frames
#: How a part USD's frame relates to upstream's OBJ frame: ``p_usd = R @ p_obj``.
#:
#: ``obj``   -- a straight conversion of the OBJ (``tools/convert_assets.py``).
#: ``uwlab`` -- UWLab's USDs (what ``assets/`` holds): the leg's long axis and the top's
#:              thickness run along USD ``z`` where the OBJ has ``y``, same ranges (bolt at
#:              -z, plate at -z), i.e. ``Rx(+90 deg)``. UWLab's ``metadata.yaml`` gives the
#:              seated leg the identity orientation and z = 0.047223 in the top's frame,
#:              against 0.046875 here (``tests/test_catalog.py``), so both parts were
#:              rotated the same way.
#: Which frame a part's file uses is a row of :mod:`.catalog`.
MESH_FRAMES: Dict[str, np.ndarray] = {
    "obj": np.eye(3),
    "uwlab": rot_mat((np.pi / 2, 0.0, 0.0)),
}


def mesh_frame(parts: Sequence["PartSpec"] = None) -> str:
    """The one mesh frame the furniture's asset files share (the catalog says which)."""
    names = {catalog.mesh_frame_of(p.name) for p in (parts if parts is not None else PARTS)}
    if len(names) != 1:
        raise ValueError(f"the parts' USDs use different mesh frames {sorted(names)}; one is required")
    return names.pop()


def reset_pose_world(
    part: PartSpec, frame: Optional[str] = None
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """The part's from-scratch reset pose in the robot-base frame: (pos xyz, quat wxyz).

    ``frame`` picks the USD's mesh frame (see :data:`MESH_FRAMES`); the pose returned is that
    of the USD prim: ``R_world_usd = R_world_obj @ R^T``.
    """
    return tag_pose_world(part.reset_pos_tag, part.reset_rot_tag, frame)


def tag_pose_world(
    pos_tag: Sequence[float], rot_tag: np.ndarray, frame: Optional[str] = None
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """A part pose given as upstream gives it (tag frame, OBJ frame) -> the USD prim's pose in
    the robot-base frame: (pos xyz, quat wxyz)."""
    R = MESH_FRAMES[frame or mesh_frame()]
    pos, quat = tag_to_world(pos_tag, np.asarray(rot_tag)[:3, :3] @ R.T)
    return tuple(float(v) for v in pos), tuple(float(v) for v in quat)


def pairs_for(task: str) -> List[Tuple[int, int]]:
    if task == "one_leg":
        return list(ONE_LEG_PAIRS)
    if task == "square_table":
        return list(SQUARE_TABLE_PAIRS)
    raise ValueError(f"unknown furniture task {task!r}; expected 'one_leg' or 'square_table'")


def pos_threshold_in(frame: Optional[str] = None, threshold: Sequence[float] = ASSEMBLED_POS_THRESHOLD) -> np.ndarray:
    """Upstream's per-axis tolerance, given in the OBJ frame's axes, carried to ``frame``'s:
    the axes are permuted by the frame rotation, so the 5 mm stays on the seat axis (OBJ y,
    uwlab z) and the 10 mm on the in-plane ones."""
    R = MESH_FRAMES[frame or mesh_frame()]
    return np.abs(R @ np.asarray(threshold, dtype=np.float64))


def rel_pose_targets(pairs: Sequence[Tuple[int, int]], frame: Optional[str] = None) -> np.ndarray:
    """``(num_pairs, K, 3)`` assembled positions, one row of K candidates per pair, in the
    USD frame: with both parts' frames rotated by the same ``R``, the relative rotation stays
    the identity and the relative translation becomes ``R @ t``."""
    R = MESH_FRAMES[frame or mesh_frame()]
    return np.stack([ASSEMBLED_REL_POSES[pair] @ R.T for pair in pairs])
