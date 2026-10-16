"""FurnitureBench's coordinate frames, and the scene layout they imply.

Pure numpy, no Isaac Lab -- so the numbers here are testable on a laptop.

FurnitureBench defines every part pose in the **AprilTag frame** (``base_tag``): a frame
printed on the table in front of the robot, whose ``z`` points *down* into the table. Its
sim builds the world as

    table (1.3 x 0.8 x 0.03) centred at (0, 0, 0.4)  ->  surface at z = 0.415
    robot base at (-0.3, 0, 0.43)                    ->  0.015 m above the surface (the clamp)
    base_tag at robot + (0.3015, 0, -0.015), R = Rz(pi/2) Ry(0) Rx(pi)

(``furniture_sim_env.create_envs`` and ``config["robot"]["tag_base_from_robot_base"]``).

This port puts the **world origin at the robot base**, z up, which is also the frame every
observation is published in. Everything FurnitureBench specifies in the tag frame goes
through :func:`tag_to_world` exactly once, at import time, and the raw tag-frame numbers are
kept verbatim in :mod:`.square_table` so a diff against upstream stays trivial.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

__all__ = [
    "ROBOT_HEIGHT",
    "RESIP_ROBOT_HEIGHT",
    "RESIP_BASE_Z_TO_PORT",
    "TABLE_SURFACE_Z",
    "TABLE_SIZE",
    "TABLE_POS",
    "TAG_POS_W",
    "TAG_ROT_W",
    "OBSTACLE_FRONT_SIZE",
    "OBSTACLE_FRONT_POS",
    "OBSTACLE_SIDE_SIZE",
    "OBSTACLE_SIDE_POS",
    "RESET_JOINTS",
    "MAX_GRIPPER_WIDTH",
    "EE_POSITION_LIMITS",
    "PART_POSITION_LIMITS_TAG",
    "rot_mat",
    "mat_to_quat_wxyz",
    "tag_to_world",
]

#: height of the bench clamp the robot sits on, above the table surface (FurnitureBench)
ROBOT_HEIGHT = 0.015

#: ResiP's fork of FurnitureBench (ankile/furniture-bench, ``furniture_bench/config.py``) sets
#: ``ROBOT_HEIGHT = 0.0`` ("not using bench clamp"): its robot base sits on the table surface,
#: 15 mm lower than here, while the tag, the table and the parts are where they are here. So a
#: ResiP robot-base-frame z (EE poses, EE targets) reads 15 mm higher than this port's for the
#: same physical pose; tag-frame part poses are unaffected. Add this to a ResiP base-frame z.
RESIP_ROBOT_HEIGHT = 0.0
RESIP_BASE_Z_TO_PORT = RESIP_ROBOT_HEIGHT - ROBOT_HEIGHT

#: the table surface, in the robot-base frame
TABLE_SURFACE_Z = -ROBOT_HEIGHT
#: FurnitureBench's ``table.obj`` is a 1.3 x 0.8 x 0.03 slab; its centre is 0.3 m in front
#: of the robot base (the base sits at x = -0.3 in a table-centred world)
TABLE_SIZE = (1.3, 0.8, 0.03)
TABLE_POS = (0.3, 0.0, TABLE_SURFACE_Z - TABLE_SIZE[2] / 2)

#: ``tag_base_from_robot_base``: translation and the (x, y, z) Euler angles FurnitureBench
#: feeds its ``rot_mat`` (R = Rz @ Ry @ Rx)
TAG_POS_W = (0.23 + 0.0715, 0.0, -ROBOT_HEIGHT)
_TAG_EULER = (math.pi, 0.0, math.pi / 2)

#: the three fixed obstacles, as boxes in the robot-base frame. FurnitureBench spawns them
#: relative to the base tag, rotated 90 deg about z, 0.015 above the surface:
#:   front: tag.x + 0.37 + 0.01, y = 0            (0.35 x 0.02 x 0.03 before the rotation)
#:   sides: tag.x + 0.37 + 0.01 - 0.075, y = +-0.175 (0.02 x 0.17 x 0.03 before the rotation)
OBSTACLE_FRONT_SIZE = (0.02, 0.35, 0.03)
OBSTACLE_FRONT_POS = (TAG_POS_W[0] + 0.37 + 0.01, 0.0, TABLE_SURFACE_Z + 0.015)
OBSTACLE_SIDE_SIZE = (0.17, 0.02, 0.03)
OBSTACLE_SIDE_POS = {
    "obstacle_right": (TAG_POS_W[0] + 0.37 + 0.01 - 0.075, -0.175, TABLE_SURFACE_Z + 0.015),
    "obstacle_left": (TAG_POS_W[0] + 0.37 + 0.01 - 0.075, 0.175, TABLE_SURFACE_Z + 0.015),
}

#: ``config["robot"]["reset_joints"]``
RESET_JOINTS = (
    -0.02630888,
    0.3758795,
    0.12485036,
    -2.1383357,
    -0.09431414,
    2.49649072,
    0.01921718,
)
#: ``config["robot"]["max_gripper_width"]["one_leg"]``
MAX_GRIPPER_WIDTH = 0.065
#: ``config["robot"]["position_limits"]``, end-effector, robot-base frame
EE_POSITION_LIMITS = ((0.3, 0.8), (-0.55, 0.55), (0.005, 0.4))
#: ``config["furniture"]["position_limits"]``, parts, **tag** frame (x, y)
PART_POSITION_LIMITS_TAG = ((-0.21, 0.21), (0.07, 0.37))


def rot_mat(angles: Sequence[float]) -> np.ndarray:
    """FurnitureBench's ``rot_mat``: R = Rz(z) @ Ry(y) @ Rx(x) for angles (x, y, z)."""
    x, y, z = angles
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx


#: rotation of the tag frame in the robot-base frame
TAG_ROT_W = rot_mat(_TAG_EULER)


def mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z), Isaac Lab's convention."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def tag_to_world(pos_tag: Sequence[float], rot_tag: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """A pose given in the AprilTag frame -> (pos, quat_wxyz) in the robot-base frame.

    ``rot_tag`` may be 3x3 or a 4x4 homogeneous matrix (FurnitureBench's ``hom=True``).
    """
    R_tag = np.asarray(rot_tag, dtype=np.float64)[:3, :3]
    pos = np.asarray(TAG_POS_W) + TAG_ROT_W @ np.asarray(pos_tag, dtype=np.float64)
    return pos, mat_to_quat_wxyz(TAG_ROT_W @ R_tag)
