"""Observation terms: FurnitureBench's ``robot_state`` and ``parts_poses``, in the robot-base
frame, quaternions ``(w, x, y, z)``.

Upstream's state observation is ``ee_pos``, ``ee_quat`` (x, y, z, w), ``ee_pos_vel``,
``ee_ori_vel``, ``gripper_width`` plus ``parts_poses`` in the AprilTag frame. This port keeps
the same quantities and changes two conventions, both deliberate: poses are in the robot-base
frame (what the controller and the parts share), and quaternions are Isaac Lab's ``wxyz``.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Sequence

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_apply_inverse, quat_mul, subtract_frame_transforms

from .state import parts_pose_b

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@functools.lru_cache(maxsize=None)
def _cached(values: tuple, device: str, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(values, device=device, dtype=dtype)


def const(values: Sequence, device, dtype=torch.float32) -> torch.Tensor:
    """A constant as a device tensor, built once per (value, device, dtype): building it inside an observation
    term is a host-to-device copy, i.e. a host synchronization, on every call."""
    as_t = torch.as_tensor(values, dtype=torch.float64)
    return _cached(tuple(as_t.reshape(-1).tolist()), str(device), dtype).reshape(tuple(as_t.shape))


def index(ids: Sequence[int], device) -> torch.Tensor:
    """Joint / body ids as a device long tensor, built once: indexing with a Python list copies it every call."""
    return _cached(tuple(int(i) for i in ids), str(device), torch.long)


def _ee_pose_b(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, offset: tuple):
    """Pose of the end-effector body (plus a fixed offset along it) in the robot-base frame."""
    robot = env.scene[asset_cfg.name]
    body = asset_cfg.body_ids[0]
    pos_w = robot.data.body_pos_w[:, body]
    quat_w = robot.data.body_quat_w[:, body]
    off = const(offset, env.device, pos_w.dtype).expand(env.num_envs, -1)
    pos_w, quat_w = combine_frame_transforms(pos_w, quat_w, off)
    return subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, pos_w, quat_w)


def ee_pos(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="panda_hand"),
    offset: tuple = (0.0, 0.0, 0.1034),
) -> torch.Tensor:
    return _ee_pose_b(env, asset_cfg, offset)[0]


def ee_quat(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="panda_hand"),
    offset: tuple = (0.0, 0.0, 0.1034),
) -> torch.Tensor:
    return _ee_pose_b(env, asset_cfg, offset)[1]


def ee_vel(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="panda_hand"),
    offset: tuple | None = None,
) -> torch.Tensor:
    """Linear then angular velocity of the end-effector body, robot-base frame, (N, 6).

    The linear velocity is that of the body's centre of mass (``offset`` None), or of the point
    ``offset`` in the body's link frame -- ``EE_OFFSET`` is FurnitureBench's ``k_ee_link``, where
    ``ee_pos`` is and whose velocity FurnitureBench's sim reports (``furniture.ee_vel_point``)."""
    robot = env.scene[asset_cfg.name]
    body = asset_cfg.body_ids[0]
    ang_w = robot.data.body_ang_vel_w[:, body]
    if offset is None:
        lin_w = robot.data.body_lin_vel_w[:, body]
    else:
        off = const(offset, env.device, ang_w.dtype).expand(env.num_envs, -1)
        r_w = quat_apply(robot.data.body_link_quat_w[:, body], off)
        lin_w = robot.data.body_link_lin_vel_w[:, body] + torch.linalg.cross(ang_w, r_w, dim=-1)
    lin = quat_apply_inverse(robot.data.root_quat_w, lin_w)
    ang = quat_apply_inverse(robot.data.root_quat_w, ang_w)
    return torch.cat([lin, ang], dim=-1)


def gripper_width(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names="panda_finger_joint.*"),
) -> torch.Tensor:
    """Finger separation, (N, 1): the sum of the two finger joint positions."""
    robot = env.scene[asset_cfg.name]
    return robot.data.joint_pos[:, index(asset_cfg.joint_ids, env.device)].sum(dim=-1, keepdim=True)


def parts_poses(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Every part's ``[pos, quat]`` in the robot-base frame, flattened to (N, 7 * P), in the
    order of ``cfg.furniture.part_names`` (upstream's part index order)."""
    pos, quat = parts_pose_b(env)
    return torch.cat([pos, quat], dim=-1).reshape(env.num_envs, -1)


# ------------------------------------------------------------------ FurnitureBench's own layout
def ee_quat_xyzw(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="panda_hand"),
    offset: tuple = (0.0, 0.0, 0.1034),
) -> torch.Tensor:
    """``ee_quat`` in FurnitureBench's ``(x, y, z, w)`` order."""
    q = _ee_pose_b(env, asset_cfg, offset)[1]
    return q[:, [1, 2, 3, 0]]


def _parts_pose_tag(env: "ManagerBasedRLEnv"):
    """Every part's OBJ-frame pose in the AprilTag frame: ``(N, P, 3)`` and ``(N, P, 4)`` wxyz
    with ``w >= 0`` (the collector's ``read_state``, batched)."""
    from ...furniture import frames
    from ...furniture.square_table import MESH_FRAMES

    pos_b, quat_b = parts_pose_b(env)  # USD mesh frames, robot-base frame, wxyz
    N, P = pos_b.shape[:2]
    device, dtype = pos_b.device, pos_b.dtype
    tag_pos = const(frames.TAG_POS_W, device, dtype).expand(N * P, -1)
    tag_quat = const(frames.mat_to_quat_wxyz(frames.TAG_ROT_W), device, dtype).expand(N * P, -1)
    pos_t, quat_t = subtract_frame_transforms(tag_pos, tag_quat, pos_b.reshape(-1, 3), quat_b.reshape(-1, 4))
    # USD frame -> OBJ frame: R_obj = R_usd @ R_mesh (square_table.reset_pose_world's inverse)
    q_mesh = const(frames.mat_to_quat_wxyz(MESH_FRAMES[env.cfg.furniture.mesh_frame]), device, dtype)
    quat_o = canonical_quat(quat_mul(quat_t, q_mesh.expand(N * P, -1)))
    return pos_t.reshape(N, P, 3), quat_o.reshape(N, P, 4)


def parts_poses_tag(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """FurnitureBench's ``parts_poses``: ``(N, 7 P)`` of ``[pos 3, quat xyzw 4]`` per part, the
    OBJ mesh frames in the AprilTag frame -- what upstream's policies observe and what the
    demo pkls hold."""
    pos_t, quat_o = _parts_pose_tag(env)
    N, P = pos_t.shape[:2]
    return torch.cat([pos_t, quat_o[..., [1, 2, 3, 0]]], dim=-1).reshape(N, 7 * P)


def parts_poses_tag_rot6d(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """``parts_poses`` with each orientation as rot6d: ``(N, 9 P)`` of ``[pos 3, rot6d 6]``."""
    from .rot6d import quat_to_rot6d

    pos_t, quat_o = _parts_pose_tag(env)
    N, P = pos_t.shape[:2]
    return torch.cat([pos_t, quat_to_rot6d(quat_o)], dim=-1).reshape(N, 9 * P)


def parts_poses_b_rot6d(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Every part's OBJ-frame pose in the robot-base frame (the EE's frame) as ``(N, 9 P)`` of
    ``[pos 3, rot6d 6]``: ``parts_poses_tag_rot6d``'s parts, in the base frame instead of the
    AprilTag frame (``demos.resip_abs_states`` is its NumPy twin)."""
    from ...furniture import frames
    from ...furniture.square_table import MESH_FRAMES
    from .rot6d import quat_to_rot6d

    pos_b, quat_b = parts_pose_b(env)  # USD mesh frames, robot-base frame, wxyz
    N, P = pos_b.shape[:2]
    q_mesh = const(frames.mat_to_quat_wxyz(MESH_FRAMES[env.cfg.furniture.mesh_frame]), pos_b.device, pos_b.dtype)
    # USD frame -> OBJ frame: R_obj = R_usd @ R_mesh (the origins coincide)
    quat_o = canonical_quat(quat_mul(quat_b.reshape(-1, 4), q_mesh.expand(N * P, -1)))
    return torch.cat([pos_b, quat_to_rot6d(quat_o).reshape(N, P, 6)], dim=-1).reshape(N, 9 * P)


def obstacle_xy_b(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """The front obstacle's x, y in the robot-base frame, ``(N, 2)``: all that changes about it
    (``furniture.obstacle_offset`` slides the three together; their z and rotation are fixed)."""
    robot = env.scene["robot"]
    obstacle = env.scene["obstacle_front"]
    pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, obstacle.data.root_pos_w, obstacle.data.root_quat_w
    )
    return pos_b[:, :2]


def ee_rot6d(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="panda_hand"),
    offset: tuple = (0.0, 0.0, 0.1034),
) -> torch.Tensor:
    """The EE orientation (robot-base frame) as rot6d, (N, 6)."""
    from .rot6d import quat_to_rot6d

    return quat_to_rot6d(_ee_pose_b(env, asset_cfg, offset)[1])


# ------------------------------------------------------- relative to the end-effector (-FBRel-)
#: the front obstacle's z in the robot-base frame: it never changes (``obstacle_xy_b``), and the
#: recordings' ``obstacles_robot`` hold 0.0 -- the EE-relative obstacle is its point (x, y, 0)
OBSTACLE_Z_B = 0.0


def to_ee_frame(p_ee: torch.Tensor, R_ee: torch.Tensor, pos: torch.Tensor, R: torch.Tensor | None = None):
    """Base-frame positions ``pos`` ``(N, K, 3)`` (and rotations ``R`` ``(N, K, 3, 3)``) relative to
    the EE pose ``p_ee`` ``(N, 3)``, ``R_ee`` ``(N, 3, 3)``: ``R_ee^T (pos - p_ee)`` and ``R_ee^T R``.
    ``demos.ee_relative_states`` is its NumPy twin."""
    R_t = R_ee.transpose(-1, -2).unsqueeze(1)  # (N, 1, 3, 3)
    pos_e = (R_t @ (pos - p_ee.unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
    return pos_e if R is None else (pos_e, R_t @ R)


def _ee_frame(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, offset: tuple):
    """The EE pose ``ee_pos`` / ``ee_rot6d`` report: ``(N, 3)`` position and ``(N, 3, 3)`` rotation, base frame."""
    from .rot6d import quat_to_mat

    pos, quat = _ee_pose_b(env, asset_cfg, offset)
    return pos, quat_to_mat(quat)


def ee_vel_ee(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="panda_hand"),
    offset: tuple | None = None,
) -> torch.Tensor:
    """``ee_vel`` (linear then angular, of the same point: ``furniture.ee_vel_point`` sets
    ``offset`` as for ``ee_vel``) in the EE's own frame, ``(N, 6)``."""
    _, R_ee = _ee_frame(env, asset_cfg, (0.0, 0.0, 0.1034))  # the offset is a translation: R is the hand's
    vel = ee_vel(env, asset_cfg, offset).reshape(env.num_envs, 2, 3)
    return (R_ee.transpose(-1, -2).unsqueeze(1) @ vel.unsqueeze(-1)).reshape(env.num_envs, 6)


def parts_poses_ee_rot6d(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="panda_hand"),
    offset: tuple = (0.0, 0.0, 0.1034),
) -> torch.Tensor:
    """``parts_poses_b_rot6d``'s parts relative to the EE pose (``ee_pos`` / ``ee_rot6d``), ``(N, 9 P)``
    of ``[pos 3, rot6d 6]``: each position ``R_ee^T (p - p_ee)``, each orientation ``R_ee^T R``."""
    from .rot6d import mat_to_rot6d, rot6d_to_mat

    p_ee, R_ee = _ee_frame(env, asset_cfg, offset)
    parts = parts_poses_b_rot6d(env).reshape(env.num_envs, -1, 9)
    pos, R = to_ee_frame(p_ee, R_ee, parts[..., :3], rot6d_to_mat(parts[..., 3:]))
    return torch.cat([pos, mat_to_rot6d(R)], dim=-1).reshape(env.num_envs, -1)


def obstacle_pos_ee(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="panda_hand"),
    offset: tuple = (0.0, 0.0, 0.1034),
) -> torch.Tensor:
    """The front obstacle's point ``(x, y, OBSTACLE_Z_B)`` (``obstacle_xy_b``'s x, y) in the EE frame, ``(N, 3)``."""
    p_ee, R_ee = _ee_frame(env, asset_cfg, offset)
    xy = obstacle_xy_b(env)
    point = torch.cat([xy, torch.full_like(xy[:, :1], OBSTACLE_Z_B)], dim=-1)
    return to_ee_frame(p_ee, R_ee, point.unsqueeze(1)).squeeze(1)


def canonical_quat(q_wxyz: torch.Tensor) -> torch.Tensor:
    """``q`` with ``w >= 0``: the sign the collector's ``mat_to_quat_wxyz`` gives the demos'
    part quaternions. ``q`` and ``-q`` are one rotation but two input vectors to a policy."""
    return torch.where(q_wxyz[..., :1] < 0, -q_wxyz, q_wxyz)


def ee_wrench(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """The external EE wrench estimated from the joint torques (``assembly/wrench.py``),
    filtered, base frame: force then torque, (N, 6). Not in any group by default."""
    return env.wrench.filtered


def obstacle_pose_tag_rot6d(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """The front obstacle's pose in the AprilTag frame as ``[pos 3, rot6d 6]``, ``(N, 9)``: what
    ResiP's policies observe after the five parts (its ``parts_poses`` is 42 wide, the front
    obstacle appended), here in the parts' rot6d form. It moves with ``furniture.obstacle_offset``."""
    from ...furniture import frames
    from .rot6d import quat_to_rot6d

    robot = env.scene["robot"]
    obstacle = env.scene["obstacle_front"]
    pos_b, quat_b = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, obstacle.data.root_pos_w, obstacle.data.root_quat_w
    )
    N, device, dtype = pos_b.shape[0], pos_b.device, pos_b.dtype
    tag_pos = const(frames.TAG_POS_W, device, dtype).expand(N, -1)
    tag_quat = const(frames.mat_to_quat_wxyz(frames.TAG_ROT_W), device, dtype).expand(N, -1)
    pos_t, quat_t = subtract_frame_transforms(tag_pos, tag_quat, pos_b, quat_b)
    return torch.cat([pos_t, quat_to_rot6d(canonical_quat(quat_t))], dim=-1)
