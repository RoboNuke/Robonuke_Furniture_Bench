"""Observation terms: FurnitureBench's ``robot_state`` and ``parts_poses``, in the robot-base
frame, quaternions ``(w, x, y, z)``.

Upstream's state observation is ``ee_pos``, ``ee_quat`` (x, y, z, w), ``ee_pos_vel``,
``ee_ori_vel``, ``gripper_width`` plus ``parts_poses`` in the AprilTag frame. This port keeps
the same quantities and changes two conventions, both deliberate: poses are in the robot-base
frame (what the controller and the parts share), and quaternions are Isaac Lab's ``wxyz``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply_inverse, quat_mul, subtract_frame_transforms

from .state import parts_pose_b

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ee_pose_b(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, offset: tuple):
    """Pose of the end-effector body (plus a fixed offset along it) in the robot-base frame."""
    robot = env.scene[asset_cfg.name]
    body = asset_cfg.body_ids[0]
    pos_w = robot.data.body_pos_w[:, body]
    quat_w = robot.data.body_quat_w[:, body]
    off = torch.tensor(offset, device=env.device, dtype=pos_w.dtype).expand(env.num_envs, -1)
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
) -> torch.Tensor:
    """Linear then angular velocity of the end-effector body, robot-base frame, (N, 6)."""
    robot = env.scene[asset_cfg.name]
    body = asset_cfg.body_ids[0]
    lin = quat_apply_inverse(robot.data.root_quat_w, robot.data.body_lin_vel_w[:, body])
    ang = quat_apply_inverse(robot.data.root_quat_w, robot.data.body_ang_vel_w[:, body])
    return torch.cat([lin, ang], dim=-1)


def gripper_width(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names="panda_finger_joint.*"),
) -> torch.Tensor:
    """Finger separation, (N, 1): the sum of the two finger joint positions."""
    robot = env.scene[asset_cfg.name]
    return robot.data.joint_pos[:, asset_cfg.joint_ids].sum(dim=-1, keepdim=True)


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
    tag_pos = torch.tensor(frames.TAG_POS_W, device=device, dtype=dtype).expand(N * P, -1)
    tag_quat = torch.tensor(frames.mat_to_quat_wxyz(frames.TAG_ROT_W), device=device, dtype=dtype).expand(N * P, -1)
    pos_t, quat_t = subtract_frame_transforms(tag_pos, tag_quat, pos_b.reshape(-1, 3), quat_b.reshape(-1, 4))
    # USD frame -> OBJ frame: R_obj = R_usd @ R_mesh (square_table.reset_pose_world's inverse)
    q_mesh = torch.tensor(frames.mat_to_quat_wxyz(MESH_FRAMES[env.cfg.furniture.mesh_frame]), device=device, dtype=dtype)
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


def ee_rot6d(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="panda_hand"),
    offset: tuple = (0.0, 0.0, 0.1034),
) -> torch.Tensor:
    """The EE orientation (robot-base frame) as rot6d, (N, 6)."""
    from .rot6d import quat_to_rot6d

    return quat_to_rot6d(_ee_pose_b(env, asset_cfg, offset)[1])


def canonical_quat(q_wxyz: torch.Tensor) -> torch.Tensor:
    """``q`` with ``w >= 0``: the sign the collector's ``mat_to_quat_wxyz`` gives the demos'
    part quaternions. ``q`` and ``-q`` are one rotation but two input vectors to a policy."""
    return torch.where(q_wxyz[..., :1] < 0, -q_wxyz, q_wxyz)


def ee_wrench(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """The external EE wrench estimated from the joint torques (``assembly/wrench.py``),
    filtered, base frame: force then torque, (N, 6). Not in any group by default."""
    return env.wrench.filtered
