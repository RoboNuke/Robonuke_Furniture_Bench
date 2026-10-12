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
from isaaclab.utils.math import combine_frame_transforms, quat_apply_inverse, subtract_frame_transforms

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
