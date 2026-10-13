"""Shared per-env task state: the part poses this step, and the assembled-pair latch.

Several managers need the same two things -- the reward (newly assembled pairs), the
success termination (all pairs assembled) and the reset event (clear the latch) -- and the
manager API gives each term only the env. So the :class:`AssemblyTracker` hangs off the env
and every term reaches it through :func:`tracker`, which also updates it **once per env
step**, whichever term asks first (the termination manager runs before the reward manager
in ``ManagerBasedRLEnv.step``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from isaaclab.utils.math import subtract_frame_transforms

from ...furniture.assembly import AssemblyTracker
from ...furniture.square_table import pairs_for, pos_threshold_in, rel_pose_targets

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_ATTR = "_furniture_assembly"


def parts_pose_w(env: "ManagerBasedRLEnv") -> Tuple[torch.Tensor, torch.Tensor]:
    """``(N, P, 3)`` positions and ``(N, P, 4)`` (w, x, y, z) orientations of the parts, world
    frame, in the order of ``env.cfg.furniture.part_names``."""
    names = env.cfg.furniture.part_names
    pos = torch.stack([env.scene[name].data.root_pos_w for name in names], dim=1)
    quat = torch.stack([env.scene[name].data.root_quat_w for name in names], dim=1)
    return pos, quat


def parts_pose_b(env: "ManagerBasedRLEnv") -> Tuple[torch.Tensor, torch.Tensor]:
    """The same, in the robot-base frame (FurnitureBench publishes part poses in a fixed
    table frame; the base frame is this port's equivalent)."""
    pos_w, quat_w = parts_pose_w(env)
    robot = env.scene["robot"]
    P = pos_w.shape[1]
    root_pos = robot.data.root_pos_w.unsqueeze(1).expand(-1, P, -1)
    root_quat = robot.data.root_quat_w.unsqueeze(1).expand(-1, P, -1)
    pos_b, quat_b = subtract_frame_transforms(
        root_pos.reshape(-1, 3), root_quat.reshape(-1, 4), pos_w.reshape(-1, 3), quat_w.reshape(-1, 4)
    )
    return pos_b.reshape(pos_w.shape), quat_b.reshape(quat_w.shape)


def tracker(env: "ManagerBasedRLEnv") -> AssemblyTracker:
    """The env's tracker, created on first use and updated once for the current step."""
    t = getattr(env, _ATTR, None)
    if t is None:
        fcfg = env.cfg.furniture
        pairs = pairs_for(fcfg.task)
        t = AssemblyTracker(
            env.num_envs,
            pairs,
            torch.as_tensor(rel_pose_targets(pairs, fcfg.mesh_frame)),
            pos_threshold_in(fcfg.mesh_frame, fcfg.assembled_pos_threshold),  # OBJ axes -> this frame's
            fcfg.ori_bound,
            env.device,
        )
        t._stamp = -1
        setattr(env, _ATTR, t)
    if t._stamp != env.common_step_counter:
        pos, quat = parts_pose_w(env)  # relative poses: any common frame will do
        t.update(pos, quat)
        t._stamp = env.common_step_counter
    return t
