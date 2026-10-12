"""Reset events: FurnitureBench's part-pose randomization, and clearing the assembly latch.

Upstream (``Furniture.randomize_init_pose`` / ``Part.randomize_init_pose``) perturbs each
part's reset pose by a uniform (x, y) offset and a uniform yaw about the table normal, then
rejects the whole draw if any two parts' footprint boxes overlap or a part leaves the parts
position limits, and redraws. (Here a yawed box is bounded by its axis-aligned box, which
is slightly more conservative than upstream's unrotated boxes.) The levels are

    low:    +-0.015 m, +-15 deg      (the default)
    medium: +-0.05 m,  +-45 deg
    high:   fixed poses from a table (``high_rand_reset_pos``), not ported

Here the draw is batched: ``max_tries`` rounds of sampling, each round re-sampling only the
envs whose draw still collides, and a part that never finds a free draw keeps its nominal
pose (upstream gives up after 300k tries and logs an error; this logs a warning).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import torch

from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

from ...furniture.square_table import PARTS
from .state import tracker

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def randomize_parts_pose(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    pos_range: float = 0.015,
    yaw_range_deg: float = 15.0,
    max_tries: int = 20,
) -> None:
    """Perturb every part's default root pose by a planar offset and a yaw, collision-free.

    Runs after ``reset_scene_to_default`` has put the parts back at their nominal poses, and
    writes root pose and zero velocity for ``env_ids``.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    if n == 0:
        return
    names = env.cfg.furniture.part_names
    specs = {p.name: p for p in PARTS}
    P = len(names)
    device = env.device

    # default_root_state is in the env's local frame: add the env origins, as
    # reset_scene_to_default does
    nominal_pos = torch.stack(
        [env.scene[name].data.default_root_state[env_ids, :3] for name in names], dim=1
    ) + env.scene.env_origins[env_ids].unsqueeze(1)  # (n, P, 3) world frame
    nominal_quat = torch.stack(
        [env.scene[name].data.default_root_state[env_ids, 3:7] for name in names], dim=1
    )
    foot = torch.tensor([specs[name].footprint for name in names], device=device)  # (P, 2)
    eye = torch.eye(P, dtype=torch.bool, device=device)

    offset = torch.zeros(n, P, 3, device=device)
    yaw = torch.zeros(n, P, device=device)
    pending = torch.ones(n, dtype=torch.bool, device=device)
    yaw_range = math.radians(yaw_range_deg)
    for _ in range(max_tries):
        idx = pending.nonzero(as_tuple=False).squeeze(-1)
        if len(idx) == 0:
            break
        offset[idx, :, :2] = (torch.rand(len(idx), P, 2, device=device) * 2 - 1) * pos_range
        yaw[idx] = (torch.rand(len(idx), P, device=device) * 2 - 1) * yaw_range
        xy = nominal_pos[idx, :, :2] + offset[idx, :, :2]  # (k, P, 2)
        # each part's footprint box, rotated by its yaw, bounded by an axis-aligned box
        c, s = torch.cos(yaw[idx]).abs(), torch.sin(yaw[idx]).abs()  # (k, P)
        hx = foot[:, 0] * c + foot[:, 1] * s
        hy = foot[:, 0] * s + foot[:, 1] * c
        dx = (xy[:, :, None, 0] - xy[:, None, :, 0]).abs()  # (k, P, P)
        dy = (xy[:, :, None, 1] - xy[:, None, :, 1]).abs()
        overlap = (dx < hx[:, :, None] + hx[:, None, :]) & (dy < hy[:, :, None] + hy[:, None, :])
        pending[idx] = (overlap & ~eye).any(dim=(1, 2))
    if pending.any():
        print(
            f"[furniture] {int(pending.sum())} env(s) found no collision-free part layout in "
            f"{max_tries} tries; using the nominal poses there",
            flush=True,
        )
        offset[pending] = 0.0
        yaw[pending] = 0.0

    zeros = torch.zeros_like(yaw)
    yaw_quat = quat_from_euler_xyz(zeros, zeros, yaw)  # (n, P, 4), about world z
    pos = nominal_pos + offset
    quat = quat_mul(yaw_quat, nominal_quat)
    vel = torch.zeros(n, 6, device=device)
    for k, name in enumerate(names):
        asset = env.scene[name]
        asset.write_root_pose_to_sim(torch.cat([pos[:, k], quat[:, k]], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(vel, env_ids=env_ids)


def reset_assembly_tracker(env: "ManagerBasedRLEnv", env_ids: torch.Tensor) -> None:
    """Clear the assembled-pair latch for the envs that restart."""
    tracker(env).reset(env_ids)
