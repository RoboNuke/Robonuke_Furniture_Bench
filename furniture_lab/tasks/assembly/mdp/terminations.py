"""Termination terms.

FurnitureSim ends an episode on ``all_assembled()`` or the step limit. FurnitureBench's
real-robot env additionally ends it when a part leaves the robot's positional limits
(``Furniture.parts_out_pos_lim``: in the robot frame, x outside (0.3, 0.8), y outside
(-0.55, 0.55) or z above 0.4; the lower z bound is not checked). :func:`part_out_of_limits`
is that rule, plus "fell below the table" for the sim; a part out there can never be
assembled, so the episode ends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...furniture.frames import EE_POSITION_LIMITS, TABLE_SURFACE_Z
from .state import parts_pose_b, tracker

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def all_assembled(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Every pair in ``should_be_assembled`` is (latched) assembled, (N,) bool."""
    return tracker(env).all_assembled()


def part_out_of_limits(env: "ManagerBasedRLEnv", margin: float = 0.05) -> torch.Tensor:
    """Any part's origin is outside upstream's robot-frame position limits (x, y box and the z
    ceiling), or ``margin`` below the table surface, (N,) bool."""
    pos, _ = parts_pose_b(env)
    (x0, x1), (y0, y1), (_, z1) = EE_POSITION_LIMITS
    out = (
        (pos[..., 0] < x0)
        | (pos[..., 0] > x1)
        | (pos[..., 1] < y0)
        | (pos[..., 1] > y1)
        | (pos[..., 2] > z1)
        | (pos[..., 2] < TABLE_SURFACE_Z - margin)
    )
    return out.any(dim=-1)


part_dropped = part_out_of_limits  # the old name
