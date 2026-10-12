"""Termination terms.

Upstream ends an episode on ``all_assembled()`` or the step limit. :func:`part_dropped` is an
addition of this port: a part that falls off the table (FurnitureBench's sim has walls on
three sides and simply lets it fall) can never be assembled, so the episode ends there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...furniture.frames import TABLE_SURFACE_Z
from .state import parts_pose_b, tracker

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def all_assembled(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Every pair in ``should_be_assembled`` is (latched) assembled, (N,) bool."""
    return tracker(env).all_assembled()


def part_dropped(env: "ManagerBasedRLEnv", margin: float = 0.05) -> torch.Tensor:
    """Any part's origin is ``margin`` below the table surface, (N,) bool."""
    pos, _ = parts_pose_b(env)
    return (pos[:, :, 2] < TABLE_SURFACE_Z - margin).any(dim=-1)
