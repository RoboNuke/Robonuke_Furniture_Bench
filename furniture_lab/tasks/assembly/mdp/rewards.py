"""Reward terms.

FurnitureBench's reward is sparse: ``compute_assemble`` pays 1 for each pair that becomes
assembled, once per pair per episode (``total_reward`` is 1.0 for ``one_leg``, 4.0 for the
full table). That is :func:`newly_assembled`. Nothing dense is defined here on purpose --
upstream has none -- so a shaping term is an explicit addition to the env cfg, not a
default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .state import tracker

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def newly_assembled(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Number of pairs that became assembled this step, (N,)."""
    return tracker(env).newly_assembled.sum(dim=-1).float()
