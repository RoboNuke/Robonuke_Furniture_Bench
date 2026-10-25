"""Reward terms.

FurnitureBench's reward is sparse: ``compute_assemble`` pays 1 for each pair that becomes
assembled, once per pair per episode (``total_reward`` is 1.0 for ``one_leg``, 4.0 for the
full table). That is :func:`newly_assembled`. Nothing dense is defined here on purpose --
upstream has none -- so a shaping term is an explicit addition to the env cfg, not a
default. :func:`each_step` is a constant per-step term, which the env cfg carries at weight 0
(off): at weight -1 it is Policy Decorator's -1 per step.
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


def each_step(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """1 on every step, the step that ends the episode included, (N,). With a negative weight a
    constant cost per step: since success ends the episode, the return then counts the steps
    to success (Policy Decorator's -1 per step, which its success step pays too)."""
    return torch.ones(env.num_envs, device=env.device)
