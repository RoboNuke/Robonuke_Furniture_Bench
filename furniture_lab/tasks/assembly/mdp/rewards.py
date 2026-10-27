"""Reward terms.

FurnitureBench's reward is sparse: ``compute_assemble`` pays 1 for each pair that becomes
assembled, once per pair per episode (``total_reward`` is 1.0 for ``one_leg``, 4.0 for the
full table). That is :func:`newly_assembled`. Nothing dense is defined here on purpose --
upstream has none -- so a shaping term is an explicit addition to the env cfg, not a
default. :func:`each_step` is a constant per-step term, which the env cfg carries at weight 0
(off): at weight -1 it is Policy Decorator's -1 per step. :func:`phase_done` pays on the step
one_leg gets past a task phase (``furniture.track_phases``); the env cfg carries one term per
phase (``corner_done``, ``grasp_done``, ``insert_done``), each at weight 0 (off).
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


#: the phases ``phase_done`` rewards, by name, as ``assembly/phases.py`` numbers them
PHASE_REWARD_NAMES = ("corner", "grasp", "insert")


def phase_done(env: "ManagerBasedRLEnv", phase: str) -> torch.Tensor:
    """1 on the step an env gets past one_leg's ``phase`` (``corner``: the top pushed into the corner
    and let go; ``grasp``: leg 4 held and lifted; ``insert``: the leg left standing in its hole),
    else 0, (N,): the criteria of ``furniture.track_phases`` (``assembly/phases.py``), which must be
    on. Each phase pays once per episode (the phases only move forward)."""
    if phase not in PHASE_REWARD_NAMES:
        raise ValueError(f"phase_done: phase must be one of {PHASE_REWARD_NAMES}, got {phase!r}")
    tracker_ = getattr(env, "phases", None)
    if tracker_ is None:
        raise ValueError(
            f"the {phase} phase reward reads one_leg's phase tracker: set task.cfg.furniture.track_phases: true"
        )
    return tracker_.newly[:, PHASE_REWARD_NAMES.index(phase)].to(torch.float32)
