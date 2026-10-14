"""The env class: Isaac Lab's ``ManagerBasedRLEnv`` plus the package's episode-metric channel.

``robonuke_rl_core`` reads ``infos["episode_metrics_to_log"]["episode/success"]`` -- a
``(num_envs,)`` tensor, valid for the envs that finished this step -- for its success rate
and best-checkpoint selection. A manager-based env publishes nothing of the kind, and the
package's Forge wrapper that would is Forge-only, so the env publishes it itself.

The value has to be captured **before** the reset clears the assembly latch for the envs
that finished, which is why it is taken in ``_reset_idx`` and attached to ``extras`` after
the step.

The env also owns the joint-torque wrench estimator (``wrench.py``, ``env.wrench``). It
has to run every physics substep, and a manager-based env gives its terms no substep hook;
the action manager's ``apply_action`` is called once per substep, just before the sim
steps, so the env runs the estimator from there -- the moment Forge computes its force
sensor too.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .mdp.state import tracker
from .wrench import JointTorqueWrench


class AssemblyEnv(ManagerBasedRLEnv):
    #: how a recorder names each termination term (robonuke_rl_core.recording.termination_causes)
    termination_labels = {"success": "success", "part_out_of_limits": "out of bounds", "time_out": "timeout"}

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        self._episode_success = None
        # furniture.part_friction / table_friction / gripper_* are plain fields a config file
        # can override after the cfg's __post_init__ ran; push them into the event terms and
        # the actuator here, where the scene has not been built yet
        cfg.apply_friction()
        cfg.apply_gripper()
        cfg.apply_visuals()
        super().__init__(cfg, render_mode, **kwargs)
        self._episode_success = torch.zeros(self.num_envs, device=self.device)
        self.wrench = JointTorqueWrench(self, cfg.furniture)
        apply_action = self.action_manager.apply_action

        def apply_action_then_estimate():
            apply_action()
            self.wrench.update()

        self.action_manager.apply_action = apply_action_then_estimate

    def _reset_idx(self, env_ids: Sequence[int]):
        if self._episode_success is not None and len(env_ids) > 0:
            # all pairs assembled at the moment the episode ended, upstream's success
            self._episode_success[env_ids] = tracker(self).all_assembled()[env_ids].float()
            if self.cfg.furniture.log_resets:
                self._log_reset(env_ids)
            self.wrench.reset(env_ids)
        super()._reset_idx(env_ids)

    def _log_reset(self, env_ids: Sequence[int]) -> None:
        """One line per reset: which envs, after how many steps, and which termination fired."""
        ids = torch.as_tensor(env_ids, device=self.device)
        causes = []
        for name in self.termination_manager.active_terms:
            fired = self.termination_manager.get_term(name)[ids]
            if fired.any():
                causes.append(f"{name}x{int(fired.sum())}")
        steps = self.episode_length_buf[ids]
        print(
            f"[reset] step {self.common_step_counter}: envs {ids.tolist()} after "
            f"{steps.min().item()}-{steps.max().item()} steps, cause {' '.join(causes) or 'external reset()'}",
            flush=True,
        )

    def step(self, action: torch.Tensor):
        obs, rew, terminated, truncated, extras = super().step(action)
        extras["episode_metrics_to_log"] = {"episode/success": self._episode_success}
        return obs, rew, terminated, truncated, extras
