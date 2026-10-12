"""The env class: Isaac Lab's ``ManagerBasedRLEnv`` plus the package's episode-metric channel.

``robonuke_rl_core`` reads ``infos["episode_metrics_to_log"]["episode/success"]`` -- a
``(num_envs,)`` tensor, valid for the envs that finished this step -- for its success rate
and best-checkpoint selection. A manager-based env publishes nothing of the kind, and the
package's Forge wrapper that would is Forge-only, so the env publishes it itself.

The value has to be captured **before** the reset clears the assembly latch for the envs
that finished, which is why it is taken in ``_reset_idx`` and attached to ``extras`` after
the step.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .mdp.state import tracker


class AssemblyEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        self._episode_success = None
        super().__init__(cfg, render_mode, **kwargs)
        self._episode_success = torch.zeros(self.num_envs, device=self.device)

    def _reset_idx(self, env_ids: Sequence[int]):
        if self._episode_success is not None and len(env_ids) > 0:
            # all pairs assembled at the moment the episode ended, upstream's success
            self._episode_success[env_ids] = tracker(self).all_assembled()[env_ids].float()
            if self.cfg.furniture.log_resets:
                self._log_reset(env_ids)
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
