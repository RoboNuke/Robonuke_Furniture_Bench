"""The env class: Isaac Lab's ``ManagerBasedRLEnv`` plus the package's episode-metric channel.

``robonuke_rl_core`` reads ``infos["episode_metrics_to_log"]["episode/success"]`` -- a
``(num_envs,)`` tensor, valid for the envs that finished this step -- for its success rate
and best-checkpoint selection. A manager-based env publishes nothing of the kind, and the
package's Forge wrapper that would is Forge-only, so the env publishes it itself.

The value has to be captured **before** the reset clears the assembly latch for the envs
that finished, which is why it is taken in ``_reset_idx`` and attached to ``extras`` after
the step. ``episode/out_of_bounds`` (1 where the ``part_out_of_limits`` termination ended the
episode; its mean over episodes, the share of out-of-bounds failures) is captured the same way,
for every cfg that has that term. ``episode/isaac_steps`` is the ending episode's length on Isaac's own clock
(``episode_length_buf`` before the reset), to check a wrapper's or learner's episode bookkeeping against.

The env also owns the joint-torque wrench estimator (``wrench.py``, ``env.wrench``). It
has to run every physics substep, and a manager-based env gives its terms no substep hook;
the action manager's ``apply_action`` is called once per substep, just before the sim
steps, so the env runs the estimator from there -- the moment Forge computes its force
sensor too.

Two opt-in eval channels, both read after the terminations and before the resets clear
anything (the termination manager's ``compute`` is wrapped for it): ``furniture.log_wrench``
publishes the wrench estimate as ``infos["eval_state"]["ee_wrench"]`` (six columns of an eval's
per-step parquet), ``furniture.track_phases`` one_leg's task phases (``phases.py``).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .mdp.events import randomness
from .mdp.state import tracker
from .phases import OneLegPhases
from .wrench import JointTorqueWrench


class AssemblyEnv(ManagerBasedRLEnv):
    #: how a recorder names each termination term (robonuke_rl_core.recording.termination_causes)
    termination_labels = {"success": "success", "part_out_of_limits": "out of bounds", "time_out": "timeout"}

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        self._episode_success = None
        #: 1 where the ending episode was ended by the part_out_of_limits termination (episode/out_of_bounds);
        #: None when the cfg has no such term
        self._episode_out_of_bounds = None
        #: furniture.track_phases: one_leg's phase tracker (phases.py); furniture.log_wrench: the
        #: wrench estimate as it was after the step's terminations (before any reset zeroed it)
        self.phases = None
        self._wrench_out = None
        #: ResiP's reset kick on the parts (furniture.randomness kick_*), made by its
        #: reset event (mdp.events.resip_reset); None until then
        self.part_kick = None
        # furniture.*_friction / gripper_* / ee_vel_point / part_masses are plain fields a config file can
        # override after the cfg's __post_init__ ran; push them into the event terms, the
        # actuator and the observation terms here, where the scene has not been built yet
        cfg.apply_assets()
        cfg.apply_friction()
        cfg.apply_gripper()
        cfg.apply_observations()
        cfg.apply_masses()
        # an unknown furniture.randomness fails here, not at the first reset; every level but
        # port runs through the reset_parts term, so a cfg without it would silently skip it
        if randomness(cfg.furniture) != "port":
            term = getattr(cfg.events, "reset_parts", None)
            if term is None or getattr(term.func, "__name__", "") != "randomize_parts_pose":
                raise ValueError(
                    f"furniture.randomness={cfg.furniture.randomness!r} needs events.reset_parts = "
                    f"mdp.randomize_parts_pose (it runs that level's reset); got {term!r}"
                )
        super().__init__(cfg, render_mode, **kwargs)
        self._episode_success = torch.zeros(self.num_envs, device=self.device)
        #: the ending episode's length on Isaac's own clock (episode_length_buf before the reset): episode/isaac_steps
        self._episode_steps = torch.zeros(self.num_envs, device=self.device)
        if "part_out_of_limits" in self.termination_manager.active_terms:
            self._episode_out_of_bounds = torch.zeros(self.num_envs, device=self.device)
        self.wrench = JointTorqueWrench(self, cfg.furniture)
        apply_action = self.action_manager.apply_action

        def apply_action_then_estimate():
            apply_action()
            if self.part_kick is not None:
                self.part_kick.apply()  # before the scene writes the wrenches and the sim steps
            self.wrench.update()

        self.action_manager.apply_action = apply_action_then_estimate
        if cfg.furniture.track_phases:
            self.phases = OneLegPhases(self, cfg.furniture)
        if self.phases is not None or cfg.furniture.log_wrench:
            compute = self.termination_manager.compute

            def compute_then_track():
                dones = compute()
                # after the physics and the terminations, before the resets clear anything
                if cfg.furniture.log_wrench:
                    self._wrench_out = self.wrench.filtered.clone()
                if self.phases is not None:
                    self.phases.update()
                    # the no-progress reset (furniture.stall_steps_*) may have ended more episodes
                    return self.termination_manager.dones
                return dones

            self.termination_manager.compute = compute_then_track

    def _reset_idx(self, env_ids: Sequence[int]):
        if self._episode_success is not None and len(env_ids) > 0:
            # all pairs assembled at the moment the episode ended, upstream's success
            self._episode_success[env_ids] = tracker(self).all_assembled()[env_ids].float()
            self._episode_steps[env_ids] = self.episode_length_buf[env_ids].float()
            if self._episode_out_of_bounds is not None:
                self._episode_out_of_bounds[env_ids] = self.termination_manager.get_term("part_out_of_limits")[env_ids].float()
            if self.cfg.furniture.log_resets:
                self._log_reset(env_ids)
            self.wrench.reset(env_ids)
            if self.phases is not None:
                self.phases.reset(env_ids)
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
        extras["episode_metrics_to_log"] = {"episode/success": self._episode_success, "episode/isaac_steps": self._episode_steps}
        if self._episode_out_of_bounds is not None:
            # the share of finished episodes a part leaving its limits ended (their mean over episodes)
            extras["episode_metrics_to_log"]["episode/out_of_bounds"] = self._episode_out_of_bounds
        state = {}
        if self._wrench_out is not None:
            state["ee_wrench"] = self._wrench_out
        if self.phases is not None:
            extras["episode_metrics_to_log"].update(self.phases.episode)
            state.update(self.phases.state)
        if state:
            extras.setdefault("eval_state", {}).update(state)
        return obs, rew, terminated, truncated, extras
