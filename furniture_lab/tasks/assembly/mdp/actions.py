"""FurnitureBench's 8-d action as a manager-based action term, and the absolute-pose variant.

Upstream's action is ``[dpos 3 (m, robot frame), dquat 4 (xyzw, the rotation delta in the
EE's own frame), gripper (< 0 opens, else closes)]``. This term takes exactly that and drives
the task's own two terms with it: the OSC ``pose_rel`` arm term (``dpos / position_scale``,
the delta rotation re-expressed in the base frame as an axis-angle ``/ orientation_scale`` --
the term multiplies the scales back) and the binary gripper (sign flipped: Isaac Lab's
closes on ``< 0``). It is the collector's ``env_action`` as a term, so a policy trained on
the demos' actions sends them to the env as they are.

:class:`FurnitureBenchAbsAction` is the ``-FBAbs-v0`` interface: ``[goal_pos 3 (base frame),
goal_rot6d 6, gripper]`` -- the absolute EE target the demo's delta commanded
(``furniture_lab.demos.fb_abs_action``), as ResiP's ``control_mode: pos`` -- driving the OSC in
``pose_abs`` mode (target ``[pos, quat wxyz]``, scales 1) and the same gripper term.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul

from .observations import _ee_pose_b

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from isaaclab.managers import SceneEntityCfg


def arm_scales(arm_cfg) -> tuple:
    """``(position_scale, orientation_scale)`` of the task's relative arm term: the OSC term's
    two fields, or the differential-IK term's per-dim ``scale`` (dims 0 and 3)."""
    if hasattr(arm_cfg, "position_scale"):
        return float(arm_cfg.position_scale), float(arm_cfg.orientation_scale)
    scale = arm_cfg.scale
    if isinstance(scale, (int, float)):
        return float(scale), float(scale)
    return float(scale[0]), float(scale[3])


class _GripperHysteresis:
    """FurnitureRLSimEnv's gripper: the command (FurnitureBench sign, > 0 closes) flips only
    when the action's sign changes and |action| > margin; otherwise the last one holds."""

    def __init__(self, num_envs: int, device, margin: float) -> None:
        self.margin = float(margin)
        self.state = -torch.ones(num_envs, 1, device=device)  # open

    def __call__(self, grip: torch.Tensor) -> torch.Tensor:
        if self.margin <= 0:
            return grip
        flip = (torch.sign(grip) != torch.sign(self.state)) & (grip.abs() > self.margin)
        self.state = torch.where(flip, torch.sign(grip), self.state)
        return self.state.clone()

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self.state[:] = -1.0
        else:
            self.state[env_ids] = -1.0


class FurnitureBenchAction(ActionTerm):
    """See the module docstring."""

    cfg: "FurnitureBenchActionCfg"

    def __init__(self, cfg: "FurnitureBenchActionCfg", env: "ManagerBasedEnv") -> None:
        super().__init__(cfg, env)
        self.arm = cfg.arm.class_type(cfg.arm, env)
        self.gripper = cfg.gripper.class_type(cfg.gripper, env)
        self._ee_cfg = SceneEntityCfg(cfg.asset_name, body_names=cfg.arm.body_name)
        self._ee_cfg.resolve(env.scene)
        self._offset = tuple(cfg.arm.body_offset.pos) if cfg.arm.body_offset is not None else (0.0, 0.0, 0.0)
        self._raw = torch.zeros(self.num_envs, 8, device=self.device)
        # what the two terms receive: the arm's 6 (dpos, axis-angle) and the gripper's 1
        self._processed = torch.zeros(self.num_envs, 7, device=self.device)
        self._grip = _GripperHysteresis(self.num_envs, self.device, getattr(env.cfg.furniture, "gripper_margin", 0.0))

    @property
    def action_dim(self) -> int:
        return 8

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw[:] = actions
        pos_scale, rot_scale = arm_scales(self.cfg.arm)
        dpos = actions[:, 0:3] / pos_scale
        q_d = actions[:, [6, 3, 4, 5]]  # xyzw -> wxyz
        _, q_ee = _ee_pose_b(self._env, self._ee_cfg, self._offset)
        q_base = quat_mul(quat_mul(q_ee, q_d), quat_conjugate(q_ee))  # the same rotation, base frame
        drot = axis_angle_from_quat(q_base) / rot_scale
        grip = -self._grip(actions[:, 7:8])  # upstream: < 0 opens; Isaac Lab's binary term: < 0 closes
        self._processed[:] = torch.cat([dpos, drot, grip], dim=-1)
        self.arm.process_actions(self._processed[:, 0:6])
        self.gripper.process_actions(self._processed[:, 6:7])

    def apply_actions(self) -> None:
        self.arm.apply_actions()
        self.gripper.apply_actions()

    def reset(self, env_ids=None) -> None:
        self.arm.reset(env_ids)
        self.gripper.reset(env_ids)
        self._grip.reset(env_ids)
        if env_ids is None:
            self._raw.zero_()
            self._processed.zero_()
        else:
            self._raw[env_ids] = 0.0
            self._processed[env_ids] = 0.0


@configclass
class FurnitureBenchActionCfg(ActionTermCfg):
    """Holds the task's arm (OSC ``pose_rel``) and gripper term cfgs; ``asset_name`` is theirs."""

    class_type: type = FurnitureBenchAction
    asset_name: str = "robot"
    arm: object = MISSING
    gripper: object = MISSING


class FurnitureBenchAbsAction(FurnitureBenchAction):
    """See the module docstring: 10-d absolute target, rot6d orientation."""

    def __init__(self, cfg: "FurnitureBenchAbsActionCfg", env: "ManagerBasedEnv") -> None:
        super().__init__(cfg, env)
        self._raw = torch.zeros(self.num_envs, 10, device=self.device)
        self._processed = torch.zeros(self.num_envs, 8, device=self.device)  # arm 7 (pos, quat), gripper 1
        self._grip = _GripperHysteresis(self.num_envs, self.device, getattr(env.cfg.furniture, "gripper_margin", 0.0))

    @property
    def action_dim(self) -> int:
        return 10

    def process_actions(self, actions: torch.Tensor) -> None:
        from .rot6d import rot6d_to_quat

        self._raw[:] = actions
        quat = rot6d_to_quat(actions[:, 3:9])  # wxyz, the OSC's pose_abs convention
        grip = -self._grip(actions[:, 9:10])  # upstream: < 0 opens; Isaac Lab's binary term: < 0 closes
        self._processed[:] = torch.cat([actions[:, 0:3], quat, grip], dim=-1)
        self.arm.process_actions(self._processed[:, 0:7])
        self.gripper.process_actions(self._processed[:, 7:8])


@configclass
class FurnitureBenchAbsActionCfg(FurnitureBenchActionCfg):
    """``arm`` must be the OSC term in ``pose_abs`` mode with unit scales."""

    class_type: type = FurnitureBenchAbsAction
