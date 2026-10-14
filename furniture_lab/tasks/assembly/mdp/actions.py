"""FurnitureBench's 8-d action as a manager-based action term.

Upstream's action is ``[dpos 3 (m, robot frame), dquat 4 (xyzw, the rotation delta in the
EE's own frame), gripper (< 0 opens, else closes)]``. This term takes exactly that and drives
the task's own two terms with it: the OSC ``pose_rel`` arm term (``dpos / position_scale``,
the delta rotation re-expressed in the base frame as an axis-angle ``/ orientation_scale`` --
the term multiplies the scales back) and the binary gripper (sign flipped: Isaac Lab's
closes on ``< 0``). It is the collector's ``env_action`` as a term, so a policy trained on
the demos' actions sends them to the env as they are.
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
        dpos = actions[:, 0:3] / self.cfg.arm.position_scale
        q_d = actions[:, [6, 3, 4, 5]]  # xyzw -> wxyz
        _, q_ee = _ee_pose_b(self._env, self._ee_cfg, self._offset)
        q_base = quat_mul(quat_mul(q_ee, q_d), quat_conjugate(q_ee))  # the same rotation, base frame
        drot = axis_angle_from_quat(q_base) / self.cfg.arm.orientation_scale
        grip = -actions[:, 7:8]  # upstream: < 0 opens; Isaac Lab's binary term: < 0 closes
        self._processed[:] = torch.cat([dpos, drot, grip], dim=-1)
        self.arm.process_actions(self._processed[:, 0:6])
        self.gripper.process_actions(self._processed[:, 6:7])

    def apply_actions(self) -> None:
        self.arm.apply_actions()
        self.gripper.apply_actions()

    def reset(self, env_ids=None) -> None:
        self.arm.reset(env_ids)
        self.gripper.reset(env_ids)
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
