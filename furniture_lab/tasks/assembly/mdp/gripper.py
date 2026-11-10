"""ResiP's effort-mode gripper (``furniture.gripper_mode: effort``).

FurnitureRLSimEnv drives each finger joint in Isaac Gym's ``DOF_MODE_EFFORT`` (no stiffness, no
damping) with the joint friction ``sim_config["robot"]["gripper_frictions"]`` (15) and the finger
travel capped at half the max gripper width (``upper = 0.0325``), and every step it applies
``+-sim_config["robot"]["gripper_torque"]`` (13 N) to both fingers: ``+`` while the gripper command
is open, ``-`` while closed (``simulate_step``; the command flips with ``grasp_margin``'s
hysteresis). Against the joint friction the fingers move at ~57 mm/s each and squeeze the leg
with the full 13 N, which then also holds them shut against what the leg pushes back with.

:class:`BinaryJointEffortAction` is Isaac Lab's binary joint action with that force as the command
(the open / close expressions are efforts); :func:`set_finger_limits` caps the finger travel at
startup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointAction
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = [
    "BinaryJointEffortAction", "BinaryJointEffortActionCfg", "BinaryJointPositionAction", "BinaryJointPositionActionCfg",
    "set_finger_limits",
]


class _TensorJointIds:
    """Isaac Lab's binary terms keep their joint ids as a Python list and index the articulation's target buffer
    with it every physics substep: each is a host-to-device copy, i.e. a host synchronization. The ids as a
    device tensor index without one."""

    def _joint_index(self) -> torch.Tensor:
        index = getattr(self, "_joint_index_t", None)
        if index is None:
            index = torch.as_tensor(list(self._joint_ids), dtype=torch.long, device=self.device)
            self._joint_index_t = index
        return index


class BinaryJointEffortAction(_TensorJointIds, BinaryJointAction):
    """The binary open / close command as a joint effort on each joint (N for the fingers)."""

    cfg: "BinaryJointEffortActionCfg"

    def apply_actions(self):
        self._asset.set_joint_effort_target(self._processed_actions, joint_ids=self._joint_index())


@configclass
class BinaryJointEffortActionCfg(BinaryJointActionCfg):
    """``open_command_expr`` / ``close_command_expr``: the effort per joint (ResiP: +13 / -13 N)."""

    class_type: type = BinaryJointEffortAction


class BinaryJointPositionAction(_TensorJointIds, BinaryJointAction):
    """Isaac Lab's ``BinaryJointPositionAction`` (the open / close command as joint position targets) with the
    target written through a device index: no host synchronization per substep."""

    cfg: "BinaryJointPositionActionCfg"

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_index())


@configclass
class BinaryJointPositionActionCfg(BinaryJointActionCfg):
    """Isaac Lab's ``BinaryJointPositionActionCfg`` with the sync-free term."""

    class_type: type = BinaryJointPositionAction


def set_finger_limits(env: "ManagerBasedEnv", env_ids: torch.Tensor, upper: float, asset_cfg: SceneEntityCfg) -> None:
    """Startup: the finger joints' travel to ``[0, upper]`` (m), as ResiP's ``franka_dof_props
    ["upper"][7:] = max_gripper_width / 2``. ``asset_cfg`` must come in the term's params (the event
    manager resolves only those): an unresolved one selects every joint, so it is refused."""
    ids = asset_cfg.joint_ids
    if isinstance(ids, slice) or ids is None:
        raise ValueError(f"set_finger_limits: asset_cfg's joints are unresolved ({ids!r}); pass asset_cfg in the term's params")
    robot = env.scene[asset_cfg.name]
    limits = robot.data.joint_pos_limits[:, ids].clone()
    limits[..., 0] = 0.0
    limits[..., 1] = float(upper)
    robot.write_joint_position_limit_to_sim(limits, joint_ids=ids)
