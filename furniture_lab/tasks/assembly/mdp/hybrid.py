"""Hybrid force/position control on joint torques: the Forge controller, on FurnitureBench.

The action is ResiP's absolute target plus robonuke_rl_core's hybrid blocks::

    [goal_pos 3 (base frame), goal_rot6d 6, gripper 1 | selection n | force target n]

with ``n = sum(force_axes)``. Every physics substep the term computes the joint torques of
robonuke_rl_core's controller (``envs/forge/control.py``, the law its Forge wrapper runs)::

    tau = J^T [ (I - S) (K e - D v) + S K_f (f_d - f) ] + nullspace

and writes them as joint effort targets: the arm's drives carry no stiffness or damping in the
env cfgs that use this term (``OneLegFBAbsResiPHybridEnvCfg``), so the torque is all there is.

* ``e``: the pose error of FurnitureBench's EE frame (``panda_hand`` + 0.1034 m) to the target,
  in the robot-base frame (the package's ``pose_error``: position, axis-angle);
* ``v``: the EE frame's twist, ``J q'`` with the same Jacobian;
* ``S``: one selection bit per force-eligible axis, ``> 0`` is force control (the package's
  ``ActionInterface``: the same split, the same convention); the target's other axes are
  position-controlled always;
* ``f``: the force the EE applies to its surroundings, the negative of the env's joint-torque
  estimate of the external wrench (``-env.wrench.filtered``, base frame; under effort control
  the torque it projects is exactly the one applied). ``f_d`` is in the same sense (negative z
  presses down), which makes ``K_f (f_d - f)`` a negative feedback: with the external wrench
  itself the law would push harder the harder it pressed (measured: a -5 N target settled at
  -0.56 N, ``K_f f_d / (1 - K_f)``);
* ``f_d``: the force action in [-1, 1] times ``force_target_bounds``; ``K_f`` is
  ``default_force_gains`` (constant gains, as the Forge hybrid runs);
* ``K`` / ``D``: :class:`HybridTorqueCfg` ``stiffness`` / ``damping`` (null: ``2 sqrt(K)``,
  Forge's rule); the nullspace pulls the arm to FurnitureBench's reset joints with
  ``kp_null`` / ``kd_null`` (Forge's 10 / 6.3246; FurnitureBench's OSC used 10, critically
  damped).

Two fields, both on by default, change that law (both off: Forge's law exactly).
``inertial_decoupling`` scales the motion branch by the
operational-space inertia ``Lambda = (J M^-1 J^T)^-1``, as Isaac Lab's OSC and FurnitureBench's
own torque-controlled env do: ``tau = J^T [ (I - S) Lambda (K e - D v) + S K_f (f_d - f) ]``. Every
axis then responds as a unit mass, so ``K`` and ``D`` are accelerations per error (1/s^2,
1/s) and can be set from a time constant. Without it the gains act on the arm's own,
direction-dependent inertia: stiff, well-damped gains are unstable at a 1/120 s physics step in
the light (wrist) directions, which is what ``scripts/hybrid_tracking.py`` measured.

``force_feedforward`` commands the force target itself plus the correction,
``S (f_d + K_f (f_d - f))``. A still EE presses with exactly the force the joint torques command
(no gravity, no acceleration), so Forge's ``K_f (f_d - f)`` settles at ``K_f f_d / (1 + K_f)``
(~9 % of the target at K_f 0.1) and this at ``f_d`` for any ``K_f``. With the env's estimate
removing the arm's own inertia (``furniture.wrench_inertial``), ``K_f`` also lowers the EE's
apparent mass on the force axis by ``1 + K_f``, which is what limits it: ``--mode press``
(the truth from PhysX's joint reaction wrench) holds every target 0-10 N within 0.01 N at K_f 0.1
and 0.2, switched in contact or 0.5-2 cm above it; K_f 0.5 rings after an impact, K_f >= 1.5
in contact. The default 0.1 keeps a 5x margin.

The links carry no gravity in this env, so there is no gravity term. The gripper is the
abs term's (binary, with ResiP's hysteresis). At a reset every axis returns to position control
with a zero force target.
"""

from __future__ import annotations

from dataclasses import MISSING
from types import SimpleNamespace
from typing import TYPE_CHECKING, List, Optional

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from .actions import _GripperHysteresis
from .observations import _ee_pose_b

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = ["HybridTorqueCfg", "HybridTorqueAbsAction", "HybridTorqueAbsActionCfg", "POSE_DIMS", "decoupled_wrench", "RESIP_TRACKING_GAINS"]

#: the pose block: ResiP's absolute target ``[goal_pos 3, goal_rot6d 6, gripper 1]``
POSE_DIMS = 10


def decoupled_wrench(selection: torch.Tensor, motion: torch.Tensor, lam: torch.Tensor, force: torch.Tensor) -> torch.Tensor:
    """``(I - S) Lambda (I - S) motion + S force``: the motion branch (``K e - D v``, per axis) is
    zeroed on the force axes **before** the operational-space inertia ``lam`` couples the axes, and
    masked again after it (Isaac Lab's OSC order); the force branch owns the force axes. ``(N, 6)``
    each, ``lam`` ``(N, 6, 6)``."""
    free = 1.0 - selection
    coupled = (lam @ (free * motion).unsqueeze(-1)).squeeze(-1)
    return free * coupled + selection * force


#: gains that make the torque controller track like ResiP's differential IK into joint PD
#: 1000 / 200 (``scripts/hybrid_tracking.py``, with ``inertial_decoupling``): K 300 / 500 and
#: D 3.5 / 4.5 sqrt(K) on translation / rotation, a heavily damped response with a dominant
#: time constant near the reference's 0.2 s. Over the script's sequence, the 15 deg roll and pitch
#: steps aside, the EE path is within 0.24 mm RMS (1.4 mm max) and 0.06 deg RMS (0.3 deg max) of
#: the reference's; on those two steps the reference's own EE swings ~19 mm off its fixed position
#: target while this holds it within 2.5 mm.
RESIP_TRACKING_GAINS = {
    "stiffness": [300.0, 300.0, 300.0, 500.0, 500.0, 500.0],
    "damping": [60.62, 60.62, 60.62, 100.62, 100.62, 100.62],
}


@configclass
class HybridTorqueCfg:
    """The controller's fields (``task.cfg.hybrid.*``). Defaults are Forge's where Forge has
    one (``ctrl.default_task_prop_gains``, the nullspace gains) and robonuke_rl_core's
    ``controller`` defaults for the force side."""

    #: ``[x, y, z, Rx, Ry, Rz]`` axes the force branch may take; the selection and force
    #: blocks are as wide as its sum
    force_axes: List[int] = [1, 1, 1, 0, 0, 0]
    #: the force action in [-1, 1] scales to +/- these, N and N m
    force_target_bounds: List[float] = [50.0, 50.0, 50.0, 5.0, 5.0, 5.0]
    #: K_f per axis; on x, y, z stable to ~0.3 with the feedforward and the inertial wrench
    #: estimate (the module docstring)
    default_force_gains: List[float] = [0.1, 0.1, 0.1, 0.01, 0.01, 0.01]
    #: K per axis, N/m and N m/rad (Forge: 565 / 28)
    stiffness: List[float] = [565.0, 565.0, 565.0, 28.0, 28.0, 28.0]
    #: D per axis; null: 2 sqrt(K) (Forge's task_deriv_gains)
    damping: Optional[List[float]] = None
    #: nullspace posture gains toward the reset joints (Forge's kp_null / kd_null)
    kp_null: float = 10.0
    kd_null: float = 6.3246
    #: scale the motion branch by the operational-space inertia (see the module docstring);
    #: K and D are then accelerations per error. Off: Forge's law
    inertial_decoupling: bool = True
    #: command the force target itself plus the correction, ``f_d + K_f (f_d - f)``: a still
    #: EE settles at ``f_d`` for any ``K_f``. Off: Forge's ``K_f (f_d - f)``, which settles at
    #: ``K_f f_d / (1 + K_f)``
    force_feedforward: bool = True

    def controller_cfg(self):
        """robonuke_rl_core's ``ControllerCfg`` for the split: pose and force, constant gains,
        the 10-wide abs target as the pose block."""
        from robonuke_rl_core.envs.cfg import ControllerCfg

        return ControllerCfg(
            enabled=True,
            use_pose=True,
            use_force=True,
            gain_mapping="constant",
            native_action_dim=POSE_DIMS,
            force_axes=list(self.force_axes),
            force_target_bounds=list(self.force_target_bounds),
            default_force_gains=list(self.default_force_gains),
        )

    def gains(self, device) -> tuple:
        """``(K, D)`` as ``(6,)`` tensors."""
        stiffness = torch.tensor(list(self.stiffness), dtype=torch.float32, device=device)
        if self.damping is None:
            damping = 2.0 * torch.sqrt(stiffness)
        else:
            damping = torch.tensor(list(self.damping), dtype=torch.float32, device=device)
        return stiffness, damping

    def validate(self) -> None:
        self.controller_cfg().validate(None)  # the package's own field checks
        for name in ("stiffness",) + (("damping",) if self.damping is not None else ()):
            values = list(getattr(self, name))
            if len(values) != 6 or min(values) < 0:
                raise ValueError(f"hybrid.{name} must be six values >= 0, got {values}")
        if self.kp_null < 0 or self.kd_null < 0:
            raise ValueError(f"hybrid.kp_null / kd_null must be >= 0, got {self.kp_null}, {self.kd_null}")


class HybridTorqueAbsAction(ActionTerm):
    """See the module docstring."""

    cfg: "HybridTorqueAbsActionCfg"

    def __init__(self, cfg: "HybridTorqueAbsActionCfg", env: "ManagerBasedEnv") -> None:
        super().__init__(cfg, env)
        from robonuke_rl_core.envs.interface import ActionInterface

        from ...furniture import frames
        from ..wrench import EE_OFFSET

        hc = env.cfg.hybrid
        hc.validate()
        self.interface = ActionInterface(hc.controller_cfg(), device=self.device)
        self._dim = self.interface.action_dim
        self.robot = env.scene[cfg.asset_name]
        self.joint_ids, _ = self.robot.find_joints(cfg.joint_names)
        self.joint_ids = list(self.joint_ids)
        self._ee_cfg = SceneEntityCfg(cfg.asset_name, body_names=cfg.body_name)
        self._ee_cfg.resolve(env.scene)
        self._offset = tuple(EE_OFFSET)
        self.stiffness, self.damping = hc.gains(self.device)
        self.inertial_decoupling = bool(hc.inertial_decoupling)
        self.force_feedforward = bool(hc.force_feedforward)
        self._ctrl = SimpleNamespace(
            default_dof_pos_tensor=list(frames.RESET_JOINTS), kp_null=float(hc.kp_null), kd_null=float(hc.kd_null)
        )
        self.gripper = cfg.gripper.class_type(cfg.gripper, env)
        self._grip = _GripperHysteresis(self.num_envs, self.device, getattr(env.cfg.furniture, "gripper_margin", 0.0))

        n = self.num_envs
        self._raw = torch.zeros(n, self._dim, device=self.device)
        self._target_pos = torch.zeros(n, 3, device=self.device)
        self._target_quat = torch.zeros(n, 4, device=self.device)
        self._target_quat[:, 0] = 1.0
        self._selection = torch.zeros(n, 6, device=self.device)
        self._force_target = torch.zeros(n, 6, device=self.device)
        self._force_gains = torch.zeros(n, 6, device=self.device)
        self.torque = torch.zeros(n, len(self.joint_ids), device=self.device)
        #: the last substep's wrench command, for the logs and the tuning script
        self.wrench = torch.zeros(n, 6, device=self.device)

    # ------------------------------------------------------------------ the term
    @property
    def action_dim(self) -> int:
        return self._dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def selection(self) -> torch.Tensor:
        """``(N, 6)`` of 0/1, 1 = force-controlled, as the controller applies it."""
        return self._selection

    def process_actions(self, actions: torch.Tensor) -> None:
        from .rot6d import rot6d_to_quat

        self._raw[:] = actions
        pose, selection, force_target, _, force_gains = self.interface.split(actions)
        self._target_pos[:] = pose[:, 0:3]
        self._target_quat[:] = rot6d_to_quat(pose[:, 3:9])  # wxyz
        self._selection[:] = selection
        self._force_target[:] = force_target
        self._force_gains[:] = force_gains
        grip = -self._grip(pose[:, 9:10])  # upstream: < 0 opens; Isaac Lab's binary term: < 0 closes
        self.gripper.process_actions(grip)

    def apply_actions(self) -> None:
        self.torque[:] = self.compute_torque()
        self.robot.set_joint_effort_target(self.torque, joint_ids=self.joint_ids)
        self.gripper.apply_actions()

    def compute_torque(self) -> torch.Tensor:
        """This substep's joint torques from the current state (see the module docstring)."""
        from robonuke_rl_core.envs.forge import control

        env = self._env
        ee_pos, ee_quat = _ee_pose_b(env, self._ee_cfg, self._offset)
        jacobian = env.wrench.jacobian_b()
        joint_vel = self.robot.data.joint_vel[:, self.joint_ids]
        twist = (jacobian @ joint_vel.unsqueeze(-1)).squeeze(-1)
        delta_pose = control.pose_error(ee_pos, ee_quat, self._target_pos, self._target_quat)
        n = self.num_envs
        mass = env.wrench.mass_matrix()
        stiffness = self.stiffness.expand(n, -1)
        damping = self.damping.expand(n, -1)
        if self.inertial_decoupling:
            motion = control.pose_wrench(delta_pose, twist[:, 0:3], twist[:, 3:6], stiffness, damping)
            lam = torch.inverse(jacobian @ torch.inverse(mass) @ jacobian.transpose(1, 2))
            force = control.force_wrench(self._force_target, self.applied_force(), self._force_gains)
            wrench = decoupled_wrench(self._selection, motion, lam, force)
        else:
            wrench = control.unified_wrench(
                force_selection=self._selection,
                delta_pose=delta_pose,
                linvel=twist[:, 0:3],
                angvel=twist[:, 3:6],
                stiffness=stiffness,
                damping=damping,
                force_target=self._force_target,
                force_measured=self.applied_force(),
                force_gains=self._force_gains,
            )
        if self.force_feedforward:
            wrench = wrench + self._selection * self._force_target
        self.wrench[:] = wrench
        return control.dof_torque_from_wrench(
            wrench=wrench,
            jacobian=jacobian,
            arm_mass_matrix=mass,
            dof_pos=self.robot.data.joint_pos[:, self.joint_ids],
            dof_vel=joint_vel,
            ctrl=self._ctrl,
            device=self.device,
        )

    def applied_force(self) -> torch.Tensor:
        """The wrench the EE applies to its surroundings: minus the estimated external one."""
        return -self._env.wrench.filtered

    def reset(self, env_ids=None) -> None:
        self.gripper.reset(env_ids)
        self._grip.reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0.0
        # back to position control with no force target; a step's action sets the targets
        # before any torque is applied
        self._selection[ids] = 0.0
        self._force_target[ids] = 0.0


@configclass
class HybridTorqueAbsActionCfg(ActionTermCfg):
    """The arm joints, the EE body (the offset is FurnitureBench's EE frame), the gripper term;
    the controller's fields are the env cfg's ``hybrid``."""

    class_type: type = HybridTorqueAbsAction
    asset_name: str = "robot"
    joint_names: List[str] = ["panda_joint.*"]
    body_name: str = "panda_hand"
    gripper: object = MISSING
