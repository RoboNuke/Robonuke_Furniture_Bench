"""External end-effector wrench estimated from the joint torques, the way the real robot does it.

The arm's equation of motion is ``M(q) q'' + c(q, q') + g(q) = tau_J + J^T F_ext``, with
``F_ext`` the wrench the environment exerts on the end-effector. Rearranged and projected::

    tau_ext = M q'' + c + g - tau_J
    F_ext   = (J^T)^+ tau_ext          (robot-base frame, at the end-effector point)

On the Franka that is ``O_F_ext_hat_K`` from ``tau_ext_hat_filtered``. Here:

* ``tau_J`` is what a joint torque sensor reads: the torque the drive put into the joint,
  ``robot.data.applied_torque`` (the arm runs effort-controlled, so it is exactly the OSC's
  torque after the effort clamp);
* the robot's links have gravity disabled in this env (the OSC does no gravity
  compensation), so there is no ``g`` to remove; the Coriolis and inertial terms are
  optional (``FurnitureCfg.wrench_coriolis`` / ``wrench_inertial``, both off: the plain
  "project the torques" estimate), ``q''`` being Isaac Lab's finite-differenced
  ``joint_acc``;
* the projection is the Moore-Penrose pseudo-inverse of ``J^T`` (``wrench_projection:
  pinv``) or the dynamically consistent one ``Lambda J M^-1`` (``dynamic``), ``J`` the
  geometric Jacobian of FurnitureBench's EE frame (``panda_hand`` + 0.1034 m) in the base
  frame;
* noise is added where it arises on the robot: Gaussian on each joint torque before the
  projection (``wrench_noise_joint_torque``, N m), and, if wanted, on the projected wrench
  (``wrench_noise``, N / N m); then an EMA with ``wrench_smoothing`` (Forge's
  ``ft_smoothing_factor`` is 0.25), run every physics substep exactly as Forge runs its
  filter, and zeroed on reset.

The estimate is read with ``env.wrench.filtered`` (``(N, 6)``: force then torque) and
``env.wrench.raw``; the observation term ``mdp.ee_wrench`` exposes it to an observation
group. ``scripts/wrench_probe.py`` checks it against PhysX's own joint reaction wrench.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["project_torques", "ema", "JointTorqueWrench"]

#: FurnitureBench's EE frame relative to ``panda_hand`` (``k_ee_link``)
EE_OFFSET = (0.0, 0.0, 0.1034)


# ---------------------------------------------------------------------- the math (no sim)
def project_torques(jacobian: torch.Tensor, tau_ext: torch.Tensor, mass_matrix: torch.Tensor | None = None) -> torch.Tensor:
    """``F = (J^T)^+ tau_ext`` per env; with ``mass_matrix`` the dynamically consistent
    inverse ``F = (J M^-1 J^T)^-1 J M^-1 tau_ext``. ``jacobian`` is ``(N, 6, n)``."""
    if mass_matrix is None:
        return (torch.linalg.pinv(jacobian.transpose(1, 2)) @ tau_ext.unsqueeze(-1)).squeeze(-1)
    m_inv = torch.linalg.inv(mass_matrix)
    j_m_inv = jacobian @ m_inv
    lam = torch.linalg.inv(j_m_inv @ jacobian.transpose(1, 2))
    return (lam @ j_m_inv @ tau_ext.unsqueeze(-1)).squeeze(-1)


def ema(state: torch.Tensor, value: torch.Tensor, alpha: float) -> torch.Tensor:
    """Forge's filter: ``alpha * value + (1 - alpha) * state``."""
    return alpha * value + (1.0 - alpha) * state


def _std(values: Sequence[float] | float, width: int, device, what: str) -> torch.Tensor:
    v = torch.as_tensor(values, dtype=torch.float32, device=device).reshape(-1)
    if what == "wrench" and v.numel() == 2:  # (force, torque)
        v = torch.cat((v[0].repeat(3), v[1].repeat(3)))
    if v.numel() == 1:
        v = v.repeat(width)
    if v.numel() != width or (v < 0).any():
        raise ValueError(f"{what} noise must be one value or {width} non-negative values, got {values}")
    return v


# ---------------------------------------------------------------------- the estimator
class JointTorqueWrench:
    """Per-physics-step estimator on the env's robot; see the module docstring."""

    def __init__(self, env: "ManagerBasedRLEnv", cfg) -> None:
        self.env = env
        self.cfg = cfg
        robot = env.scene["robot"]
        self.robot = robot
        self.joint_ids, _ = robot.find_joints("panda_joint.*")
        self.joint_ids = list(self.joint_ids)
        body_ids, _ = robot.find_bodies("panda_hand")
        self.body_idx = int(body_ids[0])
        self.parent_idx = int(robot.find_bodies("panda_link7")[0][0])
        # PhysX's jacobian array skips the fixed root body
        self.jacobi_body_idx = self.body_idx - 1 if robot.is_fixed_base else self.body_idx
        self.jacobi_joint_ids = self.joint_ids if robot.is_fixed_base else [i + 6 for i in self.joint_ids]
        n, dev = env.num_envs, env.device
        self.offset = torch.tensor(EE_OFFSET, device=dev, dtype=torch.float32).expand(n, -1)
        self.alpha = float(cfg.wrench_smoothing)
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"furniture.wrench_smoothing must be in (0, 1], got {self.alpha}")
        if cfg.wrench_projection not in ("pinv", "dynamic"):
            raise ValueError(f"furniture.wrench_projection must be pinv or dynamic, got {cfg.wrench_projection!r}")
        self.tau_std = _std(cfg.wrench_noise_joint_torque, len(self.joint_ids), dev, "joint torque")
        self.wrench_std = _std(cfg.wrench_noise, 6, dev, "wrench")
        if cfg.wrench_coriolis and not hasattr(robot.root_physx_view, "get_coriolis_and_centrifugal_compensation_forces"):
            raise RuntimeError("furniture.wrench_coriolis: this PhysX build has no get_coriolis_and_centrifugal_compensation_forces")
        self.raw = torch.zeros((n, 6), device=dev)
        self.filtered = torch.zeros((n, 6), device=dev)
        self.tau_ext = torch.zeros((n, len(self.joint_ids)), device=dev)
        self.updates = 0

    # ------------------------------------------------------------------ pieces
    def jacobian_b(self) -> torch.Tensor:
        """Geometric Jacobian of the EE frame (hand + offset), rows in the base frame, (N, 6, 7)."""
        from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_apply_inverse, quat_inv, skew_symmetric_matrix

        robot = self.robot
        jac = robot.root_physx_view.get_jacobians()[:, self.jacobi_body_idx, :, self.jacobi_joint_ids].clone()
        base_inv = matrix_from_quat(quat_inv(robot.data.root_quat_w))
        jac[:, 0:3, :] = base_inv @ jac[:, 0:3, :]
        jac[:, 3:6, :] = base_inv @ jac[:, 3:6, :]
        # the offset point moves with the hand: v_ee = v_hand + w x r
        r_w = quat_apply(robot.data.body_quat_w[:, self.body_idx], self.offset)
        r_b = quat_apply_inverse(robot.data.root_quat_w, r_w)
        jac[:, 0:3, :] = jac[:, 0:3, :] - skew_symmetric_matrix(r_b) @ jac[:, 3:6, :]
        return jac

    def ee_pos_b(self) -> torch.Tensor:
        """The EE point the wrench is taken about, base frame."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        robot = self.robot
        p_w = robot.data.body_pos_w[:, self.body_idx] + quat_apply(robot.data.body_quat_w[:, self.body_idx], self.offset)
        return quat_apply_inverse(robot.data.root_quat_w, p_w - robot.data.root_pos_w)

    def external_torques(self) -> torch.Tensor:
        """``tau_ext = [M q''] + [c] - tau_J`` over the arm joints, noise on ``tau_J``."""
        robot, ids = self.robot, self.joint_ids
        tau_j = robot.data.applied_torque[:, ids]
        if self.tau_std.any():
            tau_j = tau_j + torch.randn_like(tau_j) * self.tau_std
        tau = -tau_j
        if self.cfg.wrench_coriolis:
            tau = tau + robot.root_physx_view.get_coriolis_and_centrifugal_compensation_forces()[:, ids]
        if self.cfg.wrench_inertial:
            tau = tau + (self.mass_matrix() @ robot.data.joint_acc[:, ids].unsqueeze(-1)).squeeze(-1)
        return tau

    def mass_matrix(self) -> torch.Tensor:
        ids = self.joint_ids
        return self.robot.root_physx_view.get_generalized_mass_matrices()[:, ids, :][:, :, ids]

    # ------------------------------------------------------------------ the step
    def update(self) -> None:
        """One physics substep: estimate, add wrench noise, filter."""
        self.tau_ext = self.external_torques()
        mass = self.mass_matrix() if self.cfg.wrench_projection == "dynamic" else None
        wrench = project_torques(self.jacobian_b(), self.tau_ext, mass)
        if self.wrench_std.any():
            wrench = wrench + torch.randn_like(wrench) * self.wrench_std
        self.raw = wrench
        self.filtered = ema(self.filtered, wrench, self.alpha)
        self.updates += 1

    def reset(self, env_ids) -> None:
        self.raw[env_ids] = 0.0
        self.filtered[env_ids] = 0.0

    # ------------------------------------------------------------------ the reference
    def physx_reference(self) -> torch.Tensor:
        """PhysX's own reaction wrench through ``panda_hand``'s joint, as the external wrench
        on the EE in the base frame: Isaac Lab documents ``body_incoming_joint_wrench_b`` as
        the wrench the parent applies to the child in the **parent** (link 7) frame, so the
        wrench on the hand from the outside is its negative (the hand and fingers have no
        gravity); rotated into the base frame and its moment moved to the EE point. For
        ``scripts/wrench_probe.py``."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        robot = self.robot
        w = robot.data.body_incoming_joint_wrench_b[:, self.body_idx]
        parent_quat_w = robot.data.body_quat_w[:, self.parent_idx]
        f_w, m_w = quat_apply(parent_quat_w, -w[:, 0:3]), quat_apply(parent_quat_w, -w[:, 3:6])
        f_b, m_b = quat_apply_inverse(robot.data.root_quat_w, f_w), quat_apply_inverse(robot.data.root_quat_w, m_w)
        hand_b = quat_apply_inverse(robot.data.root_quat_w, robot.data.body_pos_w[:, self.body_idx] - robot.data.root_pos_w)
        m_ee = m_b + torch.cross(hand_b - self.ee_pos_b(), f_b, dim=-1)
        return torch.cat((f_b, m_ee), dim=-1)
