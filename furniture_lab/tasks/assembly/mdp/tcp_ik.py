"""Isaac Lab's differential-IK action term with the TCP Jacobian fixed (``furniture_lab.tcp_jacobian``)
and the per-substep solve kept on the GPU.

Isaac Lab's term adds the ``body_offset`` lever to the hand's root-frame Jacobian in the hand's
own frame; this one rotates it into the root frame first, giving the Jacobian ResiP's sim reads
off its ``k_ee_link``. The TCP pose (``_compute_frame_pose``), the controller's pose error, the
scaling and the relative / absolute commands are Isaac Lab's, unchanged. The offset must have no
rotation (the angular rows of one rigid body need none).

Isaac Lab's ``apply_actions`` runs once per physics substep and synchronizes with the host six times
on the way: its joint and Jacobian indices are Python lists (each a host-to-device copy), it tests
``ee_quat_curr.norm() != 0`` on the host, and ``torch.linalg.pinv`` (or ``torch.inverse`` for its
``dls``) reads cusolver's status back. :class:`SyncFreeDifferentialIKController` solves the damped
least squares with :func:`furniture_lab.linalg.damped_pinv_apply` instead (``ik_method: dls``,
``lambda_val`` the damping; ``dls`` with ``lambda_val`` 0 is the pseudo-inverse), and
:meth:`TcpDifferentialIKAction.apply_actions` indexes with device tensors and folds the
zero-quaternion guard (the first substep before the sim has written body poses) into a
``torch.where``. Same solve, same 120 Hz, no round trip.
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

from ....linalg import damped_pinv_apply
from ....tcp_jacobian import tcp_jacobian

__all__ = ["SyncFreeDifferentialIKController", "TcpDifferentialIKAction"]


class SyncFreeDifferentialIKController(DifferentialIKController):
    """``DifferentialIKController`` whose ``dls`` solve never synchronizes with the host. Only ``dls`` is
    accepted: ``pinv`` and ``svd`` are SVDs (two host syncs each) and ``trans`` is not a solve."""

    def __init__(self, cfg, num_envs: int, device: str) -> None:
        super().__init__(cfg, num_envs, device)
        if cfg.ik_method != "dls":
            raise ValueError(
                f"{type(self).__name__}: ik_method must be 'dls' (damped least squares, lambda_val the damping; 0 "
                f"is the pseudo-inverse), got {cfg.ik_method!r}: the other methods synchronize with the host"
            )
        self.damping = float(cfg.ik_params["lambda_val"])
        if self.damping < 0:
            raise ValueError(f"ik_params['lambda_val'] must be >= 0, got {self.damping}")

    def _compute_delta_joint_pos(self, delta_pose: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
        return damped_pinv_apply(jacobian, delta_pose, self.damping)


class TcpDifferentialIKAction(DifferentialInverseKinematicsAction):
    """``DifferentialInverseKinematicsAction`` with the TCP lever in the root frame and a sync-free substep."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        if self._offset_rot is not None and not torch.allclose(
            self._offset_rot, torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand_as(self._offset_rot)
        ):
            raise ValueError(f"{type(self).__name__}: body_offset.rot must be identity, got {self.cfg.body_offset.rot}")
        self._ik_controller = SyncFreeDifferentialIKController(cfg=self.cfg.controller, num_envs=self.num_envs, device=self.device)
        # the joint indices as device tensors: a Python list is copied to the device on every indexing
        self._joint_index = self._as_index(self._joint_ids)
        self._jacobi_joint_index = self._as_index(self._jacobi_joint_ids)

    def _as_index(self, ids):
        return ids if isinstance(ids, slice) else torch.as_tensor(list(ids), dtype=torch.long, device=self.device)

    @property
    def jacobian_w(self) -> torch.Tensor:
        return self._asset.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._jacobi_joint_index]

    def _compute_frame_jacobian(self) -> torch.Tensor:
        jacobian = self.jacobian_b
        if self.cfg.body_offset is None:
            return jacobian
        root_quat_w = self._asset.data.root_quat_w
        body_quat_w = self._asset.data.body_quat_w[:, self._body_idx]
        body_quat_b = math_utils.quat_mul(math_utils.quat_inv(root_quat_w), body_quat_w)
        return tcp_jacobian(jacobian, body_quat_b, self._offset_pos)

    def apply_actions(self) -> None:
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_index]
        jacobian = self._compute_frame_jacobian()
        joint_pos_des = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
        # Isaac Lab holds the joints while the body pose is still unset (a zero quaternion, before the first sim
        # step); the same per env, on the device
        valid = ee_quat_curr.norm(dim=-1, keepdim=True) > 0
        joint_pos_des = torch.where(valid, joint_pos_des, joint_pos)
        self._asset.set_joint_position_target(joint_pos_des, self._joint_index)
