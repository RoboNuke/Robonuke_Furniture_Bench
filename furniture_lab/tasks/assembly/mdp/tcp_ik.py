"""Isaac Lab's differential-IK action term with the TCP Jacobian fixed (``furniture_lab.tcp_jacobian``).

Isaac Lab's term adds the ``body_offset`` lever to the hand's root-frame Jacobian in the hand's
own frame; this one rotates it into the root frame first, giving the Jacobian ResiP's sim reads
off its ``k_ee_link``. Everything else -- the TCP pose (``_compute_frame_pose``), the controller,
the scaling, relative / absolute commands -- is Isaac Lab's, unchanged. The offset must have no
rotation (the angular rows of one rigid body need none).
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

from ....tcp_jacobian import tcp_jacobian

__all__ = ["TcpDifferentialIKAction"]


class TcpDifferentialIKAction(DifferentialInverseKinematicsAction):
    """``DifferentialInverseKinematicsAction`` with the TCP lever in the root frame."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        if self._offset_rot is not None and not torch.allclose(
            self._offset_rot, torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand_as(self._offset_rot)
        ):
            raise ValueError(f"{type(self).__name__}: body_offset.rot must be identity, got {self.cfg.body_offset.rot}")

    def _compute_frame_jacobian(self) -> torch.Tensor:
        jacobian = self.jacobian_b
        if self.cfg.body_offset is None:
            return jacobian
        root_quat_w = self._asset.data.root_quat_w
        body_quat_w = self._asset.data.body_quat_w[:, self._body_idx]
        body_quat_b = math_utils.quat_mul(math_utils.quat_inv(root_quat_w), body_quat_w)
        return tcp_jacobian(jacobian, body_quat_b, self._offset_pos)
