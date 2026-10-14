"""FurnitureBench's assembled check, batched over envs in torch.

Upstream (``Furniture.is_assembled_idx``) does, per env and per pair ``(i, j)`` in
``should_be_assembled``::

    rel = inv(pose_i) @ pose_j                        # part j in part i's frame
    assembled = any over the pair's candidate poses T of
        |rel.pos - T.pos| <= assembled_pos_threshold   (per axis)
        and cos(rel.R[:, k], T.R[:, k]) >= ori_bound   for k in x, y, z

Every candidate pose has the identity rotation, so the orientation test is just
``diag(rel.R) >= ori_bound`` -- which is what :func:`rotation_diag` computes straight from
the quaternion. Once a pair tests assembled it stays assembled for the rest of the episode
(upstream's ``assembled_set``), and the reward is the number of pairs that became assembled
this step (``compute_assemble``); :class:`AssemblyTracker` keeps that latch.

The tracker can be made as strict as this port's demo collector (``hold_steps`` and the
``hand_clear`` argument of :meth:`AssemblyTracker.update`): a pair then latches only after
it has tested assembled for ``hold_steps`` consecutive steps with the hand released and
clear of it. ``hold_steps=1`` without ``hand_clear`` is upstream's single-step latch.

Quaternions are ``(w, x, y, z)``, Isaac Lab's convention. Pure torch: no env, no Isaac Lab.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

__all__ = [
    "quat_conjugate",
    "quat_mul",
    "quat_rotate_inverse",
    "relative_pose",
    "rotation_diag",
    "pairs_assembled",
    "AssemblyTracker",
]


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """R(q)^T v."""
    w = q[..., :1]
    xyz = q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v - w * t + torch.cross(xyz, t, dim=-1)


def relative_pose(
    pos_a: torch.Tensor, quat_a: torch.Tensor, pos_b: torch.Tensor, quat_b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pose of ``b`` in ``a``'s frame: ``inv(T_a) @ T_b``."""
    rel_pos = quat_rotate_inverse(quat_a, pos_b - pos_a)
    rel_quat = quat_mul(quat_conjugate(quat_a), quat_b)
    return rel_pos, rel_quat


def rotation_diag(q: torch.Tensor) -> torch.Tensor:
    """``(R00, R11, R22)`` of the rotation matrix of a unit quaternion.

    ``R_kk`` is the cosine between the rotated frame's ``k`` axis and the identity's, which is
    exactly upstream's ``cosine_sim(rot[:, k], I[:, k])``.
    """
    x, y, z = q[..., 1], q[..., 2], q[..., 3]
    return torch.stack(
        [1 - 2 * (y * y + z * z), 1 - 2 * (x * x + z * z), 1 - 2 * (x * x + y * y)], dim=-1
    )


def pairs_assembled(
    parts_pos: torch.Tensor,
    parts_quat: torch.Tensor,
    pairs: torch.Tensor,
    targets: torch.Tensor,
    pos_threshold: torch.Tensor,
    ori_bound: float,
) -> torch.Tensor:
    """Which pairs are currently in an assembled pose.

    Args:
        parts_pos: ``(N, P, 3)`` part positions, any common frame.
        parts_quat: ``(N, P, 4)`` part orientations (w, x, y, z).
        pairs: ``(M, 2)`` long tensor of ``(i, j)`` part indices.
        targets: ``(M, K, 3)`` candidate assembled positions of ``j`` in ``i``'s frame.
        pos_threshold: ``(3,)`` per-axis tolerance in ``i``'s frame.
        ori_bound: minimum cosine between each axis of ``j`` and the same axis of ``i``.

    Returns:
        ``(N, M)`` bool.
    """
    pos_i = parts_pos[:, pairs[:, 0]]  # (N, M, 3)
    pos_j = parts_pos[:, pairs[:, 1]]
    quat_i = parts_quat[:, pairs[:, 0]]
    quat_j = parts_quat[:, pairs[:, 1]]
    rel_pos, rel_quat = relative_pose(pos_i, quat_i, pos_j, quat_j)
    ori_ok = (rotation_diag(rel_quat) >= ori_bound).all(dim=-1)  # (N, M)
    err = (rel_pos.unsqueeze(2) - targets.unsqueeze(0)).abs()  # (N, M, K, 3)
    pos_ok = (err <= pos_threshold).all(dim=-1).any(dim=-1)  # (N, M)
    return ori_ok & pos_ok


class AssemblyTracker:
    """The per-env latch of assembled pairs, and the sparse reward it yields.

    Call :meth:`update` once per env step with the current part poses; read
    :attr:`assembled`, :attr:`newly_assembled` and :meth:`all_assembled` after it. Call
    :meth:`reset` for the envs that restart.
    """

    def __init__(
        self,
        num_envs: int,
        pairs: Sequence[Tuple[int, int]],
        targets: torch.Tensor,
        pos_threshold: Sequence[float],
        ori_bound: float,
        device: torch.device | str,
        hold_steps: int = 1,
    ) -> None:
        if hold_steps < 1:
            raise ValueError(f"hold_steps must be >= 1, got {hold_steps}")
        self.hold_steps = int(hold_steps)
        self.pairs = torch.as_tensor(list(pairs), dtype=torch.long, device=device)
        self.targets = torch.as_tensor(targets, dtype=torch.float32, device=device)
        self.pos_threshold = torch.as_tensor(pos_threshold, dtype=torch.float32, device=device)
        self.ori_bound = float(ori_bound)
        self.assembled = torch.zeros(num_envs, len(pairs), dtype=torch.bool, device=device)
        self.newly_assembled = torch.zeros_like(self.assembled)
        #: the raw check this step, and how many consecutive steps each pair has qualified
        self.now = torch.zeros_like(self.assembled)
        self.streak = torch.zeros(num_envs, len(pairs), dtype=torch.long, device=device)

    def update(
        self, parts_pos: torch.Tensor, parts_quat: torch.Tensor, hand_clear: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Latch the pairs that qualified ``hold_steps`` steps in a row -- assembled now and,
        when ``hand_clear`` (``(N,)`` or ``(N, M)`` bool) is given, the hand released and clear;
        returns the count newly assembled per env."""
        self.now = pairs_assembled(
            parts_pos, parts_quat, self.pairs, self.targets, self.pos_threshold, self.ori_bound
        )
        qualifies = self.now
        if hand_clear is not None:
            clear = hand_clear if hand_clear.dim() == 2 else hand_clear.unsqueeze(-1)
            qualifies = qualifies & clear
        self.streak = torch.where(qualifies, self.streak + 1, torch.zeros_like(self.streak))
        confirmed = self.streak >= self.hold_steps
        self.newly_assembled = confirmed & ~self.assembled
        self.assembled |= confirmed
        return self.newly_assembled.sum(dim=-1).float()

    def all_assembled(self) -> torch.Tensor:
        return self.assembled.all(dim=-1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.assembled.zero_()
            self.newly_assembled.zero_()
            self.now.zero_()
            self.streak.zero_()
        else:
            self.assembled[env_ids] = False
            self.newly_assembled[env_ids] = False
            self.now[env_ids] = False
            self.streak[env_ids] = 0
