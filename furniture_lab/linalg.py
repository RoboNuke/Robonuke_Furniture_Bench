"""Batched small solves that never synchronize with the host, plain torch.

``torch.linalg.pinv``, ``inv``, ``solve`` and ``lstsq`` all read cusolver's ``info`` back to the host to
raise on a singular input: one CPU-GPU round trip per call, two for the SVD-based ``pinv``. Run every
physics substep for the IK and the wrench estimator (24 substeps per control step), those round trips
stall the CPU behind the GPU on every one. The ``*_ex`` variants return ``info`` as a tensor instead,
so the solve stays on the device and a failed row is handled with ``torch.where``.

The Jacobians here are ``(N, 6, 7)`` of full row rank, so the Moore-Penrose pseudo-inverse is
``J^+ = J^T (J J^T)^-1`` and ``(J^T)^+ = (J J^T)^-1 J`` exactly; ``J J^T`` is symmetric positive
definite and a Cholesky solve gives it. A damping ``lambda`` adds ``lambda^2 I``: Tikhonov / damped
least squares, which scales the response along a singular direction ``sigma`` by
``sigma^2 / (sigma^2 + lambda^2)`` -- indistinguishable from the pseudo-inverse where
``sigma >> lambda`` and bounded instead of exploding at a singularity. The factorization runs in
float64 (a ``6 x 6`` per env: nothing), so a damping of ``1e-4`` on singular values of ``1e-2`` and
up is a relative change below ``1e-4``, less than float32's own rounding of the Jacobian.
"""

from __future__ import annotations

import torch

__all__ = ["spd_solve", "damped_pinv_apply", "damped_pinv_t_apply"]


def spd_solve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Solve ``a x = b`` for symmetric positive definite ``a`` ``(N, k, k)`` and ``b`` ``(N, k, m)``, in
    float64, with no host synchronization. A row whose factorization fails (``a`` not positive
    definite: a NaN input, or no damping at an exact singularity) gets ``x = 0``: the hold-still
    answer, instead of NaNs that would spread through the sim."""
    if a.dim() != 3 or a.shape[-1] != a.shape[-2]:
        raise ValueError(f"spd_solve: a must be (N, k, k), got {tuple(a.shape)}")
    if b.dim() != 3 or b.shape[:2] != a.shape[:2]:
        raise ValueError(f"spd_solve: b must be (N, k, m) matching a {tuple(a.shape)}, got {tuple(b.shape)}")
    factor, info = torch.linalg.cholesky_ex(a.to(torch.float64))
    x = torch.cholesky_solve(b.to(torch.float64), factor)
    x = torch.where((info == 0).view(-1, 1, 1), x, torch.zeros_like(x))
    return x.to(b.dtype)


def _gram(jacobian: torch.Tensor, damping: float) -> torch.Tensor:
    """``J J^T + damping^2 I`` in float64."""
    if jacobian.dim() != 3:
        raise ValueError(f"jacobian must be (N, m, n), got {tuple(jacobian.shape)}")
    if damping < 0:
        raise ValueError(f"damping must be >= 0, got {damping}")
    j = jacobian.to(torch.float64)
    gram = j @ j.transpose(1, 2)
    if damping:
        gram = gram + (damping * damping) * torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
    return gram


def damped_pinv_apply(jacobian: torch.Tensor, x: torch.Tensor, damping: float = 0.0) -> torch.Tensor:
    """``J^+ x`` as ``J^T (J J^T + damping^2 I)^-1 x``: ``jacobian`` ``(N, m, n)`` with ``m <= n`` and full
    row rank, ``x`` ``(N, m)`` -> ``(N, n)``. At ``damping`` 0 this is the pseudo-inverse applied to ``x``
    (the least-norm solution of ``J dq = x``); the IK's joint step."""
    if x.shape != jacobian.shape[:2]:
        raise ValueError(f"x must be {tuple(jacobian.shape[:2])} for jacobian {tuple(jacobian.shape)}, got {tuple(x.shape)}")
    y = spd_solve(_gram(jacobian, damping), x.to(torch.float64).unsqueeze(-1))
    return (jacobian.to(torch.float64).transpose(1, 2) @ y).squeeze(-1).to(x.dtype)


def damped_pinv_t_apply(jacobian: torch.Tensor, tau: torch.Tensor, damping: float = 0.0) -> torch.Tensor:
    """``(J^T)^+ tau`` as ``(J J^T + damping^2 I)^-1 J tau``: ``jacobian`` ``(N, m, n)`` with ``m <= n`` and full
    row rank, ``tau`` ``(N, n)`` -> ``(N, m)``. At ``damping`` 0 this is the pseudo-inverse of ``J^T`` applied
    to ``tau``; the wrench estimator's projection of joint torques to an end-effector wrench."""
    if tau.shape != (jacobian.shape[0], jacobian.shape[2]):
        raise ValueError(f"tau must be {(jacobian.shape[0], jacobian.shape[2])} for jacobian {tuple(jacobian.shape)}, got {tuple(tau.shape)}")
    rhs = jacobian.to(torch.float64) @ tau.to(torch.float64).unsqueeze(-1)
    return spd_solve(_gram(jacobian, damping), rhs).squeeze(-1).to(tau.dtype)
