"""The 6-d rotation representation (Zhou et al. 2019), torch, no Isaac Lab.

A rotation is its matrix's first two **rows**, flattened: ``[R00 R01 R02 R10 R11 R12]``
(pytorch3d's ``matrix_to_rotation_6d`` convention). Back to a matrix by Gram-Schmidt: the
first row normalized, the second made orthogonal to it and normalized, the third their cross
product. Continuous, sign-free, and any 6 numbers decode to a valid rotation -- which is why
it is the representation for both the absolute-pose action and the orientations in the
observation of the ``-FBAbs-v0`` task. ``furniture_lab/demos.py`` holds the NumPy twin for
the demo pkls; the two must agree (tested).
"""

from __future__ import annotations

import torch

__all__ = ["mat_to_rot6d", "rot6d_to_mat", "quat_to_mat", "mat_to_quat", "quat_to_rot6d", "rot6d_to_quat", "axis_angle_to_mat"]


def mat_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """``(..., 3, 3)`` -> ``(..., 6)``: the first two rows."""
    return R[..., :2, :].reshape(*R.shape[:-2], 6)


def rot6d_to_mat(d6: torch.Tensor) -> torch.Tensor:
    """``(..., 6)`` -> ``(..., 3, 3)`` by Gram-Schmidt on the two rows."""
    a1, a2 = d6[..., 0:3], d6[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """``(..., 4)`` wxyz unit quaternion -> ``(..., 3, 3)``."""
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), dim=-1),
            torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), dim=-1),
            torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), dim=-1),
        ),
        dim=-2,
    )


def mat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """``(..., 3, 3)`` -> ``(..., 4)`` wxyz with ``w >= 0`` (Shepperd's method, batched)."""
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    # the four candidate squared components, each >= 0 up to rounding
    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack(
                (1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22), dim=-1
            ),
            min=0.0,
        )
    )
    # one quaternion per candidate (scaled); pick the best-conditioned one per row
    quat_by_wxyz = torch.stack(
        (
            torch.stack((q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21), dim=-1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2), dim=-1),
        ),
        dim=-2,
    )
    quat_candidates = quat_by_wxyz / (2.0 * q_abs[..., None].clamp(min=0.1))
    best = q_abs.argmax(dim=-1)
    q = torch.gather(quat_candidates, -2, best[..., None, None].expand(*best.shape, 1, 4)).squeeze(-2)
    q = torch.where(q[..., :1] < 0, -q, q)
    return q / q.norm(dim=-1, keepdim=True)


def quat_to_rot6d(q: torch.Tensor) -> torch.Tensor:
    return mat_to_rot6d(quat_to_mat(q))


def rot6d_to_quat(d6: torch.Tensor) -> torch.Tensor:
    return mat_to_quat(rot6d_to_mat(d6))


def axis_angle_to_mat(v: torch.Tensor) -> torch.Tensor:
    """``(..., 3)`` axis-angle (the unit axis times the angle, rad) -> ``(..., 3, 3)`` by Rodrigues'
    formula; the zero vector is the identity (``demos.mat_to_axis_angle`` is the inverse)."""
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    zero = torch.zeros_like(x)
    K = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(*v.shape[:-1], 3, 3)
    theta = v.norm(dim=-1)[..., None, None]
    small = theta < 1e-6
    t = torch.where(small, torch.ones_like(theta), theta)
    # sin(t) / t and (1 - cos t) / t^2, by their Taylor series near 0
    a = torch.where(small, 1 - theta**2 / 6, torch.sin(t) / t)
    b = torch.where(small, 0.5 - theta**2 / 24, (1 - torch.cos(t)) / t**2)
    eye = torch.eye(3, dtype=v.dtype, device=v.device).expand(K.shape)
    return eye + a * K + b * (K @ K)
