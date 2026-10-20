"""The Jacobian of a point fixed on a body (the TCP on the hand), torch, no Isaac Lab.

A point at ``r`` in the body's frame moves at ``v_p = v + omega x (R r)``: the lever is the
offset *rotated into the frame the Jacobian is expressed in* (here the robot root). Isaac Lab's
``DifferentialInverseKinematicsAction._compute_frame_jacobian`` (and the OSC term's
``_compute_ee_jacobian``) uses the body-frame ``r`` instead, which is right only when the body
is not rotated relative to the root. With the hand pointing down (~180 deg about x) the lever
comes out on the wrong side of the hand -- 2 x 0.1034 m wrong -- and every tilting command
drags the TCP sideways by ~0.207 m x the angle until the next solve corrects it. ResiP's sim
reads the TCP link's own Jacobian from the physics engine (``k_ee_link``), so it has no
such error; :func:`tcp_jacobian` gives that Jacobian.
"""

from __future__ import annotations

import torch

__all__ = ["quat_wxyz_to_mat", "skew", "tcp_jacobian"]


def quat_wxyz_to_mat(q: torch.Tensor) -> torch.Tensor:
    """``(..., 4)`` unit quaternion wxyz -> ``(..., 3, 3)``."""
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), dim=-1),
            torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), dim=-1),
            torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), dim=-1),
        ),
        dim=-2,
    )


def skew(v: torch.Tensor) -> torch.Tensor:
    """``(..., 3)`` -> ``(..., 3, 3)`` with ``skew(a) @ b == a x b``."""
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack((torch.stack((o, -z, y), -1), torch.stack((z, o, -x), -1), torch.stack((-y, x, o), -1)), -2)


def tcp_jacobian(jacobian_b: torch.Tensor, body_quat_b: torch.Tensor, offset_body: torch.Tensor) -> torch.Tensor:
    """The Jacobian of the point ``offset_body`` (``(N, 3)``, in the body's frame) on a body.

    ``jacobian_b`` ``(N, 6, n)``: the body origin's ``[linear; angular]`` Jacobian in the root
    frame; ``body_quat_b`` ``(N, 4)`` wxyz: the body's orientation in the root frame. Returns a new
    ``(N, 6, n)``: ``v_p = v - (R r) x omega`` rows on top, the angular rows unchanged (one
    rigid body has one angular velocity)."""
    r = (quat_wxyz_to_mat(body_quat_b) @ offset_body.unsqueeze(-1)).squeeze(-1)  # the lever, root frame
    J = jacobian_b.clone()
    J[:, 0:3, :] = jacobian_b[:, 0:3, :] - skew(r) @ jacobian_b[:, 3:, :]
    return J
