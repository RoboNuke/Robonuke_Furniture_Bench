"""furniture_lab.tcp_jacobian against finite differences of a random 7-joint arm's TCP pose:
the fixed lever (rotated into the root frame) matches; Isaac Lab's body-frame lever does not
whenever the hand is rotated."""

import numpy as np
import torch

from furniture_lab.tcp_jacobian import skew, tcp_jacobian

R_OFF = np.array([0.0, 0.0, 0.1034])


def rot(axis, a):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


rng = np.random.default_rng(0)
AXES = rng.normal(size=(7, 3))
LINKS = rng.normal(size=(7, 3)) * 0.15


def fk(q):
    """The hand's pose (R, p) for joint angles q: revolute joints about random axes."""
    R, p = np.eye(3), np.zeros(3)
    for a, l, qi in zip(AXES, LINKS, q):
        R = R @ rot(a, qi)
        p = p + R @ l
    return R, p


def fd_jacobians(q, eps=1e-6):
    """Finite-difference [linear; angular] Jacobians of the hand origin and of the TCP."""
    R0, p0 = fk(q)
    J_body, J_tcp = np.zeros((6, 7)), np.zeros((6, 7))
    for i in range(7):
        dq = np.zeros(7)
        dq[i] = eps
        R1, p1 = fk(q + dq)
        dR = R1 @ R0.T
        w = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * eps)
        J_body[:3, i], J_body[3:, i] = (p1 - p0) / eps, w
        J_tcp[:3, i] = ((p1 + R1 @ R_OFF) - (p0 + R0 @ R_OFF)) / eps
        J_tcp[3:, i] = w
    return R0, J_body, J_tcp


def quat_wxyz(R):
    from furniture_lab.tasks.furniture.frames import mat_to_quat_wxyz

    return mat_to_quat_wxyz(R)


def test_the_tcp_jacobian_matches_finite_differences_and_isaacs_lever_does_not():
    for q in rng.uniform(-1.5, 1.5, size=(5, 7)):
        R0, J_body, J_tcp = fd_jacobians(q)
        J = tcp_jacobian(
            torch.tensor(J_body)[None], torch.tensor(quat_wxyz(R0))[None], torch.tensor(R_OFF)[None]
        )[0].numpy()
        assert np.abs(J - J_tcp).max() < 1e-4
        isaac = J_body.copy()
        isaac[:3] += -skew(torch.tensor(R_OFF)).numpy() @ J_body[3:]  # the body-frame lever
        assert np.abs(isaac - J_tcp).max() > 1e-2  # the hand is rotated: Isaac's lever is off


def test_hand_pointing_down_a_pure_tilt_no_longer_drags_the_tcp():
    """The numbers of the bug: hand at 180 deg about x, a 1 deg tilt about x, the TCP held."""
    R_hand = np.diag([1.0, -1.0, -1.0])
    J_body = torch.eye(6, dtype=torch.float64)[None]  # 'joints' = the hand origin's twist
    true = tcp_jacobian(J_body, torch.tensor(quat_wxyz(R_hand))[None], torch.tensor(R_OFF)[None])[0].numpy()
    cmd = np.array([0, 0, 0, np.radians(1.0), 0, 0])
    assert np.abs((true @ (np.linalg.pinv(true) @ cmd))[:3]).max() < 1e-12
    isaac = np.eye(6)
    isaac[:3] += -skew(torch.tensor(R_OFF)).numpy() @ np.eye(6)[3:]
    drag = (true @ (np.linalg.pinv(isaac) @ cmd))[:3]
    assert abs(np.linalg.norm(drag) - 2 * 0.1034 * np.radians(1.0)) < 1e-6  # 3.6 mm
