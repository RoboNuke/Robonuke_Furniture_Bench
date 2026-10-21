"""mdp.ee_vel: the centre-of-mass velocity by default; with an offset, the velocity of that point
of the body (FurnitureBench's k_ee_link), checked against a finite difference of the point."""

import importlib
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import isaaclab_stub  # noqa: E402


def _quat_apply(q, v):  # wxyz
    w, u = q[..., :1], q[..., 1:]
    t = 2.0 * torch.linalg.cross(u, v, dim=-1)
    return v + w * t + torch.linalg.cross(u, t, dim=-1)


def _quat_apply_inverse(q, v):
    return _quat_apply(torch.cat((q[..., :1], -q[..., 1:]), dim=-1), v)


def _quat_axis_angle(axis, angle):
    axis = torch.as_tensor(axis, dtype=torch.float64)
    axis = axis / axis.norm()
    return torch.cat((torch.tensor([math.cos(angle / 2)], dtype=torch.float64), math.sin(angle / 2) * axis))


@pytest.fixture
def obs(monkeypatch):
    uninstall = isaaclab_stub.install()
    try:
        mod = importlib.import_module("furniture_lab.tasks.assembly.mdp.observations")
        monkeypatch.setattr(mod, "quat_apply", _quat_apply)
        monkeypatch.setattr(mod, "quat_apply_inverse", _quat_apply_inverse)
        yield mod
    finally:
        uninstall()


def _env(link_quat, link_lin, com_lin, ang, root_quat):
    data = SimpleNamespace(
        body_link_quat_w=link_quat[None, None], body_link_lin_vel_w=link_lin[None, None],
        body_lin_vel_w=com_lin[None, None], body_ang_vel_w=ang[None, None], root_quat_w=root_quat[None],
    )
    return SimpleNamespace(scene={"robot": SimpleNamespace(data=data)}, device="cpu", num_envs=1)


def test_without_offset_it_is_the_centre_of_mass_velocity(obs):
    ident = torch.tensor([1.0, 0, 0, 0], dtype=torch.float64)
    com = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
    ang = torch.tensor([0.5, 0.0, -1.0], dtype=torch.float64)
    env = _env(ident, torch.zeros(3, dtype=torch.float64), com, ang, ident)
    out = obs.ee_vel(env, SimpleNamespace(name="robot", body_ids=[0]))
    assert torch.allclose(out[0], torch.cat((com, ang)))


def test_with_offset_it_is_the_points_velocity(obs):
    """A body turning at w with its link origin moving at v: the point p_link + R off moves
    at v + w x (R off); seen from a base rotated about z."""
    off = (0.0, 0.0, 0.1034)
    q_link = _quat_axis_angle([0.3, -1.0, 0.4], 2.1)
    v_link = torch.tensor([0.05, 0.02, -0.03], dtype=torch.float64)
    w = torch.tensor([0.4, -0.7, 1.1], dtype=torch.float64)
    q_root = _quat_axis_angle([0.0, 0.0, 1.0], 0.6)

    def point(t):  # the offset point in the world at time t (constant w and v)
        th = w.norm() * t
        q = _quat_axis_angle(w, th)
        q_t = torch.cat((q[:1] * q_link[:1] - (q[1:] * q_link[1:]).sum(-1, keepdim=True),
                         q[:1] * q_link[1:] + q_link[:1] * q[1:] + torch.linalg.cross(q[1:], q_link[1:], dim=-1)))
        return v_link * t + _quat_apply(q_t, torch.tensor(off, dtype=torch.float64))

    dt = 1e-6
    v_fd = (point(dt) - point(-dt)) / (2 * dt)
    env = _env(q_link, v_link, torch.full((3,), 9.0, dtype=torch.float64), w, q_root)
    out = obs.ee_vel(env, SimpleNamespace(name="robot", body_ids=[0]), offset=off)
    assert torch.allclose(out[0, :3], _quat_apply_inverse(q_root, v_fd), atol=1e-6)
    assert torch.allclose(out[0, 3:], _quat_apply_inverse(q_root, w))
