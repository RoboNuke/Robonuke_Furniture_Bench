"""The FurnitureBench 8-d action term's mapping onto the arm and gripper terms, with real
quaternion math substituted for the stub's no-ops and fake sub-terms recording their input."""

import math
from types import SimpleNamespace

import pytest
import torch

from tests import isaaclab_stub


@pytest.fixture
def actions_module():
    uninstall = isaaclab_stub.install()
    from furniture_lab.tasks.assembly.mdp import actions

    yield actions
    uninstall()


def quat_mul(a, b):
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ), dim=-1)


def quat_conjugate(q):
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0])


def axis_angle_from_quat(q):
    w, v = q[:, 0:1], q[:, 1:4]
    angle = 2.0 * torch.atan2(v.norm(dim=-1, keepdim=True), w)
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9) * angle


class Recorder:
    def __init__(self):
        self.received = None

    def process_actions(self, a):
        self.received = a.clone()

    def apply_actions(self):
        pass

    def reset(self, env_ids=None):
        pass


def make_term(actions_module, monkeypatch, ee_quat_wxyz):
    monkeypatch.setattr(actions_module, "quat_mul", quat_mul)
    monkeypatch.setattr(actions_module, "quat_conjugate", quat_conjugate)
    monkeypatch.setattr(actions_module, "axis_angle_from_quat", axis_angle_from_quat)
    monkeypatch.setattr(actions_module, "_ee_pose_b", lambda env, cfg, off: (None, torch.tensor([ee_quat_wxyz])))
    term = actions_module.FurnitureBenchAction.__new__(actions_module.FurnitureBenchAction)
    term.cfg = SimpleNamespace(arm=SimpleNamespace(position_scale=0.025, orientation_scale=0.1))
    term.arm, term.gripper = Recorder(), Recorder()
    term._env, term._ee_cfg, term._offset = None, None, (0, 0, 0)
    term.num_envs, term.device = 1, "cpu"
    term._raw = torch.zeros(1, 8)
    term._processed = torch.zeros(1, 7)
    return term


def test_the_arm_gets_six_dims_and_the_gripper_one_with_the_sign_flipped(actions_module, monkeypatch):
    term = make_term(actions_module, monkeypatch, (1.0, 0.0, 0.0, 0.0))
    # 2.5 cm in x, a 0.1 rad yaw (xyzw quaternion), gripper +1 = upstream "close"
    half = 0.05
    a = torch.tensor([[0.025, 0.0, 0.0, 0.0, 0.0, math.sin(half), math.cos(half), 1.0]])
    term.process_actions(a)
    assert term.arm.received.shape == (1, 6) and term.gripper.received.shape == (1, 1)
    assert torch.allclose(term.arm.received[0, 0:3], torch.tensor([1.0, 0.0, 0.0]))  # / position_scale
    assert torch.allclose(term.arm.received[0, 3:6], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6)  # 0.1 rad / 0.1
    assert term.gripper.received[0, 0] == -1.0  # Isaac Lab's binary term closes on < 0
    assert term.action_dim == 8 and term.raw_actions.shape == (1, 8)


def test_the_rotation_delta_is_reexpressed_in_the_base_frame(actions_module, monkeypatch):
    # EE rotated 90 deg about base z: a delta about the EE's x axis is a delta about base y
    s = math.sqrt(0.5)
    term = make_term(actions_module, monkeypatch, (s, 0.0, 0.0, s))
    half = 0.05
    a = torch.tensor([[0.0, 0.0, 0.0, math.sin(half), 0.0, 0.0, math.cos(half), -1.0]])
    term.process_actions(a)
    assert torch.allclose(term.arm.received[0, 3:6], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
    assert term.gripper.received[0, 0] == 1.0


def test_part_quaternions_are_canonical_like_the_collectors(actions_module):
    from furniture_lab.tasks.assembly.mdp.observations import canonical_quat

    q = torch.tensor([[-0.5, 0.5, 0.5, 0.5], [0.5, -0.5, 0.5, -0.5], [0.0, 1.0, 0.0, 0.0]])
    out = canonical_quat(q)
    assert out.tolist() == [[0.5, -0.5, -0.5, -0.5], [0.5, -0.5, 0.5, -0.5], [0.0, 1.0, 0.0, 0.0]]
