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


def test_rot6d_torch_and_numpy_agree_and_roundtrip(actions_module):
    import numpy as np

    from furniture_lab import demos
    from furniture_lab.tasks.assembly.mdp import rot6d

    g = torch.Generator().manual_seed(1)
    q = torch.randn(64, 4, generator=g)
    q = q / q.norm(dim=-1, keepdim=True)
    d6 = rot6d.quat_to_rot6d(q)
    for i in range(64):
        R_np = demos.quat_to_mat(q[i].numpy().astype(np.float64))
        assert np.allclose(demos.mat_to_rot6d(R_np), d6[i].numpy(), atol=1e-5)
        assert np.allclose(demos.rot6d_to_mat(d6[i].numpy()), R_np, atol=1e-5)
    back = rot6d.rot6d_to_quat(d6)
    assert torch.allclose(torch.minimum((back - q).abs().amax(-1), (back + q).abs().amax(-1)), torch.zeros(64), atol=1e-5)
    assert (back[:, 0] >= 0).all()


def test_the_abs_term_sends_the_target_pose_and_the_flipped_gripper(actions_module, monkeypatch):
    from furniture_lab.tasks.assembly.mdp import rot6d

    term = actions_module.FurnitureBenchAbsAction.__new__(actions_module.FurnitureBenchAbsAction)
    term.arm, term.gripper = Recorder(), Recorder()
    term.num_envs, term.device = 1, "cpu"
    term._raw = torch.zeros(1, 10)
    term._processed = torch.zeros(1, 8)
    s = math.sqrt(0.5)
    q = torch.tensor([[s, 0.0, 0.0, s]])  # 90 deg about z
    a = torch.cat([torch.tensor([[0.5, 0.1, 0.2]]), rot6d.quat_to_rot6d(q), torch.tensor([[1.0]])], dim=-1)
    term.process_actions(a)
    assert term.arm.received.shape == (1, 7) and term.gripper.received.shape == (1, 1)
    assert torch.allclose(term.arm.received[0, 0:3], torch.tensor([0.5, 0.1, 0.2]))
    assert torch.allclose(term.arm.received[0, 3:7], q[0], atol=1e-6)  # wxyz
    assert term.gripper.received[0, 0] == -1.0
    assert term.action_dim == 10


def test_fb_abs_action_is_the_pose_the_delta_commanded():
    import numpy as np

    from furniture_lab import demos

    ee_pos = np.array([0.5, 0.0, 0.3], dtype=np.float32)
    ee_q_wxyz = np.array([0.0, 1.0, 0.0, 0.0])  # hand pointing down
    half = 0.05
    dq_wxyz = np.array([math.cos(half), 0.0, 0.0, math.sin(half)])  # 0.1 rad about the EE's z
    obs = {"robot_state": {"ee_pos": ee_pos, "ee_quat": ee_q_wxyz[[1, 2, 3, 0]]}}
    act = np.array([0.01, 0.0, -0.02, *dq_wxyz[[1, 2, 3, 0]], 1.0], dtype=np.float32)
    out = demos.fb_abs_action(obs, act)
    assert out.shape == (10,)
    assert np.allclose(out[0:3], [0.51, 0.0, 0.28])
    R_goal = demos.quat_to_mat(ee_q_wxyz) @ demos.quat_to_mat(dq_wxyz)
    assert np.allclose(demos.rot6d_to_mat(out[3:9]), R_goal, atol=1e-6)
    assert out[9] == 1.0


def test_fb_abs_state_layout(actions_module):
    import numpy as np

    from furniture_lab import demos

    rs = {"ee_pos": np.zeros(3), "ee_quat": [0, 0, 0, 1.0], "ee_pos_vel": np.ones(3), "ee_ori_vel": 2 * np.ones(3), "gripper_width": [0.04]}
    parts = np.concatenate([[i, 0, 0, 0, 0, 0, 1.0] for i in range(5)])
    s = demos.fb_abs_state({"robot_state": rs, "parts_poses": parts})
    assert s.shape == (61,)
    assert np.allclose(s[3:9], [1, 0, 0, 0, 1, 0])  # identity rotation's rot6d
    assert s[15] == 0.04 and s[16] == 0 and s[16 + 9] == 1  # parts: pos x of part 0 and 1
    assert np.allclose(s[19:25], [1, 0, 0, 0, 1, 0])
