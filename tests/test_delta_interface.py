"""The -FBDelta-ResiP-v0 interface: the FBAbs ResiP observation, the action relative to the end-effector
with an axis-angle rotation.

``mdp.FurnitureBenchDeltaAction`` against its NumPy twin ``demos.ee_delta_actions`` (which converts the
demos), with the env's own inputs faked; the axis-angle maps; the task's cfg wiring. No Isaac Sim.
"""

import numpy as np
import pytest
import torch

from tests.test_fb_action import Recorder
from tests.test_relative_interface import N, POS_LIMIT, cfg_module, mdp, plain, random_resip_states, small_rotations  # noqa: F401


def make_delta_term(actions_module):
    term = actions_module.FurnitureBenchDeltaAction.__new__(actions_module.FurnitureBenchDeltaAction)
    term.arm, term.gripper = Recorder(), Recorder()
    term.num_envs, term.device = N, "cpu"
    term._env, term._ee_cfg, term._offset = None, None, (0.0, 0.0, 0.1034)
    term.cfg = type("Cfg", (), {"pos_limit": POS_LIMIT})()
    term._raw = torch.zeros(N, 7)
    term._processed = torch.zeros(N, 8)
    term._grip = actions_module._GripperHysteresis(N, "cpu", 0.0)
    return term


def same_rotation(q, want):
    """wxyz quaternions equal up to sign, per row."""
    return torch.minimum((q - want).abs().amax(-1), (q + want).abs().amax(-1))


def test_the_axis_angle_maps_are_each_others_inverse(mdp):
    from furniture_lab import demos

    g = torch.Generator().manual_seed(5)
    axis = torch.randn(N, 3, generator=g, dtype=torch.float64)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    angle = torch.cat([torch.zeros(1), torch.tensor([1e-8, 1e-3]), torch.rand(N - 3, generator=g) * 3.0]).double()
    v = axis * angle[:, None]
    R = mdp.rot6d.axis_angle_to_mat(v)
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3, dtype=torch.float64).expand(N, 3, 3), atol=1e-12)
    assert torch.allclose(R[0], torch.eye(3, dtype=torch.float64))  # zero: the identity
    back = np.stack([demos.mat_to_axis_angle(r) for r in R.numpy()])
    assert np.allclose(back, v.numpy(), atol=1e-9)
    # the rotation about the z axis by 90 deg
    Rz = mdp.rot6d.axis_angle_to_mat(torch.tensor([0.0, 0.0, np.pi / 2], dtype=torch.float64))
    assert torch.allclose(Rz, torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float64), atol=1e-12)


def test_the_delta_action_term_inverts_the_demos_conversion(mdp, monkeypatch):
    """action.abs -> ee_delta_actions -> FurnitureBenchDeltaAction gives the arm action.abs's target back,
    for targets inside the limits (a few cm, up to 30 deg from the EE)."""
    from furniture_lab import demos

    rot6d = mdp.rot6d
    states, p_ee, q_ee = random_resip_states(rot6d)
    g = torch.Generator().manual_seed(3)
    R_goal = rot6d.quat_to_mat(q_ee) @ small_rotations(rot6d, g)
    goal = torch.cat(
        [p_ee + torch.randn(N, 3, generator=g, dtype=torch.float64) * 0.02, rot6d.mat_to_rot6d(R_goal), torch.sign(torch.randn(N, 1, generator=g, dtype=torch.float64))],
        dim=-1,
    )
    delta = torch.as_tensor(demos.ee_delta_actions(states.numpy(), goal.numpy()), dtype=torch.float32)
    assert delta.shape == (N, 7) and delta[:, 0:6].abs().max() < 1.0  # inside the limits: nothing clipped
    monkeypatch.setattr(mdp.actions, "_ee_pose_b", lambda env, cfg, offset: (p_ee.float(), q_ee.float()))
    term = make_delta_term(mdp.actions)
    term.process_actions(delta)
    sent = term.arm.received  # [pos 3, quat wxyz 4]
    assert torch.allclose(sent[:, 0:3], goal[:, 0:3].float(), atol=1e-5)
    assert torch.allclose(same_rotation(sent[:, 3:7], rot6d.rot6d_to_quat(goal[:, 3:9].float())), torch.zeros(N), atol=1e-5)
    assert torch.equal(term.gripper.received, -goal[:, 9:10].float())  # flipped as the absolute term flips it
    assert torch.equal(term.raw_actions, delta)  # the delta action, as the policy sent it


def test_a_zero_delta_action_holds_the_ee(mdp, monkeypatch):
    _, p_ee, q_ee = random_resip_states(mdp.rot6d)
    monkeypatch.setattr(mdp.actions, "_ee_pose_b", lambda env, cfg, offset: (p_ee.float(), q_ee.float()))
    term = make_delta_term(mdp.actions)
    term.process_actions(torch.tensor([0.0] * 6 + [-1.0]).expand(N, -1).clone())
    sent = term.arm.received
    assert torch.allclose(sent[:, 0:3], p_ee.float(), atol=1e-6)
    assert torch.allclose(same_rotation(sent[:, 3:7], q_ee.float()), torch.zeros(N), atol=1e-5)


def test_dpos_moves_along_the_ees_axes_in_units_of_the_limit_and_drot_turns_about_them(mdp, monkeypatch):
    rot6d = mdp.rot6d
    _, p_ee, q_ee = random_resip_states(rot6d)
    monkeypatch.setattr(mdp.actions, "_ee_pose_b", lambda env, cfg, offset: (p_ee.float(), q_ee.float()))
    term = make_delta_term(mdp.actions)
    term.process_actions(torch.tensor([0.5, 0.0, 0.0, 0.0, 0.0, 0.3, -1.0]).expand(N, -1).clone())
    R_ee = rot6d.quat_to_mat(q_ee.float())
    assert torch.allclose(term.arm.received[:, 0:3], p_ee.float() + 0.5 * POS_LIMIT * R_ee[:, :, 0], atol=1e-6)
    # 0.3 rad about the EE's own z: R_goal = R_ee Rz(0.3)
    c, s = np.cos(0.3), np.sin(0.3)
    Rz = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    assert torch.allclose(same_rotation(term.arm.received[:, 3:7], rot6d.mat_to_quat(R_ee @ Rz)), torch.zeros(N), atol=1e-5)


def test_a_delta_action_past_the_limit_is_clipped_to_it_by_the_term_and_the_conversion(mdp, monkeypatch):
    """A target 0.3 m along the EE's z converts to dpos z = 1, and the term sends any dpos z >= 1 to
    pos_limit along that axis; the gripper command is never clipped."""
    from furniture_lab import demos

    rot6d = mdp.rot6d
    states, p_ee, q_ee = random_resip_states(rot6d)
    R_ee = rot6d.quat_to_mat(q_ee)
    far = torch.cat([p_ee + 0.3 * R_ee[:, :, 2], rot6d.mat_to_rot6d(R_ee), torch.ones(N, 1, dtype=torch.float64)], dim=-1)
    delta = demos.ee_delta_actions(states.numpy(), far.numpy())
    assert np.allclose(delta, np.tile([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], (N, 1)), atol=1e-6)
    monkeypatch.setattr(mdp.actions, "_ee_pose_b", lambda env, cfg, offset: (p_ee.float(), q_ee.float()))
    term = make_delta_term(mdp.actions)
    term.process_actions(torch.tensor([0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 3.0]).expand(N, -1).clone())
    assert torch.allclose(term.arm.received[:, 0:3], (p_ee + POS_LIMIT * R_ee[:, :, 2]).float(), atol=1e-6)
    assert torch.equal(term.gripper.received, torch.full((N, 1), -3.0))  # sign flipped as the absolute term flips it, unclipped


def test_the_delta_task_is_the_resip_task_with_only_the_action_changed(cfg_module):
    from furniture_lab import demos

    delta, ab = cfg_module.OneLegFBDeltaResiPEnvCfg(), cfg_module.OneLegFBAbsResiPEnvCfg()
    fb = delta.actions.fb
    assert fb.class_type.__name__ == "FurnitureBenchDeltaAction" and cfg_module.FBDELTA_ACTION_DIM == 7
    assert fb.pos_limit == POS_LIMIT == demos.REL_POS_LIMIT  # what the demos are converted with
    # the arm takes absolute targets, as the FBAbs task's does (the term converts the delta)
    assert fb.arm.controller.use_relative_mode is False and fb.arm.class_type.__name__ == "TcpDifferentialIKAction"
    assert fb.arm.body_offset.pos == cfg_module.EE_OFFSET
    assert plain(fb.gripper) == plain(ab.actions.fb.gripper)
    # everything but the action is the FBAbs ResiP task's, the observation included
    for field in ("observations", "decimation", "episode_length_s", "furniture", "events", "rewards", "terminations", "scene"):
        assert plain(getattr(delta, field)) == plain(getattr(ab, field)), field
    assert cfg_module.fbabs_obs_layout(delta) == cfg_module.FBABS_RESIP_OBS_LAYOUT
    assert ab.actions.fb.class_type.__name__ == "FurnitureBenchAbsAction"  # the FBAbs task's own action is untouched


def test_the_delta_task_id_is_registered(cfg_module):
    import importlib

    import gymnasium as gym

    import furniture_lab.tasks  # noqa: F401

    spec = gym.spec("Isaac-Furniture-OneLeg-FBDelta-ResiP-v0")
    mod, attr = spec.kwargs["env_cfg_entry_point"].split(":")
    assert getattr(importlib.import_module(mod), attr).__name__ == "OneLegFBDeltaResiPEnvCfg"
