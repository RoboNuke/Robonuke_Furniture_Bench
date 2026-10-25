"""The -FBRel-ResiP-v0 interface: observation and action relative to the end-effector.

The torch terms (``mdp.ee_vel_ee``, ``mdp.parts_poses_ee_rot6d``, ``mdp.obstacle_pos_ee``,
``mdp.FurnitureBenchRelAction``) against their NumPy twins (``demos.ee_relative_states`` /
``demos.ee_relative_actions``), which convert the demos, with the env's own inputs faked; and the
task's cfg wiring. No Isaac Sim.
"""

import importlib
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests import isaaclab_stub
from tests.test_fb_action import Recorder

N = 16
#: the term's default (FurnitureBenchRelActionCfg.pos_limit, demos.REL_POS_LIMIT)
POS_LIMIT = 0.12


@pytest.fixture
def mdp():
    uninstall = isaaclab_stub.install()
    from furniture_lab.tasks.assembly.mdp import actions, observations, rot6d

    yield SimpleNamespace(actions=actions, observations=observations, rot6d=rot6d)
    uninstall()


def random_resip_states(rot6d, n=N, seed=0):
    """``(n, 63)`` ResiP observations (``FBABS_RESIP_OBS_LAYOUT``) with real rotations, plus the
    EE's ``(n, 3)`` position and ``(n, 4)`` wxyz quaternion they hold."""
    g = torch.Generator().manual_seed(seed)

    def quats(k):
        q = torch.randn(k, 4, generator=g, dtype=torch.float64)
        q = q / q.norm(dim=-1, keepdim=True)
        return torch.where(q[:, :1] < 0, -q, q)

    p_ee = torch.rand(n, 3, generator=g, dtype=torch.float64) * 0.6 + torch.tensor([0.3, -0.3, 0.0], dtype=torch.float64)
    q_ee = quats(n)
    vel = torch.randn(n, 6, generator=g, dtype=torch.float64) * 0.1
    width = torch.rand(n, 1, generator=g, dtype=torch.float64) * 0.08
    parts = torch.cat(
        [torch.rand(n * 5, 3, generator=g, dtype=torch.float64) * 0.5, rot6d.quat_to_rot6d(quats(n * 5))], dim=-1
    ).reshape(n, 45)
    xy = torch.rand(n, 2, generator=g, dtype=torch.float64) * 0.1 + torch.tensor([0.65, -0.02], dtype=torch.float64)
    states = torch.cat([p_ee, rot6d.quat_to_rot6d(q_ee), vel, width, parts, xy], dim=-1)
    return states, p_ee, q_ee


def test_the_relative_observation_terms_match_their_numpy_twin(mdp, monkeypatch):
    from furniture_lab import demos

    obs = mdp.observations
    states, p_ee, q_ee = random_resip_states(mdp.rot6d)
    # the env's inputs, as the absolute ResiP terms give them
    monkeypatch.setattr(obs, "_ee_pose_b", lambda env, asset_cfg, offset: (p_ee, q_ee))
    monkeypatch.setattr(obs, "ee_vel", lambda env, asset_cfg, offset=None: states[:, 9:15])
    monkeypatch.setattr(obs, "parts_poses_b_rot6d", lambda env: states[:, 16:61])
    monkeypatch.setattr(obs, "obstacle_xy_b", lambda env: states[:, 61:63])
    env = SimpleNamespace(num_envs=N)
    torch_obs = torch.cat(
        [obs.ee_vel_ee(env, None), states[:, 15:16], obs.parts_poses_ee_rot6d(env, None, None), obs.obstacle_pos_ee(env, None, None)],
        dim=-1,
    )
    twin = demos.ee_relative_states(states.numpy())
    assert twin.shape == (N, 55) and torch_obs.shape == (N, 55)
    assert np.allclose(torch_obs.numpy(), twin, atol=1e-5)


def test_the_relative_observation_is_the_absolute_one_when_the_ee_is_the_base_frame(mdp):
    from furniture_lab import demos

    states, _, _ = random_resip_states(mdp.rot6d)
    states[:, 0:3] = 0.0
    states[:, 3:9] = torch.tensor([1.0, 0, 0, 0, 1.0, 0], dtype=torch.float64)
    rel = demos.ee_relative_states(states.numpy())
    assert np.allclose(rel[:, 0:7], states[:, 9:16].numpy(), atol=1e-6)  # velocities and width as they were
    assert np.allclose(rel[:, 7:52], states[:, 16:61].numpy(), atol=1e-6)  # the parts as they were
    assert np.allclose(rel[:, 52:55], np.concatenate([states[:, 61:63].numpy(), np.zeros((N, 1))], axis=1), atol=1e-6)


def test_the_relative_observation_turns_with_the_ee(mdp):
    """A part straight ahead of the EE along its own x stays at (d, 0, 0) however the EE is turned."""
    from furniture_lab import demos

    rot6d = mdp.rot6d
    states, p_ee, q_ee = random_resip_states(rot6d)
    R_ee = rot6d.quat_to_mat(q_ee)
    states[:, 16:19] = p_ee + 0.2 * R_ee[:, :, 0]  # 0.2 m along the EE's x axis
    states[:, 19:25] = rot6d.mat_to_rot6d(R_ee)  # the part oriented as the EE
    rel = demos.ee_relative_states(states.numpy())
    assert np.allclose(rel[:, 7:10], np.tile([0.2, 0.0, 0.0], (N, 1)), atol=1e-6)
    assert np.allclose(rel[:, 10:16], np.tile([1.0, 0, 0, 0, 1.0, 0], (N, 1)), atol=1e-6)


def make_rel_term(actions_module):
    term = actions_module.FurnitureBenchRelAction.__new__(actions_module.FurnitureBenchRelAction)
    term.arm, term.gripper = Recorder(), Recorder()
    term.num_envs, term.device = N, "cpu"
    term._env, term._ee_cfg, term._offset = None, None, (0.0, 0.0, 0.1034)
    term.cfg = SimpleNamespace(pos_limit=POS_LIMIT)
    term._raw = torch.zeros(N, 10)
    term._processed = torch.zeros(N, 8)
    term._grip = actions_module._GripperHysteresis(N, "cpu", 0.0)
    return term


def small_rotations(rot6d, g, max_deg=30.0):
    """``(N, 3, 3)`` rotations about random axes by up to ``max_deg``."""
    axis = torch.randn(N, 3, generator=g, dtype=torch.float64)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    half = torch.deg2rad(torch.rand(N, 1, generator=g, dtype=torch.float64) * max_deg) / 2
    return rot6d.quat_to_mat(torch.cat([half.cos(), half.sin() * axis], dim=-1))


def test_the_relative_action_term_inverts_the_demos_conversion(mdp, monkeypatch):
    """action.abs -> ee_relative_actions -> FurnitureBenchRelAction gives the arm action.abs's target
    back, for targets inside the limits (a few cm, up to 30 deg from the EE)."""
    from furniture_lab import demos

    rot6d = mdp.rot6d
    states, p_ee, q_ee = random_resip_states(rot6d)
    g = torch.Generator().manual_seed(3)
    R_goal = rot6d.quat_to_mat(q_ee) @ small_rotations(rot6d, g)
    goal = torch.cat(
        [p_ee + torch.randn(N, 3, generator=g, dtype=torch.float64) * 0.02, rot6d.mat_to_rot6d(R_goal), torch.sign(torch.randn(N, 1, generator=g, dtype=torch.float64))],
        dim=-1,
    )
    rel = torch.as_tensor(demos.ee_relative_actions(states.numpy(), goal.numpy()), dtype=torch.float32)
    assert rel[:, 0:9].abs().max() < 1.0  # inside the limits: nothing clipped
    monkeypatch.setattr(mdp.actions, "_ee_pose_b", lambda env, cfg, offset: (p_ee.float(), q_ee.float()))
    term = make_rel_term(mdp.actions)
    term.process_actions(rel)
    sent = term.arm.received  # [pos 3, quat wxyz 4]
    assert torch.allclose(sent[:, 0:3], goal[:, 0:3].float(), atol=1e-5)
    want_q = rot6d.rot6d_to_quat(goal[:, 3:9].float())
    assert torch.allclose(torch.minimum((sent[:, 3:7] - want_q).abs().amax(-1), (sent[:, 3:7] + want_q).abs().amax(-1)), torch.zeros(N), atol=1e-5)
    assert torch.equal(term.gripper.received, -goal[:, 9:10].float())  # flipped as the absolute term flips it
    assert torch.equal(term.raw_actions, rel)  # the relative action, as the policy sent it


def test_a_zero_relative_action_holds_the_ee(mdp, monkeypatch):
    rot6d = mdp.rot6d
    _, p_ee, q_ee = random_resip_states(rot6d)
    monkeypatch.setattr(mdp.actions, "_ee_pose_b", lambda env, cfg, offset: (p_ee.float(), q_ee.float()))
    term = make_rel_term(mdp.actions)
    hold = torch.tensor([0.0] * 9 + [-1.0]).expand(N, -1).clone()
    term.process_actions(hold)
    sent = term.arm.received
    assert torch.allclose(sent[:, 0:3], p_ee.float(), atol=1e-6)
    q = q_ee.float()
    assert torch.allclose(torch.minimum((sent[:, 3:7] - q).abs().amax(-1), (sent[:, 3:7] + q).abs().amax(-1)), torch.zeros(N), atol=1e-5)


def test_a_dpos_along_the_ees_x_moves_the_target_along_it_in_units_of_the_limit(mdp, monkeypatch):
    rot6d = mdp.rot6d
    _, p_ee, q_ee = random_resip_states(rot6d)
    monkeypatch.setattr(mdp.actions, "_ee_pose_b", lambda env, cfg, offset: (p_ee.float(), q_ee.float()))
    term = make_rel_term(mdp.actions)
    step = torch.tensor([0.5] + [0.0] * 8 + [-1.0]).expand(N, -1).clone()
    term.process_actions(step)
    x_axis = rot6d.quat_to_mat(q_ee.float())[:, :, 0]
    assert torch.allclose(term.arm.received[:, 0:3], p_ee.float() + 0.5 * POS_LIMIT * x_axis, atol=1e-6)


def test_an_action_past_the_limit_is_clipped_to_it_by_the_term_and_the_conversion(mdp, monkeypatch):
    """A target 0.3 m along the EE's z converts to dpos z = 1, and the term sends any dpos z >= 1 to
    pos_limit along that axis; the gripper command is never clipped."""
    from furniture_lab import demos

    rot6d = mdp.rot6d
    states, p_ee, q_ee = random_resip_states(rot6d)
    R_ee = rot6d.quat_to_mat(q_ee)
    far = torch.cat([p_ee + 0.3 * R_ee[:, :, 2], rot6d.mat_to_rot6d(R_ee), torch.ones(N, 1, dtype=torch.float64)], dim=-1)
    rel = demos.ee_relative_actions(states.numpy(), far.numpy())
    assert np.allclose(rel, np.tile([0.0, 0.0, 1.0] + [0.0] * 6 + [1.0], (N, 1)), atol=1e-6)
    monkeypatch.setattr(mdp.actions, "_ee_pose_b", lambda env, cfg, offset: (p_ee.float(), q_ee.float()))
    term = make_rel_term(mdp.actions)
    term.process_actions(torch.tensor([0.0, 0.0, 2.5] + [0.0] * 6 + [3.0]).expand(N, -1).clone())
    assert torch.allclose(term.arm.received[:, 0:3], (p_ee + POS_LIMIT * R_ee[:, :, 2]).float(), atol=1e-6)
    assert torch.equal(term.gripper.received, torch.full((N, 1), -3.0))  # sign flipped as the absolute term flips it, unclipped


def test_the_relative_action_constants_agree_with_their_numpy_twins(mdp):
    from furniture_lab import demos

    assert demos.REL_POS_LIMIT == POS_LIMIT
    assert mdp.actions.ROT6D_IDENTITY == demos.ROT6D_IDENTITY == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def plain(value):
    """A cfg as nested dicts / lists, for comparing two cfgs field by field (the stub's configclass has no ==)."""
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "__dict__") and not isinstance(value, type) and not callable(value):
        return {k: plain(v) for k, v in vars(value).items()}
    return value


@pytest.fixture
def cfg_module():
    uninstall = isaaclab_stub.install()
    try:
        yield importlib.import_module("furniture_lab.tasks.assembly.assembly_env_cfg")
    finally:
        uninstall()


def test_the_relative_task_is_the_resip_task_with_the_relative_interface(cfg_module):
    rel, ab = cfg_module.OneLegFBRelResiPEnvCfg(), cfg_module.OneLegFBAbsResiPEnvCfg()
    pol = rel.observations.policy
    terms = [n for n, v in vars(pol).items() if hasattr(v, "func")]
    assert terms == ["ee_vel", "gripper_width", "parts_poses", "obstacle_pos"]
    assert [getattr(pol, t).func.__name__ for t in terms] == ["ee_vel_ee", "gripper_width", "parts_poses_ee_rot6d", "obstacle_pos_ee"]
    assert sum(w for _, w in cfg_module.FBREL_RESIP_OBS_LAYOUT) == 55 and cfg_module.FBREL_ACTION_DIM == 10
    # furniture.ee_vel_point reaches the relative velocity term as it does the absolute one
    assert pol.ee_vel.params["offset"] == ab.observations.policy.ee_vel.params["offset"] == cfg_module.EE_OFFSET
    assert pol.enable_corruption is False and pol.concatenate_terms is True
    fb = rel.actions.fb
    assert fb.class_type.__name__ == "FurnitureBenchRelAction"
    assert fb.pos_limit == POS_LIMIT  # demos.REL_POS_LIMIT, what the demos are converted with
    # the arm takes absolute targets, as the FBAbs task's does (the term converts the delta)
    assert fb.arm.controller.use_relative_mode is False and fb.arm.class_type.__name__ == "TcpDifferentialIKAction"
    assert fb.arm.body_offset.pos == cfg_module.EE_OFFSET
    # everything but the interface is the FBAbs ResiP task's
    for field in ("decimation", "episode_length_s", "furniture", "events", "rewards", "terminations"):
        assert plain(getattr(rel, field)) == plain(getattr(ab, field)), field
    assert plain(rel.scene) == plain(ab.scene)
    assert plain(fb.gripper) == plain(ab.actions.fb.gripper)
    # the FBAbs task's own interface is untouched
    assert ab.actions.fb.class_type.__name__ == "FurnitureBenchAbsAction"
    assert cfg_module.fbabs_obs_layout(ab) == cfg_module.FBABS_RESIP_OBS_LAYOUT


def test_the_relative_task_id_is_registered(cfg_module):
    import gymnasium as gym

    import furniture_lab.tasks  # noqa: F401

    spec = gym.spec("Isaac-Furniture-OneLeg-FBRel-ResiP-v0")
    mod, attr = spec.kwargs["env_cfg_entry_point"].split(":")
    assert getattr(importlib.import_module(mod), attr).__name__ == "OneLegFBRelResiPEnvCfg"
