"""The env cfg's wiring, checked against stubbed Isaac Lab modules (no Isaac Sim needed).

What this can catch: a part missing from the scene, a term that forgets its ``asset_cfg``,
events out of order, a registered task pointing at a name that does not exist, the robot's
arm not switched to effort control. What it cannot: anything about simulation.
"""

import importlib
import os
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import isaaclab_stub  # noqa: E402

from furniture_lab.tasks.furniture import frames, square_table as st  # noqa: E402


@pytest.fixture
def cfg_module():
    uninstall = isaaclab_stub.install()
    try:
        yield importlib.import_module("furniture_lab.tasks.assembly.assembly_env_cfg")
    finally:
        uninstall()


def test_scene_holds_every_part_at_its_reset_pose(cfg_module):
    cfg = cfg_module.OneLegEnvCfg()
    for part in st.PARTS:
        entry = getattr(cfg.scene, part.name)
        pos, quat = st.reset_pose_world(part)
        assert entry.prim_path == "{ENV_REGEX_NS}/" + part.name
        assert tuple(entry.init_state.pos) == pytest.approx(pos)
        assert tuple(entry.init_state.rot) == pytest.approx(quat)
        assert os.path.isfile(entry.spawn.usd_path)  # the repo's own assets/
    assert cfg.furniture.part_names == st.PART_NAMES
    assert cfg.furniture.task == "one_leg"
    assert cfg.furniture.mesh_frame == "uwlab"
    assert cfg_module.SquareTableEnvCfg().furniture.task == "square_table"


def test_robot_arm_is_effort_controlled_with_reset_joints(cfg_module):
    robot = cfg_module.OneLegEnvCfg().scene.robot
    assert robot.prim_path == "{ENV_REGEX_NS}/Robot"
    assert robot.spawn.rigid_props.disable_gravity is True
    for group in ("panda_shoulder", "panda_forearm"):
        assert robot.actuators[group].stiffness == 0.0 and robot.actuators[group].damping == 0.0
    assert robot.actuators["panda_hand"].stiffness > 0
    jp = robot.init_state.joint_pos
    assert [jp[f"panda_joint{i}"] for i in range(1, 8)] == pytest.approx(frames.RESET_JOINTS)
    assert jp["panda_finger_joint.*"] == pytest.approx(frames.MAX_GRIPPER_WIDTH / 2)


def test_actions_use_upstream_gains_and_tcp(cfg_module):
    actions = cfg_module.OneLegEnvCfg().actions
    assert actions.arm.body_name == "panda_hand"
    assert tuple(actions.arm.body_offset.pos) == (0.0, 0.0, 0.1034)
    ctrl = actions.arm.controller_cfg
    assert ctrl.target_types == ["pose_rel"]
    assert tuple(ctrl.motion_stiffness_task) == (90.0, 90.0, 90.0, 70.0, 60.0, 80.0)
    assert ctrl.nullspace_control == "position" and ctrl.nullspace_stiffness == 10.0
    assert actions.arm.nullspace_joint_pos_target == "default"
    assert actions.gripper.open_command_expr["panda_finger_joint.*"] == pytest.approx(0.0325)
    assert actions.gripper.close_command_expr["panda_finger_joint.*"] == 0.0


def test_observation_terms_name_their_bodies(cfg_module):
    policy = cfg_module.OneLegEnvCfg().observations.policy
    # a SceneEntityCfg with body/joint names is only resolved when passed as a param
    for name in ("ee_pos", "ee_quat", "ee_vel"):
        term = getattr(policy, name)
        assert term.params["asset_cfg"].body_names == "panda_hand"
    assert policy.gripper_width.params["asset_cfg"].joint_names == "panda_finger_joint.*"
    assert policy.ee_pos.params["offset"] == (0.0, 0.0, 0.1034)
    assert policy.concatenate_terms is True and policy.enable_corruption is False


def test_reset_events_run_in_order(cfg_module):
    events = cfg_module.OneLegEnvCfg().events
    names = [k for k, v in vars(events).items() if getattr(v, "mode", None) == "reset"]
    assert names == ["reset_scene", "reset_parts", "reset_assembly"]
    assert events.reset_parts.params == {"pos_range": 0.015, "yaw_range_deg": 15.0, "max_tries": 20}
    startup = [v for v in vars(events).values() if getattr(v, "mode", None) == "startup"]
    friction = [v for v in startup if "static_friction_range" in v.params]
    mass = [v for v in startup if "mass_distribution_params" in v.params]
    assert sorted(v.params["asset_cfg"]._args[0] for v in friction) == sorted(st.PART_NAMES)
    assert all(v.params["static_friction_range"] == (st.PART_FRICTION,) * 2 for v in friction)
    assert {v.params["asset_cfg"]._args[0]: v.params["mass_distribution_params"][0] for v in mass} == {
        p.name: p.mass for p in st.PARTS
    }
    assert all(v.params["operation"] == "abs" for v in mass)


def test_timing_matches_upstream(cfg_module):
    cfg = cfg_module.OneLegEnvCfg()
    assert cfg.sim.dt == pytest.approx(1 / 120)
    assert cfg.decimation == 6  # 20 Hz policy
    assert cfg.sim.physx.bounce_threshold_velocity == 0.02
    assert cfg.sim.physx.friction_offset_threshold == 0.01
    assert cfg.sim.physx.friction_correlation_distance == 0.0005


def test_terminations_and_reward(cfg_module):
    cfg = cfg_module.OneLegEnvCfg()
    assert cfg.terminations.time_out.time_out is True
    assert cfg.terminations.success.func.__name__ == "all_assembled"
    assert cfg.rewards.assembled.weight == 1.0


def test_registry_entry_points_resolve(cfg_module):
    import gymnasium as gym

    import furniture_lab.tasks  # noqa: F401

    for task, cls in (("Isaac-Furniture-OneLeg-v0", "OneLegEnvCfg"), ("Isaac-Furniture-SquareTable-v0", "SquareTableEnvCfg")):
        spec = gym.spec(task)
        mod, attr = spec.kwargs["env_cfg_entry_point"].split(":")
        assert getattr(importlib.import_module(mod), attr).__name__ == cls
        mod, attr = spec.entry_point.split(":")
        assert hasattr(importlib.import_module(mod), attr)


def test_part_out_of_limits_is_upstreams_rule(cfg_module, monkeypatch):
    """A part on the table but outside upstream's robot-frame limits ends the episode; the old
    'fell 5 cm below the table' rule alone never fired, since the table is 1.3 m wide."""
    import torch

    from furniture_lab.tasks.assembly.mdp import terminations

    quat = torch.tensor([[[1.0, 0, 0, 0]] * 2])
    cases = {
        "on the table, inside": ([[0.54, 0.0, 0.001], [0.37, 0.2, 0.0]], False),
        "slid toward the robot": ([[0.54, 0.0, 0.001], [0.28, 0.2, 0.0]], True),
        "slid sideways": ([[0.54, 0.0, 0.001], [0.37, 0.56, 0.0]], True),
        "thrown up": ([[0.54, 0.0, 0.41], [0.37, 0.2, 0.0]], True),
        "below the table": ([[0.54, 0.0, 0.001], [0.37, 0.2, -0.1]], True),
        "slightly below (detection slop)": ([[0.54, 0.0, 0.001], [0.37, 0.2, -0.03]], False),
    }
    for name, (pos, expect) in cases.items():
        monkeypatch.setattr(terminations, "parts_pose_b", lambda env, p=pos: (torch.tensor([p]), quat))
        assert bool(terminations.part_out_of_limits(env=None)[0]) is expect, name
    assert cfg_module.TerminationsCfg().part_out_of_limits.func is terminations.part_out_of_limits


def test_friction_knobs_reach_the_events_and_the_table(cfg_module):
    cfg = cfg_module.OneLegEnvCfg()
    assert cfg.events.leg4_friction.params["static_friction_range"] == (st.PART_FRICTION, st.PART_FRICTION)
    assert cfg.sim.physics_material.static_friction == st.TABLE_FRICTION
    cfg.furniture.part_friction = 0.5
    cfg.furniture.table_friction = 0.4
    cfg.apply_friction()
    for name in st.PART_NAMES:
        term = getattr(cfg.events, name.replace("square_table_", "") + "_friction")
        assert term.params["static_friction_range"] == (0.5, 0.5)
        assert term.params["dynamic_friction_range"] == (0.5, 0.5)
    assert cfg.sim.physics_material.static_friction == 0.4
    assert cfg.sim.physics_material.dynamic_friction == 0.4
