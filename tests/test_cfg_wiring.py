"""The env cfg's wiring, checked against stubbed Isaac Lab modules (no Isaac Sim needed).

What this can catch: a part missing from the scene, a term that forgets its ``asset_cfg``,
events out of order, a registered task pointing at a name that does not exist, the robot's
arm not switched to effort control. What it cannot: anything about simulation.
"""

import importlib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import isaaclab_stub  # noqa: E402

from furniture_lab.tasks.furniture import frames, square_table as st  # noqa: E402


@pytest.fixture
def cfg_module(tmp_path, monkeypatch):
    for name in ("square_table_top", "square_table_leg"):
        (tmp_path / f"{name}.usd").touch()
    monkeypatch.setenv("FURNITURE_ASSETS_DIR", str(tmp_path))
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
        assert entry.spawn.mass_props.mass == part.mass
        assert entry.spawn.usd_path.endswith(".usd")
    assert cfg.furniture.part_names == st.PART_NAMES
    assert cfg.furniture.task == "one_leg"
    assert cfg_module.SquareTableEnvCfg().furniture.task == "square_table"


def test_shared_leg_usd_is_used_when_no_per_leg_file(cfg_module):
    cfg = cfg_module.OneLegEnvCfg()
    assert cfg.scene.square_table_leg3.spawn.usd_path.endswith("square_table_leg.usd")


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
    assert sorted(v.params["asset_cfg"]._args[0] for v in startup) == sorted(st.PART_NAMES)
    assert all(v.params["static_friction_range"] == (st.PART_FRICTION,) * 2 for v in startup)


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


def test_missing_assets_dir_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("FURNITURE_ASSETS_DIR", raising=False)
    uninstall = isaaclab_stub.install()
    try:
        with pytest.raises(RuntimeError, match="FURNITURE_ASSETS_DIR"):
            importlib.import_module("furniture_lab.tasks.assembly.assembly_env_cfg")
    finally:
        uninstall()


def test_uwlab_file_names_are_accepted(tmp_path, monkeypatch):
    for name in ("square_table_top", "square_leg"):
        (tmp_path / f"{name}.usd").touch()
    monkeypatch.setenv("FURNITURE_ASSETS_DIR", str(tmp_path))
    monkeypatch.setenv("FURNITURE_MESH_FRAME", "uwlab")
    uninstall = isaaclab_stub.install()
    try:
        mod = importlib.import_module("furniture_lab.tasks.assembly.assembly_env_cfg")
        cfg = mod.OneLegEnvCfg()
        assert cfg.scene.square_table_leg4.spawn.usd_path.endswith("square_leg.usd")
        assert cfg.furniture.mesh_frame == "uwlab"
    finally:
        uninstall()
