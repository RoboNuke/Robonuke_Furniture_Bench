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
    assert names == ["reset_scene", "reset_obstacles", "reset_parts", "reset_assembly"]
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


def test_gripper_knobs_reach_the_finger_actuator(cfg_module):
    cfg = cfg_module.OneLegEnvCfg()
    act = cfg.scene.robot.actuators["panda_hand"]
    assert act.stiffness == 600.0 and act.damping == 40.0
    cfg.furniture.gripper_stiffness = 2000.0
    cfg.furniture.gripper_effort_limit = 200.0
    cfg.apply_gripper()
    assert act.stiffness == 2000.0
    for name in ("effort_limit", "effort_limit_sim"):
        if hasattr(act, name):
            assert getattr(act, name) == 200.0


def test_furniturebench_interface_cfg(cfg_module):
    """Isaac-Furniture-OneLeg-FB-v0: upstream's 49-d state observation in its order, no
    previous action, and the 8-d action term driving the task's own arm and gripper terms."""
    cfg = cfg_module.OneLegFBEnvCfg()
    terms = [n for n, v in vars(cfg.observations.policy).items() if hasattr(v, "func")]
    assert terms == ["ee_pos", "ee_quat", "ee_vel", "gripper_width", "parts_poses"]
    assert cfg.observations.policy.ee_quat.func.__name__ == "ee_quat_xyzw"
    assert cfg.observations.policy.parts_poses.func.__name__ == "parts_poses_tag"
    assert cfg_module.FB_OBS_DIM == 49 and cfg_module.FB_ACTION_DIM == 8
    assert [n for n, _ in cfg_module.FB_OBS_LAYOUT] == ["ee_pos", "ee_quat", "ee_pos_vel", "ee_ori_vel", "gripper_width", "parts_poses"]
    fb = cfg.actions.fb
    own = cfg_module.ActionsCfg()  # the task's own terms (a configclass has no class attributes)
    assert fb.arm.body_name == own.arm.body_name and fb.arm.position_scale == own.arm.position_scale
    assert fb.gripper.joint_names == own.gripper.joint_names
    assert fb.class_type.__name__ == "FurnitureBenchAction"
    # the task env's success termination and the rest are inherited unchanged
    assert cfg.terminations.success.func.__name__ == "all_assembled"
    assert cfg.furniture.task == "one_leg"


def test_furniturebench_abs_interface_cfg(cfg_module):
    """Isaac-Furniture-OneLeg-FBAbs-v0: rot6d everywhere (61), the 10-d absolute action on a
    pose_abs OSC term with unit scales."""
    cfg = cfg_module.OneLegFBAbsEnvCfg()
    terms = [n for n, v in vars(cfg.observations.policy).items() if hasattr(v, "func")]
    assert terms == ["ee_pos", "ee_rot6d", "ee_vel", "gripper_width", "parts_poses"]
    assert cfg.observations.policy.parts_poses.func.__name__ == "parts_poses_tag_rot6d"
    assert cfg_module.FBABS_OBS_DIM == 61 and cfg_module.FBABS_ACTION_DIM == 10
    fb = cfg.actions.fb
    assert fb.class_type.__name__ == "FurnitureBenchAbsAction"
    assert fb.arm.controller_cfg.target_types == ["pose_abs"]
    assert fb.arm.position_scale == 1.0 and fb.arm.orientation_scale == 1.0
    # the shared task cfg is untouched
    own = cfg_module.ActionsCfg().arm
    assert own.controller_cfg.target_types == ["pose_rel"] and own.position_scale == 0.025



def test_the_table_is_dark_grey(cfg_module):
    cfg = cfg_module.OneLegFBAbsEnvCfg()
    assert tuple(cfg.scene.table.spawn.visual_material.diffuse_color) == (0.08, 0.08, 0.08)


def test_success_mode_defaults_to_soft(cfg_module):
    cfg = cfg_module.OneLegFBAbsEnvCfg()
    assert cfg.furniture.success_mode == "soft" and cfg.furniture.success_hold_steps == 10



def test_resip_sim_cfgs(cfg_module):
    """ResiP's FurnitureRLSimEnv: 10 Hz, differential IK into joint PD 1000/200, FurnitureSim
    friction, a 13 N gripper with hysteresis, 70 s episodes -- on every interface."""
    for name, absolute in (("OneLegResiPEnvCfg", False), ("OneLegFBResiPEnvCfg", False), ("OneLegFBAbsResiPEnvCfg", True)):
        cfg = getattr(cfg_module, name)()
        assert cfg.decimation == 12 and abs(cfg.sim.dt - 1 / 120) < 1e-12 and cfg.episode_length_s == 70.0
        assert cfg.scene.robot.actuators["panda_shoulder"].stiffness == 1000.0
        assert cfg.scene.robot.actuators["panda_forearm"].damping == 200.0
        f = cfg.furniture
        assert (f.part_friction, f.table_friction, f.gripper_effort_limit, f.gripper_margin) == (0.15, 0.10, 13.0, 0.019)
        assert f.obstacle_offset == 0.02 and cfg.events.reset_obstacles is not None
        assert f.gripper_velocity_limit == 0.2 and cfg.scene.robot.actuators["panda_hand"].velocity_limit_sim == 0.2
        assert cfg.scene.robot.actuators["panda_hand"].stiffness == 600.0 and cfg.scene.robot.actuators["panda_hand"].damping == 40.0
        arm = cfg.actions.fb.arm if hasattr(cfg.actions, "fb") else cfg.actions.arm
        assert arm.controller.use_relative_mode is (not absolute) and arm.controller.ik_method == "pinv"
        assert arm.body_offset.pos == cfg_module.EE_OFFSET
    # the ResiP FBAbs observation: the parts in the robot-base frame, then the front obstacle's x, y (63)
    resip = cfg_module.OneLegFBAbsResiPEnvCfg()
    pol = resip.observations.policy
    assert pol.obstacle_xy is not None and pol.parts_poses.func.__name__ == "parts_poses_b_rot6d"
    assert cfg_module.fbabs_obs_layout(resip) == cfg_module.FBABS_RESIP_OBS_LAYOUT
    assert sum(w for _, w in cfg_module.FBABS_RESIP_OBS_LAYOUT) == 63
    # the port's cfgs are untouched
    port = cfg_module.OneLegFBAbsEnvCfg()
    assert cfg_module.fbabs_obs_layout(port) == cfg_module.FBABS_OBS_LAYOUT
    assert port.furniture.obstacle_offset == 0.0
    assert port.decimation == 6 and port.scene.robot.actuators["panda_shoulder"].stiffness == 0.0


def test_the_hybrid_cfg_is_resip_with_the_arm_on_joint_torques(cfg_module):
    """``OneLegFBAbsResiPHybridEnvCfg``: ResiP's sim and interface, the arm's drives with no
    stiffness or damping, one hybrid torque term (no IK arm term beside it) and the gripper."""
    cfg = cfg_module.OneLegFBAbsResiPHybridEnvCfg()
    assert cfg.decimation == 12 and cfg.episode_length_s == 70.0
    for group in ("panda_shoulder", "panda_forearm"):
        assert cfg.scene.robot.actuators[group].stiffness == 0.0
        assert cfg.scene.robot.actuators[group].damping == 0.0
    assert cfg.scene.robot.actuators["panda_hand"].stiffness > 0
    assert not hasattr(cfg.actions, "arm") and not hasattr(cfg.actions, "fb")
    term = cfg.actions.hybrid
    assert term.class_type.__name__ == "HybridTorqueAbsAction"
    assert term.body_name == "panda_hand" and term.joint_names == ["panda_joint.*"]
    assert term.gripper is not None
    assert cfg.furniture.gripper_margin == 0.019
    # the plain ResiP cfg is untouched by the hybrid one
    assert cfg_module.OneLegFBAbsResiPEnvCfg().scene.robot.actuators["panda_shoulder"].stiffness == 1000.0
    assert cfg.hybrid.force_axes == [1, 1, 1, 0, 0, 0]
    # the gains tuned to track like ResiP's IK + joint PD (scripts/hybrid_tracking.py)
    assert cfg.hybrid.inertial_decoupling is True and cfg.hybrid.force_feedforward is True
    assert list(cfg.hybrid.default_force_gains)[0:3] == [0.1] * 3
    # the wrench estimate removes the arm's own inertia here, and only here
    assert cfg.furniture.wrench_inertial is True
    assert cfg_module.OneLegFBAbsResiPEnvCfg().furniture.wrench_inertial is False
    assert list(cfg.hybrid.stiffness) == [300.0] * 3 + [500.0] * 3


def test_the_hybrid_task_id_is_registered(cfg_module):
    import gymnasium as gym

    import furniture_lab.tasks  # noqa: F401

    spec = gym.spec("Isaac-Furniture-OneLeg-FBAbs-ResiP-Hybrid-v0")
    mod, attr = spec.kwargs["env_cfg_entry_point"].split(":")
    assert getattr(importlib.import_module(mod), attr).__name__ == "OneLegFBAbsResiPHybridEnvCfg"
