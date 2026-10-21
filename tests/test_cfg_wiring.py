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
        pos, quat = st.reset_pose_world(part, cfg.furniture.mesh_frame)  # the default assets' frame
        assert entry.prim_path == "{ENV_REGEX_NS}/" + part.name
        assert tuple(entry.init_state.pos) == pytest.approx(pos)
        assert tuple(entry.init_state.rot) == pytest.approx(quat)
        assert os.path.isfile(entry.spawn.usd_path)  # the repo's own assets/
    assert cfg.furniture.part_names == st.PART_NAMES
    assert cfg.furniture.task == "one_leg"
    assert cfg.furniture.mesh_frame == "obj"  # part_assets furniturebench: FurnitureBench's OBJ frame
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
    # the parts' friction events (the robot's own is furniture.robot_friction's: 1.0 by default)
    friction = [v for v in startup if "static_friction_range" in v.params and v.params["asset_cfg"]._args[0] != "robot"]
    assert events.robot_friction.params["static_friction_range"] == (1.0, 1.0)
    mass = [v for v in startup if "mass_distribution_params" in v.params]
    assert sorted(v.params["asset_cfg"]._args[0] for v in friction) == sorted(st.PART_NAMES)
    assert all(v.params["static_friction_range"] == (st.PART_FRICTION,) * 2 for v in friction)
    # furniture.part_masses: what FurnitureSim simulates (st.ISAAC_GYM_MASS), not PartSpec.mass
    assert {v.params["asset_cfg"]._args[0]: v.params["mass_distribution_params"] for v in mass} == {
        p.name: (st.ISAAC_GYM_MASS[p.name],) * 2 for p in st.PARTS
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


OBSTACLES = ("obstacle_front", "obstacle_left", "obstacle_right")


def test_obstacle_and_robot_friction_default_to_furnituresims(cfg_module):
    """Defaults on every cfg (apply_resip included): FurnitureSim's 1.0 on the obstacle boxes and a
    startup event giving every robot shape 1.0. The earlier values (0.10, -1: not set) still work."""
    for name in ("OneLegEnvCfg", "OneLegFBAbsEnvCfg", "OneLegFBAbsResiPEnvCfg"):
        cfg = getattr(cfg_module, name)()
        assert cfg.furniture.obstacle_friction == cfg.furniture.robot_friction == cfg_module.RESIP_CONTACT_FRICTION == 1.0
        for ob in OBSTACLES:
            mat = getattr(cfg.scene, ob).spawn.physics_material
            assert (mat.static_friction, mat.dynamic_friction) == (1.0, 1.0)
        assert cfg.events.robot_friction.params["static_friction_range"] == (1.0, 1.0)
    cfg = cfg_module.OneLegFBAbsResiPEnvCfg()
    cfg.furniture.obstacle_friction = st.TABLE_FRICTION
    cfg.apply_friction()
    assert getattr(cfg.scene, "obstacle_front").spawn.physics_material.static_friction == st.TABLE_FRICTION
    cfg.furniture.obstacle_friction = 1.0
    cfg.apply_friction()
    for ob in OBSTACLES:
        mat = getattr(cfg.scene, ob).spawn.physics_material
        assert (mat.static_friction, mat.dynamic_friction) == (1.0, 1.0)
    term = cfg.events.robot_friction
    assert term.mode == "startup" and term.params["static_friction_range"] == (1.0, 1.0)
    assert term.params["dynamic_friction_range"] == (1.0, 1.0)
    assert term.params["asset_cfg"]._args == ("robot",) and not hasattr(term.params["asset_cfg"], "body_names")
    # the table box and the parts are untouched
    assert cfg.scene.table.spawn.physics_material.static_friction == st.TABLE_FRICTION
    assert cfg.events.leg4_friction.params["static_friction_range"] == (st.PART_FRICTION, st.PART_FRICTION)
    cfg.furniture.robot_friction = -1.0
    cfg.apply_friction()
    assert cfg.events.robot_friction is None
    cfg.furniture.robot_friction = 0.0  # zero is a friction, not "off"
    cfg.apply_friction()
    assert cfg.events.robot_friction.params["static_friction_range"] == (0.0, 0.0)


def test_ee_vel_point_defaults_to_the_ee(cfg_module):
    """``ee`` (default): the ee_vel term at EE_OFFSET, on every group; ``hand_com``: no offset."""
    for name in ("OneLegEnvCfg", "OneLegFBEnvCfg", "OneLegFBAbsEnvCfg", "OneLegFBAbsResiPEnvCfg"):
        cfg = getattr(cfg_module, name)()
        assert cfg.furniture.ee_vel_point == "ee"
        assert cfg.observations.policy.ee_vel.params["offset"] == cfg_module.EE_OFFSET
        cfg.furniture.ee_vel_point = "hand_com"
        cfg.apply_observations()
        assert "offset" not in cfg.observations.policy.ee_vel.params
        cfg.furniture.ee_vel_point = "ee"
        cfg.apply_observations()
        assert cfg.observations.policy.ee_vel.params["offset"] == cfg_module.EE_OFFSET
        assert cfg.observations.policy.ee_vel.params["asset_cfg"].body_names == "panda_hand"
        cfg.furniture.ee_vel_point = "hand_com"
        cfg.apply_observations()
        assert "offset" not in cfg.observations.policy.ee_vel.params
    cfg.furniture.ee_vel_point = "tcp"
    with pytest.raises(ValueError, match="ee_vel_point"):
        cfg.apply_observations()


def test_reset_and_masses_default_to_furniturebench(cfg_module):
    """Every cfg, apply_resip included: ``furniture.randomness`` FurnitureBench's ``low`` (run by the
    reset_parts term) and ``part_masses`` what FurnitureSim simulates; bad values fail at construction."""
    for name in ("OneLegEnvCfg", "OneLegFBAbsEnvCfg", "OneLegFBAbsResiPEnvCfg", "OneLegFBAbsResiPHybridEnvCfg"):
        cfg = getattr(cfg_module, name)()
        assert cfg.furniture.randomness == "low" and cfg.furniture.high_layout == 0
        assert cfg.furniture.part_masses == tuple(st.ISAAC_GYM_MASS[n] for n in st.PART_NAMES)
        assert cfg.events.reset_parts.func.__name__ == "randomize_parts_pose"
    assert st.ISAAC_GYM_MASS == {"square_table_top": 0.4115, "square_table_leg1": 0.0264, "square_table_leg2": 0.0265,
                                 "square_table_leg3": 0.0264, "square_table_leg4": 0.0291}
    cfg = cfg_module.OneLegEnvCfg()
    cfg.furniture.part_masses = (0.151, 0.0231, 0.0231, 0.0231, 0.0231)
    cfg.apply_masses()
    assert cfg.events.top_mass.params["mass_distribution_params"] == (0.151, 0.151)
    for bad, match in (({"randomness": "medium"}, "furniture.randomness"), ({"high_layout": 3}, "high_layout"),
                       ({"part_masses": (0.4, 0.02)}, "part_masses")):
        with pytest.raises(ValueError, match=match):
            cfg_module.OneLegEnvCfg(furniture=cfg_module.FurnitureCfg(task="one_leg", **bad))


def test_gripper_mode_switches_the_drive_the_term_and_the_travel(cfg_module):
    """``position`` (the port tasks' default): the PD drive and the binary position term, no
    friction or travel change; ``effort``: ResiP's; and back."""
    cfg = cfg_module.OneLegFBAbsEnvCfg()
    hand = cfg.scene.robot.actuators["panda_hand"]
    assert cfg.furniture.gripper_mode == "position" and hand.stiffness == 600.0 and hand.friction is None
    assert cfg.actions.fb.gripper.close_command_expr == {"panda_finger_joint.*": 0.0} and cfg.events.finger_limits is None
    cfg.furniture.gripper_mode, cfg.furniture.gripper_effort = "effort", 10.0
    with pytest.raises(ValueError, match="robot_self_collisions"):
        cfg.apply_gripper()  # effort needs the robot's self-collisions off
    cfg.furniture.robot_self_collisions = False
    cfg.apply_gripper()
    assert (hand.stiffness, hand.damping, hand.friction, hand.dynamic_friction) == (0.0, 0.0, 8.0, 8.0)
    assert cfg.actions.fb.gripper.close_command_expr == {"panda_finger_joint.*": -10.0}
    assert cfg.events.finger_limits.params["upper"] == pytest.approx(frames.MAX_GRIPPER_WIDTH / 2)
    cfg.furniture.gripper_mode = "position"
    cfg.apply_gripper()
    assert hand.stiffness == 600.0 and hand.friction is None and cfg.events.finger_limits is None
    assert cfg.actions.fb.gripper.open_command_expr == {"panda_finger_joint.*": pytest.approx(0.0325)}
    cfg.furniture.gripper_mode = "torque"
    with pytest.raises(ValueError, match="gripper_mode"):
        cfg.apply_gripper()


def test_part_assets_switch_the_usds_the_mesh_frame_and_the_nominal_poses(cfg_module):
    from furniture_lab.tasks.furniture import catalog

    cfg = cfg_module.OneLegFBAbsResiPEnvCfg()
    assert cfg.furniture.part_assets == "furniturebench" and cfg.furniture.mesh_frame == "obj"  # the default
    for part in st.PARTS:
        entry = getattr(cfg.scene, part.name)
        assert entry.spawn.usd_path.endswith(f"assets/FurnitureBench/{part.name}.usd") and os.path.isfile(entry.spawn.usd_path)
        pos, quat = st.reset_pose_world(part, "obj")
        assert tuple(entry.init_state.pos) == pytest.approx(pos) and tuple(entry.init_state.rot) == pytest.approx(quat)
    cfg.furniture.part_assets = "furniturebench_top1024"  # the top's SDF at 1024, the rest the same
    cfg.apply_assets()
    assert cfg.furniture.mesh_frame == "obj" and cfg.scene.square_table_top.spawn.usd_path.endswith("square_table_top_sdf1024.usd")
    assert cfg.scene.square_table_leg4.spawn.usd_path.endswith("assets/FurnitureBench/square_table_leg4.usd")
    assert os.path.isfile(cfg.scene.square_table_top.spawn.usd_path)
    cfg.furniture.part_assets = "uwlab"  # the earlier files, frame and poses
    cfg.apply_assets()
    assert cfg.furniture.mesh_frame == "uwlab" and "SquareLeg" in cfg.scene.square_table_leg4.spawn.usd_path
    for part in st.PARTS:
        entry = getattr(cfg.scene, part.name)
        assert entry.spawn.usd_path == catalog.usd_path(part.name, "uwlab")
        assert tuple(entry.init_state.rot) == pytest.approx(st.reset_pose_world(part, "uwlab")[1])
    cfg.furniture.part_assets = "obj"
    with pytest.raises(ValueError, match="part_assets"):
        cfg.apply_assets()


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
        assert f.gripper_velocity_limit == 0.057 and cfg.scene.robot.actuators["panda_hand"].velocity_limit_sim == 0.057
        # ResiP's gripper: effort mode, +-13 N per finger, travel 0.0325, their finger speed, self-collisions off
        hand = cfg.scene.robot.actuators["panda_hand"]
        assert f.gripper_mode == "effort" and (hand.stiffness, hand.damping) == (0.0, 0.0)
        assert (hand.friction, hand.dynamic_friction) == (8.0, 8.0) and f.gripper_effort == 13.0
        assert f.robot_self_collisions is False
        grip = cfg.actions.fb.gripper if hasattr(cfg.actions, "fb") else cfg.actions.gripper
        assert grip.class_type.__name__ == "BinaryJointEffortAction"
        assert grip.open_command_expr == {"panda_finger_joint.*": 13.0} and grip.close_command_expr == {"panda_finger_joint.*": -13.0}
        lim = cfg.events.finger_limits
        assert lim.mode == "startup" and lim.params["upper"] == 0.0325
        assert lim.params["asset_cfg"]._args == ("robot",) and lim.params["asset_cfg"].joint_names == "panda_finger_joint.*"
        arm = cfg.actions.fb.arm if hasattr(cfg.actions, "fb") else cfg.actions.arm
        assert arm.controller.use_relative_mode is (not absolute) and arm.controller.ik_method == "pinv"
        assert arm.class_type.__name__ == "TcpDifferentialIKAction"  # the TCP Jacobian fixed
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


def test_ik_jacobian_switches_the_arm_term(cfg_module):
    """furniture.ik_jacobian: tcp (default) = TcpDifferentialIKAction, isaaclab = Isaac Lab's own term,
    on every ResiP interface; anything else fails; the OSC cfgs are left alone."""
    for name in ("OneLegResiPEnvCfg", "OneLegFBResiPEnvCfg", "OneLegFBAbsResiPEnvCfg"):
        cfg = getattr(cfg_module, name)()
        arm = cfg.actions.fb.arm if hasattr(cfg.actions, "fb") else cfg.actions.arm
        assert cfg.furniture.ik_jacobian == "tcp" and arm.class_type.__name__ == "TcpDifferentialIKAction"
        cfg.furniture.ik_jacobian = "isaaclab"
        cfg.apply_ik()
        assert arm.class_type.__name__ == "DifferentialInverseKinematicsAction"
        cfg.furniture.ik_jacobian = "tcp"
        cfg.apply_ik()
        assert arm.class_type.__name__ == "TcpDifferentialIKAction"
        cfg.furniture.ik_jacobian = "TCP"
        with pytest.raises(ValueError, match="ik_jacobian"):
            cfg.apply_ik()
    port = cfg_module.OneLegFBAbsEnvCfg()
    arm = port.actions.fb.arm
    before = getattr(arm, "class_type", None)
    port.furniture.ik_jacobian = "isaaclab"
    port.apply_ik()
    assert port.actions.fb.arm is arm and getattr(arm, "class_type", None) is before


def test_the_hybrid_cfg_is_resip_with_the_arm_on_joint_torques(cfg_module):
    """``OneLegFBAbsResiPHybridEnvCfg``: ResiP's sim and interface, the arm's drives with no
    stiffness or damping, one hybrid torque term (no IK arm term beside it) and the gripper."""
    cfg = cfg_module.OneLegFBAbsResiPHybridEnvCfg()
    assert cfg.decimation == 12 and cfg.episode_length_s == 70.0
    for group in ("panda_shoulder", "panda_forearm"):
        assert cfg.scene.robot.actuators[group].stiffness == 0.0
        assert cfg.scene.robot.actuators[group].damping == 0.0
    # the fingers keep ResiP's effort-mode gripper (no drive gains, +-13 N), not the arm's torque law
    assert cfg.furniture.gripper_mode == "effort" and cfg.scene.robot.actuators["panda_hand"].friction == 8.0
    assert not hasattr(cfg.actions, "arm") and not hasattr(cfg.actions, "fb")
    term = cfg.actions.hybrid
    assert term.class_type.__name__ == "HybridTorqueAbsAction"
    assert term.body_name == "panda_hand" and term.joint_names == ["panda_joint.*"]
    assert term.gripper is not None and term.gripper.class_type.__name__ == "BinaryJointEffortAction"
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
