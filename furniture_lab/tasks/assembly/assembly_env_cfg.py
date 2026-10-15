"""FurnitureBench's square-table assembly as a manager-based Isaac Lab env.

One scene and MDP for the family; the task picks the pairs to assemble::

    OneLegEnvCfg       should_be_assembled = [(0, 4)]              reward 1.0
    SquareTableEnvCfg  should_be_assembled = [(0, 4), (0, 3), (0, 1), (0, 2)]   reward 4.0

What is upstream and what is this port's choice:

* **Scene**: upstream's layout to the millimetre (:mod:`..furniture.frames`), a Panda at the
  origin, all five parts at their from-scratch reset poses, the table and the three
  obstacles as boxes. Part USDs are the repo's ``assets/`` (``furniture.catalog`` maps part
  to file and names the file's mesh frame, ``square_table.MESH_FRAMES``).
* **Physics**: upstream's Isaac Gym ``sim_params`` (TGS solver, bounce 0.02,
  friction offset 0.01 / correlation 0.0005, contact offset 2 mm), friction 0.15 on parts
  and 0.10 on the table. Upstream steps at dt 1/60 with 2 substeps and 3 steps per action,
  which is a 1/120 physics step and a **20 Hz** policy (despite ``config["robot"]["hz"] =
  10``: ``sim_steps`` divides by ``substeps``). So: ``sim.dt = 1/120``, ``decimation = 6``.
* **Control**: upstream's OSC (``kp = [90, 90, 90, 70, 60, 80]``, critical damping,
  nullspace to the reset joints with gain 10) is Isaac Lab's ``OperationalSpaceController``
  with the same gains. The action is a **delta pose + gripper**: upstream clamps a 7-D delta
  (position + quaternion) to [-1, 1] and adds it to the EE pose in metres; here it is 6-D
  (position + axis-angle) scaled by ``position_scale`` / ``orientation_scale`` -- those two
  scales are this port's, not upstream's. Gripper: ``< 0`` closes (Isaac Lab's binary
  action; upstream's sign is the opposite).
* **Observations**: upstream's state observation (EE pose and velocity, gripper width, part
  poses) in the robot-base frame, ``wxyz`` quaternions, plus the last action.
* **Reward / termination**: upstream's sparse assembled reward and success; ``time_out``
  at ``episode_length_s`` (upstream's 3000-step cap is 150 s; 30 s is this port's default)
  and a ``part_out_of_limits`` termination (FurnitureBench's real-robot rule; FurnitureSim lacks it).
* **Reset**: ``low`` randomness by default (+-1.5 cm, +-15 deg); ``medium`` is a param change
  (``randomize_parts_pose``: 0.05, 45). ``high`` (a table of fixed poses) is not ported.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import List, Tuple

import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKControllerCfg, OperationalSpaceControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
    OperationalSpaceControllerActionCfg,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from ..furniture import assets, frames
from ..furniture.square_table import (
    ASSEMBLED_POS_THRESHOLD,
    ORI_BOUND,
    PART_FRICTION,
    PART_NAMES,
    PARTS,
    TABLE_FRICTION,
    mesh_frame,
)
from . import mdp

#: FurnitureBench's ``k_ee_link``: ``panda_hand`` + 0.1034 m along its z
EE_OFFSET = (0.0, 0.0, 0.1034)
EE_BODY = SceneEntityCfg("robot", body_names="panda_hand")


# ------------------------------------------------------------------------------- scene
@configclass
class AssemblySceneCfg(InteractiveSceneCfg):
    """Robot at the origin, the work table, three obstacles, five parts."""

    ground = assets.ground_cfg()
    light = assets.light_cfg()

    # a mid grey, so the white parts read against it
    table = assets.fixture_cfg("Table", frames.TABLE_SIZE, frames.TABLE_POS, TABLE_FRICTION, color=(0.08, 0.08, 0.08))
    obstacle_front = assets.fixture_cfg(
        "ObstacleFront", frames.OBSTACLE_FRONT_SIZE, frames.OBSTACLE_FRONT_POS, color=(0.3, 0.3, 0.3)
    )
    obstacle_left = assets.fixture_cfg(
        "ObstacleLeft", frames.OBSTACLE_SIDE_SIZE, frames.OBSTACLE_SIDE_POS["obstacle_left"], color=(0.3, 0.3, 0.3)
    )
    obstacle_right = assets.fixture_cfg(
        "ObstacleRight", frames.OBSTACLE_SIDE_SIZE, frames.OBSTACLE_SIDE_POS["obstacle_right"], color=(0.3, 0.3, 0.3)
    )

    robot = assets.FRANKA_OSC_CFG

    # the parts, attribute names == PART_NAMES == upstream's part index order
    square_table_top = assets.part_cfg(PARTS[0])
    square_table_leg1 = assets.part_cfg(PARTS[1])
    square_table_leg2 = assets.part_cfg(PARTS[2])
    square_table_leg3 = assets.part_cfg(PARTS[3])
    square_table_leg4 = assets.part_cfg(PARTS[4])


# ------------------------------------------------------------------------------- mdp
@configclass
class FurnitureCfg:
    """What the task-state terms read: which pairs count, and upstream's tolerances."""

    task: str = MISSING  # "one_leg" | "square_table"
    part_names: List[str] = PART_NAMES
    #: the mesh frame the part USDs use (from the catalog); the scene's reset orientations
    #: and the assembled targets are both expressed in it
    mesh_frame: str = mesh_frame()
    #: upstream's numbers, per axis of the top's OBJ frame (y = seat axis, 5 mm); the tracker
    #: carries them into ``mesh_frame`` (uwlab: z is the seat axis)
    assembled_pos_threshold: Tuple[float, float, float] = ASSEMBLED_POS_THRESHOLD
    ori_bound: float = ORI_BOUND
    #: ``soft`` (default, what papers report): FurnitureBench's latch, the first single step
    #: the check passes, hand anywhere (the other three fields unused). ``strict``: success
    #: as the demo collector's verdict (``mdp/state.py``) -- a pair counts
    #: once it has tested assembled ``success_hold_steps`` consecutive steps with the fingers
    #: open wider than ``success_release_width`` (m) and the TCP farther than
    #: ``success_clearance`` (m) from the inserted part.
    success_mode: str = "soft"
    success_hold_steps: int = 10
    success_release_width: float = 0.05
    success_clearance: float = 0.08
    #: print a line on every reset (which envs, after how many steps, which termination
    #: fired). For watching; off for training, where it would print every step.
    log_resets: bool = False
    #: friction of the parts and of the table/obstacles (PhysX averages a pair, so part on
    #: table = (part + table) / 2). Defaults are FurnitureSim's: 0.15 and 0.10, i.e. 0.125
    #: between a part and the table -- icy; wood on laminate is 0.3-0.5. Applied in
    #: ``AssemblyEnvCfg.__post_init__`` to the startup events and the sim material.
    part_friction: float = PART_FRICTION
    table_friction: float = TABLE_FRICTION
    #: the finger drives. Isaac Lab's Franka: stiffness 2000 N/m, effort limit 200 N, each
    #: finger its own drive -- the first pad to touch a leg that is held by its thread shoves
    #: it over (~25 N on a 3 cm lever) before the other pad arrives, and the fingers close
    #: through it (fail2.pkl: width 0.044 -> 0.000, leg thrown). Upstream drives the gripper
    #: at 13 N (effort) or 200 N/m (position). A real hand grips at 20-70 N.
    gripper_stiffness: float = 600.0  # N/m: ~9 N squeeze on a 30 mm leg
    gripper_damping: float = 40.0
    gripper_effort_limit: float = 30.0  # N per finger
    #: the FB action terms' gripper hysteresis: the command flips only when the action's sign
    #: changes and its size exceeds this (ResiP / FurnitureRLSimEnv's grasp_margin, 0.019);
    #: 0 = flip on the sign, Isaac Lab's binary term as is
    gripper_margin: float = 0.0
    #: the external EE wrench estimated from the joint torques every physics substep
    #: (``assembly/wrench.py``; ``env.wrench.filtered``). EMA factor (Forge: 0.25);
    #: Gaussian noise std on each joint torque before the projection (N m: one value or
    #: seven) and on the projected wrench (one value, (force, torque), or six);
    #: whether the Coriolis / inertial (M q'') terms are removed from the torques (the
    #: robot carries no gravity in this env); the projection, Moore-Penrose (pinv) or
    #: dynamically consistent (dynamic)
    wrench_smoothing: float = 0.25
    wrench_noise_joint_torque: Tuple[float, ...] = (0.0,)
    wrench_noise: Tuple[float, ...] = (0.0,)
    wrench_coriolis: bool = False
    wrench_inertial: bool = False
    wrench_projection: str = "pinv"


@configclass
class ActionsCfg:
    arm = OperationalSpaceControllerActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        body_offset=OperationalSpaceControllerActionCfg.OffsetCfg(pos=EE_OFFSET),
        controller_cfg=OperationalSpaceControllerCfg(
            target_types=["pose_rel"],
            impedance_mode="fixed",
            inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=False,
            gravity_compensation=False,  # the robot's links have gravity disabled instead
            motion_stiffness_task=(90.0, 90.0, 90.0, 70.0, 60.0, 80.0),
            motion_damping_ratio_task=1.0,
            nullspace_control="position",
            nullspace_stiffness=10.0,
            nullspace_damping_ratio=1.0,
        ),
        nullspace_joint_pos_target="default",
        # this port's action scaling: an action of 1 is 2.5 cm / 0.1 rad of EE target delta
        position_scale=0.025,
        orientation_scale=0.1,
    )
    gripper = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_joint.*"],
        open_command_expr={"panda_finger_joint.*": frames.MAX_GRIPPER_WIDTH / 2},
        close_command_expr={"panda_finger_joint.*": 0.0},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        ee_pos = ObsTerm(func=mdp.ee_pos, params={"asset_cfg": EE_BODY, "offset": EE_OFFSET})
        ee_quat = ObsTerm(func=mdp.ee_quat, params={"asset_cfg": EE_BODY, "offset": EE_OFFSET})
        ee_vel = ObsTerm(func=mdp.ee_vel, params={"asset_cfg": EE_BODY})
        gripper_width = ObsTerm(
            func=mdp.gripper_width,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names="panda_finger_joint.*")},
        )
        parts_poses = ObsTerm(func=mdp.parts_poses)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


def _mass_event(part: str, mass: float) -> EventTerm:
    """Set a part's mass once at startup (UWLab's USDs have no MassAPI for the spawn to edit)."""
    return EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(part),
            "mass_distribution_params": (mass, mass),
            "operation": "abs",
            "recompute_inertia": True,
        },
    )


def _friction_event(part: str, friction: float) -> EventTerm:
    """Set a part's friction once at startup (a USD spawn has no material field)."""
    return EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (friction, friction),
            "dynamic_friction_range": (friction, friction),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "asset_cfg": SceneEntityCfg(part),
        },
    )


@configclass
class EventCfg:
    # -- startup: upstream's measured masses and part friction
    top_mass = _mass_event("square_table_top", PARTS[0].mass)
    leg1_mass = _mass_event("square_table_leg1", PARTS[1].mass)
    leg2_mass = _mass_event("square_table_leg2", PARTS[2].mass)
    leg3_mass = _mass_event("square_table_leg3", PARTS[3].mass)
    leg4_mass = _mass_event("square_table_leg4", PARTS[4].mass)
    top_friction = _friction_event("square_table_top", PART_FRICTION)
    leg1_friction = _friction_event("square_table_leg1", PART_FRICTION)
    leg2_friction = _friction_event("square_table_leg2", PART_FRICTION)
    leg3_friction = _friction_event("square_table_leg3", PART_FRICTION)
    leg4_friction = _friction_event("square_table_leg4", PART_FRICTION)

    # -- reset, in this order: everything to its default, then the parts perturbed, then
    #    the assembly latch cleared
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_parts = EventTerm(
        func=mdp.randomize_parts_pose,
        mode="reset",
        params={"pos_range": 0.015, "yaw_range_deg": 15.0, "max_tries": 20},
    )
    reset_assembly = EventTerm(func=mdp.reset_assembly_tracker, mode="reset")


@configclass
class RewardsCfg:
    assembled = RewTerm(func=mdp.newly_assembled, weight=1.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.all_assembled)
    part_out_of_limits = DoneTerm(func=mdp.part_out_of_limits, params={"margin": 0.05})


# ------------------------------------------------------------------------------- env
@configclass
class AssemblyEnvCfg(ManagerBasedRLEnvCfg):
    scene: AssemblySceneCfg = AssemblySceneCfg(num_envs=64, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    furniture: FurnitureCfg = FurnitureCfg()

    # upstream: dt 1/60 x 2 substeps, 3 sim steps per action -> 1/120 physics, 20 Hz policy
    decimation: int = 6
    episode_length_s: float = 30.0
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=6,
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            bounce_threshold_velocity=0.02,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.0005,
            # the SDF part meshes generate many contacts per pair; UWLab's budgets
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_found_lost_aggregate_pairs_capacity=2**22,
            gpu_total_aggregate_pairs_capacity=2**23,
            gpu_collision_stack_size=2**30,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=TABLE_FRICTION, dynamic_friction=TABLE_FRICTION, restitution=0.0
        ),
    )

    def __post_init__(self):
        self.viewer.eye = (1.5, 0.0, 0.8)
        self.viewer.lookat = (0.4, 0.0, 0.0)
        if self.furniture.task is MISSING:
            raise ValueError("furniture.task is unset: use OneLegEnvCfg or SquareTableEnvCfg")
        self.apply_friction()
        self.apply_gripper()

    def apply_gripper(self) -> None:
        """Push ``furniture.gripper_*`` into the finger actuator (call again after changing them)."""
        act = self.scene.robot.actuators["panda_hand"]
        act.stiffness = float(self.furniture.gripper_stiffness)
        act.damping = float(self.furniture.gripper_damping)
        for name in ("effort_limit", "effort_limit_sim"):
            if hasattr(act, name):
                setattr(act, name, float(self.furniture.gripper_effort_limit))

    def apply_friction(self) -> None:
        """Push ``furniture.part_friction`` / ``table_friction`` into the startup friction
        events and the sim's default material (call again after changing them)."""
        f = float(self.furniture.part_friction)
        for name in PART_NAMES:
            term = getattr(self.events, f"{name.replace('square_table_', '')}_friction", None)
            if term is not None:
                term.params["static_friction_range"] = (f, f)
                term.params["dynamic_friction_range"] = (f, f)
        self.sim.physics_material.static_friction = float(self.furniture.table_friction)
        self.sim.physics_material.dynamic_friction = float(self.furniture.table_friction)


@configclass
class OneLegEnvCfg(AssemblyEnvCfg):
    furniture: FurnitureCfg = FurnitureCfg(task="one_leg")


# ------------------------------------------------------- FurnitureBench's own interface
#: FurnitureBench's DEFAULT_STATE_OBS, in order, with widths: the layout of the policy
#: observation below and of the demo pkls' ``observation.state``
FB_OBS_LAYOUT = (("ee_pos", 3), ("ee_quat", 4), ("ee_pos_vel", 3), ("ee_ori_vel", 3), ("gripper_width", 1), ("parts_poses", 35))
FB_OBS_DIM = sum(w for _, w in FB_OBS_LAYOUT)
FB_ACTION_DIM = 8


@configclass
class FBObservationsCfg:
    """Upstream's state observation, verbatim: ``ee_pos, ee_quat (xyzw), ee_pos_vel,
    ee_ori_vel, gripper_width, parts_poses (AprilTag frame, OBJ mesh frames, xyzw)`` -- 49
    dims, no previous action. What the demos hold and what a policy trained on them sees."""

    @configclass
    class PolicyCfg(ObsGroup):
        ee_pos = ObsTerm(func=mdp.ee_pos, params={"asset_cfg": EE_BODY, "offset": EE_OFFSET})
        ee_quat = ObsTerm(func=mdp.ee_quat_xyzw, params={"asset_cfg": EE_BODY, "offset": EE_OFFSET})
        ee_vel = ObsTerm(func=mdp.ee_vel, params={"asset_cfg": EE_BODY})  # pos_vel then ori_vel
        gripper_width = ObsTerm(
            func=mdp.gripper_width,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names="panda_finger_joint.*")},
        )
        parts_poses = ObsTerm(func=mdp.parts_poses_tag)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class FBActionsCfg:
    """Upstream's 8-d action ``[dpos, dquat xyzw (EE frame), gripper]`` driving the task's own
    OSC and gripper terms (``mdp.FurnitureBenchAction``)."""

    # the task's own terms, from an instance: a configclass turns its class attributes into
    # dataclass fields, so ``ActionsCfg.arm`` does not exist on the class
    fb = mdp.FurnitureBenchActionCfg(asset_name="robot", arm=ActionsCfg().arm, gripper=ActionsCfg().gripper)


@configclass
class OneLegFBEnvCfg(OneLegEnvCfg):
    """``one_leg`` with FurnitureBench's own observation (49) and action (8): the interface the
    scripted demos are recorded in, for policies trained on them."""

    observations: FBObservationsCfg = FBObservationsCfg()
    actions: FBActionsCfg = FBActionsCfg()


# ------------------------------------------- the absolute-pose, rot6d variant (ResiP's)
#: ``-FBAbs-v0``: every orientation as rot6d (``mdp/rot6d.py``); the observation is 61 wide
FBABS_OBS_LAYOUT = (("ee_pos", 3), ("ee_rot6d", 6), ("ee_pos_vel", 3), ("ee_ori_vel", 3), ("gripper_width", 1), ("parts_poses", 45))
FBABS_OBS_DIM = sum(w for _, w in FBABS_OBS_LAYOUT)
#: ``[goal_pos 3 (base frame), goal_rot6d 6, gripper]``
FBABS_ACTION_DIM = 10


def abs_arm_cfg() -> OperationalSpaceControllerActionCfg:
    """The task's OSC term in ``pose_abs`` mode: the action is the target ``[pos, quat wxyz]``
    in the base frame, unit scales (as ``replay_env_cfg.OneLegReplayAbsCfg``)."""
    arm = ActionsCfg().arm
    arm.controller_cfg.target_types = ["pose_abs"]
    arm.position_scale = 1.0
    arm.orientation_scale = 1.0
    return arm


@configclass
class FBAbsObservationsCfg:
    """``FBObservationsCfg`` with the EE and every part's orientation as rot6d."""

    @configclass
    class PolicyCfg(ObsGroup):
        ee_pos = ObsTerm(func=mdp.ee_pos, params={"asset_cfg": EE_BODY, "offset": EE_OFFSET})
        ee_rot6d = ObsTerm(func=mdp.ee_rot6d, params={"asset_cfg": EE_BODY, "offset": EE_OFFSET})
        ee_vel = ObsTerm(func=mdp.ee_vel, params={"asset_cfg": EE_BODY})  # pos_vel then ori_vel
        gripper_width = ObsTerm(
            func=mdp.gripper_width,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names="panda_finger_joint.*")},
        )
        parts_poses = ObsTerm(func=mdp.parts_poses_tag_rot6d)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class FBAbsActionsCfg:
    """The 10-d absolute target ``[goal_pos, goal_rot6d, gripper]`` (``mdp.FurnitureBenchAbsAction``)."""

    fb = mdp.FurnitureBenchAbsActionCfg(asset_name="robot", arm=abs_arm_cfg(), gripper=ActionsCfg().gripper)


@configclass
class OneLegFBAbsEnvCfg(OneLegEnvCfg):
    """``one_leg`` with absolute EE targets and rot6d orientations everywhere (observation 61,
    action 10): ResiP's interface, for policies trained on the demos' ``*_abs`` columns."""

    observations: FBAbsObservationsCfg = FBAbsObservationsCfg()
    actions: FBAbsActionsCfg = FBAbsActionsCfg()


# --------------------------------------------- ResiP's sim (FurnitureRLSimEnv), for the demos
#: what ResiP's BC and RL ran in (``furniture_bench/envs/furniture_rl_sim_env.py`` of their
#: fork): 10 Hz policy (dt 1/60 x 6 there; 1/120 x 12 here, the same 0.1 s with this port's
#: physics step), the arm on differential IK into stiff joint PD (1000 / 200), the finger
#: drive 600 N/m capped at 13 N, friction 0.15 parts / 0.10 table (FurnitureSim's), and the gripper
#: hysteresis. Episodes 70 s (their 700-step rollout cap at 10 Hz).
RESIP_DECIMATION = 12
RESIP_ARM_STIFFNESS = 1000.0
RESIP_ARM_DAMPING = 200.0


def dik_arm_cfg(absolute: bool) -> DifferentialInverseKinematicsActionCfg:
    """ResiP's arm controller as an Isaac Lab term: differential IK (least squares,
    ``pinv``) of FurnitureBench's EE frame to joint position targets. Relative: the 6-d
    ``[dpos, axis-angle]`` in the base frame, scaled like the OSC term (an action of 1 is
    2.5 cm / 0.1 rad); absolute: the 7-d target ``[pos, quat wxyz]`` in the base frame."""
    return DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=EE_OFFSET),
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=not absolute, ik_method="pinv"),
        scale=1.0 if absolute else (0.025, 0.025, 0.025, 0.1, 0.1, 0.1),
    )


def apply_resip(cfg) -> None:
    """Turn a one_leg cfg into ResiP's sim (see above), in place; call from ``__post_init__``."""
    cfg.decimation = RESIP_DECIMATION
    cfg.sim.dt = 1.0 / 120.0
    cfg.sim.render_interval = RESIP_DECIMATION
    cfg.episode_length_s = 70.0
    for group in ("panda_shoulder", "panda_forearm"):
        cfg.scene.robot.actuators[group].stiffness = RESIP_ARM_STIFFNESS
        cfg.scene.robot.actuators[group].damping = RESIP_ARM_DAMPING
    f = cfg.furniture
    f.part_friction, f.table_friction = 0.15, 0.10
    # finger drive capped at ResiP's 13 N; 600 N/m (as the port: a stiffer drive pushed the
    # fingers into the leg), ~9 N squeeze on a 30 mm leg -- the setting the resip demos use
    f.gripper_stiffness, f.gripper_damping, f.gripper_effort_limit = 600.0, 40.0, 13.0
    f.gripper_margin = 0.019
    cfg.apply_friction()
    cfg.apply_gripper()
    fb = getattr(cfg.actions, "fb", None)
    if getattr(cfg.actions, "hybrid", None) is not None:
        pass  # the hybrid torque term drives the arm itself (OneLegFBAbsResiPHybridEnvCfg)
    elif fb is not None:
        fb.arm = dik_arm_cfg(absolute=fb.class_type.__name__ == "FurnitureBenchAbsAction")
    else:
        cfg.actions.arm = dik_arm_cfg(absolute=False)


@configclass
class OneLegResiPEnvCfg(OneLegEnvCfg):
    """``one_leg`` (this port's 7-d action / 56-d observation) in ResiP's sim."""

    def __post_init__(self):
        super().__post_init__()
        apply_resip(self)


@configclass
class OneLegFBResiPEnvCfg(OneLegFBEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_resip(self)


@configclass
class OneLegFBAbsResiPEnvCfg(OneLegFBAbsEnvCfg):
    """ResiP's interface (absolute rot6d targets, rot6d observation) in ResiP's sim."""

    def __post_init__(self):
        super().__post_init__()
        apply_resip(self)


# ------------------------------------- hybrid force/position control on joint torques
@configclass
class FBAbsHybridActionsCfg:
    """``[goal_pos 3, goal_rot6d 6, gripper 1 | selection n | force n]`` driving the arm by joint
    torques (``mdp.HybridTorqueAbsAction``; the controller's fields are the cfg's ``hybrid``)."""

    hybrid = mdp.HybridTorqueAbsActionCfg(asset_name="robot", gripper=ActionsCfg().gripper)


@configclass
class OneLegFBAbsResiPHybridEnvCfg(OneLegFBAbsResiPEnvCfg):
    """ResiP's sim and interface with the arm on robonuke_rl_core's hybrid force/position law
    (``mdp/hybrid.py``) instead of differential IK into stiff joint PD: everything else is
    ``OneLegFBAbsResiPEnvCfg``'s (10 Hz, friction, gripper, observation 61). The arm's drives get
    no stiffness or damping, so the term's joint torques are all that moves it. The controller
    starts at ``mdp.hybrid.RESIP_TRACKING_GAINS`` (inertial decoupling on, K / D tuned so the EE
    tracks targets as ResiP's IK + joint PD does; the force branch with its feedforward). The
    wrench estimate removes the arm's own inertia here (``furniture.wrench_inertial``), which the
    force branch reads; the other cfgs keep it off."""

    actions: FBAbsHybridActionsCfg = FBAbsHybridActionsCfg()
    hybrid: mdp.HybridTorqueCfg = mdp.HybridTorqueCfg(**mdp.hybrid.RESIP_TRACKING_GAINS)

    def __post_init__(self):
        super().__post_init__()
        self.furniture.wrench_inertial = True
        for group in ("panda_shoulder", "panda_forearm"):
            self.scene.robot.actuators[group].stiffness = 0.0
            self.scene.robot.actuators[group].damping = 0.0


@configclass
class SquareTableEnvCfg(AssemblyEnvCfg):
    furniture: FurnitureCfg = FurnitureCfg(task="square_table")
