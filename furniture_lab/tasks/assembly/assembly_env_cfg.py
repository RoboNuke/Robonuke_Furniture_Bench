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
* **Reset**: ``furniture.randomness`` picks the level, by default FurnitureBench's ``low``
  exactly (+-1.5 cm, +-15 deg in the tag frame, collision-free, inside the parts' position limits);
  ``med`` (+-5 cm, +-45 deg) and ``high`` (recorded layouts) too, ResiP's FurnitureRLSimEnv reset as
  ``kick_low`` / ``kick_med`` / ``kick_high``, the port's earlier uniform layout as ``port``
  (``mdp/events.py``). Part masses: ``furniture.part_masses``, by default what FurnitureSim simulates.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import List, Optional, Tuple

import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKControllerCfg, OperationalSpaceControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg, OperationalSpaceControllerActionCfg
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
    ISAAC_GYM_MASS,
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
    # kinematic rigid objects (not static fixtures) so a replay can place them per demo
    obstacle_front = assets.obstacle_cfg("ObstacleFront", frames.OBSTACLE_FRONT_SIZE, frames.OBSTACLE_FRONT_POS)
    obstacle_left = assets.obstacle_cfg("ObstacleLeft", frames.OBSTACLE_SIDE_SIZE, frames.OBSTACLE_SIDE_POS["obstacle_left"])
    obstacle_right = assets.obstacle_cfg("ObstacleRight", frames.OBSTACLE_SIDE_SIZE, frames.OBSTACLE_SIDE_POS["obstacle_right"])

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
    #: m: at every reset the three obstacles move together by one uniform (x, y) offset in
    #: [-r, r] (ResiP's FurnitureRLSimEnv, low randomness: 0.02; ``apply_resip`` sets it).
    #: 0 keeps them at ``frames``' fixed poses. Not used when ``randomness`` is a ResiP level
    obstacle_offset: float = 0.0
    #: the reset distribution, read at every reset (``mdp/events.py``). ``port`` (every dataset
    #: before 2026-10-20): every part uniform +-1.5 cm / +-15 deg yaw, redrawn while footprints
    #: overlap (no margin, no position limits; ``events.reset_parts.params``), the obstacles by
    #: ``obstacle_offset``, the arm at its reset joints. ``low`` (default) / ``med`` / ``high``:
    #: FurnitureBench's own levels exactly (FurnitureSimEnv) -- each part's tag-frame x, y
    #: +-1.5 / +-5 cm and yaw +-15 / +-45 deg, redrawn until collision-free (1 cm margin) and inside
    #: the parts' position limits; ``high`` one of three recorded layouts (``high_layout``); the
    #: arm at its reset joints, the obstacles fixed. ``kick_low`` / ``kick_med`` / ``kick_high``:
    #: ResiP's FurnitureRLSimEnv levels exactly (``mdp.events.RESIP_RANDOMNESS``) -- the parts at
    #: their nominal poses, each kicked once by a random xy force and z torque; every arm joint
    #: +-5 / 10 / 13 deg; the obstacles +-2 / 4 / 6 cm. ``obstacle_offset`` is used by ``port`` only
    randomness: str = "low"
    #: ``randomness: high``: which of FurnitureBench's three recorded layouts (0, 1, 2; their
    #: FurnitureSimEnv's default, ``--high-random-idx``, is 0), or -1: one at random per env at
    #: every reset
    high_layout: int = 0
    success_mode: str = "soft"
    success_hold_steps: int = 10
    success_release_width: float = 0.05
    success_clearance: float = 0.08
    #: print a line on every reset (which envs, after how many steps, which termination
    #: fired). For watching; off for training, where it would print every step.
    log_resets: bool = False
    #: publish the wrench estimate (``env.wrench.filtered``: force N then torque N m, robot-base
    #: frame) every step as ``infos["eval_state"]["ee_wrench"]``: six columns of an eval's
    #: per-step parquet (``assembly_env.py``)
    log_wrench: bool = False
    #: one_leg only: track the task phase per env (``assembly/phases.py``): Corner -> Grasp ->
    #: Insert -> Screw, forward only, then success / failure / timeout; published in
    #: ``infos["eval_state"]`` every step (the parquet, the ``one_leg`` video overlay) and in the
    #: episode metrics at the end. The phase_* fields below are its thresholds:
    track_phases: bool = False
    #: Corner is done once the gripper lets go of the top: it held the top (fingers closed narrower
    #: than ``phase_top_grip_width`` m with the TCP within ``phase_top_grip_dist`` m of the top's
    #: centre and at most ``phase_top_grip_height`` m above it), then opened wider than
    #: ``phase_release_width``. The top's distance from the corner goal is only measured
    phase_top_grip_width: float = 0.02
    phase_top_grip_dist: float = 0.13
    phase_top_grip_height: float = 0.03
    #: m: Grasp is done once leg 4 is held (the TCP within ``phase_held_dist`` of its centre) and
    #: has risen ``phase_lift`` above where it lay at the episode's start
    phase_lift: float = 0.03
    phase_held_dist: float = 0.03
    #: Insert is done once the fingers are open wider than ``phase_release_width`` (m; Corner's let
    #: go as well) and the leg
    #: has stayed in its hole (off-axis <= ``phase_hole_xz`` m, tilt <= ``phase_tilt_deg`` from the
    #: seat axis) for ``phase_hold_steps`` consecutive steps: 1 (default) is the first such step, as
    #: FurnitureBench has no hold; more is opt-in
    phase_release_width: float = 0.05
    phase_hole_xz: float = 0.013
    phase_tilt_deg: float = 16.0
    phase_hold_steps: int = 1
    #: display only: the overlay's force bar is full at this (N)
    phase_force_scale: float = 50.0
    #: no-progress reset (track_phases): end an episode as a failure once leg 4 has moved at most ``stall_pos_mm``
    #: (mm) and turned at most ``stall_rot_deg`` (deg) per step, step to step, for this many consecutive steps of Insert
    #: (``stall_steps_insert``) or Screw (``stall_steps_screw``); 0 is off for that phase. Published every step as
    #: ``eval_state`` ``phase/still`` (1: no progress this step), ``phase/still_steps`` and ``phase/stalled`` (1: the
    #: episode ends for lack of progress), and at the episode's end as ``phase/stalled``
    stall_steps_insert: int = 0
    stall_steps_screw: int = 0
    stall_pos_mm: float = 0.5
    stall_rot_deg: float = 1.0
    #: each part's mass (kg), in ``part_names`` order, set by the startup mass events
    #: (``apply_masses``). Default: what FurnitureSim simulates (``ISAAC_GYM_MASS``: top 411.5 g,
    #: legs 26.4 / 26.5 / 26.4 / 29.1 g); ``PartSpec.mass`` holds the real parts' 151 g / 23.1 g
    #: (every dataset before 2026-10-20)
    part_masses: Tuple[float, ...] = tuple(ISAAC_GYM_MASS[name] for name in PART_NAMES)
    #: friction of the parts and of the table/obstacles (PhysX averages a pair, so part on
    #: table = (part + table) / 2). Defaults are FurnitureSim's: 0.15 and 0.10, i.e. 0.125
    #: between a part and the table -- icy; wood on laminate is 0.3-0.5. Applied in
    #: ``AssemblyEnvCfg.__post_init__`` to the startup events and the sim material.
    #: ``table_friction`` reaches only the sim's default material, i.e. the shapes with no material
    #: of their own (the ground, and the robot unless its USD binds one or ``robot_friction`` is
    #: set); the table box keeps ``TABLE_FRICTION`` and the obstacles ``obstacle_friction``.
    part_friction: float = PART_FRICTION
    table_friction: float = TABLE_FRICTION
    #: friction of the three obstacle boxes. Default 1.0 (since 2026-10-20; ``TABLE_FRICTION`` 0.10
    #: before): FurnitureSim / ResiP's sim sets none on its obstacles, so they keep Isaac Gym's asset
    #: default 1.0 (``RESIP_CONTACT_FRICTION``): a top pushed into the corner grips the walls there
    obstacle_friction: float = 1.0
    #: friction of every robot shape (fingers, hand, links), set once at startup. Default 1.0 (since
    #: 2026-10-20): FurnitureSim / ResiP's sim sets none on the Franka, so its fingers grip at Isaac
    #: Gym's asset default 1.0 (``RESIP_CONTACT_FRICTION``). Negative (-1, the earlier default): not
    #: set -- whatever the robot USD binds, else the sim's default material. (A float, not None: a
    #: config file can only override a field with a value of its type.)
    robot_friction: float = 1.0
    #: the point the observed EE linear velocity (the ``ee_vel`` term: ``ee_pos_vel``) is of.
    #: ``ee`` (default since 2026-10-20): the EE point ``ee_pos`` is at (``EE_OFFSET`` on panda_hand,
    #: FurnitureBench's ``k_ee_link``), what FurnitureBench's / ResiP's sim observes; ``hand_com``
    #: (every dataset before): panda_hand's centre of mass. Applied by ``apply_observations``
    ee_vel_point: str = "ee"
    #: the finger drives. Isaac Lab's Franka: stiffness 2000 N/m, effort limit 200 N, each
    #: finger its own drive -- the first pad to touch a leg that is held by its thread shoves
    #: it over (~25 N on a 3 cm lever) before the other pad arrives, and the fingers close
    #: through it (fail2.pkl: width 0.044 -> 0.000, leg thrown). Upstream drives the gripper
    #: at 13 N (effort) or 200 N/m (position). A real hand grips at 20-70 N.
    gripper_stiffness: float = 600.0  # N/m: ~9 N squeeze on a 30 mm leg
    gripper_damping: float = 40.0
    gripper_effort_limit: float = 30.0  # N per finger
    #: max finger joint speed (m/s per finger; PhysX max joint velocity), None = the asset's.
    #: (ResiP's effort-mode fingers -- 13 N against DOF friction 15 -- close at ~0.057 m/s)
    gripper_velocity_limit: Optional[float] = None
    #: ``position``: the finger drive is the PD above (``gripper_stiffness`` / ``gripper_damping``,
    #: capped at ``gripper_effort_limit``) on the binary open / close target. ``effort`` (what
    #: ``apply_resip`` sets): ResiP's FurnitureRLSimEnv gripper (``mdp/gripper.py``) -- no drive gains,
    #: a constant ``+gripper_effort`` N on each finger while open and ``-gripper_effort`` while closed,
    #: the finger joints' friction coefficient ``gripper_joint_friction`` (static and dynamic), the
    #: travel capped at half ``frames.MAX_GRIPPER_WIDTH`` (65 mm open); ``gripper_velocity_limit``
    #: still applies. Needs ``robot_self_collisions`` off (see there).
    gripper_mode: str = "position"
    gripper_effort: float = 13.0
    #: effort mode's finger joint friction (PhysX: resists up to coefficient x the force the joint
    #: transmits). ResiP's Isaac Gym sets 15; under this PhysX (5) 15 freezes a free finger against the
    #: 13 N (0.55 mm per step) and anything up to 12 does not slow it at all, so their finger speed
    #: comes from ``gripper_velocity_limit`` (``RESIP_FINGER_SPEED``) and this keeps a friction that
    #: resists the leg pushing a closed finger open (8 x the contact load) while a free finger moves
    gripper_joint_friction: float = 8.0
    #: the part USDs (``catalog.ASSET_SETS``). ``furniturebench`` (default since 2026-10-20):
    #: FurnitureBench's own OBJs converted as ResiP's URDFs give them to Isaac Gym
    #: (``assets/FurnitureBench``, ``tools/convert_assets.py``): one collision mesh per part, an SDF at
    #: 512 on the top and leg 4, convex hulls on legs 1-3 (with ResiP's checkpoint: insertion failures
    #: 40% -> 4%, the released leg's lean 10 -> 4.5 deg). ``uwlab`` (every dataset before): the repo's
    #: UWLab files -- each part split into convex hulls (the top's plate and walls, the leg's body)
    #: and SDF meshes at PhysX's default resolution on the hole blocks and the bolt.
    #: ``furniturebench_top1024``: ``furniturebench`` with the top's SDF at 1024 (0.16 mm cells; 512
    #: gives 0.32 mm, about the thread's 0.38 mm crest-to-root clearance). ``apply_assets`` sets the
    #: USDs, ``mesh_frame`` and the nominal poses
    part_assets: str = "furniturebench"
    #: the robot's self-collisions (Isaac Lab's Franka: on). Isaac Lab's Franka USD has finger 2 touch
    #: another of the robot's links once it opens past ~24.7 mm: a position drive pushes through, a
    #: 13 N effort stalls there (the gripper opens to 57 mm, not 65), so effort mode needs them off
    #: (``apply_resip`` turns them off; ResiP's own Franka opens to 65 mm)
    robot_self_collisions: bool = True
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
    #: dynamically consistent (dynamic), solved on the device with ``wrench_damping`` added as
    #: damping^2 I to its 6 x 6 system (0: the exact pseudo-inverse; a small value keeps it
    #: bounded at a singular arm pose)
    wrench_smoothing: float = 0.25
    wrench_noise_joint_torque: Tuple[float, ...] = (0.0,)
    wrench_noise: Tuple[float, ...] = (0.0,)
    wrench_coriolis: bool = False
    wrench_inertial: bool = False
    wrench_projection: str = "pinv"
    wrench_damping: float = 1.0e-4


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
    # the port's binary position term (mdp/gripper.py): Isaac Lab's with the target written through a device
    # index instead of a Python list, which would synchronize with the host every physics substep
    gripper = mdp.BinaryJointPositionActionCfg(
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
    # every robot shape's friction: set by apply_friction from furniture.robot_friction (< 0: off)
    robot_friction: Optional[EventTerm] = None
    # the finger travel capped at ResiP's 0.0325 m: set by apply_gripper in effort mode (else off)
    finger_limits: Optional[EventTerm] = None

    # -- reset, in this order: everything to its default, then the obstacles offset and the
    #    parts perturbed, then
    #    the assembly latch cleared
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    # the three obstacles moved together by up to furniture.obstacle_offset (0: not at all)
    reset_obstacles = EventTerm(func=mdp.randomize_obstacles, mode="reset")
    reset_parts = EventTerm(
        func=mdp.randomize_parts_pose,
        mode="reset",
        params={"pos_range": 0.015, "yaw_range_deg": 15.0, "max_tries": 20},
    )
    reset_assembly = EventTerm(func=mdp.reset_assembly_tracker, mode="reset")


@configclass
class RewardsCfg:
    assembled = RewTerm(func=mdp.newly_assembled, weight=1.0)
    # a constant per-step term, off at weight 0 (the reward manager skips a zero-weight term);
    # task.cfg.rewards.step.weight: -1.0 is Policy Decorator's -1 per step (the manager scales every
    # term by step_dt; robonuke_rl_core divides it back out)
    step = RewTerm(func=mdp.each_step, weight=0.0)
    # one_leg's phase rewards, each off at weight 0: weight w on the step the episode gets past that
    # phase (furniture.track_phases' criteria, assembly/phases.py; it must be on), once per episode
    # -- corner: the top pushed into the corner and let go, grasp: leg 4 held and lifted, insert:
    # the leg left standing in its hole. Success is ``assembled``
    corner_done = RewTerm(func=mdp.phase_done, weight=0.0, params={"phase": "corner"})
    grasp_done = RewTerm(func=mdp.phase_done, weight=0.0, params={"phase": "grasp"})
    insert_done = RewTerm(func=mdp.phase_done, weight=0.0, params={"phase": "insert"})


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
        self.apply_assets()
        self.apply_friction()
        self.apply_gripper()
        self.apply_observations()
        self.apply_masses()
        mdp.randomness(self.furniture)  # an unknown furniture.randomness fails here
        if self.furniture.high_layout not in (-1, 0, 1, 2):
            raise ValueError(f"furniture.high_layout must be 0, 1, 2 or -1 (random), got {self.furniture.high_layout}")

    def apply_assets(self) -> None:
        """Push ``furniture.part_assets`` into the parts' USDs, ``furniture.mesh_frame`` and the
        parts' nominal (default) poses, which are expressed in the USDs' mesh frame (call again after
        changing it)."""
        from ..furniture import catalog
        from ..furniture.square_table import reset_pose_world

        assets = self.furniture.part_assets
        if assets not in catalog.ASSET_SETS:
            raise ValueError(f"furniture.part_assets must be one of {sorted(catalog.ASSET_SETS)}, got {assets!r}")
        specs = {p.name: p for p in PARTS}
        frames_used = {catalog.mesh_frame_of(name, assets) for name in self.furniture.part_names}
        if len(frames_used) != 1:
            raise ValueError(f"furniture.part_assets {assets!r}: the parts use different mesh frames {sorted(frames_used)}")
        frame = frames_used.pop()
        self.furniture.mesh_frame = frame
        for name in self.furniture.part_names:
            part = getattr(self.scene, name)
            part.spawn.usd_path = catalog.usd_path(name, assets)
            pos, quat = reset_pose_world(specs[name], frame)
            part.init_state.pos, part.init_state.rot = pos, quat

    def apply_masses(self) -> None:
        """Push ``furniture.part_masses`` into the startup mass events (call again after changing it)."""
        names, masses = list(self.furniture.part_names), tuple(float(m) for m in self.furniture.part_masses)
        if len(masses) != len(names) or min(masses) <= 0.0:
            raise ValueError(f"furniture.part_masses needs one positive mass per part {names}, got {masses}")
        for name, mass in zip(names, masses):
            term = getattr(self.events, f"{name.replace('square_table_', '')}_mass", None)
            if term is None:
                raise ValueError(f"no startup mass event for {name} (events.{name.replace('square_table_', '')}_mass)")
            term.params["mass_distribution_params"] = (mass, mass)

    def apply_observations(self) -> None:
        """Push ``furniture.ee_vel_point`` into every observation group's ``ee_vel`` term (call
        again after changing it)."""
        point = self.furniture.ee_vel_point
        if point not in ("hand_com", "ee"):
            raise ValueError(f"furniture.ee_vel_point must be 'hand_com' or 'ee', got {point!r}")
        for group in vars(self.observations).values():
            term = getattr(group, "ee_vel", None)
            if term is None or not hasattr(term, "params"):
                continue
            if point == "ee":
                term.params["offset"] = EE_OFFSET
            else:
                term.params.pop("offset", None)

    def apply_gripper(self) -> None:
        """Push ``furniture.gripper_*`` into the finger actuator, the gripper action term(s) and the
        finger travel (call again after changing them)."""
        f = self.furniture
        if f.gripper_mode not in ("position", "effort"):
            raise ValueError(f"furniture.gripper_mode must be 'position' or 'effort', got {f.gripper_mode!r}")
        effort = f.gripper_mode == "effort"
        if effort and f.robot_self_collisions:
            raise ValueError(
                "furniture.gripper_mode 'effort' needs furniture.robot_self_collisions false: with them on, "
                "Isaac Lab's Franka finger 2 stalls at ~24.7 mm against another link under a 13 N effort"
            )
        props = getattr(self.scene.robot.spawn, "articulation_props", None)
        if props is not None:
            props.enabled_self_collisions = bool(f.robot_self_collisions)
        act = self.scene.robot.actuators["panda_hand"]
        act.stiffness = 0.0 if effort else float(f.gripper_stiffness)
        act.damping = 0.0 if effort else float(f.gripper_damping)
        for name in ("effort_limit", "effort_limit_sim"):
            if hasattr(act, name):
                setattr(act, name, float(f.gripper_effort if effort else f.gripper_effort_limit))
        # the joint friction: ResiP's in effort mode; the asset's (None) otherwise, as always
        act.friction = float(f.gripper_joint_friction) if effort else None
        act.dynamic_friction = float(f.gripper_joint_friction) if effort else None
        if f.gripper_velocity_limit is not None:
            act.velocity_limit_sim = float(f.gripper_velocity_limit)
        # the gripper term inside each action group that has one: the effort term (+-gripper_effort),
        # or back to the task's binary position term when an effort one is there
        for group in (self.actions, getattr(self.actions, "fb", None), getattr(self.actions, "hybrid", None)):
            term = getattr(group, "gripper", None) if group is not None else None
            if term is None:
                continue
            is_effort = getattr(term, "class_type", None) is mdp.BinaryJointEffortAction
            if effort:
                group.gripper = mdp.BinaryJointEffortActionCfg(
                    asset_name="robot",
                    joint_names=["panda_finger_joint.*"],
                    open_command_expr={"panda_finger_joint.*": float(f.gripper_effort)},
                    close_command_expr={"panda_finger_joint.*": -float(f.gripper_effort)},
                )
            elif is_effort:
                group.gripper = ActionsCfg().gripper
        self.events.finger_limits = (
            EventTerm(
                func=mdp.set_finger_limits,
                mode="startup",
                params={"upper": frames.MAX_GRIPPER_WIDTH / 2, "asset_cfg": SceneEntityCfg("robot", joint_names="panda_finger_joint.*")},
            )
            if effort
            else None
        )

    def apply_friction(self) -> None:
        """Push ``furniture.part_friction`` / ``table_friction`` into the startup friction
        events and the sim's default material, ``obstacle_friction`` into the obstacle boxes and
        ``robot_friction`` into a startup event on the robot (call again after changing them)."""
        f = float(self.furniture.part_friction)
        for name in PART_NAMES:
            term = getattr(self.events, f"{name.replace('square_table_', '')}_friction", None)
            if term is not None:
                term.params["static_friction_range"] = (f, f)
                term.params["dynamic_friction_range"] = (f, f)
        self.sim.physics_material.static_friction = float(self.furniture.table_friction)
        self.sim.physics_material.dynamic_friction = float(self.furniture.table_friction)
        for name in ("obstacle_front", "obstacle_left", "obstacle_right"):
            obstacle = getattr(self.scene, name, None)
            if obstacle is not None:
                obstacle.spawn.physics_material.static_friction = float(self.furniture.obstacle_friction)
                obstacle.spawn.physics_material.dynamic_friction = float(self.furniture.obstacle_friction)
        r = float(self.furniture.robot_friction)
        self.events.robot_friction = None if r < 0 else _friction_event("robot", r)


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
#: the ResiP ``-FBAbs-ResiP-v0`` observation, every pose in the robot-base frame: the robot state
#: of ``FBABS_OBS_LAYOUT``, the parts in the base frame (``parts_poses_b``: the same 5 x ``[pos,
#: rot6d]``, not the tag frame), then the front obstacle's base-frame x, y (it slides per episode;
#: its z and rotation never change), 63
FBABS_RESIP_OBS_LAYOUT = FBABS_OBS_LAYOUT[:-1] + (("parts_poses_b", 45), ("obstacle_xy", 2))
#: the ``-FBRel-ResiP-v0`` observation, every pose relative to the EE (the EE pose itself is not
#: observed): the EE's linear and angular velocity in its own frame, the finger separation, each
#: part ``[R_ee^T (p - p_ee), rot6d(R_ee^T R)]``, the front obstacle's point (x, y, 0) in the EE
#: frame, 55 (``demos.ee_relative_states`` builds it from the 63-wide ResiP observation)
FBREL_RESIP_OBS_LAYOUT = (
    ("ee_pos_vel", 3), ("ee_ori_vel", 3), ("gripper_width", 1), ("parts_poses_ee", 45), ("obstacle_pos_ee", 3)
)
#: ``[dpos 3, drot6d 6, gripper]``, relative to the current EE pose in its own frame
FBREL_ACTION_DIM = 10
#: the ``-FBDelta-ResiP-v0`` action: ``[dpos 3, drot 3 (axis-angle), gripper]``, relative to the
#: current EE pose in its own frame (its observation is the FBAbs ResiP one, ``FBABS_RESIP_OBS_LAYOUT``)
FBDELTA_ACTION_DIM = 7


def fbabs_obs_layout(env_cfg) -> tuple:
    """The ``-FBAbs-`` observation layout of ``env_cfg``: with the obstacle term when its policy
    group has one (the ResiP task), else ``FBABS_OBS_LAYOUT``."""
    policy = getattr(getattr(env_cfg, "observations", None), "policy", None)
    return FBABS_RESIP_OBS_LAYOUT if getattr(policy, "obstacle_xy", None) is not None else FBABS_OBS_LAYOUT


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
class FBAbsResiPObservationsCfg:
    """``FBAbsObservationsCfg`` with every pose in the robot-base frame -- the parts in it too, not
    the tag frame -- plus the front obstacle's base-frame x, y (``FBABS_RESIP_OBS_LAYOUT``, 63)."""

    @configclass
    class PolicyCfg(FBAbsObservationsCfg.PolicyCfg):
        parts_poses = ObsTerm(func=mdp.parts_poses_b_rot6d)
        obstacle_xy = ObsTerm(func=mdp.obstacle_xy_b)

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
#: drive 600 N/m capped at 13 N (speed limit 0.2 m/s), the obstacles moved +-2 cm per reset, friction 0.15 parts / 0.10 table (FurnitureSim's), and the gripper
#: hysteresis. Episodes 70 s (their 700-step rollout cap at 10 Hz).
RESIP_DECIMATION = 12
RESIP_ARM_STIFFNESS = 1000.0
RESIP_ARM_DAMPING = 200.0
#: ResiP's obstacle randomization at low randomness (m, uniform +- in x and y, all three together)
RESIP_OBSTACLE_OFFSET = 0.02
#: the finger speed limit (m/s per finger) of the port's earlier position-mode gripper
RESIP_GRIPPER_SPEED = 0.2
#: ResiP's fingers' speed (m/s each), measured in their demos: 11.3 mm of finger separation per 0.1 s
#: step opening and closing (their 13 N effort against their joint friction); effort mode's cap
RESIP_FINGER_SPEED = 0.057
#: the position gripper's finger speed cap (m/s per finger), apply_resip's default
POSITION_FINGER_SPEED = 0.2
#: the friction of every shape ResiP's sim sets none on -- the obstacles and the whole Franka,
#: fingers included: Isaac Gym's asset default (their URDFs loaded and read back, 1.0 on every
#: shape). Not applied by ``apply_resip``; ``furniture.obstacle_friction`` / ``robot_friction`` opt in
RESIP_CONTACT_FRICTION = 1.0


#: the differential IK's damping (``ik_params["lambda_val"]``): a least-squares step ``J^T (J J^T + lambda^2 I)^-1 dx``
#: solved on the device every physics substep (``mdp/tcp_ik.py``). Against the pseudo-inverse (``pinv``, what the
#: datasets and runs before 2026-11-10 used) it scales the step along a singular direction sigma by
#: sigma^2 / (sigma^2 + lambda^2): below 1e-4 relative for sigma >= 0.01, i.e. the same controller, and bounded
#: instead of unbounded at a singular pose
RESIP_IK_DAMPING = 1.0e-4


def dik_arm_cfg(absolute: bool) -> DifferentialInverseKinematicsActionCfg:
    """ResiP's arm controller as an Isaac Lab term: differential IK (least squares, ``dls`` with
    ``RESIP_IK_DAMPING``) of FurnitureBench's EE frame to joint position targets, with the TCP
    Jacobian fixed and the solve kept on the device (``mdp.tcp_ik.TcpDifferentialIKAction``).
    Relative: the 6-d ``[dpos, axis-angle]`` in the base frame, scaled like the OSC term (an
    action of 1 is 2.5 cm / 0.1 rad); absolute: the 7-d target ``[pos, quat wxyz]`` in the base
    frame."""
    from .mdp.tcp_ik import TcpDifferentialIKAction

    cfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=EE_OFFSET),
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=not absolute, ik_method="dls", ik_params={"lambda_val": RESIP_IK_DAMPING}
        ),
        scale=1.0 if absolute else (0.025, 0.025, 0.025, 0.1, 0.1, 0.1),
    )
    # Isaac Lab's term puts the TCP offset's lever in the hand frame on a root-frame Jacobian;
    # this one rotates it into the root frame (the Jacobian ResiP's k_ee_link gives)
    cfg.class_type = TcpDifferentialIKAction
    return cfg


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
    # the gripper every successful dataset and run used: gripper_mode "position", a 600 N/m drive capped at
    # 13 N and a 0.2 m/s cap, which closes and opens in about one step, with the robot's self-collisions on.
    # ResiP's own gripper is opt-in: gripper_mode "effort" (+-13 N per finger, the finger travel capped at
    # 0.0325, mdp/gripper.py), robot_self_collisions false (see FurnitureCfg.robot_self_collisions) and
    # gripper_velocity_limit RESIP_FINGER_SPEED, their fingers' measured speed
    f.gripper_mode, f.gripper_effort, f.gripper_joint_friction = "position", 13.0, 8.0
    f.robot_self_collisions = True
    f.gripper_stiffness, f.gripper_damping, f.gripper_effort_limit = 600.0, 40.0, 13.0
    f.gripper_velocity_limit = POSITION_FINGER_SPEED
    f.gripper_margin = 0.019
    f.obstacle_offset = RESIP_OBSTACLE_OFFSET
    cfg.apply_friction()
    cfg.apply_gripper()
    fb = getattr(cfg.actions, "fb", None)
    if getattr(cfg.actions, "hybrid", None) is not None:
        pass  # the hybrid torque term drives the arm itself (OneLegFBAbsResiPHybridEnvCfg)
    elif fb is not None:
        # the absolute term and its subclasses (the -FBRel- term sends absolute targets too)
        fb.arm = dik_arm_cfg(absolute=issubclass(fb.class_type, mdp.FurnitureBenchAbsAction))
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
    """ResiP's interface (absolute rot6d targets; rot6d observation, every pose in the robot-base
    frame, with the front obstacle's x, y: 63) in ResiP's sim."""

    observations: FBAbsResiPObservationsCfg = FBAbsResiPObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        apply_resip(self)


# ------------------------------------- everything relative to the end-effector (-FBRel-)
@configclass
class FBRelResiPObservationsCfg:
    """The ResiP observation with every pose relative to the EE and no EE pose
    (``FBREL_RESIP_OBS_LAYOUT``, 55)."""

    @configclass
    class PolicyCfg(ObsGroup):
        # named ee_vel so FurnitureCfg.ee_vel_point reaches it (apply_observations)
        ee_vel = ObsTerm(func=mdp.ee_vel_ee, params={"asset_cfg": EE_BODY})
        gripper_width = ObsTerm(
            func=mdp.gripper_width,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names="panda_finger_joint.*")},
        )
        parts_poses = ObsTerm(func=mdp.parts_poses_ee_rot6d, params={"asset_cfg": EE_BODY, "offset": EE_OFFSET})
        obstacle_pos = ObsTerm(func=mdp.obstacle_pos_ee, params={"asset_cfg": EE_BODY, "offset": EE_OFFSET})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class FBRelActionsCfg:
    """``[dpos 3, drot6d 6, gripper]`` relative to the EE, in its frame (``mdp.FurnitureBenchRelAction``)."""

    fb = mdp.FurnitureBenchRelActionCfg(asset_name="robot", arm=abs_arm_cfg(), gripper=ActionsCfg().gripper)


@configclass
class OneLegFBRelResiPEnvCfg(OneLegFBAbsResiPEnvCfg):
    """``OneLegFBAbsResiPEnvCfg`` (ResiP's sim, the same physics, arm controller and gripper) with
    the observation and the action relative to the end-effector: observation 55
    (``FBREL_RESIP_OBS_LAYOUT``), action 10 (``FBREL_ACTION_DIM``). Only the interface differs, so
    switching a config between the two is its task name."""

    observations: FBRelResiPObservationsCfg = FBRelResiPObservationsCfg()
    actions: FBRelActionsCfg = FBRelActionsCfg()


@configclass
class FBDeltaActionsCfg:
    """``[dpos 3, drot 3 (axis-angle), gripper]`` relative to the EE, in its frame (``mdp.FurnitureBenchDeltaAction``)."""

    fb = mdp.FurnitureBenchDeltaActionCfg(asset_name="robot", arm=abs_arm_cfg(), gripper=ActionsCfg().gripper)


@configclass
class OneLegFBDeltaResiPEnvCfg(OneLegFBAbsResiPEnvCfg):
    """``OneLegFBAbsResiPEnvCfg`` (ResiP's sim, the same physics, arm controller, gripper and
    observation, ``FBABS_RESIP_OBS_LAYOUT``) with the action relative to the end-effector: 7
    (``FBDELTA_ACTION_DIM``). Only the action differs, so switching a config between the two is
    its task name."""

    actions: FBDeltaActionsCfg = FBDeltaActionsCfg()


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
    ``OneLegFBAbsResiPEnvCfg``'s (10 Hz, friction, gripper, observation 63). The arm's drives get
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
