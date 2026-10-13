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
from isaaclab.controllers import OperationalSpaceControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
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
    table = assets.fixture_cfg("Table", frames.TABLE_SIZE, frames.TABLE_POS, TABLE_FRICTION, color=(0.35, 0.35, 0.35))
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
    #: print a line on every reset (which envs, after how many steps, which termination
    #: fired). For watching; off for training, where it would print every step.
    log_resets: bool = False
    #: friction of the parts and of the table/obstacles (PhysX averages a pair, so part on
    #: table = (part + table) / 2). Defaults are FurnitureSim's: 0.15 and 0.10, i.e. 0.125
    #: between a part and the table -- icy; wood on laminate is 0.3-0.5. Applied in
    #: ``AssemblyEnvCfg.__post_init__`` to the startup events and the sim material.
    part_friction: float = PART_FRICTION
    table_friction: float = TABLE_FRICTION


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


@configclass
class SquareTableEnvCfg(AssemblyEnvCfg):
    furniture: FurnitureCfg = FurnitureCfg(task="square_table")
