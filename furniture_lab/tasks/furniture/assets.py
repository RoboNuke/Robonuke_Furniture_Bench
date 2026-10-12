"""Isaac Lab asset configs for the FurnitureBench scene: parts, fixtures and the robot.

Imports Isaac Lab, so only import this after ``AppLauncher``.

**Part USDs** are the repo's own, under ``assets/`` (UWLab's files and layout); which folder
serves which part is a row of :mod:`.catalog`, and that row also names the mesh frame the
file uses (``square_table.MESH_FRAMES``). Nothing is read from the environment.

The robot is Isaac Lab's Panda with the arm joints switched to pure effort control (zero PD
in the implicit actuators) so the operational-space action term owns the torque, as in
Isaac Lab's own OSC example, and with gravity disabled on its links in place of gravity
compensation. The gripper keeps its PD and is driven by a binary position action.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

from . import catalog, frames
from .square_table import PART_FRICTION, TABLE_FRICTION, PartSpec, reset_pose_world

__all__ = [
    "part_usd_path",
    "part_cfg",
    "fixture_cfg",
    "ground_cfg",
    "light_cfg",
    "FRANKA_OSC_CFG",
]

def part_usd_path(name: str) -> str:
    """The part's USD in the repo's ``assets/`` folder (:mod:`.catalog`)."""
    return catalog.usd_path(name)


def part_cfg(part: PartSpec) -> RigidObjectCfg:
    """A furniture part as a free rigid body at its FurnitureBench reset pose (expressed for
    the frame its USD uses).

    The solver iteration counts and contact offsets mirror FurnitureBench's Isaac Gym
    ``sim_params`` (20 position / 1 velocity iterations, contact offset 2 mm, rest offset
    0). Mass and friction are set by startup events in the env cfg
    (``randomize_rigid_body_mass`` / ``randomize_rigid_body_material`` at fixed values): a
    USD spawn has no physics-material field, and UWLab's files carry no ``MassAPI`` for a
    spawn-time ``mass_props`` to modify.
    """
    pos, quat = reset_pose_world(part)
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + part.name,
        spawn=sim_utils.UsdFileCfg(
            usd_path=part_usd_path(part.name),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                kinematic_enabled=False,
                solver_position_iteration_count=20,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
                enable_gyroscopic_forces=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=quat),
    )


def fixture_cfg(
    prim_name: str, size: tuple, pos: tuple, friction: float = TABLE_FRICTION, color=(0.6, 0.6, 0.6)
) -> AssetBaseCfg:
    """A static box: the work table or one of the three obstacles."""
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/" + prim_name,
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=friction, dynamic_friction=friction, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def ground_cfg() -> AssetBaseCfg:
    """A floor well below the table, so a part that falls off stops somewhere.

    FurnitureBench's world has its ground at z = 0 with the robot base at z = 0.43."""
    return AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.43)),
    )


def light_cfg() -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=2500.0),
    )


# ---------------------------------------------------------------------------------- robot
FRANKA_OSC_CFG: ArticulationCfg = FRANKA_PANDA_CFG.copy()
FRANKA_OSC_CFG.prim_path = "{ENV_REGEX_NS}/Robot"
# gravity off on the links instead of gravity compensation in the controller (Isaac Lab's
# OSC example does the same)
FRANKA_OSC_CFG.spawn.rigid_props.disable_gravity = True
# FurnitureBench's reset joints; fingers at half its 65 mm max gripper width
FRANKA_OSC_CFG.init_state = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        **{f"panda_joint{i + 1}": float(q) for i, q in enumerate(frames.RESET_JOINTS)},
        "panda_finger_joint.*": frames.MAX_GRIPPER_WIDTH / 2,
    },
)
# pure effort control on the arm: the OSC action term writes joint efforts
for _group in ("panda_shoulder", "panda_forearm"):
    FRANKA_OSC_CFG.actuators[_group].stiffness = 0.0
    FRANKA_OSC_CFG.actuators[_group].damping = 0.0
