"""Isaac Lab asset configs for the FurnitureBench scene: parts, fixtures and the robot.

Imports Isaac Lab, so only import this after ``AppLauncher``.

**Part USDs** come from ``$FURNITURE_ASSETS_DIR``: one ``<part name>.usd`` per part
(``square_table_top.usd``, ``square_table_leg1.usd`` ... ``square_table_leg4.usd``). If a
per-leg file is missing, ``square_table_leg.usd`` or UWLab's ``square_leg.usd`` serves every
leg. Make them with ``tools/convert_assets.py`` (SDF collision, which the thread needs), or
point the variable at a folder holding UWLab's two files and set
``FURNITURE_MESH_FRAME=uwlab`` (their meshes are rotated relative to the OBJs; see
``square_table.MESH_FRAMES``). ``tools/check_usd.py`` shows what a file holds.

The robot is Isaac Lab's Panda with the arm joints switched to pure effort control (zero PD
in the implicit actuators) so the operational-space action term owns the torque, as in
Isaac Lab's own OSC example, and with gravity disabled on its links in place of gravity
compensation. The gripper keeps its PD and is driven by a binary position action.
"""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

from . import frames
from .square_table import PART_FRICTION, TABLE_FRICTION, PartSpec, reset_pose_world

__all__ = [
    "ASSETS_DIR_VAR",
    "assets_dir",
    "part_usd_path",
    "part_cfg",
    "fixture_cfg",
    "ground_cfg",
    "light_cfg",
    "FRANKA_OSC_CFG",
]

ASSETS_DIR_VAR = "FURNITURE_ASSETS_DIR"


def assets_dir() -> Path:
    value = os.environ.get(ASSETS_DIR_VAR)
    if not value:
        raise RuntimeError(
            f"{ASSETS_DIR_VAR} is not set. Point it at the folder holding the part USDs "
            "(see tools/convert_assets.py)."
        )
    return Path(value).expanduser()


#: file names tried for a part, in order: its own, then shared ones (UWLab's ``square_leg.usd``
#: serves every leg)
_CANDIDATES = {
    "square_table_top": ("square_table_top.usd",),
    **{f"square_table_leg{i}": (f"square_table_leg{i}.usd", "square_table_leg.usd", "square_leg.usd") for i in range(1, 5)},
}


def part_usd_path(name: str) -> str:
    """``<assets>/<name>.usd``, or a shared leg file for a leg without its own."""
    root = assets_dir()
    for candidate in _CANDIDATES[name]:
        path = root / candidate
        if path.exists():
            return str(path)
    raise FileNotFoundError(
        f"no USD for part {name!r} in {root}: looked for " + ", ".join(_CANDIDATES[name])
    )


def part_cfg(part: PartSpec) -> RigidObjectCfg:
    """A furniture part as a free rigid body at its FurnitureBench reset pose.

    The solver iteration counts and contact offsets mirror FurnitureBench's Isaac Gym
    ``sim_params`` (20 position / 1 velocity iterations, contact offset 2 mm, rest offset
    0). The mass is upstream's measured one. Friction is set by the startup event in the
    env cfg (``randomize_rigid_body_material`` at a fixed value), because a USD spawn has no
    physics-material field.
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
            mass_props=sim_utils.MassPropertiesCfg(mass=part.mass),
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
