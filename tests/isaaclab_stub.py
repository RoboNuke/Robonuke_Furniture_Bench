"""A minimal stand-in for the Isaac Lab modules the task package imports, so the env cfg's
wiring (scene entries, term parameters, event order, registered names) can be checked on a
machine without Isaac Sim. Nothing here simulates anything.

``install()`` puts the stubs in ``sys.modules`` and returns a function that removes them
again, together with every ``furniture_lab.tasks.assembly*`` module imported against them.
"""

from __future__ import annotations

import sys
import types
from dataclasses import MISSING  # noqa: F401  (re-exported for convenience)


def _kwargs_class(name):
    def __init__(self, *args, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._args = args
        post = getattr(self, "__post_init__", None)
        if post is not None:
            post()

    def copy(self):
        import copy as _copy

        return _copy.deepcopy(self)

    def replace(self, **kwargs):
        new = self.copy()
        for key, value in kwargs.items():
            setattr(new, key, value)
        return new

    return type(name, (), {"__init__": __init__, "copy": copy, "replace": replace})


def configclass(cls):
    """Like Isaac Lab's: class attributes become instance defaults, kwargs override them."""

    def __init__(self, **kwargs):
        # a plain (non-configclass) base with its own __init__ sets things like `viewer`
        for base in reversed(cls.__mro__):
            if base is not object and "__init__" in vars(base) and not getattr(base, "_stub_cc", False):
                base.__init__(self)
        for base in reversed(cls.__mro__):
            for key, value in vars(base).items():
                if key.startswith("__") or isinstance(value, (type, types.FunctionType, property)):
                    continue  # dunders, nested cfg classes and methods are not fields
                setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)
        post = getattr(self, "__post_init__", None)
        if post is not None:
            post()

    if "__init__" not in vars(cls):
        cls.__init__ = __init__
    cls._stub_cc = True
    cls.copy = lambda self: __import__("copy").deepcopy(self)
    return cls


def install():
    mods = {}

    def module(name, **attrs):
        m = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(m, key, value)
        mods[name] = m
        return m

    # ---- isaaclab.sim
    sim = module(
        "isaaclab.sim",
        **{
            n: _kwargs_class(n)
            for n in (
                "UsdFileCfg", "RigidBodyPropertiesCfg", "MassPropertiesCfg",
                "CollisionPropertiesCfg", "CuboidCfg", "RigidBodyMaterialCfg",
                "PreviewSurfaceCfg", "GroundPlaneCfg", "DomeLightCfg", "SimulationCfg",
                "PhysxCfg", "SDFMeshPropertiesCfg",
            )
        },
    )
    # ---- isaaclab.assets
    Articulation = _kwargs_class("ArticulationCfg")
    Articulation.InitialStateCfg = _kwargs_class("InitialStateCfg")
    AssetBase = _kwargs_class("AssetBaseCfg")
    AssetBase.InitialStateCfg = _kwargs_class("InitialStateCfg")
    RigidObject = _kwargs_class("RigidObjectCfg")
    RigidObject.InitialStateCfg = _kwargs_class("InitialStateCfg")
    module("isaaclab.assets", ArticulationCfg=Articulation, AssetBaseCfg=AssetBase, RigidObjectCfg=RigidObject)
    # ---- isaaclab_assets.robots.franka
    franka = Articulation(
        prim_path="",
        spawn=sim.UsdFileCfg(usd_path="panda.usd", rigid_props=sim.RigidBodyPropertiesCfg(disable_gravity=False)),
        init_state=Articulation.InitialStateCfg(joint_pos={}),
        actuators={
            "panda_shoulder": _kwargs_class("Act")(stiffness=80.0, damping=4.0),
            "panda_forearm": _kwargs_class("Act")(stiffness=80.0, damping=4.0),
            "panda_hand": _kwargs_class("Act")(stiffness=2e3, damping=1e2),
        },
    )
    module("isaaclab_assets")
    module("isaaclab_assets.robots")
    module("isaaclab_assets.robots.franka", FRANKA_PANDA_CFG=franka)
    # ---- controllers, managers, scene, utils
    module("isaaclab.controllers", OperationalSpaceControllerCfg=_kwargs_class("OperationalSpaceControllerCfg"))
    module(
        "isaaclab.managers",
        **{
            n: _kwargs_class(n)
            for n in ("EventTermCfg", "ObservationTermCfg", "RewardTermCfg", "TerminationTermCfg", "SceneEntityCfg")
        },
        ObservationGroupCfg=type("ObservationGroupCfg", (), {}),
    )
    module("isaaclab.scene", InteractiveSceneCfg=type("InteractiveSceneCfg", (), {}))
    module("isaaclab.utils", configclass=configclass)
    module(
        "isaaclab.utils.math",
        **{n: (lambda *a, **k: None) for n in (
            "subtract_frame_transforms", "combine_frame_transforms", "quat_apply_inverse",
            "quat_from_euler_xyz", "quat_mul",
        )},
    )
    # ---- envs and stock mdp
    stock = ("last_action", "time_out", "reset_scene_to_default", "randomize_rigid_body_material", "randomize_rigid_body_mass")
    mdp = module("isaaclab.envs.mdp", __all__=list(stock), **{n: (lambda *a, **k: None) for n in stock})
    for n in stock:
        getattr(mdp, n).__name__ = n
    osc_action = _kwargs_class("OperationalSpaceControllerActionCfg")
    osc_action.OffsetCfg = _kwargs_class("OffsetCfg")
    module(
        "isaaclab.envs.mdp.actions.actions_cfg",
        BinaryJointPositionActionCfg=_kwargs_class("BinaryJointPositionActionCfg"),
        OperationalSpaceControllerActionCfg=osc_action,
    )
    module("isaaclab.envs.mdp.actions")

    class ManagerBasedRLEnvCfg:
        def __init__(self, **kwargs):
            self.viewer = types.SimpleNamespace(eye=None, lookat=None)
            for key, value in kwargs.items():
                setattr(self, key, value)

    module("isaaclab.envs", ManagerBasedRLEnvCfg=ManagerBasedRLEnvCfg, ManagerBasedRLEnv=type("ManagerBasedRLEnv", (), {}))
    module("isaaclab", sim=sim)

    saved = {name: sys.modules.get(name) for name in mods}
    sys.modules.update(mods)
    # the configclass stub must init ManagerBasedRLEnvCfg subclasses too
    mods["isaaclab.envs"].ManagerBasedRLEnvCfg.__init__ = ManagerBasedRLEnvCfg.__init__

    def uninstall():
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
        for name in [m for m in sys.modules if m.startswith("furniture_lab.tasks.assembly") or m == "furniture_lab.tasks.furniture.assets"]:
            sys.modules.pop(name)

    return uninstall
