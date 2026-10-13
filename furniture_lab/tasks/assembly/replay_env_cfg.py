"""Env cfg variants for replaying FurnitureBench demos (``scripts/replay_demo.py``).

Both are ``OneLegEnvCfg`` with two changes:

* **terminations**: only the time limit. The task env ends an episode (and resets the parts)
  the moment ``success`` fires, which would yank the leg out from under a replay; here the
  assembled flag is read by the script instead.
* **episode length**: long enough for any demo (FurnitureSim's scripted ``one_leg`` cap is
  600 steps = 30 s; 120 s here).

``OneLegReplayAbsCfg`` additionally drives the arm with **absolute** EE pose targets
(``pose_abs``: position + wxyz quaternion in the robot-base frame, unscaled), for the
``ee_track`` mode. ``OneLegReplayRelCfg`` keeps the task's own ``pose_rel`` action, for the
``actions`` mode, so what the policy would see is what the demo drives.
"""

from __future__ import annotations

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .assembly_env_cfg import OneLegEnvCfg


@configclass
class ReplayTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class OneLegReplayRelCfg(OneLegEnvCfg):
    terminations: ReplayTerminationsCfg = ReplayTerminationsCfg()
    episode_length_s: float = 120.0


@configclass
class OneLegReplayAbsCfg(OneLegReplayRelCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.arm.controller_cfg.target_types = ["pose_abs"]
        self.actions.arm.position_scale = 1.0
        self.actions.arm.orientation_scale = 1.0
