"""one_leg's task phases, per env and per step (``furniture.track_phases``), for eval playback.

The phases follow the task, not the scripted expert's gripper commands (``failures.py`` sorts
those), so they read the same for a learned policy:

* **Corner** (the start): pushing the top into the corner. Done once the gripper lets go of the
  top: it has held the top (fingers closed narrower than ``phase_top_grip_width`` with the TCP
  within ``phase_top_grip_dist`` of the top's centre and at most ``phase_top_grip_height`` above
  it) and then opens wider than ``phase_release_width``. How well the corner was done does not
  decide it: the top's distance from the corner goal (the servo expert's, ``failures.py``'s: 2 mm
  inside both obstacle walls, wherever the obstacles moved this episode) is a measurement.
* **Grasp**: reaching over, grabbing leg 4 and lifting it. Done once the leg is held (the TCP
  within ``phase_held_dist`` of its centre) and has risen ``phase_lift`` above where it lay at
  the episode's start.
* **Insert**: done once the fingers are open wider than ``phase_release_width`` and the leg has
  stayed in its hole -- off-axis at most ``phase_hole_xz``, tilted at most ``phase_tilt_deg``
  from the top's seat axis, ``failures.py``'s test -- for ``phase_hold_steps`` consecutive steps.
* **Screw**: turning the leg in place, until the env's assembled check (success). Its turn is
  counted positive in the screwing direction (into the seat, a right-hand thread).

A phase only ever moves forward. The episode then ends in success (all assembled), timeout (the
``time_out`` term) or failure (any other termination).

Published, after the terminations and before any reset clears the state (``AssemblyEnv``):

* every step, ``infos["eval_state"]`` (the eval's per-step parquet and the overlays' ``step.state``):
  ``phase/id`` (0-3 the phase, 4 / 5 / 6 success / failure / timeout on the step the episode
  ends), the measurements each phase is judged by (``phase/corner_mm`` (the top from the corner
  goal), ``top_held`` (1 once the top was held), ``top_grip_mm`` (the finger width until the top
  is held, then the narrowest while holding it), ``lift_mm``, ``held_mm``, ``hole_mm``,
  ``tilt_deg``, ``open_mm`` (the finger width), ``hold_steps``, ``height_mm`` / ``ori_deg`` (the
  leg against its assembled pose: height above the seat, worst axis angle), ``turned_deg`` (about
  the seat axis since Screw began, positive screwing in)), ``track_mm`` (the absolute action's
  target position against the TCP; NaN for a relative action),
  ``force_N`` (the estimated EE force's magnitude, ``env.wrench``), the goals
  (``phase_goal/<measurement>``) and ``phase/max_steps``;
* at episode end, ``infos["episode_metrics_to_log"]`` (one value per episode in the eval's
  summary): ``phase/reached`` (the furthest phase, 4 = success), ``phase/end`` (4 / 5 / 6) and
  ``phase/<corner|grasp|insert>_step`` (the step it was done on; NaN if never),
  ``phase/corner_release_mm`` (the top's distance from the corner goal when the gripper let go
  of it; NaN if never) and ``phase/passed_<corner|grasp|insert|success>`` (1 if the episode got
  past that stage, else 0: their mean over episodes is the share of rollouts reaching it).

:attr:`OneLegPhases.newly` ``(N, 3)`` marks the phases (Corner, Grasp, Insert) each env got past
this step, for the phase rewards (``mdp.phase_done``): the update runs inside the termination
manager's compute, before the reward manager's in the same step.

The geometry is pure torch (below, tested on CPU); :class:`OneLegPhases` reads the env.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict

import torch

from ..furniture.assembly import quat_conjugate, quat_rotate_inverse, rotation_diag

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["PHASES", "ENDS", "advance", "end_code", "corner_error", "holds_top", "corner_done", "leg_in_hole", "OneLegPhases"]

#: phase ids 0-3, then the three ends 4-6
PHASES = ("corner", "grasp", "insert", "screw")
ENDS = ("success", "failure", "timeout")
CORNER, GRASP, INSERT, SCREW, SUCCESS, FAILURE, TIMEOUT = range(7)


def advance(phase: torch.Tensor, corner_ok: torch.Tensor, grasp_ok: torch.Tensor, insert_ok: torch.Tensor) -> torch.Tensor:
    """The phases after this step's checks: forward only, and through more than one in a step
    when their checks pass together."""
    phase = torch.where((phase == CORNER) & corner_ok, torch.full_like(phase, GRASP), phase)
    phase = torch.where((phase == GRASP) & grasp_ok, torch.full_like(phase, INSERT), phase)
    return torch.where((phase == INSERT) & insert_ok, torch.full_like(phase, SCREW), phase)


def end_code(success: torch.Tensor, time_out: torch.Tensor) -> torch.Tensor:
    """How an episode that ended this step ended: success over timeout over failure."""
    code = torch.where(time_out, torch.full(success.shape, TIMEOUT, dtype=torch.long, device=success.device),
                       torch.full(success.shape, FAILURE, dtype=torch.long, device=success.device))
    return torch.where(success, torch.full_like(code, SUCCESS), code)


def corner_error(top_xy: torch.Tensor, front_x: torch.Tensor, left_y: torch.Tensor) -> torch.Tensor:
    """The top's (x, y) distance from the servo expert's corner goal (robot-base frame), the
    obstacles where they are this episode (``failures._corner_error``)."""
    from ...failures import CORNER_CLEARANCE, TOP_HALF_WIDTH
    from ..furniture.frames import OBSTACLE_FRONT_SIZE, OBSTACLE_SIDE_SIZE

    gx = front_x - OBSTACLE_FRONT_SIZE[0] / 2 - TOP_HALF_WIDTH - CORNER_CLEARANCE
    gy = left_y - OBSTACLE_SIDE_SIZE[1] / 2 - TOP_HALF_WIDTH - CORNER_CLEARANCE
    return torch.hypot(top_xy[:, 0] - gx, top_xy[:, 1] - gy)


def holds_top(width: torch.Tensor, tcp: torch.Tensor, top: torch.Tensor, grip_width: float, grip_dist: float,
              grip_height: float) -> torch.Tensor:
    """Whether the gripper holds the top this step: fingers ``width`` (N,) closed narrower than
    ``grip_width``, the TCP (N, 3) within ``grip_dist`` of the top's centre (N, 3) and at most
    ``grip_height`` above it (same frame, z up). It takes the top by its edge: the TCP sits
    70-85 mm from the centre at the top's own height, the fingers 5-10 mm apart."""
    near = (tcp - top).norm(dim=-1) < grip_dist
    low = (tcp[:, 2] - top[:, 2]) < grip_height
    return (width < grip_width) & near & low


def corner_done(top_held: torch.Tensor, opened: torch.Tensor) -> torch.Tensor:
    """Corner's check: the top was held (``top_held``, latched) and the fingers are now open
    (``opened``) -- the gripper let go of it, wherever the top ended up."""
    return top_held & opened


def _rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """R(q) v."""
    return quat_rotate_inverse(quat_conjugate(q), v)


def leg_in_hole(rel_pos: torch.Tensor, rel_quat: torch.Tensor, hole: torch.Tensor, axis: torch.Tensor, ref: torch.Tensor):
    """Leg 4 against its hole from its pose in the top's frame (``rel_pos`` (N, 3), ``rel_quat``
    (N, 4) wxyz). ``hole``: the seated leg's position, ``axis``: the seat axis, ``ref``: an axis
    perpendicular to it, all (3,) in the same frame. Returns, per env: the off-axis distance (m),
    the height above the seated position along the axis (m), the tilt of the leg's axis from the
    seat axis (deg) and the leg's angle about the axis (rad, of ``ref`` turned by the leg)."""
    n = rel_pos.shape[0]
    a, r = axis.expand(n, 3), ref.expand(n, 3)
    d = rel_pos - hole
    height = (d * a).sum(-1)
    xz = (d - height.unsqueeze(-1) * a).norm(dim=-1)
    tilt = torch.rad2deg(torch.acos(((_rotate(rel_quat, a)) * a).sum(-1).clamp(-1.0, 1.0)))
    v = _rotate(rel_quat, r)
    v = v - (v * a).sum(-1, keepdim=True) * a
    angle = torch.atan2((torch.cross(r, v, dim=-1) * a).sum(-1), (r * v).sum(-1))
    return xz, height, tilt, angle


def ori_error_deg(rel_quat: torch.Tensor) -> torch.Tensor:
    """The worst axis's angle (deg) between the leg's frame and its assembled orientation (the
    identity): what upstream's ``cos >= ori_bound`` per axis bounds."""
    return torch.rad2deg(torch.acos(rotation_diag(rel_quat).min(dim=-1).values.clamp(-1.0, 1.0)))


class OneLegPhases:
    """The per-env phase state of one_leg, updated once per env step (see the module docstring)."""

    def __init__(self, env: "ManagerBasedRLEnv", fcfg) -> None:
        import numpy as np

        from ...failures import HOLE
        from ..furniture.square_table import MESH_FRAMES, pairs_for
        from .mdp.actions import FurnitureBenchAbsAction

        if fcfg.task != "one_leg":
            raise ValueError(f"furniture.track_phases is one_leg's phases; furniture.task is {fcfg.task!r}")
        for name in ("phase_top_grip_width", "phase_top_grip_dist", "phase_top_grip_height", "phase_lift",
                     "phase_held_dist", "phase_release_width", "phase_hole_xz", "phase_tilt_deg", "phase_force_scale"):
            if not getattr(fcfg, name) > 0:
                raise ValueError(f"furniture.{name} must be > 0, got {getattr(fcfg, name)}")
        if int(fcfg.phase_hold_steps) < 1:
            raise ValueError(f"furniture.phase_hold_steps must be >= 1, got {fcfg.phase_hold_steps}")
        self.env, self.cfg = env, fcfg
        n, dev = env.num_envs, env.device
        # the OBJ frame's seat axis (y), an axis across it (x) and the seated leg, in the USDs' frame
        R = np.asarray(MESH_FRAMES[fcfg.mesh_frame], dtype=np.float64)
        as_t = lambda v: torch.tensor(R @ np.asarray(v, dtype=np.float64), dtype=torch.float32, device=dev)  # noqa: E731
        self.axis, self.ref, self.hole = as_t([0.0, 1.0, 0.0]), as_t([1.0, 0.0, 0.0]), as_t(HOLE)
        (self.top, self.leg), = pairs_for("one_leg")
        robot = env.scene["robot"]
        self.fingers = robot.find_joints("panda_finger_joint.*")[0]
        self.hand = robot.find_bodies("panda_hand")[0][0]
        terms = [env.action_manager.get_term(name) for name in env.action_manager.active_terms]
        self.target_term = next((t for t in terms if isinstance(t, FurnitureBenchAbsAction)), None)

        self.phase = torch.zeros(n, dtype=torch.long, device=dev)
        self.rest_z = torch.zeros(n, device=dev)
        self.need_rest = torch.ones(n, dtype=torch.bool, device=dev)
        self.top_held = torch.zeros(n, dtype=torch.bool, device=dev)
        self.top_grip = torch.full((n,), math.inf, device=dev)
        self.corner_release = torch.full((n,), math.nan, device=dev)
        self.hold = torch.zeros(n, dtype=torch.long, device=dev)
        self.angle_prev = torch.zeros(n, device=dev)
        self.turned = torch.zeros(n, device=dev)
        self.done_step = torch.full((n, 3), math.nan, device=dev)
        #: the phases (Corner, Grasp, Insert) each env got past on the last update
        self.newly = torch.zeros((n, 3), dtype=torch.bool, device=dev)
        full = lambda v: torch.full((n,), float(v), device=dev)  # noqa: E731
        self.goals = {
            "phase_goal/top_grip_mm": full(fcfg.phase_top_grip_width * 1000),
            "phase_goal/lift_mm": full(fcfg.phase_lift * 1000),
            "phase_goal/held_mm": full(fcfg.phase_held_dist * 1000),
            "phase_goal/hole_mm": full(fcfg.phase_hole_xz * 1000),
            "phase_goal/tilt_deg": full(fcfg.phase_tilt_deg),
            "phase_goal/open_mm": full(fcfg.phase_release_width * 1000),
            "phase_goal/hold_steps": full(fcfg.phase_hold_steps),
            "phase_goal/force_N": full(fcfg.phase_force_scale),
            "phase/max_steps": full(env.max_episode_length),
        }
        #: what :meth:`update` published last (``infos["eval_state"]`` / ``["episode_metrics_to_log"]``)
        self.state: Dict[str, torch.Tensor] = {}
        self.episode: Dict[str, torch.Tensor] = {}

    def reset(self, env_ids) -> None:
        """The envs that restart: back to Corner, every reference cleared (the published values
        stay this step's)."""
        self.phase[env_ids] = CORNER
        self.need_rest[env_ids] = True
        self.top_held[env_ids] = False
        self.top_grip[env_ids] = math.inf
        self.corner_release[env_ids] = math.nan
        self.hold[env_ids] = 0
        self.turned[env_ids] = 0.0
        self.done_step[env_ids] = math.nan

    def update(self) -> None:
        """One step: measure, advance, publish. Call after the terminations, before the resets."""
        from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

        from ..furniture.assembly import relative_pose
        from .mdp.state import EE_OFFSET, parts_pose_b, parts_pose_w, tracker

        env, f = self.env, self.cfg
        robot = env.scene["robot"]
        root_pos, root_quat = robot.data.root_pos_w, robot.data.root_quat_w
        pos_w, quat_w = parts_pose_w(env)
        pos_b, _ = parts_pose_b(env)
        top_b, leg_b = pos_b[:, self.top], pos_b[:, self.leg]
        if bool(self.need_rest.any()):
            self.rest_z = torch.where(self.need_rest, leg_b[:, 2], self.rest_z)
            self.need_rest[:] = False
        off = torch.tensor(EE_OFFSET, device=env.device, dtype=root_pos.dtype).expand(env.num_envs, -1)
        tcp_w, tcp_q = combine_frame_transforms(robot.data.body_pos_w[:, self.hand], robot.data.body_quat_w[:, self.hand], off)
        tcp_b, _ = subtract_frame_transforms(root_pos, root_quat, tcp_w, tcp_q)
        width = robot.data.joint_pos[:, self.fingers].sum(dim=-1)
        front_b, _ = subtract_frame_transforms(root_pos, root_quat, env.scene["obstacle_front"].data.root_pos_w)
        left_b, _ = subtract_frame_transforms(root_pos, root_quat, env.scene["obstacle_left"].data.root_pos_w)

        corner = corner_error(top_b[:, :2], front_b[:, 0], left_b[:, 1])
        lift = leg_b[:, 2] - self.rest_z
        held = (tcp_b - leg_b).norm(dim=-1)
        rel_pos, rel_quat = relative_pose(pos_w[:, self.top], quat_w[:, self.top], pos_w[:, self.leg], quat_w[:, self.leg])
        xz, height, tilt, angle = leg_in_hole(rel_pos, rel_quat, self.hole, self.axis, self.ref)
        released = width > f.phase_release_width
        # Corner: the narrowest finger width while holding the top, then letting go of it
        holding = (self.phase == CORNER) & holds_top(
            width, tcp_b, top_b, f.phase_top_grip_width, f.phase_top_grip_dist, f.phase_top_grip_height
        )
        self.top_held |= holding
        self.top_grip = torch.where(holding, torch.minimum(self.top_grip, width), self.top_grip)
        in_hole = (xz <= f.phase_hole_xz) & (tilt <= f.phase_tilt_deg)
        self.hold = torch.where((self.phase == INSERT) & released & in_hole, self.hold + 1, torch.zeros_like(self.hold))

        old = self.phase
        self.phase = advance(
            old, corner_done(self.top_held, released), (held <= f.phase_held_dist) & (lift >= f.phase_lift), self.hold >= f.phase_hold_steps
        )
        steps = env.episode_length_buf.to(torch.float32)
        for k in (CORNER, GRASP, INSERT):
            newly = (old <= k) & (self.phase > k)
            self.newly[:, k] = newly
            self.done_step[:, k] = torch.where(newly, steps, self.done_step[:, k])
        left_corner = (old == CORNER) & (self.phase > CORNER)
        self.corner_release = torch.where(left_corner, corner, self.corner_release)
        # the leg's turn about the seat axis, unwrapped, from the step Screw began. Positive is
        # screwing in: the seated leg sits along -axis from where it enters (height > 0 is above
        # the seat), and a right-hand thread advances along -axis when turned positively about
        # -axis, i.e. negatively about the axis ``angle`` is measured about (every successful
        # eval episode turned it -436 to -453 deg that way)
        screwing = old == SCREW
        step_turn = -torch.atan2(torch.sin(angle - self.angle_prev), torch.cos(angle - self.angle_prev))
        self.turned = torch.where(screwing, self.turned + step_turn, torch.zeros_like(self.turned))
        self.angle_prev = angle

        nan = torch.full_like(corner, math.nan)
        if self.target_term is not None:
            track = (self.target_term.raw_actions[:, 0:3] - tcp_b).norm(dim=-1)
        else:
            track = nan
        force = env.wrench.filtered[:, 0:3].norm(dim=-1)

        tm = env.termination_manager
        done = tm.dones
        success = tracker(env).all_assembled()
        time_out = tm.get_term("time_out") if "time_out" in tm.active_terms else torch.zeros_like(done)
        end = end_code(success, time_out)
        shown = torch.where(done, end, self.phase)
        self.state = {
            "phase/id": shown.to(torch.float32),
            "phase/corner_mm": corner * 1000,
            "phase/top_held": self.top_held.to(torch.float32),
            "phase/top_grip_mm": torch.where(self.top_held, self.top_grip, width) * 1000,
            "phase/lift_mm": lift * 1000,
            "phase/held_mm": held * 1000,
            "phase/hole_mm": xz * 1000,
            "phase/tilt_deg": tilt,
            "phase/open_mm": width * 1000,
            "phase/hold_steps": self.hold.to(torch.float32),
            "phase/height_mm": height * 1000,
            "phase/ori_deg": ori_error_deg(rel_quat),
            "phase/turned_deg": torch.rad2deg(self.turned),
            "phase/track_mm": track * 1000,
            "phase/force_N": force,
            **self.goals,
        }
        reached = torch.where(done & success, torch.full_like(self.phase, SUCCESS), self.phase)
        self.episode = {
            "phase/reached": reached.to(torch.float32),
            "phase/end": torch.where(done, end.to(torch.float32), nan),
            **{f"phase/{name}_step": self.done_step[:, k].clone() for k, name in enumerate(PHASES[:3])},
            "phase/corner_release_mm": self.corner_release * 1000,
            **{f"phase/passed_{name}": (reached > k).to(torch.float32) for k, name in enumerate(PHASES[:3])},
            "phase/passed_success": (reached == SUCCESS).to(torch.float32),
        }
