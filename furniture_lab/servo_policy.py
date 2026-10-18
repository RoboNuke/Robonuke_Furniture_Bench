"""``one_leg`` scripted policy that closes the loop on the **object** whenever one is held.

Upstream's state machine (:mod:`.scripted_policy`) servos the hand and assumes the part
follows it. Here the free-hand phases are upstream's states, reused as they are, but as soon
as the fingers hold something the command comes from that object's pose error:

* the hand goal is the object's saturated-P step carried through the grasp as a rigid
  transform -- translation passes straight through, a rotation of the object about its own
  centre makes the TCP orbit (:class:`ObjectServo`);
* the table top is slid into the corner against both walls (the goal sits 2 mm inside
  them, so contact squares it) and released when it stops moving;
* phases hand over on error (position, tilt, the leg's own height or yaw no longer
  changing, the gripper's width), never on a step count; only the episode has a clock;
* the leg is lifted upright, carried over the hole, lowered until its own height stops
  changing, settled and released -- its axis is held vertical the whole way;
* screwing: the leg's goal is upright, on the hole axis, with its yaw led by at most
  ``lead_deg`` beyond where the leg actually is, so the controller can never store more
  than that much torque against a sticky thread; progress and stalls are read off the
  leg's yaw, not the wrist.

Interface and output are those of :class:`.scripted_policy.OneLegScriptedPolicy`
(``act(ScriptedState) -> ScriptedAction``), so the collector and the pkl format are
unchanged. Pure numpy.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .scripted_policy import (
    APRIL_TO_ROBOT,
    ASSEMBLED_POS_THRESHOLD,
    MAX_GRIPPER_WIDTH,
    ORI_BOUND,
    OneLegScriptedPolicy,
    ScriptedAction,
    ScriptedState,
    SquareTableLegPart,
    SquareTableTopPart,
    axisangle_to_quat,
    get_mat,
    hom,
    is_similar_pose,
    mat_to_quat,
    quat_conj,
    quat_mul,
)
from .tasks.furniture import frames

__all__ = ["ObjectServo", "ServoOneLegPolicy", "rotvec_to_mat", "mat_to_rotvec", "rotate_onto", "upright_goal"]


# ------------------------------------------------------------------------------ rotations
def rotvec_to_mat(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    a = float(np.linalg.norm(v))
    if a < 1e-12:
        return np.eye(3)
    k = v / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def mat_to_rotvec(R: np.ndarray) -> np.ndarray:
    """Log map, the short way."""
    q = mat_to_quat(R)
    if q[0] < 0:
        q = -q
    s = float(np.linalg.norm(q[1:]))
    if s < 1e-12:
        return np.zeros(3)
    return q[1:] / s * (2.0 * np.arctan2(s, q[0]))


def yaw_of(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


def unwrap(prev: float, now: float) -> float:
    return prev + ((now - prev + np.pi) % (2 * np.pi) - np.pi)


#: an upright leg (OBJ y, the leg's axis, pointing up; bolt down) with its x face normal
#: at world yaw 0 -- upstream's ``_hole_target`` orientation
LEG_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def rotate_onto(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The smallest rotation taking unit vector ``u`` onto unit vector ``v``."""
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    axis = np.cross(u, v)
    n = float(np.linalg.norm(axis))
    angle = float(np.arctan2(n, u @ v))
    if n < 1e-9:
        return np.eye(3) if angle < 1.0 else rotvec_to_mat(np.array([1.0, 0.0, 0.0]) * np.pi)
    return rotvec_to_mat(axis / n * angle)


def upright_goal(R_leg: np.ndarray, dyaw: float = 0.0) -> np.ndarray:
    """The leg's orientation with its axis (OBJ y) made vertical by the smallest rotation
    (yaw untouched), then yawed by ``dyaw`` about the world vertical."""
    y = R_leg[:, 1]
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(y, z)
    n = float(np.linalg.norm(axis))
    angle = float(np.arccos(np.clip(np.dot(y, z), -1.0, 1.0)))
    if n < 1e-9:
        R_up = R_leg if angle < 1e-6 else rotvec_to_mat(np.array([np.pi, 0.0, 0.0])) @ R_leg
    else:
        R_up = rotvec_to_mat(axis / n * angle) @ R_leg
    return get_mat([0, 0, 0], [0, 0, dyaw])[:3, :3] @ R_up


def leg_yaw(R_leg: np.ndarray) -> float:
    """World yaw of an (approximately upright) leg's x face normal."""
    return yaw_of(R_leg)


def leg_tilt_deg(R_leg: np.ndarray) -> float:
    """Angle of the leg's axis (OBJ y) from vertical."""
    return float(np.degrees(np.arccos(np.clip(R_leg[2, 1], -1.0, 1.0))))


# ------------------------------------------------------------------------------ the servo
class ObjectServo:
    """Hand goal from a held object's pose error.

    One step: ``dp = sat(kp_pos * (goal_pos - obj_pos))``, ``dR = exp(sat(kp_rot * log(goal_R
    obj_R^T)))``. The object is rigid in the fingers, so the hand must move with it: the TCP
    orbits the object's centre, ``ee' = obj + dp + dR (ee - obj)``, ``R_ee' = dR R_ee``.

    The result is a *relative* command to the OSC, which closes only ~20 % of it per 20 Hz
    step (measured: 3 mm asked -> 0.6 mm moved, 6 deg -> 1.1 deg). So the gains are ~5 and
    the caps bound the command, not the motion: ``max_cmd_pos`` of 40 mm moves the object
    ~8 mm per step. Upstream's free-hand deltas do the same thing (x2, up to 115 mm).
    """

    def __init__(self, kp_pos: float = 5.0, kp_rot: float = 5.0, max_cmd_pos: float = 0.04, max_cmd_rot_deg: float = 30.0):
        self.kp_pos = kp_pos
        self.kp_rot = kp_rot
        self.max_step_pos = max_cmd_pos
        self.max_step_rot = np.radians(max_cmd_rot_deg)

    @staticmethod
    def _sat(v: np.ndarray, cap: float) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return v if n <= cap else v * (cap / n)

    def step(self, obj_pos, obj_R, goal_pos, goal_R, ee_pos, ee_R) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Returns (ee goal position, ee goal rotation, position error m, rotation error rad)."""
        e_p = np.asarray(goal_pos, dtype=np.float64) - obj_pos
        e_r = mat_to_rotvec(goal_R @ obj_R.T)
        dp = self._sat(self.kp_pos * e_p, self.max_step_pos)
        dR = rotvec_to_mat(self._sat(self.kp_rot * e_r, self.max_step_rot))
        ee_goal_pos = obj_pos + dp + dR @ (ee_pos - obj_pos)
        ee_goal_R = dR @ ee_R
        return ee_goal_pos, ee_goal_R, float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r))


# ------------------------------------------------------------------------------ the policy
class ServoOneLegPolicy:
    """Free hand: upstream's states. Holding: the object servo. See the module docstring."""

    #: upstream's free-hand states for the top (then it is held: 'push' in upstream)
    TOP_FREE = ("reach_body_grasp_xy", "reach_body_grasp_z", "pick_body")
    #: ... and for the leg on the floor (then held: 'lift_up') and at the regrasp (then
    #: held: 'screw_align')
    LEG_FREE = ("reach_leg_floor_xy", "reach_leg_ori", "reach_leg_floor_z", "pick_leg")
    REGRASP_FREE = ("pre_grasp", "grasp_lower", "screw_grasp")

    TOP_HALF_WIDTH = 0.08125

    def __init__(
        self,
        seed: int = 0,
        noise: bool = False,
        pos_step: float = 0.008,
        rot_step_deg: float = 6.0,
        lead_deg: float = 10.0,
        press: float = 0.003,
        insert_rate: float = 0.002,
        insert_floor: float = 0.003,
        insert_stop: float = 0.0005,
        insert_lift: float = 0.0,
        release_dz: float = 0.05,
        upright_step_deg: float | None = None,
        screw_upright: bool = False,
        screw_pivot: bool = False,
        pivot_step_deg: float | None = None,
        insert_yaw_deg: float = 15.0,
        corner_clearance: float = -0.002,
        hover_dz: float = 0.04,
        q7_margin: float = 0.08,
        wind_step_deg: float = 9.0,
        assembled_hold_steps: int = 5,
        gain: float = 5.0,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.top = SquareTableTopPart(self.rng, noise)
        self.leg = SquareTableLegPart("square_table_leg4", OneLegScriptedPolicy.LEG4_ASSEMBLED_POS, self.rng, noise)
        self.leg.q7_margin = q7_margin
        self.leg.hover_dz = hover_dz
        # ``gain``: how much bigger a command is than the motion wanted from it -- 5 for the
        # port's OSC, which closes ~20 % of a command per step (see ObjectServo); 1 for a
        # controller that closes it all (ResiP's differential IK into stiff joint PD)
        self.gain = float(gain)
        self.leg.wind_step_deg = self.gain * wind_step_deg  # a command
        # pos_step / rot_step_deg are the wanted motion per step; the command is gain x that
        self.servo = ObjectServo(
            kp_pos=self.gain, kp_rot=self.gain, max_cmd_pos=self.gain * pos_step, max_cmd_rot_deg=self.gain * rot_step_deg
        )
        #: standing the grasped leg up in the air: its own rotation cap (default rot_step_deg)
        up = rot_step_deg if upright_step_deg is None else upright_step_deg
        self.upright_servo = ObjectServo(
            kp_pos=self.gain, kp_rot=self.gain, max_cmd_pos=self.gain * pos_step, max_cmd_rot_deg=self.gain * up
        )
        self.lead = np.radians(lead_deg)
        self.press = press
        self.insert_rate = insert_rate
        #: lowering: never below seat + insert_floor; "stopped" when the leg dropped less than
        #: insert_stop over 3 steps; the hand then holds insert_lift above where it stopped
        self.insert_floor = insert_floor
        self.insert_stop = insert_stop
        self.insert_lift = insert_lift
        #: after letting go of the leg, the hand rises this far straight up -- same x, y and
        #: orientation, at most pos_step per step -- before the regrasp moves it anywhere else
        self.release_dz = release_dz
        self.pos_step = pos_step
        self.rot_step = np.radians(rot_step_deg)
        #: screwing: False turns the hand about the vertical through the leg's axis and presses,
        #: nothing else; True also servos the leg upright and onto the hole axis (the old way)
        self.screw_upright = screw_upright
        #: screw regrasp with the leg leaning in its hole: grasp with the hand's axis along
        #: the leg's assumed axis (the line from the leg's centre to the hole's entry point),
        #: then turn hand and leg about that hole point -- held still -- until the line is on
        #: the hole axis, and only then screw (see _hole_point / the 'screw_pivot' phase)
        self.screw_pivot = screw_pivot
        self.pivot_step = np.radians(rot_step_deg if pivot_step_deg is None else pivot_step_deg)
        #: the leg is carried over the hole and lowered at its yaw when it stood up plus this
        #: (deg, counter-clockwise seen from above = +world yaw), a fixed goal for the whole
        #: placement: turns the thread start away from the yaw that cross-threads
        self.insert_yaw = np.radians(insert_yaw_deg)
        self._place_R = None
        self.corner_clearance = corner_clearance
        self.assembled_hold_steps = assembled_hold_steps
        # phase machine
        self.phase = "top_free"
        self.phase_steps = 0
        self.gripper = -1.0
        self.hold_pose: Optional[np.ndarray] = None
        self._leg_z_hist: list = []
        self._leg_yaw_u: Optional[float] = None
        self._yaw_hist: list = []
        self._assembled_streak = 0
        self._track: list = []
        self.steps = 0
        self.done = False

    # -- bookkeeping ---------------------------------------------------------------
    @property
    def state(self) -> str:
        if self.phase in ("top_free",):
            return self.top.state
        if self.phase in ("leg_free", "regrasp_free"):
            return self.leg.state
        return self.phase

    def _goto(self, phase: str) -> None:
        self.phase = phase
        self.phase_steps = 0
        self._leg_z_hist = []
        self._yaw_hist = []
        self._track = []

    def _opened(self, s: ScriptedState) -> bool:
        """The fingers are fully open, or have stopped opening (< 0.5 mm in 5 steps, e.g. a
        finger against the top or a wall): either way, nothing more will come of waiting."""
        if s.gripper_width >= MAX_GRIPPER_WIDTH - 0.001:
            return True
        return self.phase_steps > 3 and self._stalled(np.array([s.gripper_width]), 5, 0.0005)

    def _stalled(self, x: np.ndarray, steps: int, eps: float) -> bool:
        """``x`` (a vector of what should still be moving) has changed by less than ``eps``
        over the last ``steps`` steps -- contact, or converged."""
        self._track.append(np.asarray(x, dtype=np.float64).copy())
        if len(self._track) <= steps:
            return False
        return bool(np.linalg.norm(self._track[-1] - self._track[-1 - steps]) < eps)

    # -- geometry ------------------------------------------------------------------
    def _poses(self, s: ScriptedState):
        """Top and leg poses in the robot frame (OBJ mesh frames), and the seat point of leg 4."""
        a2r = APRIL_TO_ROBOT
        T_top = a2r @ s.parts_tag["square_table_top"]
        T_leg = a2r @ s.parts_tag["square_table_leg4"]
        seat = (T_top @ self.leg.default_assembled_pose)[:3, 3]  # leg origin when seated
        return T_top, T_leg, seat

    #: the top's half thickness along its seat axis (OBJ y): its upper face, where a hole opens
    TOP_HALF_THICKNESS = 0.015625

    def _hole_point(self, T_top: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Leg 4's hole: the point where its axis meets the top's upper face (robot frame), and
        the hole axis (the top's OBJ y, up). Where the leg *should* enter, not a contact."""
        hole = self.leg.default_assembled_pose[:3, 3]
        p = T_top @ np.array([hole[0], self.TOP_HALF_THICKNESS, hole[2], 1.0])
        return p[:3], T_top[:3, 1] / np.linalg.norm(T_top[:3, 1])

    def _leg_line(self, leg_pos: np.ndarray, T_top: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The leg's assumed axis: the unit line from the hole point up to the leg's centre;
        with the hole point and the hole axis."""
        P, h = self._hole_point(T_top)
        a = leg_pos - P
        n = float(np.linalg.norm(a))
        return (h if n < 1e-6 else a / n), P, h

    def _corner_goal(self, s: ScriptedState) -> Tuple[np.ndarray, np.ndarray]:
        """The top against the front and left obstacles (``corner_clearance`` < 0 puts the goal
        inside the walls, so it is pushed into contact and squared by them), at its nominal
        reset yaw (upstream's reset orientation, without the reset noise)."""
        # the obstacles where they are this episode (moved by furniture.obstacle_offset)
        x = s.obstacles_robot["obstacle_front"][0] - frames.OBSTACLE_FRONT_SIZE[0] / 2 - self.TOP_HALF_WIDTH - self.corner_clearance
        y = s.obstacles_robot["obstacle_left"][1] - frames.OBSTACLE_SIDE_SIZE[1] / 2 - self.TOP_HALF_WIDTH - self.corner_clearance
        T_top, _, _ = self._poses(s)
        from .tasks.furniture.square_table import PARTS

        R_goal = (APRIL_TO_ROBOT @ hom(np.asarray(PARTS[0].reset_rot_tag)[:3, :3], (0, 0, 0)))[:3, :3]
        return np.array([x, y, T_top[2, 3]]), R_goal

    def assembled(self, s: ScriptedState) -> bool:
        top = s.parts_tag["square_table_top"]
        rel = np.linalg.inv(top) @ s.parts_tag["square_table_leg4"]
        return any(
            is_similar_pose(get_mat(h, [0, 0, 0]), rel, ORI_BOUND, ASSEMBLED_POS_THRESHOLD) for h in OneLegScriptedPolicy.HOLES
        )

    # -- one step ------------------------------------------------------------------
    def act(self, s: ScriptedState) -> ScriptedAction:
        self.steps += 1
        self.phase_steps += 1
        ee_pos, ee_R = s.ee_pos.copy(), s.ee_rot.copy()
        ee_quat = mat_to_quat(ee_R)
        goal_pos, goal_R, grip = ee_pos.copy(), ee_R.copy(), self.gripper
        skill_complete = 0
        assembled_flag = False
        upstream_scaling = False  # free-hand states use upstream's delta scaling

        T_top, T_leg, seat = self._poses(s)
        leg_pos, leg_R = T_leg[:3, 3].copy(), T_leg[:3, :3].copy()

        # the held check (FurnitureBench's rule, held a few steps)
        self._assembled_streak = self._assembled_streak + 1 if self.assembled(s) else 0
        if self.phase not in ("neutral", "done") and self._assembled_streak >= self.assembled_hold_steps and self.phase not in ("top_free", "top_servo", "top_release", "top_up"):
            self.gripper = -1.0
            self.hold_pose = hom(ee_R, ee_pos)
            self._goto("neutral")
            assembled_flag = True
            skill_complete = 1

        # ---------------------------------------------------------------- the table top
        if self.phase == "top_free":
            goal_pos, goal_R, grip, skill_complete = self.top.pre_assemble(s)
            upstream_scaling = True
            if self.top.state not in self.TOP_FREE:  # 'push': the top is in the fingers
                self.gripper = 1.0
                grip = 1.0
                self._goto("top_servo")
        elif self.phase == "top_servo":
            tgt_pos, tgt_R = self._corner_goal(s)
            goal_pos, goal_R, ep, er = self.servo.step(T_top[:3, 3], T_top[:3, :3], tgt_pos, tgt_R, ee_pos, ee_R)
            goal_pos[2] = ee_pos[2]  # slide it: no lift, no press
            # done when it has stopped moving near the goal: the goal is inside the walls,
            # so that is the walls holding it (converged-in-free-space also stalls)
            against_wall = self._stalled(np.array([T_top[0, 3], T_top[1, 3], 0.1 * yaw_of(T_top[:3, :3])]), 5, 0.0003) and ep < 0.015
            if against_wall:
                self.hold_pose = hom(ee_R, ee_pos)
                self.gripper = -1.0
                self._goto("top_release")
                skill_complete = 1
        elif self.phase == "top_release":
            goal_pos, goal_R = self.hold_pose[:3, 3].copy(), self.hold_pose[:3, :3].copy()
            if self._opened(s):
                self._goto("top_up")
        elif self.phase == "top_up":
            # straight up to z 0.10, ramped (a one-step jump lets the joint PD take a
            # joint-space path and swing the open fingers through the top)
            goal_pos = self.hold_pose[:3, 3].copy()
            goal_pos[2] = min(0.10, max(ee_pos[2], goal_pos[2]) + self.pos_step)
            goal_R = self.hold_pose[:3, :3].copy()
            if abs(ee_pos[2] - 0.10) < 0.01:
                self.leg.reset()
                self.leg._state = "reach_leg_floor_xy"
                self.leg.gripper_action = -1
                self._goto("leg_free")

        # ---------------------------------------------------------------- the leg: pick
        elif self.phase == "leg_free":
            goal_pos, goal_R, grip, skill_complete = self.leg.fsm_step(s, "square_table_top")
            upstream_scaling = True
            if self.leg.state not in self.LEG_FREE:  # 'lift_up': the leg is in the fingers
                self.gripper = 1.0
                grip = 1.0
                self._goto("leg_lift")

        # ---------------------------------------------------------------- the leg: carry
        elif self.phase == "leg_lift":
            # straight up, orientation as grasped (upstream lifts before it turns the leg)
            tgt = np.array([leg_pos[0], leg_pos[1], seat[2] + 0.10])
            goal_pos, goal_R, ep, er = self.servo.step(leg_pos, leg_R, tgt, leg_R, ee_pos, ee_R)
            if ep < 0.01:
                self._goto("leg_upright")
        elif self.phase == "leg_upright":
            # stand it up in the air: axis vertical, bolt down, by the smallest rotation
            tgt = np.array([leg_pos[0], leg_pos[1], seat[2] + 0.10])
            goal_pos, goal_R, ep, er = self.upright_servo.step(leg_pos, leg_R, tgt, upright_goal(leg_R), ee_pos, ee_R)
            if ep < 0.01 and leg_tilt_deg(leg_R) < 3.0:
                self._place_R = upright_goal(leg_R, self.insert_yaw)
                self._goto("leg_over")
        elif self.phase == "leg_over":
            # bolt tip just above the hole: origin at seat + 30 mm (bolt is 25.4 mm long)
            tgt = np.array([seat[0], seat[1], seat[2] + 0.030])
            goal_pos, goal_R, ep, er = self.servo.step(leg_pos, leg_R, tgt, self._place_R, ee_pos, ee_R)
            if ep < 0.002 and leg_tilt_deg(leg_R) < 2.0 and er < np.radians(3.0):  # at the placement yaw too
                self._goto("leg_lower")
        elif self.phase == "leg_lower":
            # down at insert_rate until the leg's own height stops changing (the thread holds it)
            self._leg_z_hist.append(leg_pos[2])
            z_cmd = max(leg_pos[2] - self.insert_rate, seat[2] + self.insert_floor)
            tgt = np.array([seat[0], seat[1], z_cmd])
            goal_pos, goal_R, ep, er = self.servo.step(leg_pos, leg_R, tgt, self._place_R, ee_pos, ee_R)
            h = self._leg_z_hist
            stopped = len(h) > 3 and (h[-4] - h[-1]) < self.insert_stop  # the thread holds it
            if stopped or leg_pos[2] <= seat[2] + self.insert_floor + 0.0001:
                hold = ee_pos.copy()
                hold[2] += self.insert_lift
                self.hold_pose = hom(ee_R, hold)
                self._goto("leg_settle")
        elif self.phase == "leg_settle":
            # hold until the leg has stopped moving (position and axis), then let go
            goal_pos, goal_R = self.hold_pose[:3, 3].copy(), self.hold_pose[:3, :3].copy()
            if self._stalled(np.concatenate([leg_pos, 0.05 * leg_R[:, 1]]), 3, 0.0005):
                self.gripper = -1.0
                self._goto("leg_release")
        elif self.phase == "leg_release":
            goal_pos, goal_R = self.hold_pose[:3, 3].copy(), self.hold_pose[:3, :3].copy()
            if self._opened(s):
                self._goto("release_up")
        elif self.phase == "release_up":
            goal_pos, goal_R, rose = self._straight_up(ee_pos)
            if rose:
                self._start_regrasp()

        # ---------------------------------------------------------------- the leg: screw
        elif self.phase == "regrasp_free":
            goal_pos, goal_R, grip, skill_complete = self.leg.fsm_step(s, "square_table_top")
            upstream_scaling = True
            if self.screw_pivot and self.leg.state in self.REGRASP_FREE:
                # the hand's axis along the leg's assumed axis: the upstream targets (straight
                # down onto the leg, hovering above it) tilted with the leg, about its centre
                a, _, _ = self._leg_line(leg_pos, T_top)
                R_tilt = rotate_onto(np.array([0.0, 0.0, 1.0]), a)
                goal_pos = leg_pos + R_tilt @ (goal_pos - leg_pos)
                goal_R = R_tilt @ goal_R
            if self.leg.state not in self.REGRASP_FREE:  # 'screw_align': the leg is in the fingers
                self.gripper = 1.0
                grip = 1.0
                self._leg_yaw_u = leg_yaw(leg_R)
                self._screw_dz = ee_pos[2] - leg_pos[2]  # the grasp height on the leg
                self._goto("screw_pivot" if self.screw_pivot else "screw")
        elif self.phase == "screw_pivot":
            # hand and leg turned together about the hole point, which stays put, until the
            # leg's assumed axis is on the hole axis; then screw
            a, P, h = self._leg_line(leg_pos, T_top)
            angle = float(np.arccos(np.clip(a @ h, -1.0, 1.0)))
            if angle < np.radians(1.0) or self.phase_steps > 40:
                self._leg_yaw_u = leg_yaw(leg_R)
                self._screw_dz = ee_pos[2] - leg_pos[2]
                self._goto("screw")
            else:
                axis = np.cross(a, h)
                axis /= np.linalg.norm(axis)
                step = min(self.gain * angle, self.gain * self.pivot_step)
                R_step = rotvec_to_mat(axis * step)
                goal_pos = P + R_step @ (ee_pos - P)
                goal_R = R_step @ ee_R
        elif self.phase == "screw":
            self._leg_yaw_u = unwrap(self._leg_yaw_u, leg_yaw(leg_R))
            self._yaw_hist.append(self._leg_yaw_u)
            # lead the leg's own yaw clockwise (tightening = -world yaw), keep it upright and
            # on the hole axis, press a little
            if self.screw_upright:
                tgt_R = upright_goal(leg_R, -self.lead)
                tgt = np.array([seat[0], seat[1], leg_pos[2] - self.press])
                goal_pos, goal_R, ep, er = self.servo.step(leg_pos, leg_R, tgt, tgt_R, ee_pos, ee_R)
            else:
                # turn only: the hand yaws clockwise about the vertical through the leg's
                # centre (tightening = -world yaw), tilt left as it is, pressing down a little
                Rz = get_mat([0, 0, 0], [0, 0, -self.gain * self.rot_step])[:3, :3]
                c = np.array([leg_pos[0], leg_pos[1], ee_pos[2]])
                goal_pos = c + Rz @ (ee_pos - c)
                # the press is relative to the leg (the grasp height minus press), so it does
                # not pile up while the thread holds the leg
                goal_pos[2] = leg_pos[2] + self._screw_dz - self.press
                goal_R = Rz @ ee_R
            q7 = float(s.joints[6]) if s.joints is not None else -np.inf
            h = self._yaw_hist
            stalled = len(h) > 8 and (h[-9] - h[-1]) < np.radians(2.0)  # < 2 deg in 8 steps
            at_limit = q7 >= self.leg.Q7_MAX - self.leg.q7_margin
            if at_limit or stalled:
                self.hold_pose = hom(ee_R, ee_pos)
                self.gripper = -1.0
                self._goto("screw_release")
        elif self.phase == "screw_release":
            goal_pos, goal_R = self.hold_pose[:3, 3].copy(), self.hold_pose[:3, :3].copy()
            if self._opened(s):
                self._goto("screw_release_up")
        elif self.phase == "screw_release_up":
            goal_pos, goal_R, rose = self._straight_up(ee_pos)
            if rose:
                self._start_regrasp()

        # ---------------------------------------------------------------- the end
        elif self.phase == "neutral":
            goal_pos = self.hold_pose[:3, 3].copy()
            goal_pos[2] = 0.15
            goal_R = self.hold_pose[:3, :3].copy()
            if abs(ee_pos[2] - 0.15) < 0.01:
                self._goto("done")
                self.done = True
        elif self.phase == "done":
            self.done = True

        grip = float(self.gripper if not upstream_scaling else grip)

        # -- goal -> action
        delta_pos = goal_pos - ee_pos
        if upstream_scaling:  # upstream's get_assembly_action scaling and clamp
            sign = np.sign(delta_pos)
            delta_pos = np.abs(delta_pos) * 2
            delta_pos = np.where(delta_pos > 0.03, 0.03 + (delta_pos - 0.03) * 1.5, delta_pos) * sign
            max_delta = 0.115 * np.ones(3)
            max_delta[2] -= 0.04
            delta_pos = np.clip(delta_pos, -max_delta, max_delta)
        delta_quat = quat_mul(quat_conj(ee_quat), mat_to_quat(goal_R))  # in the EE frame
        return ScriptedAction(
            delta_pos, delta_quat, grip, goal_pos, goal_R, skill_complete, self.state, assembled_flag, self.done
        )

    def _straight_up(self, ee_pos: np.ndarray):
        """The hold pose's x, y and orientation, z ramped up by at most pos_step a step to
        release_dz above it; and whether the hand has got there (within 3 mm)."""
        goal_pos, goal_R = self.hold_pose[:3, 3].copy(), self.hold_pose[:3, :3].copy()
        top = goal_pos[2] + self.release_dz
        goal_pos[2] = min(top, max(ee_pos[2], goal_pos[2]) + self.pos_step)
        return goal_pos, goal_R, ee_pos[2] >= top - 0.003

    def _start_regrasp(self) -> None:
        """Upstream-style free-hand regrasp states of SquareTableLegPart: hover over the leg
        with the hand wound back and face-aligned, lower, close."""
        self.leg._state = "pre_grasp"
        self.leg.first_setting_target = True
        self.leg.gripper_action = -1
        self.leg.prev_cnt = self.leg.curr_cnt
        self._goto("regrasp_free")
