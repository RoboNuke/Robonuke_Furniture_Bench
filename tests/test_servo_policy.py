"""The object-servo policy against an ideal kinematic world: the hand reaches its goal each
step, a held part moves rigidly with it, the leg rides a helix once its bolt is in the hole
(pitch 9 mm/turn, stop 9 mm above the seat), joint 7 follows the hand's yaw."""

import numpy as np

from furniture_lab import scripted_policy as sp
from furniture_lab import servo_policy as sv
from furniture_lab.tasks.furniture import frames, square_table as st

PITCH = 0.009
STOP = 0.009


class IdealWorld:
    def __init__(self):
        a2r = sp.APRIL_TO_ROBOT
        self.ee = sp.hom(sp.rot_mat((np.pi, 0, 0)), np.array([0.5, 0.0, 0.2]))
        self.width = 0.065
        self.joints = np.array(frames.RESET_JOINTS[:7], dtype=float)
        # the top at its reset pose; leg 4 lying on the floor so that standing it up by the
        # smallest rotation leaves its x face at yaw 0 (= the top's)
        self.parts = {p.name: a2r @ sp.hom(np.asarray(p.reset_rot_tag)[:3, :3], p.reset_pos_tag) for p in st.PARTS}
        top = self.parts["square_table_top"]
        leg = self.parts["square_table_leg4"]
        R_floor = sp.rot_mat((np.pi / 2, 0, 0)) @ sv.LEG_UP
        self.parts["square_table_leg4"] = sp.hom(R_floor, leg[:3, 3])
        self.held = None
        self.engaged = False
        self.helix_yaw0 = None  # the leg's unwrapped yaw at which it sits STOP above the seat
        self.leg_yaw_u = sv.yaw_of(R_floor)
        self.obst = {"obstacle_front": np.array(frames.OBSTACLE_FRONT_POS)}
        self.obst.update({k: np.array(v) for k, v in frames.OBSTACLE_SIDE_POS.items()})

    @property
    def seat(self):
        """Leg 4's seated origin, from the top's CURRENT pose (the top gets moved)."""
        top = self.parts["square_table_top"]
        return (top @ sp.get_mat(sp.OneLegScriptedPolicy.LEG4_ASSEMBLED_POS, [0, 0, 0]))[:3, 3]

    def state(self):
        r2a = sp.ROBOT_TO_APRIL
        return sp.ScriptedState(
            self.ee[:3, 3].copy(), self.ee[:3, :3].copy(), self.width,
            {k: r2a @ v for k, v in self.parts.items()}, self.obst, self.joints.copy(),
        )

    def step(self, a: sp.ScriptedAction):
        old = self.ee.copy()
        new = sp.hom(a.goal_rot, a.goal_pos)
        dyaw = (sv.yaw_of(new[:3, :3]) - sv.yaw_of(old[:3, :3]) + np.pi) % (2 * np.pi) - np.pi
        self.joints[6] = np.clip(self.joints[6] - dyaw, -2.8973, 2.8973)
        self.ee = new
        tcp = new[:3, 3]
        # the gripper
        if a.gripper > 0:
            near_leg = np.linalg.norm(tcp - self.parts["square_table_leg4"][:3, 3]) < 0.03
            top = self.parts["square_table_top"]
            near_top = np.linalg.norm(tcp[:2] - top[:2, 3]) < 0.12 and abs(tcp[2] - top[2, 3]) < 0.03
            if near_leg:
                self.width, self.held = 0.03, "square_table_leg4"
            elif near_top:
                self.width, self.held = 0.01, "square_table_top"
            else:
                self.width, self.held = 0.0, None
        else:
            self.width, self.held = 0.065, None
        # a held part moves with the hand
        if self.held is not None:
            T = self.parts[self.held]
            T_new = new @ np.linalg.inv(old) @ T
            if self.held == "square_table_top":
                T_new[2, 3] = T[2, 3]  # slides on the table
            self.parts[self.held] = T_new
        # the leg on its thread
        leg = self.parts["square_table_leg4"]
        seat = self.seat
        over = np.linalg.norm(leg[:2, 3] - seat[:2]) < 0.006 and sv.leg_tilt_deg(leg[:3, :3]) < 10.0
        if over and leg[2, 3] <= seat[2] + STOP + 0.0005:
            self.leg_yaw_u = sv.unwrap(self.leg_yaw_u, sv.yaw_of(leg[:3, :3]))
            if not self.engaged:
                self.engaged, self.helix_yaw0 = True, self.leg_yaw_u
            # clockwise (negative yaw) takes it down the helix, never below the seat; the leg
            # sits at the helix height whether held (the hand is compliant) or released
            z_helix = seat[2] + STOP + PITCH * (self.leg_yaw_u - self.helix_yaw0) / (2 * np.pi)
            dz = max(z_helix, seat[2]) - leg[2, 3]
            leg[2, 3] += dz
            if self.held == "square_table_leg4":
                self.ee[2, 3] += dz  # the fingers hold it: the hand rides with the leg
        elif self.held != "square_table_leg4":
            leg[2, 3] = max(leg[2, 3], 0.0)  # on the floor


def test_servo_policy_assembles_in_the_ideal_world():
    w = IdealWorld()
    pol = sv.ServoOneLegPolicy(seed=0)
    phases, last = [], None
    for i in range(1500):
        s = w.state()
        a = pol.act(s)
        if pol.phase != last:
            phases.append(pol.phase)
            last = pol.phase
        if a.done:
            break
        w.step(a)
    assert a.done, f"not done after {i} steps, phase {pol.phase}, phases {phases}"
    assert pol.assembled(w.state())
    # the top was placed, not pushed into the obstacle: at the corner goal, right yaw
    top = w.parts["square_table_top"]
    goal_pos, goal_R = pol._corner_goal(w.state())
    assert np.linalg.norm(top[:2, 3] - goal_pos[:2]) < 0.005
    assert abs(sv.yaw_of(top[:3, :3]) - sv.yaw_of(goal_R)) < np.radians(3)
    # the phases, in order, with at least one regrasp (one wrist range is < a full turn)
    want = ["top_free", "top_servo", "top_release", "top_up", "leg_free", "leg_lift", "leg_upright", "leg_over",
            "leg_lower", "leg_settle", "leg_release", "regrasp_free", "screw", "screw_release", "regrasp_free", "screw"]
    assert phases[: len(want)] == want, phases
    assert phases[-2:] == ["neutral", "done"]
    # the leg went in upright and ended seated
    leg = w.parts["square_table_leg4"]
    assert sv.leg_tilt_deg(leg[:3, :3]) < 2.0
    assert abs(leg[2, 3] - w.seat[2]) < 0.001


def test_object_servo_orbits_the_tcp_about_the_object():
    servo = sv.ObjectServo(max_step_pos=0.01, max_step_rot_deg=5.0)
    obj_pos, obj_R = np.array([0.5, 0.1, 0.05]), np.eye(3)
    ee_pos, ee_R = obj_pos + np.array([0.0, 0.08, 0.0]), sp.rot_mat((np.pi, 0, 0))
    # a pure yaw goal of +30 deg: the TCP must move on the circle around the object
    goal_R = sp.get_mat([0, 0, 0], [0, 0, np.radians(30)])[:3, :3]
    p, R, ep, er = servo.step(obj_pos, obj_R, obj_pos, goal_R, ee_pos, ee_R)
    assert abs(np.linalg.norm(p - obj_pos) - 0.08) < 1e-9  # same radius
    assert abs(ep) < 1e-12 and abs(er - np.radians(30)) < 1e-9
    assert abs(np.degrees(sv.yaw_of(R @ ee_R.T)) - 5.0) < 1e-6  # one saturated step of 5 deg
    # a pure translation goal: passes straight through, saturated
    p, R, _, _ = servo.step(obj_pos, obj_R, obj_pos + np.array([0.05, 0, 0]), obj_R, ee_pos, ee_R)
    assert np.allclose(p - ee_pos, [0.01, 0, 0]) and np.allclose(R, ee_R)
