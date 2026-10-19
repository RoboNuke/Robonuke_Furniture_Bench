"""The object-servo policy against an ideal kinematic world: the hand closes a fixed
fraction of its relative command each step (the OSC's ~20 %), a held part moves rigidly with
it, the leg rides a helix once its bolt is in the hole (pitch 9 mm/turn, stop 9 mm above the
seat), joint 7 follows the hand's yaw."""

import numpy as np

from furniture_lab import scripted_policy as sp
from furniture_lab import servo_policy as sv
from furniture_lab.tasks.furniture import frames, square_table as st

PITCH = 0.009
STOP = 0.009
OSC_FRACTION = 0.2  # of a relative pose command, closed per step


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
        # the controller: a fraction of the commanded delta (the action, not the goal)
        pos = old[:3, 3] + OSC_FRACTION * a.delta_pos
        dq = a.delta_quat  # in the EE frame
        rv = sp.quat_to_axisangle(dq) if hasattr(sp, "quat_to_axisangle") else sv.mat_to_rotvec(sp.quat_to_mat(dq))
        R = old[:3, :3] @ sv.rotvec_to_mat(OSC_FRACTION * rv)
        new = sp.hom(R, pos)
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
                # the walls: the front and left obstacles' inner faces stop a square top
                hw = 0.08125
                x_wall = frames.OBSTACLE_FRONT_POS[0] - frames.OBSTACLE_FRONT_SIZE[0] / 2
                y_wall = frames.OBSTACLE_SIDE_POS["obstacle_left"][1] - frames.OBSTACLE_SIDE_SIZE[1] / 2
                T_new[0, 3] = min(T_new[0, 3], x_wall - hw)
                T_new[1, 3] = min(T_new[1, 3], y_wall - hw)
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
    # the top ends against both walls (the goal is 2 mm inside them), square
    top = w.parts["square_table_top"]
    hw = 0.08125
    x_wall = frames.OBSTACLE_FRONT_POS[0] - frames.OBSTACLE_FRONT_SIZE[0] / 2
    y_wall = frames.OBSTACLE_SIDE_POS["obstacle_left"][1] - frames.OBSTACLE_SIDE_SIZE[1] / 2
    assert abs(top[0, 3] - (x_wall - hw)) < 0.0005 and abs(top[1, 3] - (y_wall - hw)) < 0.0005
    _, goal_R = pol._corner_goal(w.state())
    assert abs(sv.yaw_of(top[:3, :3]) - sv.yaw_of(goal_R)) < np.radians(3)
    # the phases, in order, with at least one regrasp (one wrist range is < a full turn)
    want = ["top_free", "top_servo", "top_release", "top_up", "leg_free", "leg_lift", "leg_upright", "leg_over",
            "leg_lower", "leg_settle", "leg_release", "release_up", "regrasp_free", "screw", "screw_release", "screw_release_up",
            "regrasp_free", "screw"]
    assert phases[: len(want)] == want, phases
    assert phases[-2:] == ["neutral", "done"]
    # the leg went in upright and ended seated
    leg = w.parts["square_table_leg4"]
    assert sv.leg_tilt_deg(leg[:3, :3]) < 2.0
    assert abs(leg[2, 3] - w.seat[2]) < 0.001


def test_object_servo_orbits_the_tcp_about_the_object():
    servo = sv.ObjectServo(kp_pos=1.0, kp_rot=1.0, max_cmd_pos=0.01, max_cmd_rot_deg=5.0)
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


def test_corner_goal_follows_moved_obstacles():
    """The top is pushed into wherever the obstacles are this episode (furniture.obstacle_offset)."""
    w = IdealWorld()
    pol = sv.ServoOneLegPolicy(seed=0)
    goal0, _ = pol._corner_goal(w.state())
    shift = np.array([0.013, -0.017, 0.0])
    w.obst = {k: v + shift for k, v in w.obst.items()}
    goal1, _ = pol._corner_goal(w.state())
    np.testing.assert_allclose(goal1[:2] - goal0[:2], shift[:2], atol=1e-12)


def test_screw_pivot_turns_about_the_hole_point_toward_the_hole_axis():
    """--screw_pivot: with the leg leaning in its hole, the hand turns about the hole's entry
    point (held still) so the leg's centre-to-hole line comes onto the hole axis."""
    w = IdealWorld()
    pol = sv.ServoOneLegPolicy(seed=0, screw_pivot=True, gain=1.0)
    top = w.parts["square_table_top"]
    P, h = pol._hole_point(top)
    # the leg leaning 10 deg about the hole point, the hand at its centre, fingers closed
    lean = sv.rotvec_to_mat(np.cross(h, [1.0, 0.0, 0.0]) / np.linalg.norm(np.cross(h, [1.0, 0.0, 0.0])) * np.radians(10))
    leg_c = P + lean @ (h * 0.03)
    w.parts["square_table_leg4"] = sp.hom(lean @ sv.LEG_UP, leg_c)
    w.ee = sp.hom(lean @ sp.rot_mat((np.pi, 0, 0)), leg_c)
    w.width = 0.03
    pol.phase, pol.gripper = "screw_pivot", 1.0
    a = pol.act(w.state())
    a0, _, _ = pol._leg_line(leg_c, top)
    # the commanded hand pose stays at the grasp's distance from P ...
    assert abs(np.linalg.norm(a.goal_pos - P) - np.linalg.norm(leg_c - P)) < 1e-9
    # ... also from a hand that has sunk 5 mm toward P: the plan keeps the grasp radius
    w.ee = sp.hom(w.ee[:3, :3], leg_c - 0.005 * (leg_c - P) / np.linalg.norm(leg_c - P))
    b = pol.act(w.state())
    assert abs(np.linalg.norm(b.goal_pos - P) - np.linalg.norm(leg_c - P)) < 1e-9
    # ... and the line from P to it is closer to the hole axis than the leg's was
    new = (a.goal_pos - P) / np.linalg.norm(a.goal_pos - P)
    assert np.arccos(np.clip(new @ h, -1, 1)) < np.arccos(np.clip(a0 @ h, -1, 1)) - 1e-3
    assert a.gripper > 0  # still holding


def test_rotate_onto():
    for u, v in (([0, 0, 1], [0, 1, 0]), ([1, 0, 0], [1, 0, 0]), ([0, 0, 1], [0.1, 0.2, 0.97])):
        u, v = np.asarray(u, float), np.asarray(v, float) / np.linalg.norm(v)
        np.testing.assert_allclose(sv.rotate_onto(u, v) @ u, v, atol=1e-9)


def test_servo_policy_with_screw_pivot_still_assembles_in_the_ideal_world():
    w = IdealWorld()
    pol = sv.ServoOneLegPolicy(seed=0, screw_pivot=True)
    phases = []
    for i in range(1500):
        a = pol.act(w.state())
        if not phases or phases[-1] != pol.phase:
            phases.append(pol.phase)
        if a.done:
            break
        w.step(a)
    assert a.done and pol.assembled(w.state()), phases
    assert "screw_pivot" in phases


def test_pivot_dz_raises_the_same_arc():
    """--pivot_dz: once ramped in, the target is the dz=0 arc's, raised dz along the hole axis."""
    def target(dz):
        w = IdealWorld()
        pol = sv.ServoOneLegPolicy(seed=0, screw_pivot=True, gain=1.0, pivot_dz=dz)
        top = w.parts["square_table_top"]
        P, h = pol._hole_point(top)
        ax = np.cross(h, [1.0, 0.0, 0.0])
        lean = sv.rotvec_to_mat(ax / np.linalg.norm(ax) * np.radians(10))
        c = P + lean @ (h * 0.03)
        w.parts["square_table_leg4"] = sp.hom(lean @ sv.LEG_UP, c)
        w.ee = sp.hom(lean @ sp.rot_mat((np.pi, 0, 0)), c)
        w.width = 0.03
        pol.phase, pol.gripper = "screw_pivot", 1.0
        pol.act(w.state())  # the plan, at the grasp
        pol.phase_steps = 10  # past the ramp, the hand raised with it
        w.ee = sp.hom(w.ee[:3, :3], c + dz * h)
        return pol.act(w.state()).goal_pos, h

    g0, h = target(0.0)
    g3, _ = target(0.003)
    np.testing.assert_allclose(g3 - g0, 0.003 * h, atol=2e-4)
