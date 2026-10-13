"""The ported scripted policy, on synthetic states (no sim): first targets, the assembled
check, and the goal -> action conversion."""

import numpy as np

from furniture_lab import scripted_policy as sp
from furniture_lab.tasks.furniture import frames, square_table as st


def _state(ee_pos=(0.5, 0.0, 0.2), width=0.065, q7=0.0):
    parts_tag = {p.name: sp.hom(np.asarray(p.reset_rot_tag)[:3, :3], p.reset_pos_tag) for p in st.PARTS}
    obst = {"obstacle_front": np.array(frames.OBSTACLE_FRONT_POS)}
    obst.update({k: np.array(v) for k, v in frames.OBSTACLE_SIDE_POS.items()})
    joints = np.array(frames.RESET_JOINTS[:7], dtype=float)
    joints[6] = q7
    return sp.ScriptedState(np.array(ee_pos, dtype=float), sp.rot_mat((np.pi, 0, 0)), width, parts_tag, obst, joints)


def test_frames_round_trip():
    assert np.allclose(sp.APRIL_TO_ROBOT @ sp.ROBOT_TO_APRIL, np.eye(4))
    # a part at its tag-frame reset pose maps to this port's reset position
    for part in st.PARTS:
        T_robot = sp.APRIL_TO_ROBOT @ sp.hom(np.asarray(part.reset_rot_tag)[:3, :3], part.reset_pos_tag)
        assert np.allclose(T_robot[:3, 3], st.reset_pose_world(part, "obj")[0], atol=1e-9)


def test_first_goal_is_the_top_edge_nearest_the_robot_gripper_down():
    pol = sp.OneLegScriptedPolicy(seed=0, noise=False)
    a = pol.act(_state())
    assert a.state == "reach_body_grasp_xy"
    top_x = frames.TAG_POS_W[0] + 0.24
    assert abs(a.goal_pos[0] - (top_x - 0.08125)) < 1e-6  # the near edge
    assert abs(a.goal_pos[1]) < 1e-6
    assert a.goal_pos[2] == 0.2  # keeps the EE height for the xy approach
    assert np.allclose(a.goal_rot[:, 2], [0, 0, -1])  # gripper z pointing down
    assert a.gripper == -1  # open
    # delta: upstream doubles the error, then 0.03 + 1.5 * excess, clamped at 0.11..0.12
    err = a.goal_pos[0] - 0.5
    expected = -(0.03 + (abs(err) * 2 - 0.03) * 1.5)
    assert abs(a.delta_pos[0] - expected) < 1e-9


def test_assembled_check_matches_upstream_tolerance():
    pol = sp.OneLegScriptedPolicy(seed=0, noise=False)
    s = _state()
    top = s.parts_tag["square_table_top"]
    s.parts_tag["square_table_leg4"] = top @ sp.get_mat([-0.05625, 0.046875, -0.05625], [0, 0, 0])
    assert pol.assembled(s)
    s.parts_tag["square_table_leg4"] = top @ sp.get_mat([-0.05625, 0.046875 + 0.006, -0.05625], [0, 0, 0])
    assert not pol.assembled(s)  # 6 mm along the screw axis is outside the 5 mm tolerance
    s.parts_tag["square_table_leg4"] = top @ sp.get_mat([-0.05625, 0.046875, -0.05625], [0, np.radians(25), 0])
    assert not pol.assembled(s)  # turned 25 deg about the screw axis


def test_assembled_ends_with_move_neutral_then_done():
    pol = sp.OneLegScriptedPolicy(seed=0, noise=False)
    s = _state(ee_pos=(0.5, 0.0, 0.05))
    top = s.parts_tag["square_table_top"]
    s.parts_tag["square_table_leg4"] = top @ sp.get_mat([-0.05625, 0.046875, -0.05625], [0, 0, 0])
    a = pol.act(s)
    assert a.assembled and a.skill_complete == 1 and not a.done
    a = pol.act(s)
    assert a.state == "move_neutral" and a.goal_pos[2] == 0.15 and not a.done
    s.ee_pos[2] = 0.15
    a = pol.act(s)
    assert a.done


def test_pre_assembly_advances_under_ideal_tracking():
    """Teleport the EE to each goal: the table-top phase must walk its states in order."""
    pol = sp.OneLegScriptedPolicy(seed=0, noise=False)
    s = _state()
    seen = []
    for _ in range(80):
        a = pol.act(s)
        if not seen or seen[-1] != a.state:
            seen.append(a.state)
        s.ee_pos = a.goal_pos.copy()
        s.ee_rot = a.goal_rot.copy()
        s.gripper_width = 0.0 if a.gripper > 0 else 0.065  # an ideal gripper
        if pol.top.pre_assemble_done:
            break
    assert seen[:6] == ["reach_body_grasp_xy", "reach_body_grasp_z", "pick_body", "push", "release", "go_up"]
    assert pol.top.pre_assemble_done


def _yaw(R):
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def _track(s, goal_pos, goal_rot, gripper, leg):
    """Ideal world: the hand reaches its goal; joint 7 follows the hand's yaw (+q7 = -yaw)."""
    dyaw = np.radians(_yaw(goal_rot) - _yaw(s.ee_rot))
    dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
    s.joints[6] -= dyaw
    s.ee_pos, s.ee_rot = goal_pos.copy(), goal_rot.copy()
    s.gripper_width = 0.03 if gripper > 0 else 0.065


def test_regrasp_winds_back_then_screws_with_the_wrist_only():
    """From 'insert' the leg phase hovers over the leg and winds the open hand toward joint 7's
    -limit, lowers square onto it, closes, centres it on the hole, then tightens until joint 7
    nears its +limit -- a ~300 deg turn per grasp -- with the hand staying on the hole axis."""
    pol = sp.OneLegScriptedPolicy(seed=0, noise=False)
    leg = pol.leg
    s = _state(ee_pos=(0.45, 0.03, 0.09), q7=0.3)
    leg._state, leg.first_setting_target = "insert", True
    leg.prev_pose = s.ee_pose  # what insert_wait would have left
    hole = (sp.APRIL_TO_ROBOT @ s.parts_tag["square_table_top"] @ leg.default_assembled_pose)[:3, 3]
    leg_c = (sp.APRIL_TO_ROBOT @ s.parts_tag["square_table_leg4"])[:3, 3]
    seen, q7_at_grasp, screw_steps, max_yaw_step = [], None, 0, 0.0
    for _ in range(400):
        state = leg._state  # the state this step's goal belongs to
        goal_pos, goal_rot, gripper, _ = leg.fsm_step(s, "square_table_top")
        if not seen or seen[-1] != leg._state:
            seen.append(leg._state)
        if state == "pre_grasp":
            assert gripper == -1 and abs(goal_pos[2] - (leg_c[2] + leg.hover_dz)) < 1e-9  # open, hovering
        if state == "screw_grasp" and q7_at_grasp is None:
            q7_at_grasp = s.joints[6]
            # finger pads flat on a face of the square leg: hand yaw = face yaw mod 90 deg
            face = np.degrees(leg._leg_face_yaw(s.parts_tag["square_table_leg4"]))
            assert abs(((_yaw(s.ee_rot) - face) % 90.0 + 45.0) % 90.0 - 45.0) < 1e-6
        if state == "screw":
            screw_steps += 1
            assert np.allclose(goal_pos[:2], hole[:2])  # on the hole axis
            assert np.allclose(goal_rot[:, 2], [0, 0, -1])  # gripper straight down
            assert abs(goal_pos[2] - (s.ee_pos[2] - leg.screw_press)) < 1e-9
            max_yaw_step = max(max_yaw_step, abs(((_yaw(goal_rot) - _yaw(s.ee_rot) + 180) % 360) - 180))
        if leg._state == "release":
            break
        _track(s, goal_pos, goal_rot, gripper, leg)
    assert seen == ["pre_grasp", "grasp_lower", "screw_grasp", "screw_align", "screw", "release"]
    # wound back before the grasp: within one face step (90 deg) of the -limit, never past it
    assert leg.Q7_MIN + leg.q7_margin - 0.06 <= q7_at_grasp < leg.Q7_MIN + leg.q7_margin + np.pi / 2
    assert s.joints[6] > leg.Q7_MAX - leg.q7_margin - 0.06  # tightened to the other end
    assert np.degrees(s.joints[6] - q7_at_grasp) > 200  # one grasp, 200-300 deg of screwing
    assert leg.yaw_step_deg == 4.5  # 90 deg/s at 20 Hz, from scripts/screw_calib.py
    assert max_yaw_step <= leg.yaw_step_deg + 1e-6
    assert screw_steps >= 200 / leg.yaw_step_deg


def test_screw_ends_when_joint7_stops_advancing():
    """A bottomed leg (or slipping fingers): joint 7 does not move -> release after 8 steps."""
    leg = sp.OneLegScriptedPolicy(seed=0, noise=False).leg
    s = _state(ee_pos=(0.45, 0.03, 0.09), q7=-2.5)
    leg._state, leg.first_setting_target, leg.gripper_action = "screw_align", True, 1
    for _ in range(40):
        goal_pos, goal_rot, gripper, _ = leg.fsm_step(s, "square_table_top")
        if leg._state == "release":
            break
        s.ee_pos = goal_pos.copy()  # position tracks, the wrist does not turn
    assert leg._state == "release" and s.joints[6] == -2.5


def test_regrasp_yaw_is_square_to_the_leg_and_nearest_the_limit():
    leg = sp.OneLegScriptedPolicy(seed=0, noise=False).leg
    for leg_yaw_deg in (0.0, 17.0, -40.0, 85.0):
        for q7 in (0.3, -1.0, 2.0):
            s = _state(ee_pos=(0.45, 0.03, 0.09), q7=q7)
            # stand the leg up over the hole, faces at leg_yaw (OBJ y = leg axis, x and z the faces)
            R = sp.rot_mat((np.pi / 2, 0, leg_yaw_deg * np.pi / 180))
            s.parts_tag["square_table_leg4"] = sp.ROBOT_TO_APRIL @ sp.hom(R, np.array([0.45, 0.03, 0.05]))
            face = leg._leg_face_yaw(s.parts_tag["square_table_leg4"])
            goal = leg._regrasp_q7_goal(s, s.ee_pose, s.parts_tag["square_table_leg4"])
            yaw_at_goal = leg._yaw(s.ee_rot) - (goal - q7)
            assert abs(((yaw_at_goal - face) % (np.pi / 2) + np.pi / 4) % (np.pi / 2) - np.pi / 4) < 1e-9
            assert leg.Q7_MIN + leg.q7_margin - 1e-9 <= goal < leg.Q7_MIN + leg.q7_margin + np.pi / 2


def test_insert_lowers_the_held_leg_to_contact_then_settles_then_releases():
    """The leg in the fingers descends with the hand until it stops (the thread stop, here
    9 mm above the seat); the policy must detect the stop, hold, and only then open."""
    pol = sp.OneLegScriptedPolicy(seed=0, noise=False)
    leg = pol.leg
    s = _state(ee_pos=(0.45, 0.03, 0.09))
    top = s.parts_tag["square_table_top"]
    seat = (sp.APRIL_TO_ROBOT @ top @ leg.default_assembled_pose)[:3, 3]
    stop_z = seat[2] + 0.009
    R_up = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])  # upstream's upright leg (OBJ y up)
    leg_pos = np.array([seat[0], seat[1], seat[2] + 0.020])  # held 20 mm up, as upstream leaves it
    s.parts_tag["square_table_leg4"] = sp.ROBOT_TO_APRIL @ sp.hom(R_up, leg_pos)
    s.ee_pos = leg_pos + np.array([0.0, 0.0, 0.0])
    s.gripper_width = 0.03
    leg._state, leg.first_setting_target, leg.gripper_action = "insert_lower", True, 1
    leg.prev_pose = s.ee_pose
    seen, released_at = [], None
    for i in range(80):
        state = leg._state
        goal_pos, goal_rot, gripper, _ = leg.fsm_step(s, "square_table_top")
        if not seen or seen[-1] != leg._state:
            seen.append(leg._state)
        if state == "insert_lower":
            assert gripper == 1
            assert goal_pos[2] < s.ee_pos[2] + 1e-9  # always asking to go down
        if gripper == -1 and released_at is None:
            released_at = (i, leg_pos[2] - seat[2])
            break
        # the world: the hand reaches its goal, the leg follows but cannot go below the stop
        dz = goal_pos[2] - s.ee_pos[2]
        new_leg_z = max(leg_pos[2] + dz, stop_z)
        actual_dz = new_leg_z - leg_pos[2]
        leg_pos = leg_pos + np.array([0.0, 0.0, actual_dz])
        s.ee_pos = s.ee_pos + np.array([0.0, 0.0, actual_dz])
        s.parts_tag["square_table_leg4"] = sp.ROBOT_TO_APRIL @ sp.hom(R_up, leg_pos)
    assert seen[:3] == ["insert_lower", "insert_settle", "insert_release"]
    assert released_at is not None
    assert abs(released_at[1] - 0.009) < 1e-6  # opened with the leg resting on the stop, not 20 mm up
