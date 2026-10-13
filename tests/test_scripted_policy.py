"""The ported scripted policy, on synthetic states (no sim): first targets, the assembled
check, and the goal -> action conversion."""

import numpy as np

from furniture_lab import scripted_policy as sp
from furniture_lab.tasks.furniture import frames, square_table as st


def _state(ee_pos=(0.5, 0.0, 0.2), width=0.065):
    parts_tag = {p.name: sp.hom(np.asarray(p.reset_rot_tag)[:3, :3], p.reset_pos_tag) for p in st.PARTS}
    obst = {"obstacle_front": np.array(frames.OBSTACLE_FRONT_POS)}
    obst.update({k: np.array(v) for k, v in frames.OBSTACLE_SIDE_POS.items()})
    return sp.ScriptedState(np.array(ee_pos, dtype=float), sp.rot_mat((np.pi, 0, 0)), width, parts_tag, obst)


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
    for _ in range(pol.assembled_hold_steps - 1):  # the check has to hold, not just fire once
        a = pol.act(s)
        assert not a.assembled
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
    return np.degrees(np.arctan2(R[1, 0], R[0, 0]))


def _run_turn(leg, state, s, total_deg, press, max_steps=40):
    """Drive a turn state with ideal tracking; return (ee poses visited, steps, next state)."""
    leg._state = state
    leg.first_setting_target = True
    leg.gripper_action = 1
    poses = [sp.hom(s.ee_rot, s.ee_pos)]
    for i in range(max_steps):
        goal_pos, goal_rot, gripper, _ = leg.fsm_step(s, "square_table_top")
        assert gripper == 1
        s.ee_pos, s.ee_rot = goal_pos.copy(), goal_rot.copy()
        poses.append(sp.hom(s.ee_rot, s.ee_pos))
        if leg._state != state:
            return poses, i + 1, leg._state
    raise AssertionError("turn never finished")


def test_turns_are_about_the_hole_axis_in_steps():
    """insert_screw (-start_twist) and screw (-95 deg) turn the EE about the vertical axis
    through the hole: every visited pose keeps its distance from the hole axis, is pressed
    down, and no single step exceeds turn_step_deg; the total yaw reaches the target."""
    pol = sp.OneLegScriptedPolicy(seed=0, noise=False)
    leg = pol.leg
    assert leg.insert_height == 0.077 and leg.start_twist_deg == 90.0 and leg.insert_press == 0.003
    for state, total, press, nxt in (("insert_screw", -90.0, 0.003, "insert_release"), ("screw", -95.0, 0.005, "release")):
        s = _state(ee_pos=(0.45, 0.03, 0.09))
        hole = (sp.APRIL_TO_ROBOT @ s.parts_tag["square_table_top"] @ leg.default_assembled_pose)[:3, 3]
        r0 = np.linalg.norm(s.ee_pos[:2] - hole[:2])
        poses, n, after = _run_turn(leg, state, s, total, press)
        assert after == nxt
        for a, b in zip(poses, poses[1:]):
            assert abs(np.linalg.norm(b[:2, 3] - hole[:2]) - r0) < 1e-9  # on the circle around the hole
            assert abs(b[2, 3] - (a[2, 3] - press)) < 1e-9  # pressed down each step
            d = _yaw(b[:3, :3]) - _yaw(a[:3, :3])
            assert abs(((d + 180) % 360) - 180) <= leg.turn_step_deg + 1e-6
        total_yaw = ((_yaw(poses[-1][:3, :3]) - _yaw(poses[0][:3, :3]) + 180) % 360) - 180
        assert abs(total_yaw - total) < 2.0
        assert n >= abs(total) / leg.turn_step_deg  # fed in steps, not at once
    # upstream's sequence with the twist off
    leg0 = sp.OneLegScriptedPolicy(seed=0, noise=False, start_twist_deg=0.0).leg
    leg0._state = "insert_wait"
    leg0.first_setting_target = True
    s = _state(ee_pos=(0.45, 0.03, 0.09))
    states = []
    for _ in range(12):  # insert_wait has max_len 10
        leg0.fsm_step(s, "square_table_top")
        states.append(leg0._state)
    assert "insert_release" in states and "insert_screw" not in states


def test_turn_position_never_runs_ahead_of_a_lagging_orientation():
    """Position tracks perfectly, orientation only gets halfway each step: the angle of the
    hand around the hole axis must stay within one turn step of the hand's actual yaw."""
    leg = sp.OneLegScriptedPolicy(seed=0, noise=False).leg
    s = _state(ee_pos=(0.45, 0.03, 0.09))
    hole = (sp.APRIL_TO_ROBOT @ s.parts_tag["square_table_top"] @ leg.default_assembled_pose)[:3, 3]
    off0 = s.ee_pos[:2] - hole[:2]
    R0 = s.ee_rot.copy()
    leg._state, leg.first_setting_target, leg.gripper_action = "screw", True, 1
    for _ in range(60):
        goal_pos, goal_rot, _, _ = leg.fsm_step(s, "square_table_top")
        if leg._state != "screw":
            break
        s.ee_pos = goal_pos.copy()
        # halfway toward the goal orientation, about the vertical
        want = _yaw(goal_rot @ R0.T)
        have = _yaw(s.ee_rot @ R0.T)
        s.ee_rot = sp.get_mat([0, 0, 0], [0, 0, np.radians(have + 0.5 * (want - have))])[:3, :3] @ R0
        off = s.ee_pos[:2] - hole[:2]
        around = np.degrees(np.arctan2(off[1], off[0]) - np.arctan2(off0[1], off0[0]))
        around = ((around + 180) % 360) - 180
        assert abs(around - _yaw(s.ee_rot @ R0.T)) <= leg.turn_step_deg + 1e-6
    assert leg._state == "release"
    assert abs(_yaw(s.ee_rot @ R0.T) + 95.0) < 3.0
