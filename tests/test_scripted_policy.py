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


def test_insert_screw_turns_about_the_hole_before_release():
    """The twist target is a rotation about the hole's vertical axis (the EE stays on its
    circle around the hole, 5 mm lower, yawed -90 deg), and the state leads to insert_release;
    start_twist 0 skips it (upstream's sequence)."""
    pol = sp.OneLegScriptedPolicy(seed=0, noise=False)
    leg = pol.leg
    assert leg.insert_height == 0.077 and leg.start_twist_deg == 90.0 and leg.insert_press == 0.003
    s = _state(ee_pos=(0.45, 0.03, 0.09))
    hole = (sp.APRIL_TO_ROBOT @ s.parts_tag["square_table_top"] @ leg.default_assembled_pose)[:3, 3]
    leg._state = "insert_screw"
    leg.first_setting_target = True
    leg.gripper_action = 1  # as pick_leg left it
    goal_pos, goal_rot, gripper, _ = leg.fsm_step(s, "square_table_top")
    assert gripper == 1  # the state does not open the hand
    assert abs(np.linalg.norm(s.ee_pos[:2] - hole[:2]) - np.linalg.norm(goal_pos[:2] - hole[:2])) < 1e-9
    assert abs(goal_pos[2] - (s.ee_pos[2] - leg.insert_press)) < 1e-9
    yaw = np.degrees(np.arctan2(goal_rot[1, 0], goal_rot[0, 0]) - np.arctan2(s.ee_rot[1, 0], s.ee_rot[0, 0]))
    assert abs(((yaw + 180) % 360) - 180 + 90.0) < 1e-6
    # reaching the target advances to the release
    s.ee_pos, s.ee_rot = goal_pos.copy(), goal_rot.copy()
    leg.fsm_step(s, "square_table_top")
    assert leg._state == "insert_release"
    # upstream's sequence with the twist off
    leg0 = sp.OneLegScriptedPolicy(seed=0, noise=False, start_twist_deg=0.0).leg
    leg0._state = "insert_wait"
    leg0.first_setting_target = True
    states = []
    for _ in range(12):  # insert_wait has max_len 10
        leg0.fsm_step(s, "square_table_top")
        states.append(leg0._state)
    assert "insert_release" in states and "insert_screw" not in states
