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
