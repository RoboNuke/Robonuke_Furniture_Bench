"""one_leg's phase tracker's pure parts, with no sim: phases only move forward (and through
several in one step), how an episode ended, the corner goal, when the gripper holds and lets go
of the top (Corner's check), and the leg-in-hole geometry."""

import math

import pytest
import torch

from furniture_lab.failures import CORNER_CLEARANCE, TOP_HALF_WIDTH
from furniture_lab.tasks.assembly.phases import (
    CORNER, FAILURE, GRASP, INSERT, SCREW, SUCCESS, TIMEOUT, advance, corner_done, corner_error, end_code, holds_top,
    LEG_TIP, leg_in_hole, leg_tip_off_axis, ori_error_deg,
)
from furniture_lab.tasks.furniture.frames import OBSTACLE_FRONT_SIZE, OBSTACLE_SIDE_SIZE

T, F = True, False


def b(*v):
    return torch.tensor(v, dtype=torch.bool)


def quat_axis_angle(axis, deg):
    a = torch.tensor(axis, dtype=torch.float32)
    a = a / a.norm()
    h = math.radians(deg) / 2
    return torch.cat([torch.tensor([math.cos(h)]), a * math.sin(h)]).unsqueeze(0)


def test_phases_only_move_forward_one_check_at_a_time():
    phase = torch.tensor([CORNER, CORNER, GRASP, INSERT, SCREW, GRASP])
    out = advance(phase, b(T, F, T, T, T, F), b(F, T, T, F, T, F), b(F, F, F, T, T, T))
    # corner -> grasp; a grasp check alone does nothing in corner; grasp -> insert; insert ->
    # screw; screw stays; a later phase's check never skips the current one
    assert out.tolist() == [GRASP, CORNER, INSERT, SCREW, SCREW, GRASP]


def test_checks_that_pass_together_go_through_several_phases_in_one_step():
    out = advance(torch.tensor([CORNER, CORNER]), b(T, T), b(T, T), b(T, F))
    assert out.tolist() == [SCREW, INSERT]


def test_an_episode_ends_in_success_over_timeout_over_failure():
    out = end_code(b(T, T, F, F), b(T, F, T, F))
    assert out.tolist() == [SUCCESS, SUCCESS, TIMEOUT, FAILURE]


def test_the_corner_goal_sits_inside_both_walls_wherever_the_obstacles_are():
    front_x, left_y = torch.tensor([0.70, 0.72]), torch.tensor([0.20, 0.18])
    goal = torch.stack(
        [
            front_x - OBSTACLE_FRONT_SIZE[0] / 2 - TOP_HALF_WIDTH - CORNER_CLEARANCE,
            left_y - OBSTACLE_SIDE_SIZE[1] / 2 - TOP_HALF_WIDTH - CORNER_CLEARANCE,
        ],
        dim=-1,
    )
    assert corner_error(goal, front_x, left_y).tolist() == pytest.approx([0.0, 0.0], abs=1e-7)
    off = goal + torch.tensor([[0.003, 0.004], [0.0, -0.01]])
    assert corner_error(off, front_x, left_y).tolist() == pytest.approx([0.005, 0.01], abs=1e-6)


def test_the_gripper_holds_the_top_by_its_edge_closed_near_and_low():
    top = torch.tensor([[0.60, 0.10, 0.0]]).expand(5, 3)
    tcp = torch.tensor([
        [0.68, 0.10, 0.001],  # its edge, 80 mm out at its height: held
        [0.68, 0.10, 0.001],  # the same, fingers open
        [0.75, 0.10, 0.001],  # 150 mm out: another part
        [0.68, 0.10, 0.050],  # 50 mm above it
        [0.68, 0.10, 0.001],  # just past the width
    ])
    width = torch.tensor([0.007, 0.065, 0.007, 0.007, 0.021])
    assert holds_top(width, tcp, top, 0.02, 0.13, 0.03).tolist() == [T, F, F, F, F]


def test_corner_is_done_when_the_gripper_lets_go_of_the_top_it_held():
    held, opened = b(T, T, F, F), b(T, F, T, F)
    # let go of the top; still holding it; open without ever holding it; neither
    assert corner_done(held, opened).tolist() == [T, F, F, F]


# the OBJ frame: y is the seat axis, x across it
AXIS, REF = torch.tensor([0.0, 1.0, 0.0]), torch.tensor([1.0, 0.0, 0.0])
HOLE = torch.tensor([-0.05625, 0.046875, -0.05625])
IDENTITY = torch.tensor([[1.0, 0.0, 0.0, 0.0]])


def test_the_seated_leg_is_on_axis_at_the_seat_upright_and_unturned():
    xz, height, tilt, angle = leg_in_hole(HOLE.unsqueeze(0), IDENTITY, HOLE, AXIS, REF)
    assert (xz.item(), height.item(), tilt.item(), angle.item()) == pytest.approx((0.0, 0.0, 0.0, 0.0), abs=1e-5)
    assert ori_error_deg(IDENTITY).item() == pytest.approx(0.0, abs=1e-3)


def test_offsets_along_and_across_the_seat_axis_are_height_and_off_axis_distance():
    pos = (HOLE + torch.tensor([0.003, 0.005, -0.004])).unsqueeze(0)
    xz, height, tilt, _ = leg_in_hole(pos, IDENTITY, HOLE, AXIS, REF)
    assert xz.item() == pytest.approx(0.005, abs=1e-6)
    assert height.item() == pytest.approx(0.005, abs=1e-6)
    assert tilt.item() == pytest.approx(0.0, abs=1e-3)


def test_a_lean_is_tilt_and_a_turn_about_the_seat_axis_is_angle():
    _, _, tilt, angle = leg_in_hole(HOLE.unsqueeze(0), quat_axis_angle([1, 0, 0], 20.0), HOLE, AXIS, REF)
    assert tilt.item() == pytest.approx(20.0, abs=1e-3)
    assert angle.item() == pytest.approx(0.0, abs=1e-5)  # a lean about x does not turn x

    q = quat_axis_angle([0, 1, 0], 30.0)
    _, _, tilt, angle = leg_in_hole(HOLE.unsqueeze(0), q, HOLE, AXIS, REF)
    assert tilt.item() == pytest.approx(0.0, abs=1e-3)
    assert math.degrees(angle.item()) == pytest.approx(30.0, abs=1e-3)
    assert ori_error_deg(q).item() == pytest.approx(30.0, abs=1e-3)  # x and z each 30 deg off


def test_the_tip_is_off_axis_by_the_shift_plus_the_lean_swinging_it_about_the_origin():
    # upright, shifted across the axis: the tip is off by the shift; along it: not at all
    shifted = (HOLE + torch.tensor([0.004, 0.02, 0.0])).unsqueeze(0)
    assert leg_tip_off_axis(shifted, IDENTITY, HOLE, AXIS).item() == pytest.approx(0.004, abs=1e-6)
    # leaning 20 deg about x with the origin on the axis: the tip (LEG_TIP along -axis) swings off by LEG_TIP sin 20
    off = leg_tip_off_axis(HOLE.unsqueeze(0), quat_axis_angle([1, 0, 0], 20.0), HOLE, AXIS)
    assert off.item() == pytest.approx(LEG_TIP * math.sin(math.radians(20.0)), abs=1e-6)
    # the origin moved by the tip's swing the other way (R_x(20) takes -axis to (0, -cos, -sin)): the tip is back on it
    lean = quat_axis_angle([1, 0, 0], 20.0)
    back = (HOLE + torch.tensor([0.0, 0.0, LEG_TIP * math.sin(math.radians(20.0))])).unsqueeze(0)
    assert leg_tip_off_axis(back, lean, HOLE, AXIS).item() == pytest.approx(0.0, abs=1e-6)


def test_offline_leg_measures_match_the_trackers_geometry():
    from furniture_lab.tasks.assembly.phases import leg_measures_from_states

    states = torch.zeros(3, 63)
    for p in range(5):
        states[:, 16 + 9 * p + 3 : 16 + 9 * p + 9] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    # top at the origin, leg 4 (index 4) at the hole, then 10 mm above it, then 3 mm across (OBJ frame: y the axis)
    hole = torch.tensor([-0.05625, 0.046875, -0.05625])
    states[:, 16 + 9 * 4 : 16 + 9 * 4 + 3] = hole + torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.01, 0.0], [0.003, 0.0, 0.0]])
    m = leg_measures_from_states(states, mesh_frame="obj")
    assert m["phase/height_mm"].tolist() == pytest.approx([0.0, 10.0, 0.0], abs=1e-3)
    assert m["phase/tip_mm"].tolist() == pytest.approx([0.0, 0.0, 3.0], abs=1e-3)
    assert m["phase/tilt_deg"].tolist() == pytest.approx([0.0, 0.0, 0.0], abs=1e-2)


def test_offline_phases_stay_in_corner_without_a_grasp_and_take_only_obs_shaped_input():
    from furniture_lab.tasks.assembly.phases import phases_from_states

    states = torch.zeros(20, 63)
    states[:, 3:9] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])  # identity EE and parts
    for p in range(5):
        states[:, 16 + 9 * p + 3 : 16 + 9 * p + 9] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    states[:, 15] = 0.08  # fingers wide open: never holds the top
    states[:, 0:3] = torch.tensor([0.5, 0.0, 0.5])  # the TCP far from every part
    assert phases_from_states(states, mesh_frame="obj").tolist() == [CORNER] * 20
    with pytest.raises(ValueError, match="63"):
        phases_from_states(torch.zeros(5, 61), mesh_frame="obj")


def test_the_no_progress_check_counts_still_steps_per_phase_and_stalls_at_the_limit():
    """furniture.stall_steps_*: a step on which leg 4 moves at most stall_pos_mm and turns at most stall_rot_deg since
    the last step is still; a bigger step, or a new phase, ends the run; the limit ends the episode (stalled)."""
    from types import SimpleNamespace

    import torch

    from furniture_lab.tasks.assembly.phases import INSERT, SCREW, GRASP, OneLegPhases, PHASES

    p = object.__new__(OneLegPhases)
    p.cfg = SimpleNamespace(stall_pos_mm=0.5, stall_rot_deg=1.0)
    p.stall_limit = torch.zeros(len(PHASES), dtype=torch.long)
    p.stall_limit[INSERT], p.stall_limit[SCREW] = 3, 5
    p.prev_pos, p.prev_quat = torch.zeros(2, 3), torch.tensor([[1.0, 0, 0, 0]] * 2)
    p.still_steps = torch.zeros(2, dtype=torch.long)
    p.prev_phase = torch.full((2,), -1, dtype=torch.long)
    p.phase = torch.tensor([INSERT, GRASP])
    quat = torch.tensor([[1.0, 0, 0, 0]] * 2)
    pos = torch.zeros(2, 3)
    assert p._stall(pos, quat).tolist() == [False, False]  # the phase's first step
    assert p._stall(pos + 0.0002, quat).tolist() == [True, False]  # 0.35 mm this step; Grasp has no limit
    assert p._stall(pos + 0.0004, quat).tolist() == [True, False]  # step to step: slow drift stays still
    assert not p.stalled.any()
    assert p._stall(pos + 0.0006, quat).tolist() == [True, False] and p.stalled.tolist() == [True, False]  # 3 still
    assert p._stall(pos + 0.0016, quat).tolist() == [False, False]  # 1.7 mm this step: progress
    assert p.still_steps.tolist() == [0, 0] and not p.stalled.any()
    turned = torch.tensor([[0.99995, 0.0, 0.0, 0.0075]] * 2)  # about 0.86 deg this step: within 1 deg
    assert p._stall(pos + 0.0016, turned / turned.norm(dim=-1, keepdim=True)).tolist()[0] is True
    p.phase = torch.tensor([SCREW, GRASP])
    assert p._stall(pos + 0.0016, quat).tolist() == [False, False]  # a new phase starts a new run
