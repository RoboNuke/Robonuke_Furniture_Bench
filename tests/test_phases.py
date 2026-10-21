"""one_leg's phase tracker's pure parts, with no sim: phases only move forward (and through
several in one step), how an episode ended, the corner goal, when the gripper holds and lets go
of the top (Corner's check), and the leg-in-hole geometry."""

import math

import pytest
import torch

from furniture_lab.failures import CORNER_CLEARANCE, TOP_HALF_WIDTH
from furniture_lab.tasks.assembly.phases import (
    CORNER, FAILURE, GRASP, INSERT, SCREW, SUCCESS, TIMEOUT, advance, corner_done, corner_error, end_code, holds_top,
    leg_in_hole, ori_error_deg,
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
