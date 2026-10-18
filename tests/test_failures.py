"""furniture_lab.failures on synthetic attempts: each failure mode lands on its stage, and a
clean attempt passes every test."""

import numpy as np
import pytest

from furniture_lab import failures as F
from furniture_lab.scripted_policy import ROBOT_TO_APRIL
from furniture_lab.tasks.furniture import frames


def _hom(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def _rx(deg):
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


CORNER = np.array([
    frames.OBSTACLE_FRONT_POS[0] - frames.OBSTACLE_FRONT_SIZE[0] / 2 - F.TOP_HALF_WIDTH - F.CORNER_CLEARANCE,
    frames.OBSTACLE_SIDE_POS["obstacle_left"][1] - frames.OBSTACLE_SIDE_SIZE[1] / 2 - F.TOP_HALF_WIDTH - F.CORNER_CLEARANCE,
    0.02,
])
# the top's OBJ frame has y as the seat axis: in the robot frame, y up (z) -> rotate +90 about x
R_TOP = _rx(90)


def attempt(top_off=(0, 0, 0), lift=0.05, release_height=0.01, screw=None, success=False, T=40):
    """grip: top close 2..5, leg close 8..15, screw closes from 18. ``screw(t, top, leg_rel)``
    may edit the top pose / leg-in-top pose after the leg release (t >= 15)."""
    grip = -np.ones(T)
    grip[2:5] = 1
    grip[8:15] = 1
    for c in range(18, T - 2, 6):
        grip[c : c + 3] = 1
    top = _hom(R_TOP, CORNER + np.asarray(top_off))
    leg_floor = _hom(np.eye(3), np.array([0.5, -0.1, 0.02]))
    tops, legs, ee = [], [], []
    for t in range(T + 1):
        top_t = top.copy()
        if t < 8:
            leg = leg_floor
        elif t < 15:
            up = min(t - 8, 4) / 4 * lift
            leg = _hom(np.eye(3), leg_floor[:3, 3] + [0, 0, up])
        else:
            rel = _hom(np.eye(3), F.HOLE + np.array([0.0, release_height, 0.0]))
            if t > 25:  # screwed down by the end
                rel[1, 3] = F.HOLE[1] + (0.0 if success else release_height)
            if screw is not None:
                top_t, rel = screw(t, top_t, rel)
            leg = top_t @ rel
        tops.append(ROBOT_TO_APRIL @ top_t)
        legs.append(ROBOT_TO_APRIL @ leg)
        ee.append(leg[:3, 3] if 8 <= t < 15 else np.array([0.4, 0.0, 0.3]))
    obst = {"obstacle_front": np.asarray(frames.OBSTACLE_FRONT_POS), **{k: np.asarray(v) for k, v in frames.OBSTACLE_SIDE_POS.items()}}
    return F.Attempt("synthetic", success, "", grip, np.stack(ee), np.stack(tops), np.stack(legs), obst)


def stage(a):
    return F.classify(a)["stage"]


def test_a_clean_attempt_is_a_success_that_passes_every_test():
    r = F.classify(attempt(success=True))
    assert r["stage"] == "success" and r["passed_to"] == "screwing: stalled in the hole"


def test_top_not_in_the_corner():
    assert stage(attempt(top_off=(0.03, 0, 0))) == "top_push: top not in the corner"


def test_leg_not_lifted():
    assert stage(attempt(lift=0.005)) == "leg_grasp: not lifted"


def test_not_in_the_hole():
    def off(t, top, rel):
        rel[0, 3] += 0.02
        return top, rel

    assert stage(attempt(screw=off)) == "insertion: not in the hole"


def test_screwing_top_moved_vs_leg_fell_out_vs_stalled():
    def top_slides(t, top, rel):
        if t > 22:
            top = top.copy()
            top[0, 3] += 0.02
        return top, rel

    def leg_tips(t, top, rel):
        if t > 22:
            rel = _hom(_rx(30), rel[:3, 3])
        return top, rel

    assert stage(attempt(screw=top_slides)) == "screwing: top moved"
    assert stage(attempt(screw=leg_tips)) == "screwing: leg fell out"
    assert stage(attempt()) == "screwing: stalled in the hole"


@pytest.mark.parametrize("success", [True, False])
def test_every_verdict_is_a_listed_stage(success):
    assert stage(attempt(success=success)) in F.STAGES
