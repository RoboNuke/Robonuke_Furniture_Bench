"""Demo variation (furniture_lab/variation.py) against the ideal kinematic world of
test_servo_policy: every drawn expert still assembles, DART bursts perturb only free-space
states and the expert recovers from them, and a zero-width draw is the fixed expert."""

import dataclasses
import inspect

import numpy as np
import pytest

from furniture_lab import demos as D
from furniture_lab import scripted_policy as sp
from furniture_lab import servo_policy as sv
from furniture_lab import variation as vr

from test_servo_policy import IdealWorld

DEFAULTS = {k: v.default for k, v in inspect.signature(sv.ServoOneLegPolicy).parameters.items()}
BASE = {k: DEFAULTS[k] for k in (
    "pos_step", "rot_step_deg", "wind_step_deg", "insert_rate", "carry_dz", "over_dz", "hover_dz",
    "insert_yaw_deg", "press", "release_dz", "corner_clearance", "top_up_z",
)}
BASE["screw_step_deg"] = DEFAULTS["rot_step_deg"]


def run(pol, dart=None, steps=2500):
    w = IdealWorld()
    perturbed_states = set()
    for _ in range(steps):
        a = pol.act(w.state())
        if a.done:
            break
        off = dart.offset(a.state) if dart is not None else None
        if off is not None:
            perturbed_states.add(a.state)
            dpos, dquat = vr.DartBursts.apply(a.delta_pos, a.delta_quat, off)
            a = dataclasses.replace(a, delta_pos=dpos, delta_quat=dquat)
        w.step(a)
    return w, pol, a, perturbed_states


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("pivot", [False, True])
def test_every_drawn_expert_still_assembles(seed, pivot):
    v = vr.sample_variation(np.random.default_rng(seed), BASE)
    w, pol, a, _ = run(sv.ServoOneLegPolicy(seed=seed, screw_pivot=pivot, **v))
    assert a.done and pol.assembled(w.state()), f"phase {pol.phase}, variation {v}"


def test_the_draw_stays_in_its_ranges_and_zero_width_is_the_fixed_expert():
    r = vr.VariationRanges()
    for seed in range(50):
        v = vr.sample_variation(np.random.default_rng(seed), BASE)
        assert r.pos_speed_f[0] <= v["pos_step"] / BASE["pos_step"] <= r.pos_speed_f[1]
        assert 0.030 <= v["over_dz"] <= 0.040
        assert v["corner_clearance"] <= vr.MAX_CORNER_CLEARANCE
        assert all(0.0 <= x <= 0.05 for x in v["push_via"])
    zero = vr.VariationRanges(**{f.name: ((1.0, 1.0) if f.name.endswith("_f") else (0.0, 0.0)) for f in dataclasses.fields(r)})
    v = vr.sample_variation(np.random.default_rng(0), BASE, zero)
    for k, x in BASE.items():
        assert v[k] == pytest.approx(x), k
    assert v["push_via"] == (0.0, 0.0) and v["leg_grasp_along"] == 0.0 and v["regrasp_dz"] == 0.0
    assert vr.sample_variation(np.random.default_rng(3), BASE) == vr.sample_variation(np.random.default_rng(3), BASE)


def test_the_top_grasp_slides_along_its_edge():
    def first_target(along):
        pol = sv.ServoOneLegPolicy(seed=0, noise=False, top_grasp_along=along)
        return pol.act(IdealWorld().state()).goal_pos

    d = first_target(0.03) - first_target(0.0)
    assert np.linalg.norm(d) == pytest.approx(0.03, abs=1e-6)
    assert abs(d[2]) < 1e-9  # along the (horizontal) edge


def test_dart_bursts_only_in_free_space_and_the_expert_recovers():
    dart = vr.DartBursts(np.random.default_rng(0), prob=0.2, steps=4, pos=0.01, rot_deg=3.0)
    w, pol, a, perturbed = run(sv.ServoOneLegPolicy(seed=0), dart)
    assert dart.bursts > 5 and perturbed and perturbed <= vr.DART_STATES
    assert a.done and pol.assembled(w.state())


def test_a_burst_never_carries_into_a_contact_state():
    dart = vr.DartBursts(np.random.default_rng(0), prob=1.0, steps=10, pos=0.01, rot_deg=0.0)
    assert dart.offset("leg_over") is not None
    assert dart.offset("leg_lower") is None  # contact: the burst ends here
    assert dart.left == 0


def test_replay_reads_the_executed_actions_training_the_clean_ones():
    T = 3
    obs = [
        {"robot_state": {"ee_pos": np.zeros(3), "ee_quat": np.array([0, 0, 0, 1.0]), "ee_pos_vel": np.zeros(3),
                         "ee_ori_vel": np.zeros(3), "gripper_width": np.array([0.05])},
         "parts_poses": np.tile([0, 0, 0, 0, 0, 0, 1.0], 5)}
        for _ in range(T + 1)
    ]
    clean = [np.array([0.01, 0, 0, 0, 0, 0, 1, -1.0]) for _ in range(T)]
    sent = [np.array([0.02, 0, 0, 0, 0, 0, 1, -1.0]) for _ in range(T)]
    data = {"observations": obs, "actions": clean, "actions_executed": sent}
    _, labels = D.fb_abs_frames(data)
    _, replay = D.fb_abs_frames(data, executed=True)
    assert labels[0, 0] == pytest.approx(0.01) and replay[0, 0] == pytest.approx(0.02)
    _, old = D.fb_abs_frames({"observations": obs, "actions": clean}, executed=True)  # no DART: the actions
    assert old[0, 0] == pytest.approx(0.01)
