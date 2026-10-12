"""The batched assembled check must agree with FurnitureBench's per-env numpy one.

The reference below is ``furniture_bench.utils.pose.is_similar_pose`` and
``Furniture.is_assembled_idx``, re-typed here so the test needs no Isaac Gym.
"""

import numpy as np
import pytest
import torch

from furniture_lab.tasks.furniture import square_table as st
from furniture_lab.tasks.furniture.assembly import (
    AssemblyTracker,
    pairs_assembled,
    quat_mul,
    relative_pose,
    rotation_diag,
)
from furniture_lab.tasks.furniture.frames import mat_to_quat_wxyz, rot_mat


# ----------------------------------------------------------------- upstream reference
def _cosine_sim(w, v):
    return np.dot(w, v) / (np.linalg.norm(w) * np.linalg.norm(v))


def _is_similar_rot(rot1, rot2, ori_bound):
    return all(_cosine_sim(rot1[:, k], rot2[:, k]) >= ori_bound for k in range(3))


def _is_similar_pos(pos1, pos2, thr):
    return all(abs(pos1[k] - pos2[k]) <= thr[k] for k in range(3))


def _pose_mat(pos, quat_wxyz):
    w, x, y, z = quat_wxyz
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def _reference_assembled(pose_i, pose_j, targets, thr, ori_bound):
    rel = np.linalg.inv(_pose_mat(*pose_i)) @ _pose_mat(*pose_j)
    for t in targets:
        T = np.eye(4)
        T[:3, 3] = t
        if _is_similar_rot(T[:3, :3], rel[:3, :3], ori_bound) and _is_similar_pos(
            T[:3, 3], rel[:3, 3], thr
        ):
            return True
    return False


def _random_quat(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


# ------------------------------------------------------------------------------ tests
def test_rotation_diag_matches_matrix():
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = _random_quat(rng)
        R = _pose_mat(np.zeros(3), q)[:3, :3]
        got = rotation_diag(torch.tensor(q, dtype=torch.float64)).numpy()
        assert np.allclose(got, np.diag(R), atol=1e-12)


def test_relative_pose_matches_matrix_inverse():
    rng = np.random.default_rng(1)
    for _ in range(50):
        pa, pb = rng.normal(size=3), rng.normal(size=3)
        qa, qb = _random_quat(rng), _random_quat(rng)
        ref = np.linalg.inv(_pose_mat(pa, qa)) @ _pose_mat(pb, qb)
        rel_pos, rel_quat = relative_pose(
            *(torch.tensor(v, dtype=torch.float64) for v in (pa, qa, pb, qb))
        )
        got = _pose_mat(rel_pos.numpy(), rel_quat.numpy())
        assert np.allclose(got, ref, atol=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_pairs_assembled_agrees_with_upstream(seed):
    """Random poses near and far from every hole, with and without small rotations."""
    rng = np.random.default_rng(seed)
    pairs = st.ONE_LEG_PAIRS
    targets = st.rel_pose_targets(pairs, "obj")  # (1, 4, 3)
    thr = np.array(st.ASSEMBLED_POS_THRESHOLD)

    N = 400
    top_pos = rng.normal(scale=0.3, size=(N, 3))
    top_quat = np.stack([_random_quat(rng) for _ in range(N)])
    parts_pos = np.zeros((N, 5, 3))
    parts_quat = np.tile(np.array([1.0, 0, 0, 0]), (N, 5, 1))
    parts_pos[:, 0] = top_pos
    parts_quat[:, 0] = top_quat
    for n in range(N):
        hole = targets[0, rng.integers(4)]
        # position error: half the samples inside tolerance, half straddling it
        err = rng.uniform(-1.0, 1.0, size=3) * thr * (0.8 if n % 2 == 0 else 1.6)
        rel_pos = hole + err
        # orientation: small rotation, sometimes beyond the 0.94 cosine bound (~20 deg)
        ang = np.radians(rng.uniform(0, 12 if n % 3 else 35))
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        q_rel = np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * axis])
        T_top = _pose_mat(top_pos[n], top_quat[n])
        T_leg = T_top @ _pose_mat(rel_pos, q_rel)
        parts_pos[n, 4] = T_leg[:3, 3]
        parts_quat[n, 4] = mat_to_quat_wxyz(T_leg[:3, :3])

    got = pairs_assembled(
        torch.tensor(parts_pos, dtype=torch.float64),
        torch.tensor(parts_quat, dtype=torch.float64),
        torch.tensor(pairs),
        torch.tensor(targets, dtype=torch.float64),
        torch.tensor(thr, dtype=torch.float64),
        st.ORI_BOUND,
    )
    ref = np.array(
        [
            _reference_assembled(
                (parts_pos[n, 0], parts_quat[n, 0]),
                (parts_pos[n, 4], parts_quat[n, 4]),
                targets[0],
                thr,
                st.ORI_BOUND,
            )
            for n in range(N)
        ]
    )
    assert got.shape == (N, 1)
    assert (got[:, 0].numpy() == ref).all()
    # the sample must actually exercise both outcomes
    assert 0.2 < ref.mean() < 0.8


def test_tracker_latches_and_pays_once():
    pairs = st.ONE_LEG_PAIRS
    tracker = AssemblyTracker(
        3, pairs, st.rel_pose_targets(pairs, "obj"), st.ASSEMBLED_POS_THRESHOLD, st.ORI_BOUND, "cpu"
    )
    pos = torch.zeros(3, 5, 3)
    quat = torch.tensor([1.0, 0, 0, 0]).repeat(3, 5, 1)
    pos[:, 0] = torch.tensor([0.5, 0.0, 0.0])
    # env 1 seated in a hole, env 0 and 2 not
    pos[1, 4] = pos[1, 0] + torch.tensor([0.05625, 0.046875, 0.05625])
    pos[0, 4] = pos[0, 0] + torch.tensor([0.05625, 0.1, 0.05625])
    pos[2, 4] = pos[2, 0] + torch.tensor([0.0, 0.0, 0.0])

    r = tracker.update(pos, quat)
    assert r.tolist() == [0.0, 1.0, 0.0]
    assert tracker.all_assembled().tolist() == [False, True, False]
    # still seated: no second payment; pulled out: stays latched (upstream's assembled_set)
    assert tracker.update(pos, quat).tolist() == [0.0, 0.0, 0.0]
    pos[1, 4, 1] += 0.1
    assert tracker.update(pos, quat).tolist() == [0.0, 0.0, 0.0]
    assert tracker.all_assembled()[1]
    tracker.reset(torch.tensor([1]))
    assert not tracker.all_assembled().any()
    # re-seat after the reset: paid again
    pos[1, 4, 1] -= 0.1
    assert tracker.update(pos, quat).tolist() == [0.0, 1.0, 0.0]


def test_screwed_but_rotated_leg_is_not_assembled():
    """Upstream checks all three axes, so a leg seated at the right height but turned 45 deg
    about the screw axis (the top's y) does not count. Documented, not a choice of this port."""
    pairs = st.ONE_LEG_PAIRS
    targets = torch.tensor(st.rel_pose_targets(pairs, "obj"))  # OBJ frame: screw axis is y
    pos = torch.zeros(1, 5, 3)
    quat = torch.tensor([1.0, 0, 0, 0]).repeat(1, 5, 1)
    pos[0, 4] = torch.tensor([0.05625, 0.046875, 0.05625])
    turned = torch.tensor(mat_to_quat_wxyz(rot_mat((0, np.radians(45), 0))), dtype=torch.float32)
    quat[0, 4] = turned
    thr = torch.tensor(st.ASSEMBLED_POS_THRESHOLD)
    assert not pairs_assembled(pos, quat, torch.tensor(pairs), targets, thr, st.ORI_BOUND)[0, 0]
    quat[0, 4] = torch.tensor(mat_to_quat_wxyz(rot_mat((0, np.radians(15), 0))), dtype=torch.float32)
    assert pairs_assembled(pos, quat, torch.tensor(pairs), targets, thr, st.ORI_BOUND)[0, 0]


def test_quat_mul_is_rotation_composition():
    rng = np.random.default_rng(3)
    for _ in range(20):
        qa, qb = _random_quat(rng), _random_quat(rng)
        Ra, Rb = _pose_mat(np.zeros(3), qa)[:3, :3], _pose_mat(np.zeros(3), qb)[:3, :3]
        q = quat_mul(torch.tensor(qa), torch.tensor(qb)).numpy()
        assert np.allclose(_pose_mat(np.zeros(3), q)[:3, :3], Ra @ Rb, atol=1e-10)
