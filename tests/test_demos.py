"""The demo loader: a synthetic FurnitureBench pkl round-trips into this port's frames."""

import pickle

import numpy as np

from furniture_lab import demos
from furniture_lab.tasks.furniture import frames, square_table as st


def _quat_wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]])


def _fake_pkl(path, T=4):
    """Parts at FurnitureBench's own reset poses (tag frame), a moving EE, 8-D actions."""
    parts = []
    for part in st.PARTS:
        q = frames.mat_to_quat_wxyz(part.reset_rot_tag)
        parts.append(np.concatenate([part.reset_pos_tag, _quat_wxyz_to_xyzw(q)]))
    parts_flat = np.concatenate(parts).astype(np.float32)
    obs = []
    for t in range(T + 1):
        obs.append(
            {
                "robot_state": {
                    "ee_pos": np.array([0.5, 0.0, 0.1 + 0.01 * t], dtype=np.float32),
                    "ee_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # xyzw: 180 deg about x
                    "joint_positions": np.array(frames.RESET_JOINTS, dtype=np.float32),
                    "gripper_width": np.array([0.065], dtype=np.float32),
                },
                "parts_poses": parts_flat,
            }
        )
    acts = [np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 1.0, -1.0], dtype=np.float32) for _ in range(T)]
    data = {"observations": obs, "actions": acts, "rewards": [0.0] * (T - 1) + [1.0],
            "skills": [0.0] * T, "success": True, "furniture": "one_leg"}
    with open(path, "wb") as f:
        pickle.dump(data, f)


def test_loader_shapes_and_frames(tmp_path):
    p = tmp_path / "demo.pkl"
    _fake_pkl(p)
    d = demos.load_demo(p, frame="uwlab")
    assert d.length == 4 and d.success and d.furniture == "one_leg"
    assert d.ee_pos.shape == (5, 3) and d.ee_quat.shape == (5, 4)
    assert d.parts_pos.shape == (5, 5, 3) and d.parts_quat.shape == (5, 5, 4)
    assert d.act_dpos.shape == (4, 3) and d.act_dquat.shape == (4, 4) and d.act_grip.shape == (4,)
    # xyzw -> wxyz
    assert np.allclose(d.ee_quat[0], [0.0, 1.0, 0.0, 0.0])
    assert np.allclose(d.act_dquat[0], [1.0, 0.0, 0.0, 0.0])
    assert d.act_grip[0] == -1.0 and d.rewards[-1] == 1.0
    # a part recorded at FurnitureBench's reset pose lands on this port's reset pose
    for i, part in enumerate(st.PARTS):
        pos, quat = st.reset_pose_world(part, "uwlab")
        assert np.allclose(d.parts_pos[0, i], pos, atol=1e-6)
        assert np.allclose(d.parts_quat[0, i], quat, atol=1e-6) or np.allclose(d.parts_quat[0, i], -np.array(quat), atol=1e-6)
    assert d.part_names == st.PART_NAMES


def test_loader_accepts_torch_tensors(tmp_path):
    import torch

    p = tmp_path / "demo.pkl"
    _fake_pkl(p)
    with open(p, "rb") as f:
        data = pickle.load(f)
    for o in data["observations"]:
        o["parts_poses"] = torch.tensor(o["parts_poses"]).unsqueeze(0)  # (1, 35), as the sim env emits
        o["robot_state"] = {k: torch.tensor(v).unsqueeze(0) for k, v in o["robot_state"].items()}
    data["actions"] = [torch.tensor(a) for a in data["actions"]]
    with open(p, "wb") as f:
        pickle.dump(data, f)
    d = demos.load_demo(p, frame="uwlab")
    assert d.parts_pos.shape == (5, 5, 3) and d.gripper_width.shape == (5,)
