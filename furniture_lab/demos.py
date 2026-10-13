"""FurnitureBench demonstration files (``.pkl``), read into this port's conventions.

A demo (``furniture_bench/data/data_collector.py``) is a dict::

    observations[t]  {robot_state: {ee_pos (3), ee_quat (4, xyzw), joint_positions (7),
                                    gripper_width (1), ...}, parts_poses (7*P, xyzw, tag frame),
                      color_image1/2 ...}
    actions[t]       (8,) = [dx dy dz, dqx dqy dqz dqw, gripper]   -- the EE delta applied
                      at step t: goal_pos = ee_pos + d, goal_quat = ee_quat (x) dq, i.e. the
                      rotation delta is in the EE's own frame; gripper < 0 opens, else closes
    rewards, skills, success, furniture

``ee_pos``/``ee_quat`` are of ``k_ee_link`` (``panda_hand`` + 0.1034 m) in the robot-base frame,
which is this port's world frame. ``parts_poses`` are in the AprilTag frame and of the OBJ
mesh frames; :class:`Demo` carries them in the robot-base frame and in the catalog's mesh
frame (``uwlab``), ready to write into the scene. Quaternions come out ``(w, x, y, z)``.

Sim demos (``collect_data.py --is-sim --scripted``) run at FurnitureSim's env rate, 20 Hz --
the same as this env's -- so one demo step is one env step.

Pure numpy; no Isaac Lab.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .tasks.furniture import frames
from .tasks.furniture.square_table import MESH_FRAMES, PART_NAMES, mesh_frame

__all__ = ["Demo", "load_demo", "quat_xyzw_to_wxyz", "quat_to_mat", "mat_to_quat"]


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(w, x, y, z) -> 3x3."""
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


mat_to_quat = frames.mat_to_quat_wxyz


def _np(value: Any) -> np.ndarray:
    """A pkl may hold numpy arrays or torch tensors (sim demos are saved from tensors)."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).squeeze()


@dataclass
class Demo:
    path: str
    furniture: str
    success: bool
    #: (T, 3) / (T, 4) EE pose, robot-base frame, wxyz
    ee_pos: np.ndarray
    ee_quat: np.ndarray
    #: (T, 7) arm joints, (T,) finger separation
    joint_positions: np.ndarray
    gripper_width: np.ndarray
    #: (T, P, 3) / (T, P, 4) part poses, robot-base frame, catalog mesh frame, wxyz
    parts_pos: np.ndarray
    parts_quat: np.ndarray
    #: (T, 3) position delta (m), (T, 4) rotation delta in the EE frame (wxyz), (T,) gripper
    #: command as recorded (< 0 opens, else closes)
    act_dpos: np.ndarray
    act_dquat: np.ndarray
    act_grip: np.ndarray
    rewards: np.ndarray
    skills: np.ndarray
    part_names: List[str]
    mesh_frame: str

    @property
    def length(self) -> int:
        return int(self.act_dpos.shape[0])

    def summary(self) -> str:
        return (
            f"{Path(self.path).name}: {self.furniture}, {self.length} steps, success={self.success}, "
            f"reward sum {self.rewards.sum():.0f}, parts {len(self.part_names)}, mesh frame {self.mesh_frame}"
        )


def _parts_to_world(parts_flat: np.ndarray, R_mesh: np.ndarray) -> tuple:
    """(7P,) tag-frame xyzw poses of OBJ-frame bodies -> (P, 3), (P, 4) base-frame USD-frame wxyz."""
    P = parts_flat.shape[0] // 7
    pos = np.zeros((P, 3))
    quat = np.zeros((P, 4))
    for i in range(P):
        p = parts_flat[7 * i : 7 * i + 3]
        q = quat_xyzw_to_wxyz(parts_flat[7 * i + 3 : 7 * i + 7])
        R_obj_tag = quat_to_mat(q)
        pos[i] = np.asarray(frames.TAG_POS_W) + frames.TAG_ROT_W @ p
        quat[i] = mat_to_quat(frames.TAG_ROT_W @ R_obj_tag @ R_mesh.T)
    return pos, quat


def load_demo(path: str | Path, frame: Optional[str] = None) -> Demo:
    """Read one FurnitureBench ``.pkl`` into a :class:`Demo`."""
    path = Path(path)
    with open(path, "rb") as f:
        data: Dict[str, Any] = pickle.load(f)
    frame = frame or mesh_frame()
    R_mesh = MESH_FRAMES[frame]

    obs = data["observations"]
    acts = np.stack([_np(a) for a in data["actions"]])  # (T, 8)
    T = acts.shape[0]
    if len(obs) not in (T, T + 1):
        raise ValueError(f"{path}: {len(obs)} observations for {T} actions")
    obs = obs[: T + 1] if len(obs) == T + 1 else obs

    def rs(key):
        return np.stack([_np(o["robot_state"][key]) for o in obs])

    ee_pos = rs("ee_pos")
    ee_quat = quat_xyzw_to_wxyz(rs("ee_quat"))
    joints = rs("joint_positions")
    width = rs("gripper_width").reshape(len(obs))
    parts = [_parts_to_world(_np(o["parts_poses"]), R_mesh) for o in obs]
    parts_pos = np.stack([p for p, _ in parts])
    parts_quat = np.stack([q for _, q in parts])
    P = parts_pos.shape[1]
    names = PART_NAMES[:P] if P <= len(PART_NAMES) else [f"part{i}" for i in range(P)]

    return Demo(
        path=str(path),
        furniture=str(data.get("furniture", "?")),
        success=bool(data.get("success", False)),
        ee_pos=ee_pos,
        ee_quat=ee_quat,
        joint_positions=joints,
        gripper_width=width,
        parts_pos=parts_pos,
        parts_quat=parts_quat,
        act_dpos=acts[:, :3],
        act_dquat=quat_xyzw_to_wxyz(acts[:, 3:7]),
        act_grip=acts[:, 7],
        rewards=np.asarray([float(_np(r)) for r in data.get("rewards", [0.0] * T)]),
        skills=np.asarray([float(_np(s)) for s in data.get("skills", [0.0] * T)]),
        part_names=names,
        mesh_frame=frame,
    )
