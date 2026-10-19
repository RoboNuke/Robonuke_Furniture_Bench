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

__all__ = [
    "Demo", "load_demo", "quat_xyzw_to_wxyz", "quat_to_mat", "mat_to_quat",
    "mat_to_rot6d", "rot6d_to_mat", "resip_unscale_rotation", "FB_STATE_KEYS", "fb_state", "fb_abs_state", "fb_abs_action", "fb_abs_frames", "obstacle_tag_rot6d",
]

#: FurnitureBench's DEFAULT_STATE_OBS keys of ``robot_state``, in order (then ``parts_poses``)
FB_STATE_KEYS = ("ee_pos", "ee_quat", "ee_pos_vel", "ee_ori_vel", "gripper_width")


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
    """Read one FurnitureBench ``.pkl`` into a :class:`Demo`. Its actions are the ones the env
    was given: ``actions_executed`` when the pkl has them (a DART collection), else ``actions``."""
    path = Path(path)
    with open(path, "rb") as f:
        data: Dict[str, Any] = pickle.load(f)
    frame = frame or mesh_frame()
    R_mesh = MESH_FRAMES[frame]

    obs = data["observations"]
    acts = np.stack([_np(a) for a in data.get("actions_executed") or data["actions"]])  # (T, 8)
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


# ------------------------------------------------------------ the -FBAbs-v0 interface, NumPy
def mat_to_rot6d(R: np.ndarray) -> np.ndarray:
    """The 6-d rotation (Zhou et al.): the matrix's first two rows, flattened. The NumPy twin
    of ``furniture_lab/tasks/assembly/mdp/rot6d.py``."""
    R = np.asarray(R, dtype=np.float64)
    return R[:2, :].reshape(6)


def rot6d_to_mat(d6: np.ndarray) -> np.ndarray:
    d6 = np.asarray(d6, dtype=np.float64)
    b1 = d6[0:3] / np.linalg.norm(d6[0:3])
    b2 = d6[3:6] - np.dot(b1, d6[3:6]) * b1
    b2 = b2 / np.linalg.norm(b2)
    return np.stack((b1, b2, np.cross(b1, b2)))


def _f32(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def fb_state(obs: Dict[str, Any]) -> np.ndarray:
    """One pkl observation -> FurnitureBench's 49-d state (the ``-FB-v0`` observation)."""
    return np.concatenate([_f32(obs["robot_state"][k]) for k in FB_STATE_KEYS] + [_f32(obs["parts_poses"])])


def fb_abs_state(obs: Dict[str, Any]) -> np.ndarray:
    """One pkl observation -> the ``-FBAbs-v0`` observation (61): every orientation as rot6d.

    ``[ee_pos 3, ee_rot6d 6, ee_pos_vel 3, ee_ori_vel 3, gripper_width 1, parts 5 x (pos 3, rot6d 6)]``
    """
    rs = obs["robot_state"]
    ee_r6 = mat_to_rot6d(quat_to_mat(quat_xyzw_to_wxyz(_f32(rs["ee_quat"]))))
    parts = _f32(obs["parts_poses"]).reshape(-1, 7)
    parts_out = [np.concatenate([p[:3], mat_to_rot6d(quat_to_mat(quat_xyzw_to_wxyz(p[3:7])))]) for p in parts]
    return np.concatenate(
        [_f32(rs["ee_pos"]), ee_r6, _f32(rs["ee_pos_vel"]), _f32(rs["ee_ori_vel"]), _f32(rs["gripper_width"])] + parts_out
    ).astype(np.float32)


def fb_abs_action(obs: Dict[str, Any], action: Any) -> np.ndarray:
    """The pkl's delta action at a frame -> the absolute target it commanded (10):
    ``[goal_pos 3 (base frame), goal_rot6d 6, gripper]`` with ``goal_pos = ee_pos + dpos`` and
    ``goal_R = R(ee_quat) @ R(dq)`` (the recorded rotation delta is in the EE's own frame);
    the gripper command as recorded (FurnitureBench: > 0 closes)."""
    a = _f32(action)
    rs = obs["robot_state"]
    goal_pos = _f32(rs["ee_pos"]) + a[0:3]
    R_ee = quat_to_mat(quat_xyzw_to_wxyz(_f32(rs["ee_quat"])))
    R_goal = R_ee @ quat_to_mat(quat_xyzw_to_wxyz(a[3:7]))
    return np.concatenate([goal_pos, mat_to_rot6d(R_goal), a[7:8]]).astype(np.float32)


def obstacle_tag_rot6d(front_pos_robot: Any) -> np.ndarray:
    """The front obstacle at ``front_pos_robot`` (robot-base frame; the box is never rotated)
    -> its tag-frame ``[pos 3, rot6d 6]`` (9): the ``obstacle_pose`` observation of the ResiP
    task (``mdp.obstacle_pose_tag_rot6d``'s NumPy twin)."""
    R_t = frames.TAG_ROT_W.T
    pos_t = R_t @ (np.asarray(front_pos_robot, dtype=np.float64) - np.asarray(frames.TAG_POS_W))
    return np.concatenate([pos_t, mat_to_rot6d(R_t)]).astype(np.float32)


def fb_abs_frames(data: Dict[str, Any], obstacle: bool = False, executed: bool = False) -> tuple:
    """A loaded pkl -> ``(states (T+1, 61), actions (T, 10))`` of the ``-FBAbs-v0`` interface
    (the last observation is repeated when the pkl holds none after the last action).

    ``obstacle``: append the front obstacle's pose (``obstacle_tag_rot6d``) to every state, 70
    wide (the ResiP task's layout), from the pkl's ``obstacles_robot`` -- or, for a pkl recorded
    before the obstacles moved, their fixed pose ``frames.OBSTACLE_FRONT_POS``.

    ``executed``: the actions the env was given (``actions_executed``, a DART collection's
    perturbed commands; ``actions`` when the pkl has none) -- for replaying the episode. The
    default is ``actions``: the expert's own commands, the labels BC trains on."""
    obs = data["observations"]
    acts = (data.get("actions_executed") or data["actions"]) if executed else data["actions"]
    T = len(acts)
    states = np.stack([fb_abs_state(o) for o in obs[: T + 1]])
    if states.shape[0] == T:
        states = np.concatenate([states, states[-1:]])
    if obstacle:
        front = (data.get("obstacles_robot") or {}).get("obstacle_front", frames.OBSTACLE_FRONT_POS)
        states = np.concatenate([states, np.tile(obstacle_tag_rot6d(front), (states.shape[0], 1))], axis=1)
    actions = np.stack([fb_abs_action(obs[t], acts[t]) for t in range(T)])
    return states, actions


# ------------------------------------------------------------ ResiP's processed zarr, NumPy
#: ResiP's teleop clipped each raw step's rotation to 20 deg (robust-rearrangement
#: ``data_collector_sm.py``: ``ori_bounds_deg=20``), and its ``process_pickles.py`` then calls
#: ``clip_quat_xyzw_magnitude(..., clip_mag=0.35)``, whose ``np.linalg.norm`` has no axis: it
#: scales every rotation delta of an episode by ONE factor so the episode-wide norm
#: ``sqrt(sum ||rotvec_t||^2)`` is 0.35 rad. The steps that hit the 20 deg teleop bound all
#: land on one plateau value, so the factor is exactly ``20 deg / plateau``.
RESIP_RAW_ROT_STEP = np.deg2rad(20.0)
RESIP_EPISODE_ROT_NORM = 0.35


def resip_unscale_rotation(ee_rot6d: np.ndarray, target_rot6d: np.ndarray, min_plateau: int = 5) -> tuple:
    """One episode's ``(T, 6)`` recorded EE rot6d and ``(T, 6)`` zarr target rot6d -> the
    targets ResiP's sim was actually sent ``R_ee exp(f [log(R_ee^T R_target)])`` and ``f``.

    ``f`` is 1 when the episode's norm is under 0.35 (the clip did not fire). Raises when the
    clip fired but fewer than ``min_plateau`` steps sit on the plateau (``f`` not recoverable)."""
    from scipy.spatial.transform import Rotation

    R = np.stack([rot6d_to_mat(r) for r in ee_rot6d])
    Rt = np.stack([rot6d_to_mat(r) for r in target_rot6d])
    rv = Rotation.from_matrix(np.swapaxes(R, 1, 2) @ Rt).as_rotvec()
    mag = np.linalg.norm(rv, axis=1)
    norm = float(np.linalg.norm(rv))
    if norm < RESIP_EPISODE_ROT_NORM - 1e-4:
        f = 1.0
    else:
        plateau = mag.max()
        on = int((np.abs(mag - plateau) < 1e-5).sum())
        if on < min_plateau:
            raise ValueError(
                f"episode rotation norm {norm:.4f} (clipped) but only {on} steps on the plateau "
                f"{np.degrees(plateau):.4f} deg: the per-episode scale is not recoverable"
            )
        f = float(RESIP_RAW_ROT_STEP / plateau)
    fixed = R @ Rotation.from_rotvec(rv * f).as_matrix()
    return np.stack([mat_to_rot6d(m) for m in fixed]).astype(np.float32), f
