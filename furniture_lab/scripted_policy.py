"""FurnitureBench's scripted ``one_leg`` assembly policy, ported to numpy.

A literal port of ``furniture_bench/furniture/parts/table_top.py`` (``TableTop.pre_assemble``),
``parts/leg.py`` (``Leg.fsm_step``), the ``Part`` helpers, and the goal-to-action logic of
``FurnitureSimEnv.get_assembly_action``. The state machines are kept in **FurnitureBench's
representation** -- part poses as 4x4 matrices in the AprilTag frame, of the OBJ mesh
frames; the EE pose in the robot-base frame; ``april_to_robot`` as the bridge -- so the
numbers in them (margins, thresholds, the hole offset) mean what they meant upstream. The
caller converts this env's state into that representation (:func:`ScriptedState.from_env`
in ``scripts/collect_scripted.py``) and the returned goal back into an action.

Quaternions in this file are ``(w, x, y, z)`` (upstream is xyzw; only the products matter and
those are the same Hamilton products). Randomness (target noise, action noise, step clamps)
is upstream's, drawn from a :class:`numpy.random.Generator` so a demo is reproducible.

The sequence for ``one_leg``: push the table top into the obstacle corner and let go
(``TableTop``), then grasp leg 4 lying on the table, lift, carry over the hole, insert,
release, re-grasp from above and screw a quarter turn at a time until the assembled check
passes (``Leg``), then move up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .tasks.furniture import frames
from .tasks.furniture.square_table import ASSEMBLED_POS_THRESHOLD, ORI_BOUND

__all__ = ["hom", "rot_mat", "ScriptedState", "OneLegScriptedPolicy", "MAX_GRIPPER_WIDTH"]

MAX_GRIPPER_WIDTH = frames.MAX_GRIPPER_WIDTH  # config["robot"]["max_gripper_width"]["square_table"]
rot_mat = frames.rot_mat


# ------------------------------------------------------------------------------ small math
def hom(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(p, dtype=np.float64)[:3]
    return T


def get_mat(pos, euler) -> np.ndarray:
    """Upstream ``get_mat``: homogeneous matrix from a position and (x, y, z) Euler angles."""
    return hom(rot_mat(euler) if np.any(euler) else np.eye(3), pos)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


mat_to_quat = frames.mat_to_quat_wxyz


def axisangle_to_quat(vec) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    angle = np.linalg.norm(vec)
    if np.isclose(angle, 0.0):
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = vec / angle
    return np.concatenate([[np.cos(angle / 2)], axis * np.sin(angle / 2)])


def is_similar_rot(R1, R2, ori_bound) -> bool:
    for k in range(3):
        if np.dot(R1[:, k], R2[:, k]) / (np.linalg.norm(R1[:, k]) * np.linalg.norm(R2[:, k])) < ori_bound:
            return False
    return True


def is_similar_pose(T1, T2, ori_bound, pos_threshold) -> bool:
    if not is_similar_rot(T1[:3, :3], T2[:3, :3], ori_bound):
        return False
    return all(abs(T1[k, 3] - T2[k, 3]) <= pos_threshold[k] for k in range(3))


# ------------------------------------------------------------------------------ state in
@dataclass
class ScriptedState:
    """What the state machines read, in upstream's representation."""

    #: EE (TCP) position and rotation in the robot-base frame
    ee_pos: np.ndarray
    ee_rot: np.ndarray
    gripper_width: float
    #: part name -> 4x4 pose in the AprilTag frame, of the OBJ mesh frame
    parts_tag: Dict[str, np.ndarray]
    #: obstacle name -> position in the robot-base frame
    obstacles_robot: Dict[str, np.ndarray]
    #: the 7 arm joints (rad); the screwing states read joint 7
    joints: Optional[np.ndarray] = None

    @property
    def ee_pose(self) -> np.ndarray:
        return hom(self.ee_rot, self.ee_pos)


#: AprilTag frame -> robot-base frame (upstream's ``april_to_robot_mat``), and its inverse
APRIL_TO_ROBOT = hom(frames.TAG_ROT_W, np.asarray(frames.TAG_POS_W))
ROBOT_TO_APRIL = np.linalg.inv(APRIL_TO_ROBOT)


# ------------------------------------------------------------------------------ Part base
class _Part:
    """``furniture_bench.furniture.parts.part.Part``: counters, thresholds, target noise."""

    def __init__(self, name: str, rng: np.random.Generator, noise: bool) -> None:
        self.name = name
        self.rng = rng
        self.noise = noise
        self.pos_error_threshold = 0.01
        self.ori_error_threshold = 0.25
        self.skill_complete_next_states: Tuple[str, ...] = ()
        self.reset()

    def reset(self) -> None:
        self.pre_assemble_done = False
        self._state = ""
        self.gripper_action = -1
        self.prev_cnt = 0
        self.curr_cnt = 0
        self.first_setting_target = True
        self.prev_pose: Optional[np.ndarray] = None
        self.target: Optional[np.ndarray] = None

    @property
    def state(self) -> str:
        return self._state

    def state_no_noise(self) -> bool:
        return False

    def satisfy(self, current, target, pos_error_threshold=None, ori_error_threshold=None, max_len=25) -> bool:
        if pos_error_threshold is None:
            pos_error_threshold = self.pos_error_threshold
        if ori_error_threshold is None:
            ori_error_threshold = self.ori_error_threshold
        if (np.abs(current[:3, 3] - target[:3, 3]).sum() < pos_error_threshold) and (
            np.abs(target[:3, :3] - current[:3, :3]).sum() < ori_error_threshold
        ):
            return True
        if self.curr_cnt - self.prev_cnt >= max_len:
            return True  # upstream: "phase time out"
        return False

    def gripper_less(self, width, target_width, cnt_max=10) -> bool:
        return width <= target_width or self.curr_cnt - self.prev_cnt >= cnt_max

    def gripper_greater(self, width, target_width, cnt_max=10) -> bool:
        return width >= target_width or self.curr_cnt - self.prev_cnt >= cnt_max

    def may_transit_state(self, next_state: str) -> int:
        skill_complete = 0
        if next_state != self._state:
            self._state = next_state
            if next_state in self.skill_complete_next_states:
                skill_complete = 1
            self.first_setting_target = True
            self.prev_cnt = self.curr_cnt
        self.curr_cnt += 1
        return skill_complete

    def add_noise_first_target(self, target, pos_noise=None, ori_noise=None) -> np.ndarray:
        if self.state_no_noise():
            return target
        if self.first_setting_target:
            target = target.copy()
            if self.noise:
                target[:3, 3] += pos_noise if pos_noise is not None else self.rng.normal(0.0, 0.003, size=3)
            ori = mat_to_quat(target[:3, :3])
            if ori_noise is not None:
                ori = quat_mul(ori, ori_noise)
            elif self.noise:
                ori = quat_mul(ori, axisangle_to_quat(np.radians(self.rng.normal(0, 3, size=3))))
            self.target = hom(quat_to_mat(ori), target[:3, 3])
            self.first_setting_target = False
        return self.target


# ------------------------------------------------------------------------------ TableTop
class SquareTableTopPart(_Part):
    """``TableTop.pre_assemble`` with ``SquareTableTop``'s numbers: grasp the top by the edge
    nearest the robot, push it into the obstacle corner, release, go up."""

    def __init__(self, rng, noise) -> None:
        super().__init__("square_table_top", rng, noise)
        self.half_width = 0.08125
        self.body_grip_width = 0.01
        self.skill_complete_next_states = ("push", "go_up")
        # rel_pose_from_center, in tag-id order (SquareTableTop.__init__)
        hw = self.half_width
        self.rel_pose_from_center = [
            get_mat([0, 0, -hw], [0, 0, 0]),
            get_mat([hw, 0, 0], [0, -np.pi / 2, 0]),
            get_mat([0, 0, hw], [0, np.pi, 0]),
            get_mat([-hw, 0, 0], [0, np.pi / 2, 0]),
        ]
        self.reset()
        self._state = "reach_body_grasp_xy"

    def reset(self) -> None:
        super().reset()
        self._state = "reach_body_grasp_xy"

    def _find_closest_y(self, pose: np.ndarray) -> np.ndarray:
        closest = pose.copy()
        for rel in self.rel_pose_from_center:
            tmp = pose @ rel
            if tmp[1, 3] < closest[1, 3]:
                closest = tmp
        return closest

    def pre_assemble(self, s: ScriptedState):
        next_state = self._state
        ee_pose = s.ee_pose
        body_pose = s.parts_tag[self.name]
        a2r = APRIL_TO_ROBOT

        if self._state == "reach_body_grasp_xy":
            body_pose = self._find_closest_y(body_pose)
            rot = body_pose @ get_mat([0, 0, 0], [np.pi / 2, 0, 0])
            target_pos = (a2r @ np.append(body_pose[:3, 3], 1.0))[:3]
            target_ori = (a2r @ rot)[:3, :3]
            target_pos[2] = s.ee_pos[2]
            target = self.add_noise_first_target(hom(target_ori, target_pos))
            if self.satisfy(ee_pose, target):
                self.prev_pose = target
                next_state = "reach_body_grasp_z"
        if self._state == "reach_body_grasp_z":
            target_pos = self.prev_pose[:3, 3].copy()
            target_pos[2] = (a2r @ body_pose)[2, 3]
            target_ori = self.prev_pose[:3, :3]
            target = self.add_noise_first_target(
                hom(target_ori, target_pos),
                pos_noise=self.rng.normal(0.0, [0.01, 0.01, 0.001]) if self.noise else np.zeros(3),
            )
            if self.satisfy(ee_pose, target):
                self.prev_pose = target
                self.gripper_action = 1
                next_state = "pick_body"
        if self._state == "pick_body":
            target = self.prev_pose
            self.gripper_action = 1
            if self.gripper_less(s.gripper_width, self.body_grip_width):
                self.prev_pose = target
                next_state = "push"
        if self._state == "push":
            target_pos = np.zeros(4)
            target_pos[3] = 1.0
            for name in ("obstacle_front", "obstacle_right", "obstacle_left"):
                obstacle_pos = np.append(s.obstacles_robot[name], 1.0)
                target_pos[0] = max(obstacle_pos[0], target_pos[0])
                target_pos[1] = max(obstacle_pos[1], target_pos[1])
            # upstream: april_to_robot @ sim_to_april @ sim_pos; our positions are already
            # in the robot frame, which is what that composition yields
            target_pos[0] -= self.half_width * 2
            target_pos[1] -= self.half_width
            target_pos[2] = ee_pose[2, 3]
            target_pos = target_pos[:3]
            target_ori = np.zeros((3, 3))
            target_ori[0, 1] = 1.0
            target_ori[1, 0] = 1.0
            target_ori[2, 2] = -1.0
            target = self.add_noise_first_target(
                hom(target_ori, target_pos),
                pos_noise=self.rng.normal(0.0, [0.005, 0.005, 0.0]) if self.noise else np.zeros(3),
            )
            if self.satisfy(ee_pose, target, pos_error_threshold=0.02, ori_error_threshold=0.5):
                self.prev_pose = target
                self.gripper_action = -1
                next_state = "release"
        if self._state == "release":
            target = self.prev_pose
            self.gripper_action = -1
            if self.gripper_greater(s.gripper_width, MAX_GRIPPER_WIDTH - 0.001):
                next_state = "go_up"
        if self._state == "go_up":
            target_pos = self.prev_pose[:3, 3].copy()
            target_pos[2] = 0.1
            target = self.add_noise_first_target(hom(self.prev_pose[:3, :3], target_pos))
            if self.satisfy(ee_pose, target):
                self.prev_pose = target
                next_state = "done"
        if self._state == "done":
            self.gripper_action = -1
            self.pre_assemble_done = True
            target = self.prev_pose

        skill_complete = self.may_transit_state(next_state)
        return target[:3, 3].copy(), target[:3, :3].copy(), self.gripper_action, skill_complete


# ------------------------------------------------------------------------------ Leg
class SquareTableLegPart(_Part):
    """``Leg.fsm_step`` with ``SquareTableLeg``'s numbers."""

    def __init__(self, name: str, default_assembled_pos, rng, noise) -> None:
        super().__init__(name, rng, noise)
        self.half_width = 0.015
        self.grasp_margin_x = 0.0
        self.grasp_margin_z = 0.0
        self.default_assembled_pose = get_mat(default_assembled_pos, [0, 0, 0])
        self.skill_complete_next_states = ("lift_up", "insert")
        self.pre_assemble_done = True  # a leg has no pre-assembly phase
        # -- screwing (this port; upstream yaws -95 deg per grasp from yaw 0)
        #: the wrist cannot spin: panda_joint7 is limited to +-2.8973 rad. Each grasp winds
        #: the open hand back to Q7_MIN + margin, then tightens to Q7_MAX - margin.
        self.q7_margin = 0.15
        #: yaw commanded per step (deg); the controller is led by at most this much. 4.5 deg
        #: per 20 Hz step = 90 deg/s, the rate scripts/screw_calib.py showed screws the leg
        #: in with no slip and no wandering; upstream's one-step -95 deg makes this OSC lurch
        #: ~3 cm and pull the leg out (fail1/4/5.pkl)
        self.yaw_step_deg = 4.5
        #: press on the thread while turning (m per step, upstream 0.005)
        self.screw_press = 0.005
        #: hover above the leg centre by this much while re-orienting the open hand
        self.hover_dz = 0.06
        self._q7_prev: Optional[float] = None
        self._stall = 0
        self._state = "reach_leg_floor_xy"

    def reset(self) -> None:
        super().reset()
        self.pre_assemble_done = True
        self._state = "reach_leg_floor_xy"

    def state_no_noise(self) -> bool:
        return self._state in ("insert_wait", "insert_release", "insert")

    # panda_joint7 limits (rad), identical on the FR3 and in Isaac Lab's Franka
    Q7_MIN = -2.8973
    Q7_MAX = 2.8973

    @staticmethod
    def _yaw(R: np.ndarray) -> float:
        """World yaw of the hand's x-axis (rad)."""
        return float(np.arctan2(R[1, 0], R[0, 0]))

    @staticmethod
    def _down(yaw: float) -> np.ndarray:
        """Gripper pointing straight down, hand x-axis at world yaw ``yaw``."""
        return (get_mat([0, 0, 0], [0, 0, yaw]) @ get_mat([0, 0, 0], [np.pi, 0, 0]))[:3, :3]

    def _q7(self, s: ScriptedState) -> float:
        if s.joints is None:
            raise ValueError("ScriptedState.joints is needed for the screwing states")
        return float(s.joints[6])

    def _leg_face_yaw(self, leg_pose_tag: np.ndarray) -> float:
        """World yaw of one face normal of the (square) leg, rad; the others are +k*90 deg."""
        n = (APRIL_TO_ROBOT @ leg_pose_tag)[:3, 0]  # OBJ x: a cross-section axis (y is the leg's axis)
        return float(np.arctan2(n[1], n[0]))

    def _regrasp_q7_goal(self, s: ScriptedState, ee_pose, leg_pose_tag) -> float:
        """Joint-7 value for the regrasp: the hand yaw at which the finger pads are flat on the
        leg's faces (hand yaw = face yaw + k*90 deg) that puts joint 7 nearest its -limit
        while staying q7_margin inside it. +q7 = -world yaw."""
        psi_now = self._yaw(ee_pose[:3, :3])
        dmax = self._q7(s) - (self.Q7_MIN + self.q7_margin)  # the most the yaw may grow
        d = (self._leg_face_yaw(leg_pose_tag) - psi_now) % (np.pi / 2)  # smallest admissible growth
        d += np.floor((dmax - d) / (np.pi / 2)) * (np.pi / 2)  # the largest admissible one <= dmax
        return self._q7(s) - d

    def _yaw_toward(self, s: ScriptedState, ee_pose, q7_goal: float) -> Tuple[float, float]:
        """World yaw target that moves joint 7 toward ``q7_goal`` by at most ``yaw_step_deg``
        this step (+q7 is -world yaw, verified by FK), and the remaining joint travel (rad)."""
        remaining = q7_goal - self._q7(s)
        step = float(np.clip(remaining, -np.radians(self.yaw_step_deg), np.radians(self.yaw_step_deg)))
        return self._yaw(ee_pose[:3, :3]) - step, remaining

    @staticmethod
    def _find_down_z(mat: np.ndarray) -> np.ndarray:
        rot = get_mat([0, 0, 0], [0, np.pi / 2, 0])
        for _ in range(4):
            if mat[2, 2] > 0.8:  # z is down
                break
            mat = mat @ rot
        return mat

    @staticmethod
    def _find_leg_pose_x_look_front(leg_pose: np.ndarray) -> np.ndarray:
        best = leg_pose.copy()
        tmp = leg_pose
        rot = get_mat([0, 0, 0], [0, -np.pi / 2, 0])
        for _ in range(3):
            tmp = tmp @ rot
            if best[0, 0] < tmp[0, 0]:
                best = tmp
        return best

    def _hole_target(self, s: ScriptedState, table_pose, z: float) -> np.ndarray:
        """The leg pose that puts it over the hole, upright, at height z (robot frame)."""
        a2r = APRIL_TO_ROBOT
        table_hole_robot = a2r @ table_pose @ self.default_assembled_pose
        return np.array(
            [
                [1.0, 0.0, 0.0, table_hole_robot[0, 3]],
                [0.0, 0.0, -1.0, table_hole_robot[1, 3]],
                [0.0, 1.0, 0.0, z],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def fsm_step(self, s: ScriptedState, assemble_to: str):
        next_state = self._state
        ee_pose = s.ee_pose
        ee_pos = s.ee_pos
        a2r = APRIL_TO_ROBOT
        table_pose = s.parts_tag[assemble_to]
        leg_pose = s.parts_tag[self.name]
        margin = get_mat([0, 0, 0], [0, -np.pi / 5, 0])

        def rel_rot_mat(src, tgt):
            return tgt @ np.linalg.inv(src)

        if self._state == "reach_leg_floor_xy":
            leg_pose = self._find_down_z(leg_pose)
            pos = leg_pose[:4, 3]
            target_pos = (a2r @ pos)[:3].copy()
            target_ori = ee_pose[:3, :3]
            target_pos[2] = ee_pos[2]
            target_pos[1] += 0.01
            target_pos[0] += self.grasp_margin_x
            target = self.add_noise_first_target(hom(target_ori, target_pos), ori_noise=np.array([1.0, 0, 0, 0]))
            if self.satisfy(ee_pose, target, pos_error_threshold=0.02):
                self.prev_pose = target.copy()
                self.prev_pose[2, 3] -= 0.02
                next_state = "reach_leg_ori"
        elif self._state == "reach_leg_ori":
            rot = get_mat([0, 0, 0], [np.pi / 2, -np.pi / 2, 0])
            target_ori = (margin @ a2r @ rot)[:3, :3]
            target = hom(target_ori, self.prev_pose[:3, 3])
            if self.satisfy(ee_pose, target, pos_error_threshold=0.015):
                self.prev_pose = target
                next_state = "reach_leg_floor_z"
        elif self._state == "reach_leg_floor_z":
            target_pos = self.prev_pose[:3, 3].copy()
            target_pos[2] = (a2r @ leg_pose)[2, 3]
            target = hom(self.prev_pose[:3, :3], target_pos)
            if self.satisfy(ee_pose, target, pos_error_threshold=0.007):
                self.prev_pose = target
                next_state = "pick_leg"
        elif self._state == "pick_leg":
            target = self.prev_pose
            self.gripper_action = 1
            if self.gripper_less(s.gripper_width, 2 * self.half_width + 0.001):
                self.prev_pose = target
                next_state = "lift_up"
        elif self._state == "lift_up":
            target_pos = self.prev_pose[:3, 3] + np.array([0, 0, 0.10])
            target = self.add_noise_first_target(hom(ee_pose[:3, :3], target_pos))
            if self.satisfy(ee_pose, target, pos_error_threshold=0.02, ori_error_threshold=0.3):
                self.prev_pose = target
                next_state = "move_center"
        elif self._state == "move_center":
            target = self.add_noise_first_target(hom(self.prev_pose[:3, :3], np.array([0.5, 0.1, 0.1])))
            if self.satisfy(ee_pose, target, pos_error_threshold=0.02, ori_error_threshold=0.3):
                self.prev_pose = target
                next_state = "match_leg_ori"
        elif self._state == "match_leg_ori":
            target_ori = (margin @ get_mat([0, 0, 0], [np.pi, 0, 0]))[:3, :3]
            target = self.add_noise_first_target(hom(target_ori, np.array([0.57, 0.1, 0.12])))
            if self.satisfy(ee_pose, target, pos_error_threshold=0.02, ori_error_threshold=0.3):
                self.prev_pose = target
                next_state = "reach_table_top_xy"
        elif self._state == "reach_table_top_xy":
            leg_robot = self._find_leg_pose_x_look_front(a2r @ leg_pose)
            target_leg = self._hole_target(s, table_pose, self.prev_pose[2, 3])
            target = rel_rot_mat(leg_robot, target_leg) @ ee_pose
            if self.satisfy(ee_pose, target, pos_error_threshold=0.015, ori_error_threshold=0.3):
                self.prev_pose = target
                next_state = "reach_table_top_z"
        elif self._state == "reach_table_top_z":
            leg_robot = self._find_leg_pose_x_look_front(a2r @ leg_pose)
            target_leg = self._hole_target(s, table_pose, table_pose[2, 3] + 0.09)
            target = rel_rot_mat(leg_robot, target_leg) @ ee_pose
            if self.satisfy(ee_pose, target, pos_error_threshold=0.007, ori_error_threshold=0.15):
                self.prev_pose = target
                next_state = "insert_wait"
        elif self._state == "insert_wait":
            leg_robot = self._find_leg_pose_x_look_front(a2r @ leg_pose)
            target_leg = self._hole_target(s, table_pose, table_pose[2, 3] + 0.084)
            target = rel_rot_mat(leg_robot, target_leg) @ ee_pose
            if self.satisfy(ee_pose, target, pos_error_threshold=0.0, ori_error_threshold=0.0, max_len=10):
                self.prev_pose = target
                next_state = "insert_release"
        elif self._state == "insert_release":
            target = self.prev_pose
            self.gripper_action = -1
            if self.gripper_greater(s.gripper_width, MAX_GRIPPER_WIDTH - 0.001):
                next_state = "insert"
        elif self._state == "insert":
            target = self.prev_pose  # dummy transition for skill complete
            next_state = "pre_grasp"
        elif self._state == "release":
            target = self.prev_pose
            self.gripper_action = -1
            if self.gripper_greater(s.gripper_width, MAX_GRIPPER_WIDTH - 0.001):
                next_state = "pre_grasp"
        elif self._state == "pre_grasp":
            # hover above the leg's centre and wind the open hand back toward joint 7's -limit,
            # stopping at the last yaw where the finger pads are flat on the leg's faces, so
            # the coming grasp is square and has the most tightening travel
            target_pos = (a2r @ leg_pose[:4, 3])[:3].copy()
            target_pos[2] += self.grasp_margin_z + self.hover_dz
            yaw, remaining = self._yaw_toward(s, ee_pose, self._regrasp_q7_goal(s, ee_pose, leg_pose))
            target = hom(self._down(yaw), target_pos)
            if abs(remaining) < 0.05 and self.satisfy(ee_pose, target, pos_error_threshold=0.01, ori_error_threshold=0.3, max_len=60):
                next_state = "grasp_lower"
            elif self.curr_cnt - self.prev_cnt >= 60:
                next_state = "grasp_lower"
        elif self._state == "grasp_lower":
            # straight down onto the leg, hand square to its top, centred on its axis
            target_pos = (a2r @ leg_pose[:4, 3])[:3].copy()
            target_pos[2] += self.grasp_margin_z
            target = hom(self._down(self._yaw(ee_pose[:3, :3])), target_pos)
            if self.satisfy(ee_pose, target, pos_error_threshold=0.006, ori_error_threshold=0.3, max_len=25):
                self.prev_pose = target
                next_state = "screw_grasp"
        elif self._state == "screw_grasp":
            target = self.prev_pose
            self.gripper_action = 1
            if self.gripper_less(s.gripper_width, 2 * self.half_width + 0.001):
                self.prev_pose = target
                next_state = "screw_align"
        elif self._state == "screw_align":
            # hand (and so the leg) over the hole axis; the leg goes vertical in the hole
            hole = (a2r @ table_pose @ self.default_assembled_pose)[:3, 3]
            target_pos = np.array([hole[0], hole[1], ee_pos[2]])
            target = hom(self._down(self._yaw(ee_pose[:3, :3])), target_pos)
            if self.satisfy(ee_pose, target, pos_error_threshold=0.004, ori_error_threshold=0.3, max_len=10):
                self._q7_prev = self._q7(s)
                self._stall = 0
                next_state = "screw"
        elif self._state == "screw":
            # tighten: only the wrist turns (joint 7 axis = hand axis = leg axis once centred),
            # position held on the hole axis with a small press. Ends near joint 7's +limit,
            # or when joint 7 stops advancing (the leg has bottomed, or the fingers slip).
            hole = (a2r @ table_pose @ self.default_assembled_pose)[:3, 3]
            yaw, remaining = self._yaw_toward(s, ee_pose, self.Q7_MAX - self.q7_margin)
            target = hom(self._down(yaw), np.array([hole[0], hole[1], ee_pos[2] - self.screw_press]))
            q7 = self._q7(s)
            self._stall = self._stall + 1 if q7 - self._q7_prev < np.radians(1.0) else 0
            self._q7_prev = q7
            if remaining < 0.05 or self._stall >= 8 or self.curr_cnt - self.prev_cnt >= 80:
                self.prev_pose = hom(self._down(self._yaw(ee_pose[:3, :3])), ee_pos.copy())
                next_state = "release"
        else:
            raise RuntimeError(f"unknown leg state {self._state!r}")

        skill_complete = self.may_transit_state(next_state)
        return target[:3, 3].copy(), target[:3, :3].copy(), self.gripper_action, skill_complete


# ------------------------------------------------------------------------------ the policy
@dataclass
class ScriptedAction:
    #: upstream's 8-D action: position delta (m, robot frame), rotation delta (wxyz, EE
    #: frame), gripper (+1 close, -1 open) -- before any env-specific scaling
    delta_pos: np.ndarray
    delta_quat: np.ndarray
    gripper: float
    #: the goal the state machine asked for (robot frame), for an absolute-target controller
    goal_pos: np.ndarray
    goal_rot: np.ndarray
    skill_complete: int
    state: str
    assembled: bool = False
    done: bool = False


class OneLegScriptedPolicy:
    """``get_assembly_action`` for ``one_leg``: top pre-assembly, then leg 4 into the top."""

    #: upstream: square_table_leg4's default_assembled_pose (in the top's OBJ frame)
    LEG4_ASSEMBLED_POS = np.array([-0.05625, 0.046875, -0.05625])
    #: all four holes, for the assembled check (square_table.assembled_rel_poses)
    HOLES = np.array(
        [
            [-0.05625, 0.046875, -0.05625],
            [0.05625, 0.046875, -0.05625],
            [-0.05625, 0.046875, 0.05625],
            [0.05625, 0.046875, 0.05625],
        ]
    )

    def __init__(self, seed: int = 0, noise: bool = True) -> None:
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.top = SquareTableTopPart(self.rng, noise)
        self.leg = SquareTableLegPart("square_table_leg4", self.LEG4_ASSEMBLED_POS, self.rng, noise)
        self.assemble_idx = 0
        self.move_neutral = False
        self.steps = 0

    # -- the assembled check upstream runs inside get_assembly_action (Furniture.assembled)
    def assembled(self, s: ScriptedState) -> bool:
        rel = np.linalg.inv(s.parts_tag["square_table_top"]) @ s.parts_tag["square_table_leg4"]
        return any(
            is_similar_pose(get_mat(h, [0, 0, 0]), rel, ORI_BOUND, ASSEMBLED_POS_THRESHOLD) for h in self.HOLES
        )

    @property
    def state(self) -> str:
        if self.move_neutral:
            return "move_neutral"
        return self.top.state if not self.top.pre_assemble_done else self.leg.state

    def act(self, s: ScriptedState) -> ScriptedAction:
        self.steps += 1
        ee_quat = mat_to_quat(s.ee_rot)
        noop = ScriptedAction(np.zeros(3), np.array([1.0, 0, 0, 0]), -1.0, s.ee_pos.copy(), s.ee_rot.copy(), 0, self.state)

        if self.move_neutral:
            if s.ee_pos[2] <= 0.15 - 0.01:
                goal_pos = np.array([s.ee_pos[0], s.ee_pos[1], 0.15])
                return ScriptedAction(
                    goal_pos - s.ee_pos, np.array([1.0, 0, 0, 0]), -1.0, goal_pos, s.ee_rot.copy(), 0, "move_neutral",
                    assembled=True,
                )
            self.move_neutral = False

        if self.assemble_idx > 0:  # the one pair of one_leg is done and the hand is clear
            noop.done = True
            noop.assembled = True
            return noop

        if self.assembled(s):
            self.assemble_idx += 1
            self.move_neutral = True
            noop.assembled = True
            noop.skill_complete = 1
            return noop

        if not self.top.pre_assemble_done:
            goal_pos, goal_rot, gripper, skill_complete = self.top.pre_assemble(s)
        else:
            goal_pos, goal_rot, gripper, skill_complete = self.leg.fsm_step(s, "square_table_top")

        # -- goal -> upstream's action (get_assembly_action's scaling, clamping and noise)
        delta_pos = goal_pos - s.ee_pos
        sign = np.sign(delta_pos)
        delta_pos = np.abs(delta_pos) * 2
        for i in range(3):
            if delta_pos[i] > 0.03:
                delta_pos[i] = 0.03 + (delta_pos[i] - 0.03) * (self.rng.normal(1.5, 0.1) if self.noise else 1.5)
        delta_pos = delta_pos * sign
        max_delta = 0.11 + (0.01 * self.rng.random(3) if self.noise else 0.005 * np.ones(3))
        max_delta[2] -= 0.04
        delta_pos = np.clip(delta_pos, -max_delta, max_delta)

        delta_quat = quat_mul(quat_conj(ee_quat), mat_to_quat(goal_rot))  # in the EE frame
        if self.noise and self.leg.state_no_noise() and self.rng.random() < 0.5:
            delta_pos = self.rng.normal(delta_pos, 0.005)
            delta_quat = quat_mul(delta_quat, axisangle_to_quat(np.radians(self.rng.normal(0, 5, size=3))))
        return ScriptedAction(delta_pos, delta_quat, float(gripper), goal_pos, goal_rot, skill_complete, self.state)
