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

    def __init__(
        self,
        name: str,
        default_assembled_pos,
        rng,
        noise,
        insert_height: float = 0.077,
        start_twist_deg: float = 90.0,
        insert_press: float = 0.003,
        turn_step_deg: float = 10.0,
    ) -> None:
        super().__init__(name, rng, noise)
        #: every turn (the thread start and upstream's screw) is about the vertical axis
        #: through the hole, not the EE's own axis (an off-centre grasp would orbit the leg
        #: and pull it out), and is fed to the controller this many degrees per step.
        self.turn_step_deg = turn_step_deg
        self._turn_R0: Optional[np.ndarray] = None
        #: where the leg centre goes before release, above the top's tag-frame z. Upstream
        #: uses 0.084 (bolt tip ~5 mm above the boss; the leg drops onto the thread and, in
        #: this sim, topples). scripts/smoke.py --drop measured the free-drop thread stop at
        #: 0.072; a grasped leg meets the boss at a random thread phase and stops higher.
        self.insert_height = insert_height
        #: how far below the insert height the twist target sits (the press on the thread)
        self.insert_press = insert_press
        #: turn applied about the hole axis while still grasped, before releasing, so the
        #: thread is started and the leg stands on its own. 0 restores upstream's behaviour.
        self.start_twist_deg = start_twist_deg
        self.half_width = 0.015
        self.grasp_margin_x = 0.0
        self.grasp_margin_z = 0.0
        self.default_assembled_pose = get_mat(default_assembled_pos, [0, 0, 0])
        self.skill_complete_next_states = ("lift_up", "insert")
        self.pre_assemble_done = True  # a leg has no pre-assembly phase
        self._state = "reach_leg_floor_xy"

    def reset(self) -> None:
        super().reset()
        self.pre_assemble_done = True
        self._state = "reach_leg_floor_xy"

    def state_no_noise(self) -> bool:
        return self._state in ("insert_wait", "insert_screw", "insert_release", "insert")

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

    def _turned_deg(self, ee_pose: np.ndarray) -> float:
        """Yaw of the EE about the world vertical since the turn began (signed degrees)."""
        rel = ee_pose[:3, :3] @ self._turn_R0.T
        return float(np.degrees(np.arctan2(rel[1, 0], rel[0, 0])))

    def _turn_target(self, s: ScriptedState, table_pose, ee_pose, total_deg: float, press: float):
        """The next target of a turn about the hole axis: the EE pose rotated about the vertical
        line through the hole centre by at most ``turn_step_deg`` toward ``total_deg``
        (negative = clockwise seen from above), pressed ``press`` below its current height.
        Returns (target, finished)."""
        if self.first_setting_target:
            self._turn_R0 = ee_pose[:3, :3].copy()
            self.first_setting_target = False
        hole = (APRIL_TO_ROBOT @ table_pose @ self.default_assembled_pose)[:3, 3]
        remaining = total_deg - self._turned_deg(ee_pose)
        step = float(np.clip(remaining, -self.turn_step_deg, self.turn_step_deg))
        about = hom(np.eye(3), hole) @ get_mat([0, 0, 0], [0, 0, np.radians(step)]) @ hom(np.eye(3), -hole)
        target = about @ ee_pose
        target[2, 3] -= press
        return target, abs(remaining) < 2.0

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
            target_leg = self._hole_target(s, table_pose, table_pose[2, 3] + self.insert_height)
            target = rel_rot_mat(leg_robot, target_leg) @ ee_pose
            if self.satisfy(ee_pose, target, pos_error_threshold=0.0, ori_error_threshold=0.0, max_len=10):
                self.prev_pose = target
                next_state = "insert_screw" if self.start_twist_deg > 0 else "insert_release"
        elif self._state == "insert_screw":
            # start the thread while still holding the leg: turn it about the hole axis by
            # -start_twist, pressing down insert_press
            target, finished = self._turn_target(s, table_pose, ee_pose, -self.start_twist_deg, self.insert_press)
            if finished or self.curr_cnt - self.prev_cnt >= 40:
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
            target_ori = get_mat([0, 0, 0], [np.pi, 0, 0])[:3, :3]
            target_pos = (a2r @ leg_pose[:4, 3])[:3].copy()
            target_pos[2] += self.grasp_margin_z
            target = hom(target_ori, target_pos)
            if self.satisfy(ee_pose, target):
                self.prev_pose = target
                next_state = "screw_grasp"
        elif self._state == "screw_grasp":
            target = self.prev_pose
            self.gripper_action = 1
            if self.gripper_less(s.gripper_width, 2 * self.half_width + 0.001):
                self.prev_pose = target
                next_state = "screw"
        elif self._state == "screw":
            # upstream: yaw the EE to -(90 + 5) deg in place, pressing 5 mm; here the same turn
            # is made about the hole axis, in steps
            target, finished = self._turn_target(s, table_pose, ee_pose, -95.0, 0.005)
            if finished or self.curr_cnt - self.prev_cnt >= 25:
                self.prev_pose = target
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

    def __init__(
        self,
        seed: int = 0,
        noise: bool = True,
        assembled_hold_steps: int = 10,
        insert_height: float = 0.077,
        start_twist_deg: float = 90.0,
        insert_press: float = 0.003,
        turn_step_deg: float = 10.0,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.top = SquareTableTopPart(self.rng, noise)
        self.leg = SquareTableLegPart(
            "square_table_leg4",
            self.LEG4_ASSEMBLED_POS,
            self.rng,
            noise,
            insert_height,
            start_twist_deg,
            insert_press,
            turn_step_deg,
        )
        self.assemble_idx = 0
        self.move_neutral = False
        self.steps = 0
        #: the assembled check must hold this many consecutive steps before it counts
        #: (upstream fires on a single step, which a leg being knocked through the seat
        #: pose also satisfies)
        self.assembled_hold_steps = assembled_hold_steps
        self.assembled_streak = 0

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

        self.assembled_streak = self.assembled_streak + 1 if self.assembled(s) else 0
        if self.assembled_streak >= self.assembled_hold_steps:
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
