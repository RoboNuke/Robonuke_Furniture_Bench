"""Sort one_leg demo attempts (the collector's pkls, successes and ``fails/``) by the stage they
broke at. NumPy only, no sim: ``scripts/sort_failures.py`` runs it over a demo folder.

The stages follow the gripper commands of the scripted / servo policies: the first close grasps
the top (pushed into the obstacle corner, then released), the second grasps leg 4 (lifted,
stood up, lowered into the hole, released), and every close after that is a screw regrasp.

The geometry is FurnitureBench's, in the top's OBJ frame (``y`` the seat axis): the leg is
*in its hole* when its origin is within ``hole_xz`` of the hole's axis and it is tilted less
than ``tilt_deg`` from the top's ``y``; it is *assembled* by upstream's check
(``square_table.ASSEMBLED_POS_THRESHOLD`` / ``ORI_BOUND``, the same as the env's tracker).
A screwing failure is split by what moved: the top (``top_move`` / ``top_yaw_deg`` from where
the leg was released) or the leg leaving its hole with the top still.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import demos as D
from .scripted_policy import APRIL_TO_ROBOT, ORI_BOUND, OneLegScriptedPolicy, get_mat, is_similar_pose
from .tasks.furniture import frames
from .tasks.furniture.square_table import ASSEMBLED_POS_THRESHOLD

__all__ = ["Thresholds", "Attempt", "load_attempt", "classify", "STAGES"]

#: every verdict ``classify`` can return, in pipeline order
STAGES = (
    "success",
    "aborted",
    "top_push: never released",
    "top_push: top not in the corner",
    "leg_grasp: never grasped",
    "leg_grasp: not lifted",
    "leg_carry: dropped",
    "insertion: never released",
    "insertion: not in the hole",
    "screwing: top moved",
    "screwing: leg fell out",
    "screwing: stalled in the hole",
)

TOP_HALF_WIDTH = 0.08125  # servo_policy.ServoOneLegPolicy.TOP_HALF_WIDTH
CORNER_CLEARANCE = -0.002  # its default: the goal 2 mm inside both walls
HOLE = OneLegScriptedPolicy.LEG4_ASSEMBLED_POS


@dataclass
class Thresholds:
    #: m: the top counts as in the corner within this of the policy's corner goal (x, y)
    corner: float = 0.010
    #: m: the leg counts as lifted once it rose this far above where it was grasped
    lift: float = 0.02
    #: m: the leg counts as dropped while carried once its distance from the EE grew this much
    drop: float = 0.03
    #: m / deg: the leg is in its hole within this of the hole's axis and this tilt. The
    #: servo policy's successes release the leg leaning in its hole: 9.4-10.2 mm off-axis and
    #: 11.7-12.6 deg at release (20 of them, ResiP sim), so these sit just above that
    hole_xz: float = 0.013
    tilt_deg: float = 16.0
    #: steps after the leg release before the insertion is judged (the leg settles)
    settle: int = 5
    #: m / deg: the top moved during screwing past these (from the leg release on)
    top_move: float = 0.010
    top_yaw_deg: float = 8.0


@dataclass
class Attempt:
    path: str
    success: bool
    error: str
    #: (T,) gripper command (> 0 closes), (T+1, 3) EE position (robot frame)
    grip: np.ndarray
    ee_pos: np.ndarray
    #: (T+1, 4, 4) top / leg 4 OBJ-frame poses in the tag frame
    top: np.ndarray
    leg: np.ndarray
    #: robot-frame obstacle positions (the pkl's, or frames' fixed ones)
    obstacles: Dict[str, np.ndarray] = field(default_factory=dict)


def _pose(flat7: np.ndarray) -> np.ndarray:
    return _hom(D.quat_to_mat(D.quat_xyzw_to_wxyz(flat7[3:7])), flat7[:3])


def _hom(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def load_attempt(path: str | Path) -> Attempt:
    """One collector pkl (success or failure) -> the arrays ``classify`` reads."""
    with open(path, "rb") as f:
        data: Dict[str, Any] = pickle.load(f)
    obs = data["observations"]
    acts = np.stack([D._np(a) for a in data["actions"]])
    T = acts.shape[0]
    obs = obs[: T + 1]
    parts = np.stack([D._np(o["parts_poses"]).reshape(-1, 7) for o in obs])  # (T+1, P, 7) tag frame xyzw
    names = list(D.PART_NAMES)
    top = np.stack([_pose(p[names.index("square_table_top")]) for p in parts])
    leg = np.stack([_pose(p[names.index("square_table_leg4")]) for p in parts])
    ee = np.stack([D._np(o["robot_state"]["ee_pos"]) for o in obs])
    recorded = data.get("obstacles_robot")
    obstacles = (
        {k: np.asarray(v, dtype=np.float64) for k, v in recorded.items()}
        if recorded
        else {"obstacle_front": np.asarray(frames.OBSTACLE_FRONT_POS), **{k: np.asarray(v) for k, v in frames.OBSTACLE_SIDE_POS.items()}}
    )
    return Attempt(
        path=str(path),
        success=bool(data.get("success", False)),
        error=str(data.get("error_description", "") or ""),
        grip=acts[:, 7],
        ee_pos=ee,
        top=top,
        leg=leg,
        obstacles=obstacles,
    )


def _angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def _leg_in_top(a: Attempt, t: int) -> Dict[str, float]:
    """Leg 4 against its hole at step t, in the top's frame: off-axis distance, height above the
    seat along the seat axis, tilt from the seat axis, and upstream's assembled check."""
    rel = np.linalg.inv(a.top[t]) @ a.leg[t]
    d = rel[:3, 3] - HOLE
    tilt = float(np.degrees(np.arccos(np.clip(rel[1, 1], -1.0, 1.0))))  # leg y vs top y
    assembled = is_similar_pose(get_mat(HOLE, [0, 0, 0]), rel, ORI_BOUND, ASSEMBLED_POS_THRESHOLD)
    return {"xz": float(np.hypot(d[0], d[2])), "height": float(d[1]), "tilt": tilt, "assembled": bool(assembled)}


def _corner_error(a: Attempt, t: int) -> float:
    """The top's (x, y) distance from the policy's corner goal at step t (robot frame)."""
    p = (APRIL_TO_ROBOT @ a.top[t])[:2, 3]
    gx = a.obstacles["obstacle_front"][0] - frames.OBSTACLE_FRONT_SIZE[0] / 2 - TOP_HALF_WIDTH - CORNER_CLEARANCE
    gy = a.obstacles["obstacle_left"][1] - frames.OBSTACLE_SIDE_SIZE[1] / 2 - TOP_HALF_WIDTH - CORNER_CLEARANCE
    return float(np.hypot(p[0] - gx, p[1] - gy))


def classify(a: Attempt, th: Optional[Thresholds] = None) -> Dict[str, Any]:
    """``{"stage": one of STAGES, "step": where it was judged, ...metrics}`` for one attempt.

    A success (the pkl's verdict, or assembled at the end) keeps the metrics the stages
    measured, so the thresholds can be checked against attempts that worked."""
    r = _stages(a, th or Thresholds())
    if a.success or r.pop("assembled_end"):
        # every test passed should leave a success at "screwing: stalled in the hole"; any
        # earlier stage here means a threshold is too tight
        r.update(stage="success", step=None, passed_to=r["stage"])
    return r


def _stages(a: Attempt, th: Thresholds) -> Dict[str, Any]:
    T = len(a.grip)
    closed = a.grip > 0
    closes = [t for t in range(T) if closed[t] and (t == 0 or not closed[t - 1])]
    opens = [t for t in range(1, T) if not closed[t] and closed[t - 1]]
    out: Dict[str, Any] = {"steps": T, "closes": closes, "opens": opens}

    def verdict(stage: str, step: Optional[int], **metrics) -> Dict[str, Any]:
        return {**out, "stage": stage, "step": step, **metrics}

    end = _leg_in_top(a, T)
    out["final_height_mm"] = end["height"] * 1000
    out["assembled_end"] = end["assembled"]
    if "by key" in a.error:
        return verdict("aborted", None)

    # -- the top: first close .. first open
    if not closes or not [o for o in opens if o > closes[0]]:
        return verdict("top_push: never released", T)
    top_release = [o for o in opens if o > closes[0]][0]
    err = _corner_error(a, top_release)
    out["corner_mm"] = err * 1000
    if err > th.corner:
        return verdict("top_push: top not in the corner", top_release)

    # -- leg 4: the second close
    leg_closes = [c for c in closes if c > top_release]
    if not leg_closes:
        return verdict("leg_grasp: never grasped", T)
    grasp = leg_closes[0]
    later_opens = [o for o in opens if o > grasp]
    release = later_opens[0] if later_opens else None
    carry_end = (release - 1) if release is not None else T  # the last step the leg is held
    z_robot = np.array([(APRIL_TO_ROBOT @ a.leg[t])[2, 3] for t in range(grasp, carry_end + 1)])
    out["lift_mm"] = float(z_robot.max() - z_robot[0]) * 1000
    if z_robot.max() - z_robot[0] < th.lift:
        return verdict("leg_grasp: not lifted", grasp)
    legs_r = np.array([(APRIL_TO_ROBOT @ a.leg[t])[:3, 3] for t in range(grasp, carry_end + 1)])
    gap = np.linalg.norm(legs_r - a.ee_pos[grasp : carry_end + 1], axis=1)
    ref = gap[min(3, len(gap) - 1)]
    grown = np.nonzero(gap - ref > th.drop)[0]
    if len(grown):
        return verdict("leg_carry: dropped", grasp + int(grown[0]))
    if release is None:
        return verdict("insertion: never released", T)

    # -- the insertion: a few steps after the release
    t_ins = min(release + th.settle, T)
    ins = _leg_in_top(a, t_ins)
    out.update(insert_xz_mm=ins["xz"] * 1000, insert_tilt_deg=ins["tilt"], insert_height_mm=ins["height"] * 1000)
    if ins["xz"] > th.hole_xz or ins["tilt"] > th.tilt_deg:
        return verdict("insertion: not in the hole", t_ins)

    # -- screwing: from the release on; split by what moved
    top0 = a.top[release]
    moves = [np.linalg.norm(a.top[t][:3, 3] - top0[:3, 3]) for t in range(release, T + 1)]
    yaws = [_angle_deg(top0[:3, :3].T @ a.top[t][:3, :3]) for t in range(release, T + 1)]
    legs = [_leg_in_top(a, t) for t in range(release, T + 1)]
    out.update(
        top_move_mm=float(max(moves)) * 1000,
        top_yaw_deg=float(max(yaws)),
        screw_regrasps=len([c for c in closes if c > release]),
    )
    out_of_hole = [i for i, g in enumerate(legs) if g["xz"] > th.hole_xz or g["tilt"] > th.tilt_deg]
    moved = [i for i in range(len(moves)) if moves[i] > th.top_move or yaws[i] > th.top_yaw_deg]
    if moved and (not out_of_hole or moved[0] <= out_of_hole[0]):
        return verdict("screwing: top moved", release + moved[0])
    if out_of_hole:
        i = out_of_hole[0]
        return verdict("screwing: leg fell out", release + i, out_xz_mm=legs[i]["xz"] * 1000, out_tilt_deg=legs[i]["tilt"])
    return verdict("screwing: stalled in the hole", T)
