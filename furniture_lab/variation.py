"""Per-episode variation of the scripted ``one_leg`` expert, and DART-style action bursts, so
that 50 demos cover more than one corridor through the task. Pure NumPy.

**Variation** (``scripts/collect_scripted.py --vary``): every attempt draws its own expert
from :class:`VariationRanges` around the collector's settings -- the centre of every range is
the fixed expert, so ``--vary`` off is exactly the old collector:

* speed (factors on the per-step caps): free/held translation, held rotation (standing the
  leg up, the screw pivot), the screwing turn, winding the open hand back, the lowering rate;
* waypoints (offsets, m or deg): the carry height, the height over the hole before lowering,
  the regrasp hover, the placement yaw, the screw press, the rise after releasing the leg,
  the corner goal's clearance, the hand's height after releasing the top;
* grasps: where the lying leg is gripped along its axis, the screw regrasp's height on the
  leg, where the top's near edge is gripped along it and the hand's yaw there;
* the push: a via point back from the corner (away from both walls), so the top reaches the
  corner from different directions. The corner itself never changes: a different end
  position would be a second task mode for the same 50 demos.

The drawn values are saved in each pkl (``variation``), so the failure sorter's verdicts can
be related to them.

**DART bursts** (``--dart_prob`` > 0; Laskey et al. 2017, "DART: Noise Injection for Robust
Imitation Learning"): in the free-space states (:data:`DART_STATES`) a burst starts with
probability ``dart_prob`` per step and lasts ``dart_steps`` steps; during it a fixed random
offset -- ``dart_pos`` m on the position command, ``dart_rot`` deg on the rotation -- is added
to every command *sent*. The expert is closed loop, so it pulls the hand back afterwards. The
pkl's ``actions`` stay the expert's own (clean) commands -- the labels BC trains on, now
including recoveries -- and ``actions_executed`` holds what the env was given (for replay).
Contact states (pushing, grasping, inserting, screwing) are never perturbed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .scripted_policy import axisangle_to_quat, quat_mul

__all__ = ["VariationRanges", "sample_variation", "DART_STATES", "DartBursts"]


@dataclass(frozen=True)
class VariationRanges:
    """``(low, high)`` of a uniform draw per attempt. ``*_f``: a factor on the collector's
    value; the rest are offsets added to it (m, or deg where named), except ``push_via_*``,
    ``*_grasp_*`` and ``regrasp_dz``, which are absolute (their fixed value is 0)."""

    pos_speed_f: Tuple[float, float] = (0.75, 1.3)
    rot_speed_f: Tuple[float, float] = (0.75, 1.3)
    #: the screwing turn: narrower -- its rate was calibrated (no slip at 90 deg/s)
    screw_speed_f: Tuple[float, float] = (0.85, 1.15)
    wind_speed_f: Tuple[float, float] = (0.75, 1.3)
    insert_rate_f: Tuple[float, float] = (0.75, 1.25)
    carry_dz: Tuple[float, float] = (-0.03, 0.03)
    #: only up: the bolt is 25.4 mm long, so below the fixed 30 mm it would touch the top
    over_dz: Tuple[float, float] = (0.0, 0.010)
    hover_dz: Tuple[float, float] = (-0.01, 0.02)
    insert_yaw_deg: Tuple[float, float] = (-10.0, 10.0)
    press: Tuple[float, float] = (-0.0005, 0.0005)
    release_dz: Tuple[float, float] = (-0.02, 0.02)
    #: the goal stays inside the walls (< 0), so the push still ends against them
    corner_clearance: Tuple[float, float] = (-0.002, 0.001)
    top_up_z: Tuple[float, float] = (-0.02, 0.02)
    push_via_x: Tuple[float, float] = (0.0, 0.05)
    push_via_y: Tuple[float, float] = (0.0, 0.05)
    top_grasp_along: Tuple[float, float] = (-0.03, 0.03)
    top_grasp_yaw_deg: Tuple[float, float] = (-8.0, 8.0)
    #: the leg is 87.5 mm long; +-15 mm keeps the fingers well inside it
    leg_grasp_along: Tuple[float, float] = (-0.015, 0.015)
    #: down is limited: the fingers must stay clear of the top's surface while screwing
    regrasp_dz: Tuple[float, float] = (-0.005, 0.015)


#: the most the corner goal may come out of the walls (it must stay inside them)
MAX_CORNER_CLEARANCE = -0.0005


def sample_variation(rng: np.random.Generator, base: Dict[str, Any], ranges: Optional[VariationRanges] = None) -> Dict[str, Any]:
    """One attempt's expert: :class:`~.servo_policy.ServoOneLegPolicy` keyword arguments.

    ``base`` holds the collector's values for ``pos_step``, ``rot_step_deg``, ``screw_step_deg``,
    ``wind_step_deg``, ``insert_rate``, ``carry_dz``, ``over_dz``, ``hover_dz``,
    ``insert_yaw_deg``, ``press``, ``release_dz``, ``corner_clearance``, ``top_up_z``, and
    optionally ``upright_step_deg`` / ``pivot_step_deg`` (scaled with the held rotation when
    given). Returns those keys varied, plus ``push_via``, ``top_grasp_along``,
    ``top_grasp_yaw_deg``, ``leg_grasp_along`` and ``regrasp_dz``."""
    r = ranges or VariationRanges()
    u = lambda lo_hi: float(rng.uniform(*lo_hi))  # noqa: E731
    rot_f = u(r.rot_speed_f)
    out: Dict[str, Any] = {
        "pos_step": base["pos_step"] * u(r.pos_speed_f),
        "rot_step_deg": base["rot_step_deg"] * rot_f,
        "screw_step_deg": base["screw_step_deg"] * u(r.screw_speed_f),
        "wind_step_deg": base["wind_step_deg"] * u(r.wind_speed_f),
        "insert_rate": base["insert_rate"] * u(r.insert_rate_f),
        "carry_dz": base["carry_dz"] + u(r.carry_dz),
        "over_dz": base["over_dz"] + u(r.over_dz),
        "hover_dz": base["hover_dz"] + u(r.hover_dz),
        "insert_yaw_deg": base["insert_yaw_deg"] + u(r.insert_yaw_deg),
        "press": max(0.0, base["press"] + u(r.press)),
        "release_dz": max(0.01, base["release_dz"] + u(r.release_dz)),
        "corner_clearance": min(MAX_CORNER_CLEARANCE, base["corner_clearance"] + u(r.corner_clearance)),
        "top_up_z": base["top_up_z"] + u(r.top_up_z),
        "push_via": (u(r.push_via_x), u(r.push_via_y)),
        "top_grasp_along": u(r.top_grasp_along),
        "top_grasp_yaw_deg": u(r.top_grasp_yaw_deg),
        "leg_grasp_along": u(r.leg_grasp_along),
        "regrasp_dz": u(r.regrasp_dz),
    }
    for key in ("upright_step_deg", "pivot_step_deg"):
        if base.get(key) is not None:
            out[key] = base[key] * rot_f
    return out


def ranges_dict(ranges: Optional[VariationRanges] = None) -> Dict[str, Any]:
    return asdict(ranges or VariationRanges())


#: the free-space states (``ScriptedAction.state``) a DART burst may perturb: approaching the
#: top and the leg, rising, carrying and standing the held leg up, bringing it over the hole,
#: hovering for the regrasp. Never: pushing the top, closing on a part, lowering, inserting,
#: releasing, the screw pivot, screwing.
DART_STATES = frozenset(
    {
        "reach_body_grasp_xy",
        "top_up",
        "reach_leg_floor_xy",
        "reach_leg_ori",
        "leg_lift",
        "leg_upright",
        "leg_over",
        "release_up",
        "pre_grasp",
        "screw_release_up",
    }
)


class DartBursts:
    """Random constant-offset bursts on the commands sent (see the module docstring)."""

    def __init__(self, rng: np.random.Generator, prob: float, steps: int, pos: float, rot_deg: float) -> None:
        if not 0.0 <= prob <= 1.0 or steps < 1 or pos < 0 or rot_deg < 0:
            raise ValueError(f"DART: prob {prob} in [0, 1], steps {steps} >= 1, pos {pos} / rot {rot_deg} >= 0")
        self.rng, self.prob, self.steps = rng, float(prob), int(steps)
        self.pos, self.rot = float(pos), np.radians(float(rot_deg))
        self.left = 0
        self.dpos = np.zeros(3)
        self.drot = np.zeros(3)
        self.bursts = 0
        self.perturbed_steps = 0

    def _unit(self) -> np.ndarray:
        v = self.rng.normal(size=3)
        return v / max(float(np.linalg.norm(v)), 1e-12)

    def offset(self, state: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """This step's ``(position offset m, rotation offset rotvec rad)``, or None."""
        if state not in DART_STATES:
            self.left = 0  # a burst never carries into a contact state
            return None
        if self.left == 0 and self.prob > 0 and self.rng.random() < self.prob:
            self.left = self.steps
            self.dpos = self.pos * self._unit()
            self.drot = self.rot * self._unit()
            self.bursts += 1
        if self.left == 0:
            return None
        self.left -= 1
        self.perturbed_steps += 1
        return self.dpos.copy(), self.drot.copy()

    @staticmethod
    def apply(delta_pos: np.ndarray, delta_quat_wxyz: np.ndarray, off: Tuple[np.ndarray, np.ndarray]):
        """The perturbed ``(delta_pos, delta_quat)``: the offset added to the position command
        and composed onto the (EE-frame) rotation command."""
        dpos, drot = off
        return np.asarray(delta_pos, dtype=np.float64) + dpos, quat_mul(np.asarray(delta_quat_wxyz, dtype=np.float64), axisangle_to_quat(drot))
