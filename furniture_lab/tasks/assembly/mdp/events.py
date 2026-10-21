"""Reset events: the parts' reset randomization, the obstacles' offset, and clearing the
assembly latch. ``furniture.randomness`` picks the reset distribution, read at every reset:

* ``port`` (the default, every dataset so far): each part's (x, y) offset uniform in
  +-``pos_range`` and its yaw in +-``yaw_range_deg`` (``events.reset_parts.params``: 1.5 cm,
  15 deg), the whole draw redrawn while two parts' yawed footprint boxes overlap (no margin, no
  position limits; ``max_tries`` 20, then the nominal poses); the obstacles by
  ``furniture.obstacle_offset``; the arm at its reset joints.
* ``low`` / ``med`` / ``high``: FurnitureBench's own levels (``FurnitureSimEnv.reset_env``),
  exactly. ``low`` / ``med``: ``Furniture.randomize_init_pose`` -- each part's tag-frame (x, y)
  offset uniform in +-1.5 / +-5 cm and its yaw about the tag z in +-15 / +-45 deg, every part
  redrawn together until no two collide (``Part.is_collision``: the yawed ``reset_x_len`` x
  ``reset_y_len`` boxes' axis-aligned bounds, 1 cm margin) and every part lies inside the
  position limits (``Part.in_boundary``: tag x in [-0.21, 0.21], y in [0.07, 0.37] -- the legs
  sit on y = 0.07, so a leg only ever moves away from the robot), up to 300000 draws (here
  batched; none found raises). ``high``: one of the three recorded layouts
  (``randomize_init_pose_high``; ``furniture.high_layout``). The arm at its reset joints and the
  obstacles at their fixed poses, as there.
* ``kick_low`` / ``kick_med`` / ``kick_high``: ResiP's FurnitureRLSimEnv reset (what their BC / RL
  numbers, checkpoints and demos come from): the parts stay at their nominal poses and get one
  kick each (a world-frame xy force of random direction and magnitude, a z torque;
  :class:`PartKick`), the arm joints get uniform noise, and the obstacles the level's offset.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Optional, Sequence

import torch

from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

from ...furniture import square_table as fb
from ...furniture.frames import TAG_ROT_W
from ...furniture.square_table import PARTS
from .state import tracker

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

#: ResiP's reset levels (FurnitureRLSimEnv.__init__): the largest kick force (N, before the per-part
#: multiplier) and z torque (N m), the obstacles' offset (m, uniform +- in x and y, all three
#: together) and the arm joints' noise (deg, uniform +- per joint)
RESIP_RANDOMNESS: Dict[str, Dict[str, float]] = {
    "low": {"force": 0.2, "torque": 0.007, "obstacle_offset": 0.02, "joint_noise_deg": 5.0},
    "med": {"force": 0.5, "torque": 0.01, "obstacle_offset": 0.04, "joint_noise_deg": 10.0},
    "high": {"force": 0.75, "torque": 0.015, "obstacle_offset": 0.06, "joint_noise_deg": 13.0},
}
#: their one_leg / square_table multipliers, per part in PARTS order (top, leg1..leg4)
RESIP_KICK_FORCE_MUL = (25.0, 1.0, 1.0, 1.0, 1.0)
RESIP_KICK_TORQUE_MUL = (70.0, 1.0, 1.0, 1.0, 1.0)
#: how long the kick acts: one of their simulate() calls (dt 1/60 s; Isaac Gym applies a force for
#: the whole call, both substeps -- measured)
RESIP_KICK_DURATION = 1.0 / 60.0
#: ``furniture.randomness`` values: this port's own uniform layout, FurnitureBench's levels, ResiP's
RANDOMNESS_LEVELS = ("port", "low", "med", "high") + tuple(f"kick_{k}" for k in RESIP_RANDOMNESS)
#: candidate draws per env per batched round of the FurnitureBench layout's rejection sampling
FB_DRAWS_PER_ROUND = 128


def randomness(furniture) -> str:
    """``furniture.randomness``, checked. A cfg without the field is ``port``."""
    name = getattr(furniture, "randomness", "port")
    if name not in RANDOMNESS_LEVELS:
        raise ValueError(f"furniture.randomness must be one of {RANDOMNESS_LEVELS}, got {name!r}")
    return name


def resip_level(furniture) -> Optional[Dict[str, float]]:
    """``furniture.randomness`` ``kick_<level>`` -> ResiP's level (``RESIP_RANDOMNESS``); any other
    value -> None."""
    name = randomness(furniture)
    return RESIP_RANDOMNESS[name[len("kick_"):]] if name.startswith("kick_") else None


def randomize_parts_pose(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    pos_range: float = 0.015,
    yaw_range_deg: float = 15.0,
    max_tries: int = 20,
) -> None:
    """Perturb every part's default root pose by a planar offset and a yaw, collision-free.

    Runs after ``reset_scene_to_default`` has put the parts back at their nominal poses, and
    writes root pose and zero velocity for ``env_ids``. This is ``furniture.randomness: port``;
    any other value runs that level's reset instead (the params are then unused):
    FurnitureBench's ``low`` / ``med`` (:func:`fb_layout`) or ``high`` (:func:`fb_high_layout`),
    ResiP's ``kick_*`` (:func:`resip_reset`).
    """
    name = randomness(env.cfg.furniture)
    if name in fb.RANDOMIZE_INIT_POSE:
        pos, yaw = fb.RANDOMIZE_INIT_POSE[name]
        fb_layout(env, env_ids, pos, yaw)
        return
    if name == "high":
        fb_high_layout(env, env_ids, int(getattr(env.cfg.furniture, "high_layout", 0)))
        return
    level = resip_level(env.cfg.furniture)
    if level is not None:
        resip_reset(env, env_ids, level)
        return
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    if n == 0:
        return
    names = env.cfg.furniture.part_names
    specs = {p.name: p for p in PARTS}
    P = len(names)
    device = env.device

    # default_root_state is in the env's local frame: add the env origins, as
    # reset_scene_to_default does
    nominal_pos = torch.stack(
        [env.scene[name].data.default_root_state[env_ids, :3] for name in names], dim=1
    ) + env.scene.env_origins[env_ids].unsqueeze(1)  # (n, P, 3) world frame
    nominal_quat = torch.stack(
        [env.scene[name].data.default_root_state[env_ids, 3:7] for name in names], dim=1
    )
    foot = torch.tensor([specs[name].footprint for name in names], device=device)  # (P, 2)
    eye = torch.eye(P, dtype=torch.bool, device=device)

    offset = torch.zeros(n, P, 3, device=device)
    yaw = torch.zeros(n, P, device=device)
    pending = torch.ones(n, dtype=torch.bool, device=device)
    yaw_range = math.radians(yaw_range_deg)
    for _ in range(max_tries):
        idx = pending.nonzero(as_tuple=False).squeeze(-1)
        if len(idx) == 0:
            break
        offset[idx, :, :2] = (torch.rand(len(idx), P, 2, device=device) * 2 - 1) * pos_range
        yaw[idx] = (torch.rand(len(idx), P, device=device) * 2 - 1) * yaw_range
        xy = nominal_pos[idx, :, :2] + offset[idx, :, :2]  # (k, P, 2)
        # each part's footprint box, rotated by its yaw, bounded by an axis-aligned box
        c, s = torch.cos(yaw[idx]).abs(), torch.sin(yaw[idx]).abs()  # (k, P)
        hx = foot[:, 0] * c + foot[:, 1] * s
        hy = foot[:, 0] * s + foot[:, 1] * c
        dx = (xy[:, :, None, 0] - xy[:, None, :, 0]).abs()  # (k, P, P)
        dy = (xy[:, :, None, 1] - xy[:, None, :, 1]).abs()
        overlap = (dx < hx[:, :, None] + hx[:, None, :]) & (dy < hy[:, :, None] + hy[:, None, :])
        pending[idx] = (overlap & ~eye).any(dim=(1, 2))
    if pending.any():
        print(
            f"[furniture] {int(pending.sum())} env(s) found no collision-free part layout in "
            f"{max_tries} tries; using the nominal poses there",
            flush=True,
        )
        offset[pending] = 0.0
        yaw[pending] = 0.0

    zeros = torch.zeros_like(yaw)
    yaw_quat = quat_from_euler_xyz(zeros, zeros, yaw)  # (n, P, 4), about world z
    pos = nominal_pos + offset
    quat = quat_mul(yaw_quat, nominal_quat)
    vel = torch.zeros(n, 6, device=device)
    for k, name in enumerate(names):
        asset = env.scene[name]
        asset.write_root_pose_to_sim(torch.cat([pos[:, k], quat[:, k]], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(vel, env_ids=env_ids)


# the tag frame's z is the world's -z (or +z): a tag-frame yaw is a world yaw of that sign
if abs(abs(TAG_ROT_W[2][2]) - 1.0) > 1e-9 or abs(TAG_ROT_W[2][0]) + abs(TAG_ROT_W[2][1]) > 1e-9:
    raise ValueError("the FurnitureBench layout assumes the tag frame's z is the world's +-z")
_TAG_YAW_SIGN = float(round(TAG_ROT_W[2][2]))


def _write_parts(env, env_ids, names, pos, quat) -> None:
    vel = torch.zeros(len(env_ids), 6, device=env.device)
    for k, name in enumerate(names):
        asset = env.scene[name]
        asset.write_root_pose_to_sim(torch.cat([pos[:, k], quat[:, k]], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(vel, env_ids=env_ids)


def fb_layout(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, pos_range: float, yaw_range_deg: float) -> None:
    """FurnitureBench's ``Furniture.randomize_init_pose(from_skill=0, pos_range, rot_range)``,
    exactly: per part a tag-frame (x, y) offset ``U(-pos_range, pos_range)`` from its reset
    position and a yaw ``U(-rot, rot)`` about the tag z left-multiplied onto its reset rotation;
    every part redrawn together until ``Furniture._in_boundary`` (every part's x, y inside
    ``PARTS_POS_LIM``, bounds included) and not ``Furniture._check_collision`` (two parts collide
    unless their yawed ``RESET_LEN`` boxes' axis-aligned bounds are more than ``COLLISION_MARGIN``
    apart in x or in y), up to ``RANDOMIZE_MAX_TRIAL`` draws. Batched: each round draws
    ``FB_DRAWS_PER_ROUND`` candidates per pending env and keeps its first valid one, which is
    the same distribution as upstream's one-at-a-time loop. No valid draw at all raises."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    if n == 0:
        return
    device = env.device
    names = env.cfg.furniture.part_names
    specs = {p.name: p for p in PARTS}
    P = len(names)
    nominal_tag = torch.tensor([specs[name].reset_pos_tag[:2] for name in names], device=device)  # (P, 2)
    half = torch.tensor([fb.RESET_LEN[name] for name in names], device=device) / 2  # (P, 2)
    (x_lo, x_hi), (y_lo, y_hi) = fb.PARTS_POS_LIM
    margin = fb.COLLISION_MARGIN
    yaw_range = math.radians(yaw_range_deg)
    eye = torch.eye(P, dtype=torch.bool, device=device)

    off = torch.zeros(n, P, 2, device=device)
    yaw = torch.zeros(n, P, device=device)
    pending = torch.ones(n, dtype=torch.bool, device=device)
    draws = 0
    while draws < fb.RANDOMIZE_MAX_TRIAL:
        idx = pending.nonzero(as_tuple=False).squeeze(-1)
        if len(idx) == 0:
            break
        K = min(FB_DRAWS_PER_ROUND, fb.RANDOMIZE_MAX_TRIAL - draws)
        o = (torch.rand(len(idx), K, P, 2, device=device) * 2 - 1) * pos_range
        y = (torch.rand(len(idx), K, P, device=device) * 2 - 1) * yaw_range
        xy = nominal_tag + o  # (k, K, P, 2), tag frame
        inside = ((xy[..., 0] >= x_lo) & (xy[..., 0] <= x_hi) & (xy[..., 1] >= y_lo) & (xy[..., 1] <= y_hi)).all(-1)
        c, s = torch.cos(y).abs(), torch.sin(y).abs()
        hx = half[:, 0] * c + half[:, 1] * s  # the yawed box's axis-aligned half extents
        hy = half[:, 0] * s + half[:, 1] * c
        dx = (xy[..., :, None, 0] - xy[..., None, :, 0]).abs()  # (k, K, P, P)
        dy = (xy[..., :, None, 1] - xy[..., None, :, 1]).abs()
        hit = (dx <= hx[..., :, None] + hx[..., None, :] + margin) & (dy <= hy[..., :, None] + hy[..., None, :] + margin)
        ok = inside & ~(hit & ~eye).any(dim=(-1, -2))  # (k, K)
        found = ok.any(-1)
        first = ok.float().argmax(-1)  # the first valid candidate (argmax returns the first max)
        rows = found.nonzero(as_tuple=False).squeeze(-1)
        off[idx[rows]] = o[rows, first[rows]]
        yaw[idx[rows]] = y[rows, first[rows]]
        pending[idx[rows]] = False
        draws += K
    if pending.any():
        raise RuntimeError(
            f"FurnitureBench layout: {int(pending.sum())} env(s) found no valid draw in {fb.RANDOMIZE_MAX_TRIAL} "
            f"(+-{pos_range} m, +-{yaw_range_deg} deg)"
        )
    # tag frame -> world: offset R_tag[:, :2] (x, y), yaw about the tag z = +-world z
    R = torch.tensor(TAG_ROT_W, dtype=torch.float32, device=device)
    off_w = off @ R[:, :2].T  # (n, P, 3)
    nominal_pos = torch.stack(
        [env.scene[name].data.default_root_state[env_ids, :3] for name in names], dim=1
    ) + env.scene.env_origins[env_ids].unsqueeze(1)
    nominal_quat = torch.stack([env.scene[name].data.default_root_state[env_ids, 3:7] for name in names], dim=1)
    zeros = torch.zeros_like(yaw)
    yaw_quat = quat_from_euler_xyz(zeros, zeros, _TAG_YAW_SIGN * yaw)
    _write_parts(env, env_ids, names, nominal_pos + off_w, quat_mul(yaw_quat, nominal_quat))


#: FurnitureBench's three high-randomness layouts in the world frame, per mesh frame: (pos (3, P, 3), quat (3, P, 4))
_HIGH_WORLD: Dict[str, tuple] = {}


def fb_high_layout(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, layout: int) -> None:
    """FurnitureBench's ``high`` randomness (``Furniture.randomize_high`` /
    ``Part.randomize_init_pose_high``): every part at layout ``layout`` (0, 1, 2) of its
    recorded ``high_rand_reset_pos`` / ``high_rand_reset_ori``; -1 draws one of the three per env
    at every reset."""
    if layout not in (-1, 0, 1, 2):
        raise ValueError(f"furniture.high_layout must be 0, 1, 2 or -1 (random), got {layout}")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    if n == 0:
        return
    device = env.device
    names = env.cfg.furniture.part_names
    frame = env.cfg.furniture.mesh_frame
    if frame not in _HIGH_WORLD:
        poses = [[fb.tag_pose_world(*fb.HIGH_RAND_RESET[name][i], frame) for name in names] for i in range(3)]
        pos = torch.tensor([[p for p, _ in row] for row in poses], dtype=torch.float32)
        quat = torch.tensor([[q for _, q in row] for row in poses], dtype=torch.float32)
        _HIGH_WORLD[frame] = (pos, quat / quat.norm(dim=-1, keepdim=True))
    pos, quat = (t.to(device) for t in _HIGH_WORLD[frame])
    which = torch.randint(0, 3, (n,), device=device) if layout == -1 else torch.full((n,), layout, device=device)
    world = pos[which] + env.scene.env_origins[env_ids].unsqueeze(1)  # (n, P, 3)
    _write_parts(env, env_ids, names, world, quat[which])


#: the scene names of the three obstacles (``AssemblySceneCfg``)
OBSTACLE_NAMES = ("obstacle_front", "obstacle_left", "obstacle_right")


def randomize_obstacles(env: "ManagerBasedRLEnv", env_ids: torch.Tensor) -> None:
    """Move the three obstacles together by one uniform (x, y) offset per env, in
    ``[-r, r]`` with ``r = env.cfg.furniture.obstacle_offset`` (m; 0 leaves them at their
    defaults), as ResiP's FurnitureRLSimEnv does (``_reset_part_poses``: one offset for all
    three, z untouched; low randomness 0.02). Runs after ``reset_scene_to_default``; the field
    is read at every reset, so a YAML override of it takes effect. It is used only under
    ``furniture.randomness: port``: ResiP's ``kick_*`` levels use their own offset, FurnitureBench's
    ``low`` / ``med`` / ``high`` keep the obstacles fixed (FurnitureSimEnv never moves them)."""
    name = randomness(env.cfg.furniture)
    level = resip_level(env.cfg.furniture)
    if name == "port":
        r = float(env.cfg.furniture.obstacle_offset)
    else:
        r = 0.0 if level is None else float(level["obstacle_offset"])
    if r <= 0.0:
        return
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    if n == 0:
        return
    offset = torch.zeros(n, 3, device=env.device)
    offset[:, :2] = (torch.rand(n, 2, device=env.device) * 2 - 1) * r
    for name in OBSTACLE_NAMES:
        asset = env.scene[name]
        default = asset.data.default_root_state[env_ids]
        pos = default[:, :3] + env.scene.env_origins[env_ids] + offset
        asset.write_root_pose_to_sim(torch.cat([pos, default[:, 3:7]], dim=-1), env_ids=env_ids)


def resip_reset(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, level: Dict[str, float]) -> None:
    """ResiP's FurnitureRLSimEnv reset of the robot and the parts (``_reset_frankas``,
    ``_reset_part_poses``, ``_apply_forces_to_parts``), after ``reset_scene_to_default`` has put
    both at their defaults:

    * each arm joint at its reset value plus ``U(-a, a)``, ``a = joint_noise_deg``, at rest, its
      position target there too (the fingers keep their default, open);
    * each part at its nominal pose, kicked by a world-frame xy force of direction ``U(0, 2 pi)``
      and magnitude ``U(0, force) * RESIP_KICK_FORCE_MUL[part]`` and a z torque
      ``U(-torque, torque) * RESIP_KICK_TORQUE_MUL[part]``, at its centre of mass, for
      ``RESIP_KICK_DURATION`` (:class:`PartKick`).

    Their kick acts in the one simulate() their reset() ends with, before the first observation;
    here it acts in the first control step's physics steps, so the first observation is the
    parts' nominal pose and they slide while the first action runs."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    if n == 0:
        return
    device = env.device
    # -- the arm
    robot = env.scene["robot"]
    arm_ids, _ = robot.find_joints("panda_joint.*")
    a = math.radians(float(level["joint_noise_deg"]))
    pos = robot.data.default_joint_pos[env_ids][:, arm_ids].clone()
    pos += (torch.rand(n, len(arm_ids), device=device) * 2 - 1) * a
    robot.write_joint_state_to_sim(pos, torch.zeros_like(pos), joint_ids=arm_ids, env_ids=env_ids)
    robot.set_joint_position_target(pos, joint_ids=arm_ids, env_ids=env_ids)
    # -- the parts' kick
    P = len(env.cfg.furniture.part_names)
    if P != len(RESIP_KICK_FORCE_MUL):
        raise ValueError(f"ResiP's kick multipliers are for the 5 square-table parts, the task has {P}")
    f_mul = torch.tensor(RESIP_KICK_FORCE_MUL, device=device)
    t_mul = torch.tensor(RESIP_KICK_TORQUE_MUL, device=device)
    theta = torch.rand(n, P, device=device) * 2 * math.pi
    mag = torch.rand(n, P, device=device) * float(level["force"]) * f_mul
    force = torch.stack((mag * torch.cos(theta), mag * torch.sin(theta), torch.zeros_like(mag)), dim=-1)
    tz = (torch.rand(n, P, device=device) * 2 - 1) * float(level["torque"]) * t_mul
    torque = torch.stack((torch.zeros_like(tz), torch.zeros_like(tz), tz), dim=-1)
    kick = getattr(env, "part_kick", None)
    if kick is None:
        kick = env.part_kick = PartKick(env)
    kick.arm(env_ids, force, torque)


class PartKick:
    """The parts' reset kick (:func:`resip_reset`): a world-frame force and torque per part, held
    for the physics steps that make ``RESIP_KICK_DURATION`` and then removed. :meth:`arm` at a
    reset; :meth:`apply` before every physics step (``AssemblyEnv`` calls it with the action)."""

    def __init__(self, env: "ManagerBasedRLEnv") -> None:
        names = env.cfg.furniture.part_names
        self.assets = [env.scene[name] for name in names]
        steps = RESIP_KICK_DURATION / float(env.physics_dt)
        if abs(steps - round(steps)) > 1e-6 or round(steps) < 1:
            raise ValueError(f"physics dt {env.physics_dt} does not divide ResiP's kick duration {RESIP_KICK_DURATION}")
        self.steps = int(round(steps))
        n, P = env.num_envs, len(names)
        self.force = torch.zeros(n, P, 3, device=env.device)
        self.torque = torch.zeros(n, P, 3, device=env.device)
        self.left = torch.zeros(n, dtype=torch.long, device=env.device)
        self.active = False

    def arm(self, env_ids: torch.Tensor, force: torch.Tensor, torque: torch.Tensor) -> None:
        self.force[env_ids] = force
        self.torque[env_ids] = torque
        self.left[env_ids] = self.steps
        self.active = True

    def apply(self) -> None:
        if not self.active:
            return
        # every env written at once: an asset's wrench switch is all-or-nothing
        on = (self.left > 0)[:, None, None]
        force = torch.where(on, self.force, 0.0)
        torque = torch.where(on, self.torque, 0.0)
        for k, asset in enumerate(self.assets):
            asset.set_external_force_and_torque(force[:, k : k + 1], torque[:, k : k + 1], is_global=True)
        self.left = (self.left - 1).clamp(min=0)
        # still on: keep going; just switched off: this call wrote the zeros, done
        self.active = bool(on.any())


def reset_assembly_tracker(env: "ManagerBasedRLEnv", env_ids: torch.Tensor) -> None:
    """Clear the assembled-pair latch for the envs that restart."""
    tracker(env).reset(env_ids)
