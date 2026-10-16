"""Do the demos' absolute actions still succeed when replayed open loop?

    python scripts/replay_abs.py --demos ~/Robonuke_Furniture_Bench/demos_resip
    python scripts/replay_abs.py --demos DEMO.pkl --headless
    python scripts/replay_abs.py --zarr ~/resip_data/one_leg_teleop_low/<name>.zarr   # ResiP's demos

The BC policy trains on ``action.abs``: absolute rot6d targets (``furniture_lab.demos.fb_abs_frames``
for our pkls, the importer's own conversion; ResiP's processed zarr already holds them as
``action/pos``). This script puts the ``-FBAbs-ResiP`` env into each demo's first recorded
state, sends that demo's absolute actions one per step with no feedback, and reports whether
the leg ends up assembled (the env's own success check, soft by default).

The first state:

* **pkl** (ours): arm joints, fingers and part poses written exactly.
* **zarr** (ResiP's, no joint positions; its robot base sits 15 mm lower than here, so its
  EE and target z are shifted by ``frames.RESIP_BASE_Z_TO_PORT``): the arm is driven to the demo's first EE pose by the
  env's own differential-IK controller (``--settle`` steps of the first pose as the absolute
  target, the first gripper command), then the fingers are set to the recorded width and the
  parts written at their recorded poses. The arm's null-space posture can differ from the
  recording's (a 7-dof arm, one EE pose); the EE pose is what the actions are about. The
  settle error is printed per demo.

``--carry`` steps after each grasp (the gripper command turning positive: first the table
top, then the leg), the held part's position relative to the EE is printed for this env and
for the demo (``here minus demo``, robot-base axes, mm). The fingers centre a part across
their closing axis; along the other axes a consistent offset across demos is where the
fingers closed on it differently -- a frame offset between the demos' sim and this one. A
large one in every axis means that grasp missed.

That is the ceiling a perfect clone of these demos can reach in this env:

* most demos succeed -> the action interface reproduces the demos; the BC gap is learning.
* most fail -> what the policy is taught does not do what the demos did; look at where the
  EE error grows (per-step lines) and where the leg goes.

Some open-loop drift is physics (contact is chaotic), so a few failures are expected; a
failure rate near the BC policy's is the signal. Successful demos only (``fails/`` skipped,
zarr episodes with ``success`` 0 skipped).
"""

import argparse
import math
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
source = parser.add_mutually_exclusive_group(required=True)
source.add_argument("--demos", help="a .pkl, or a directory searched recursively (fails/ skipped)")
source.add_argument("--zarr", help="ResiP's processed .zarr (robust-rearrangement process_pickles.py)")
parser.add_argument("--task", default="Isaac-Furniture-OneLeg-FBAbs-ResiP-v0")
parser.add_argument("--max_demos", type=int, default=None)
parser.add_argument("--hold", type=int, default=20, help="steps holding the last action after the demo ends")
parser.add_argument("--settle", type=int, default=30, help="zarr: IK steps driving the arm to the first EE pose")
parser.add_argument("--carry", type=int, default=15, help="steps after each grasp to compare the held part")
parser.add_argument("--print_every", type=int, default=50, help="per-step EE error lines (0: off)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_error_magnitude  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

import furniture_lab.tasks  # noqa: E402, F401
from furniture_lab.demos import (  # noqa: E402
    _parts_to_world,
    fb_abs_frames,
    load_demo,
    mat_to_rot6d,
    quat_to_mat,
    quat_xyzw_to_wxyz,
)
from furniture_lab.tasks.assembly.assembly_env_cfg import FBABS_OBS_LAYOUT  # noqa: E402
from furniture_lab.tasks.assembly.mdp.rot6d import rot6d_to_quat  # noqa: E402
from furniture_lab.tasks.assembly.mdp.state import tracker  # noqa: E402
from furniture_lab.tasks.furniture.square_table import MESH_FRAMES, PART_NAMES, mesh_frame  # noqa: E402
from furniture_lab.tasks.furniture.frames import (  # noqa: E402
    OBSTACLE_FRONT_POS,
    RESIP_BASE_Z_TO_PORT,
    TAG_POS_W,
    TAG_ROT_W,
)

SL, _s = {}, 0
for _name, _w in FBABS_OBS_LAYOUT:
    SL[_name] = slice(_s, _s + _w)
    _s += _w


# ------------------------------------------------------------------------------ the episodes
def pkl_episodes(root: Path):
    """``(name, states (T+1, 61), actions (T, 10), start(env), recorded_success)`` per pkl."""
    if root.is_file():
        paths = [root]
    else:
        paths = sorted(p for p in root.rglob("*.pkl") if "fails" not in p.parts and not p.name.startswith("fail"))
    if not paths:
        raise SystemExit(f"no demo .pkl under {root}")
    for path in paths[: args.max_demos] if args.max_demos else paths:
        demo = load_demo(path)
        with open(path, "rb") as f:
            states, acts = fb_abs_frames(pickle.load(f))
        yield path.name, states, acts, (lambda env, demo=demo: set_state_exact(env, demo, 0)), demo.success


def zarr_states_abs(robot_state: np.ndarray, parts: np.ndarray) -> np.ndarray:
    """(T, 16) robot state + (T, 35) tag-frame xyzw part poses -> (T, 61) FBAbs observations."""
    T = robot_state.shape[0]
    p = parts.reshape(T, 5, 7)
    rot = np.stack([[mat_to_rot6d(quat_to_mat(quat_xyzw_to_wxyz(p[t, i, 3:7]))) for i in range(5)] for t in range(T)])
    return np.concatenate([robot_state, np.concatenate([p[:, :, :3], rot], axis=-1).reshape(T, 45)], axis=-1).astype(np.float32)


def check_obstacle(recorded_tag: np.ndarray) -> None:
    """Once, at the start: ResiP's recorded front-obstacle position (tag frame, every frame)
    against this scene's (``frames.OBSTACLE_FRONT_POS``, robot-base frame, into the tag frame)."""
    ours = TAG_ROT_W.T @ (np.asarray(OBSTACLE_FRONT_POS) - np.asarray(TAG_POS_W))
    theirs, spread = recorded_tag.mean(0), recorded_tag.std(0)
    d = (ours - theirs) * 1000
    print(
        f"[replay] front obstacle, tag frame: demos {np.round(theirs, 4)} (spread {np.round(spread * 1000, 2)} mm), "
        f"this scene {np.round(ours, 4)}; here minus demos x {d[0]:+.1f} y {d[1]:+.1f} z {d[2]:+.1f} mm "
        f"(robot-base axes: {np.round(TAG_ROT_W @ d, 1)} mm)",
        flush=True,
    )


def zarr_episodes(path: Path):
    """The same tuple per successful episode of ResiP's zarr; the start reached by IK."""
    import zarr

    z = zarr.open(str(path), mode="r")
    ends = np.asarray(z["episode_ends"][:], dtype=np.int64)
    robot_state = np.asarray(z["robot_state"][:], dtype=np.float32)
    parts = np.asarray(z["parts_poses"][:], dtype=np.float32)
    # ResiP's sim appends the front obstacle's pose (a 6th, fixed body) after the 5 parts
    if parts.shape[1] not in (35, 42):
        raise SystemExit(f"{path}: parts_poses is {parts.shape[1]} wide; expected 35 (5 parts) or 42 (+ obstacle)")
    if parts.shape[1] == 42:
        check_obstacle(parts[:, 35:38])
    parts = parts[:, :35]
    actions = np.asarray(z["action/pos"][:], dtype=np.float32)
    # ResiP's robot base sits 15 mm lower (frames.RESIP_ROBOT_HEIGHT): its base-frame EE z and
    # target z, into this port's base frame
    robot_state[:, 2] += RESIP_BASE_Z_TO_PORT
    actions[:, 2] += RESIP_BASE_Z_TO_PORT
    success = np.asarray(z["success"][:], dtype=np.int64) if "success" in z else np.ones(len(ends), np.int64)
    start, n = 0, 0
    for e, end in enumerate(ends):
        lo, hi = start, int(end)
        start = hi
        if not success[e]:
            continue
        if args.max_demos and n >= args.max_demos:
            break
        n += 1
        states = zarr_states_abs(robot_state[lo:hi], parts[lo:hi])
        states = np.concatenate([states, states[-1:]])  # no observation after the last action
        first = dict(ee=states[0, 0:9], width=float(robot_state[lo, 15]), parts=parts[lo], grip=float(actions[lo, 9]))
        yield f"episode {e}", states, actions[lo:hi], (lambda env, first=first: set_state_ik(env, **first)), True


# ------------------------------------------------------------------------------ the start state
def _write_parts(env, pos: np.ndarray, quat: np.ndarray, names) -> None:
    origin = env.scene.env_origins[:1]
    for i, name in enumerate(names):
        part = env.scene[name]
        p = torch.tensor(pos[i], device=env.device, dtype=torch.float32).unsqueeze(0) + origin
        q = torch.tensor(quat[i], device=env.device, dtype=torch.float32).unsqueeze(0)
        part.write_root_pose_to_sim(torch.cat([p, q], dim=-1))
        part.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))


def _settle_physics(env) -> None:
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(env.physics_dt)
    tracker(env).reset()


def set_state_exact(env, demo, t: int) -> None:
    """Env 0 into a pkl demo's state at step t (arm joints, fingers, part poses)."""
    robot = env.scene["robot"]
    arm = robot.find_joints("panda_joint.*")[0]
    fingers = robot.find_joints("panda_finger_joint.*")[0]
    q = robot.data.default_joint_pos.clone()
    q[:, arm] = torch.tensor(demo.joint_positions[t], device=env.device, dtype=q.dtype)
    q[:, fingers] = float(demo.gripper_width[t]) / 2
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    robot.set_joint_position_target(q)
    _write_parts(env, demo.parts_pos[t], demo.parts_quat[t], demo.part_names)
    _settle_physics(env)


def set_state_ik(env, ee: np.ndarray, width: float, parts: np.ndarray, grip: float) -> str:
    """Drive the arm to EE pose ``ee`` (pos 3, rot6d 6) with the env's own IK controller, then
    set the fingers to ``width`` and write the parts; returns the settle error."""
    target = torch.tensor(np.concatenate([ee, [grip]]), device=env.device, dtype=torch.float32).reshape(1, -1)
    for _ in range(args.settle):
        env.step(target)
    robot = env.scene["robot"]
    fingers = robot.find_joints("panda_finger_joint.*")[0]
    q = robot.data.joint_pos.clone()
    q[:, fingers] = width / 2
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    pos, quat = _parts_to_world(parts, MESH_FRAMES[mesh_frame()])
    _write_parts(env, pos, quat, PART_NAMES[: pos.shape[0]])
    _settle_physics(env)
    o = obs(env)
    p, r = ee_err(o, np.concatenate([ee, np.zeros(_s - 9, np.float32)]))
    return f"settle error {p * 1000:.1f} mm {r:.1f} deg"


# ------------------------------------------------------------------------------ the replay
def obs(env) -> np.ndarray:
    return env.observation_manager.compute()["policy"][0].detach().cpu().numpy()


def ee_err(a: np.ndarray, b: np.ndarray) -> tuple:
    p = float(np.linalg.norm(a[SL["ee_pos"]] - b[SL["ee_pos"]]))
    qa = rot6d_to_quat(torch.tensor(a[SL["ee_rot6d"]]).reshape(1, 6))
    qb = rot6d_to_quat(torch.tensor(b[SL["ee_rot6d"]]).reshape(1, 6))
    return p, math.degrees(float(quat_error_magnitude(qa, qb)))


#: one_leg's grasps, in order: the table top (parts slot 0), then the leg (square_table_leg4,
#: slot 4)
GRASPS = (("top", 0), ("leg", 4))


def part_from_ee(o: np.ndarray, slot: int) -> np.ndarray:
    """A part's position relative to the EE, in the robot-base frame (m). The observation
    holds the EE in the base frame and the parts in the tag frame."""
    part_tag = o[SL["parts_poses"]].reshape(-1, 9)[slot, :3]
    part_base = np.asarray(TAG_POS_W) + TAG_ROT_W @ part_tag
    return part_base - o[SL["ee_pos"]]


def replay(env, name, states, acts, start, recorded_success) -> dict:
    env.reset()
    note = start(env) or ""
    T = acts.shape[0]
    errs, first = [], None
    # each grasped part's offset from the EE once it is held: --carry steps after each close
    # (the gripper command turning positive), here and in the demo. Fingers centre a part
    # across their closing axis, so a frame offset between the demos' sim and this one shows
    # up along the other two axes
    closes = [t for t in range(T) if acts[t, 9] > 0 and (t == 0 or acts[t - 1, 9] <= 0)]
    checks = {c + args.carry: (name, slot) for c, (name, slot) in zip(closes, GRASPS) if c + args.carry < T}
    grasps = []
    for t in range(T + args.hold):
        a = torch.tensor(acts[min(t, T - 1)], device=env.device, dtype=torch.float32).reshape(1, -1)
        env.step(a)
        if first is None and bool(tracker(env).all_assembled()[0]):
            first = t
        if t in checks:
            part, slot = checks[t]
            d = (part_from_ee(obs(env), slot) - part_from_ee(states[t + 1], slot)) * 1000
            grasps.append(f"held {part} from EE, here minus demo: x {d[0]:+.1f} y {d[1]:+.1f} z {d[2]:+.1f} mm")
            print(f"[replay]   {grasps[-1]} ({args.carry} steps after that grasp, step {t})", flush=True)
        if t < T:
            p, r = ee_err(obs(env), states[t + 1])
            errs.append((p, r))
            if args.print_every and t % args.print_every == 0:
                print(f"[replay]   step {t:4d}: EE vs demo {p * 1000:6.1f} mm {r:5.1f} deg", flush=True)
    o = obs(env)
    parts_now = o[SL["parts_poses"]].reshape(-1, 9)[:, :3]
    parts_demo = states[-1][SL["parts_poses"]].reshape(-1, 9)[:, :3]
    e = np.array(errs)
    return {
        "name": name,
        "note": "; ".join(x for x in (note, *grasps) if x),
        "steps": T,
        "demo_success": recorded_success,
        "success": first is not None,
        "at": first,
        "ee_mm_mean": e[:, 0].mean() * 1000,
        "ee_mm_max": e[:, 0].max() * 1000,
        "ee_deg_max": e[:, 1].max(),
        "part_mm_end": float(np.linalg.norm(parts_now - parts_demo, axis=-1).max()) * 1000,
    }


def main() -> None:
    episodes = zarr_episodes(Path(args.zarr).expanduser()) if args.zarr else pkl_episodes(Path(args.demos).expanduser())
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 1e6
    cfg.terminations.success = None  # success read from the tracker, the replay keeps going
    cfg.terminations.part_out_of_limits = None
    env = gym.make(args.task, cfg=cfg).unwrapped
    env.reset()
    width = int(env.single_observation_space["policy"].shape[-1])
    if width != _s:
        raise SystemExit(f"{args.task} observation is {width} wide, expected the FBAbs {_s}")
    print(f"[replay] {args.zarr or args.demos} in {args.task}, success_mode={cfg.furniture.success_mode}", flush=True)

    rows = []
    for i, (name, states, acts, start, rec) in enumerate(episodes):
        print(f"[replay] {i + 1}: {name}", flush=True)
        r = replay(env, name, states, acts, start, rec)
        rows.append(r)
        print(
            f"[replay] {r['name']}: {'SUCCESS at step ' + str(r['at']) if r['success'] else 'FAIL'} "
            f"(demo {r['steps']} steps{', ' + r['note'] if r['note'] else ''}); EE vs demo mean {r['ee_mm_mean']:.1f} mm, "
            f"max {r['ee_mm_max']:.1f} mm / {r['ee_deg_max']:.1f} deg; worst part off the demo's end pose by {r['part_mm_end']:.1f} mm",
            flush=True,
        )

    n = sum(r["success"] for r in rows)
    print(f"[replay] ===== {n}/{len(rows)} demos succeed replayed open loop ({100 * n / len(rows):.0f}%)", flush=True)
    print(
        f"[replay] EE tracking vs the demos: mean {np.mean([r['ee_mm_mean'] for r in rows]):.1f} mm; "
        f"failed: {', '.join(r['name'] for r in rows if not r['success']) or 'none'}",
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()
    app.close()
