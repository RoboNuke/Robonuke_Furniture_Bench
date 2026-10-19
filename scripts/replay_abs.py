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
  EE and target z are shifted by ``frames.RESIP_BASE_Z_TO_PORT``; its sim offsets the three
  obstacles per episode -- +-2 cm in x and y at low randomness -- and records the front one, so
  they are moved to each demo's recorded place): the arm is driven to the demo's first EE pose by the
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
failure rate near the BC policy's is the signal. ``--save DIR`` writes each demo's per-step
data (this env and the demo side by side, plus the arm and finger joints) to ``DIR/<name>.npz``. Successful demos only (``fails/`` skipped,
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
parser.add_argument(
    "--save", default=None, metavar="DIR",
    help="write one .npz per demo with the per-step data, this env and the demo side by side (see save_trace)",
)
# isolating the ResiP gripper emulation and the zarr rotation rescale (defaults: the task's / on)
parser.add_argument("--grip_stiffness", type=float, default=None, help="finger drive N/m (ResiP task: 600)")
parser.add_argument("--grip_damping", type=float, default=None, help="finger drive N s/m (ResiP task: 10; old 40)")
parser.add_argument("--grip_force", type=float, default=None, help="finger effort cap N (ResiP task: 13)")
parser.add_argument("--grip_speed", type=float, default=None, help="finger speed cap m/s (ResiP task: 0.2); 0 = the asset default")
parser.add_argument("--no_rot_rescale", action="store_true", help="zarr: keep ResiP's shrunk rotation deltas as stored")
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
    obstacle_tag_rot6d,
    quat_xyzw_to_wxyz,
    resip_unscale_rotation,
)
from furniture_lab.tasks.assembly.assembly_env_cfg import FBABS_OBS_LAYOUT, fbabs_obs_layout  # noqa: E402
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
#: the task observes the front obstacle's pose after the parts (the ResiP task: 70 wide); set
#: in main() from the task's cfg, before the episodes are read
OBSTACLE_OBS = False


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
            data = pickle.load(f)
        # what the env was given when the demo was recorded (a DART demo's perturbed commands)
        states, acts = fb_abs_frames(data, obstacle=OBSTACLE_OBS, executed=True)
        # demos recorded with moving obstacles (furniture.obstacle_offset) carry where they were
        obstacles = data.get("obstacles_robot")

        def start(env, demo=demo, obstacles=obstacles):
            # before the parts settle: a pkl from before the obstacles moved had them at
            # their fixed default poses, so those go back there
            _write_obstacles_robot(env, obstacles)
            return set_state_exact(env, demo, 0)

        yield path.name, states, acts, start, demo.success


def zarr_states_abs(robot_state: np.ndarray, parts: np.ndarray) -> np.ndarray:
    """(T, 16) robot state + (T, 35) tag-frame xyzw part poses -> (T, 61) FBAbs observations."""
    T = robot_state.shape[0]
    p = parts.reshape(T, 5, 7)
    rot = np.stack([[mat_to_rot6d(quat_to_mat(quat_xyzw_to_wxyz(p[t, i, 3:7]))) for i in range(5)] for t in range(T)])
    return np.concatenate([robot_state, np.concatenate([p[:, :, :3], rot], axis=-1).reshape(T, 45)], axis=-1).astype(np.float32)


def _obstacle_states(front_tag, T: int) -> np.ndarray:
    """(T, 9) ``obstacle_pose`` rows: the recorded tag-frame front-obstacle positions (or this
    scene's fixed one for ``None``) as ``demos.obstacle_tag_rot6d`` gives them."""
    if front_tag is None:
        return np.tile(obstacle_tag_rot6d(OBSTACLE_FRONT_POS), (T, 1))
    robot = np.asarray(TAG_POS_W) + np.asarray(front_tag, dtype=np.float64) @ TAG_ROT_W.T
    return np.stack([obstacle_tag_rot6d(p) for p in robot])


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
    obstacles = None
    if parts.shape[1] == 42:
        check_obstacle(parts[:, 35:38])
        obstacles = parts[:, 35:38].copy()  # ResiP offsets the obstacles per episode (low: +-2 cm)
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
        if OBSTACLE_OBS:  # the recorded position; the rotation is this scene's (a constant)
            front_tag = obstacles[lo:hi] if obstacles is not None else None
            states = np.concatenate([states, _obstacle_states(front_tag, hi - lo)], axis=1)
        acts = actions[lo:hi].copy()
        # ResiP's processing shrank the episode's rotation deltas by one factor (demos.py)
        if not args.no_rot_rescale:
            acts[:, 3:9], f = resip_unscale_rotation(robot_state[lo:hi, 3:9], acts[:, 3:9])
            print(f"[replay] episode {e}: rotation deltas scaled back up by {f:.3f}", flush=True)
        states = np.concatenate([states, states[-1:]])  # no observation after the last action
        first = dict(
            ee=states[0, 0:9], width=float(robot_state[lo, 15]), parts=parts[lo], grip=float(actions[lo, 9]),
            obstacle_tag=None if obstacles is None else obstacles[lo],
        )
        yield f"episode {e}", states, acts, (lambda env, first=first: set_state_ik(env, **first)), True


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


def _write_obstacles(env, front_tag: np.ndarray) -> None:
    """The three obstacles where a ResiP demo had them: the front one at its recorded tag-frame
    position, the sides at ResiP's fixed offsets from it (-0.075, -/+0.175 m, world axes)."""
    front = np.asarray(TAG_POS_W) + TAG_ROT_W @ np.asarray(front_tag, dtype=np.float64)
    origin = env.scene.env_origins[:1]
    for name, off in (("obstacle_front", (0.0, 0.0)), ("obstacle_right", (-0.075, -0.175)), ("obstacle_left", (-0.075, 0.175))):
        pos = front + np.array([off[0], off[1], 0.0])
        pose = torch.tensor([[*pos, 1.0, 0.0, 0.0, 0.0]], device=env.device, dtype=torch.float32)
        pose[:, :3] += origin
        env.scene[name].write_root_pose_to_sim(pose)


def _write_obstacles_robot(env, obstacles) -> None:
    """The three obstacles at a pkl's recorded robot-base-frame positions (``obstacles_robot``),
    or, for ``None`` (a pkl recorded before they moved), at their default poses."""
    origin = env.scene.env_origins[:1]
    for name in ("obstacle_front", "obstacle_right", "obstacle_left"):
        asset = env.scene[name]
        pose = asset.data.default_root_state[:1, :7].clone()
        if obstacles is not None:
            pose[:, :3] = torch.tensor(np.asarray(obstacles[name], dtype=np.float64), device=env.device, dtype=pose.dtype)
        pose[:, :3] += origin
        asset.write_root_pose_to_sim(pose)


def set_state_ik(env, ee: np.ndarray, width: float, parts: np.ndarray, grip: float, obstacle_tag=None) -> str:
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
    if obstacle_tag is not None:
        _write_obstacles(env, obstacle_tag)
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


def robot_snapshot(env) -> dict:
    """The arm and finger joint state of env 0 (position, target, velocity, applied effort) and
    the estimated external EE wrench."""
    robot = env.scene["robot"]
    d = robot.data
    out = {}
    for key, ids in (("arm", robot.find_joints("panda_joint.*")[0]), ("finger", robot.find_joints("panda_finger_joint.*")[0])):
        out[f"{key}_q"] = d.joint_pos[0, ids].cpu().numpy()
        out[f"{key}_q_target"] = d.joint_pos_target[0, ids].cpu().numpy()
        out[f"{key}_qd"] = d.joint_vel[0, ids].cpu().numpy()
        out[f"{key}_effort"] = d.applied_torque[0, ids].cpu().numpy()
    # the estimated external EE wrench (assembly/wrench.py; base frame, N then N m; +z = the
    # parts pushing back up on the hand), for furniture_lab/failures.py
    out["ee_wrench"] = env.wrench.filtered[0].cpu().numpy()
    return out


def save_trace(path: Path, name: str, states, acts, trace: list, start_obs, start_robot, meta: dict, env) -> None:
    """One demo's replay as arrays. Per step ``t`` (the action ``acts[t]`` sent, then read):

    * ``obs_here`` (T, 61) / ``obs_demo`` (T, 61): this env's FBAbs observation after step t and
      the demo's recorded one at t+1 (same layout: ee_pos 3, ee_rot6d 6, ee_pos_vel 3,
      ee_ori_vel 3, gripper_width 1, 5 parts x [pos 3, rot6d 6] in the tag frame);
    * ``action`` (T, 10): the absolute target sent (goal pos 3, goal rot6d 6, gripper);
    * ``arm_q`` / ``arm_q_target`` / ``arm_qd`` / ``arm_effort`` (T, 7) and ``finger_*`` (T, 2):
      the joints after step t (targets as the controller last set them);
    * ``assembled`` (T,): the env's success flag;
    * ``start_obs`` (61) / ``start_*``: this env right after the start state was set (before
      step 0), ``demo_start_obs``: the demo's first recorded observation;
    * ``joint_names``, ``arm_limits`` (7, 2): for reading the joint arrays;
    * ``meta``: a JSON string (name, success, settle error, grasp offsets, obstacle).
    """
    import json

    robot = env.scene["robot"]
    arm = robot.find_joints("panda_joint.*")[0]
    arrays = {k: np.stack([row[k] for row in trace]) for k in trace[0]}
    np.savez_compressed(
        path,
        **arrays,
        obs_demo=states[1 : len(trace) + 1] if len(trace) < states.shape[0] else states[1:],
        action=acts[: len(trace)],
        start_obs=start_obs,
        demo_start_obs=states[0],
        **{f"start_{k}": v for k, v in start_robot.items()},
        joint_names=np.array(robot.joint_names),
        arm_limits=robot.data.soft_joint_pos_limits[0, arm].cpu().numpy(),
        meta=json.dumps(meta),
    )


class Skip:
    """Ends the current replay early: the N key in the viewer window, or Enter in the terminal."""

    def __init__(self):
        self.flag, self._sub = False, None
        try:  # the viewer's keyboard (not there headless)
            import carb
            import omni.appwindow

            inp = carb.input.acquire_input_interface()
            kb = omni.appwindow.get_default_app_window().get_keyboard()

            def on_key(event, *_):
                if event.type == carb.input.KeyboardEventType.KEY_PRESS and event.input == carb.input.KeyboardInput.N:
                    self.flag = True
                return True

            self._sub = inp.subscribe_to_keyboard_events(kb, on_key)
        except Exception:  # noqa: BLE001 -- headless or no app window: the terminal only
            self._sub = None
        self._tty = sys.stdin.isatty()

    def requested(self) -> bool:
        if not self.flag and self._tty:
            import select

            if select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                self.flag = True
        return self.flag

    def clear(self) -> None:
        self.flag = False


def replay(env, name, states, acts, start, recorded_success, skip=None) -> dict:
    env.reset()
    note = start(env) or ""
    trace = []
    start_obs, start_robot = obs(env), robot_snapshot(env)
    T = acts.shape[0]
    errs, first = [], None
    # each grasped part's offset from the EE once it is held: --carry steps after each close
    # (the gripper command turning positive), here and in the demo. Fingers centre a part
    # across their closing axis, so a frame offset between the demos' sim and this one shows
    # up along the other two axes
    closes = [t for t in range(T) if acts[t, 9] > 0 and (t == 0 or acts[t - 1, 9] <= 0)]
    checks = {c + args.carry: (name, slot) for c, (name, slot) in zip(closes, GRASPS) if c + args.carry < T}
    grasps = []
    skipped = None
    if skip is not None:
        skip.clear()
    for t in range(T + args.hold):
        if skip is not None and skip.requested():
            skipped = t
            print(f"[replay]   skipped by you at step {t}", flush=True)
            break
        a = torch.tensor(acts[min(t, T - 1)], device=env.device, dtype=torch.float32).reshape(1, -1)
        env.step(a)
        if first is None and bool(tracker(env).all_assembled()[0]):
            first = t
        if t in checks:
            part, slot = checks[t]
            d = (part_from_ee(obs(env), slot) - part_from_ee(states[t + 1], slot)) * 1000
            grasps.append(f"held {part} from EE, here minus demo: x {d[0]:+.1f} y {d[1]:+.1f} z {d[2]:+.1f} mm")
            print(f"[replay]   {grasps[-1]} ({args.carry} steps after that grasp, step {t})", flush=True)
        if args.save and t < T:
            trace.append({"obs_here": obs(env), "assembled": np.array(bool(tracker(env).all_assembled()[0])), **robot_snapshot(env)})
        if t < T:
            p, r = ee_err(obs(env), states[t + 1])
            errs.append((p, r))
            if args.print_every and t % args.print_every == 0:
                print(f"[replay]   step {t:4d}: EE vs demo {p * 1000:6.1f} mm {r:5.1f} deg", flush=True)
    o = obs(env)
    parts_now = o[SL["parts_poses"]].reshape(-1, 9)[:, :3]
    parts_demo = states[-1][SL["parts_poses"]].reshape(-1, 9)[:, :3]
    e = np.array(errs) if errs else np.zeros((1, 2))  # skipped before the first step
    if args.save:
        out = Path(args.save).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        meta = {"name": name, "note": note, "grasps": grasps, "success": first is not None, "at": first, "skipped_at": skipped,
                "steps": T, "task": args.task, "source": args.zarr or args.demos}
        path = out / (name.replace(" ", "_") + ".npz")
        save_trace(path, name, states, acts, trace, start_obs, start_robot, meta, env)
        print(f"[replay]   trace: {path}", flush=True)
    return {
        "name": name,
        "note": "; ".join(x for x in (note, *grasps, f"skipped at step {skipped}" if skipped is not None else "") if x),
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
    global OBSTACLE_OBS, _s
    episodes = zarr_episodes(Path(args.zarr).expanduser()) if args.zarr else pkl_episodes(Path(args.demos).expanduser())
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    layout = fbabs_obs_layout(cfg)
    OBSTACLE_OBS = any(name == "obstacle_pose" for name, _ in layout)
    _s = sum(w for _, w in layout)
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 1e6
    cfg.terminations.success = None  # success read from the tracker, the replay keeps going
    cfg.terminations.part_out_of_limits = None
    f = cfg.furniture
    for name, value in (("gripper_stiffness", args.grip_stiffness), ("gripper_damping", args.grip_damping),
                        ("gripper_effort_limit", args.grip_force)):
        if value is not None:
            setattr(f, name, value)
    if args.grip_speed is not None:
        f.gripper_velocity_limit = args.grip_speed or None
    cfg.apply_gripper()
    if args.grip_speed == 0:
        cfg.scene.robot.actuators["panda_hand"].velocity_limit_sim = None  # the asset's own
    hand = cfg.scene.robot.actuators["panda_hand"]
    print(f"[replay] fingers: stiffness {hand.stiffness}, damping {hand.damping}, effort {hand.effort_limit_sim}, "
          f"speed cap {hand.velocity_limit_sim}; zarr rotation rescale {'off' if args.no_rot_rescale else 'on'}", flush=True)
    env = gym.make(args.task, cfg=cfg).unwrapped
    env.reset()
    width = int(env.single_observation_space["policy"].shape[-1])
    if width != _s:
        raise SystemExit(f"{args.task} observation is {width} wide, expected the FBAbs {_s}")
    print(f"[replay] {args.zarr or args.demos} in {args.task}, success_mode={cfg.furniture.success_mode}", flush=True)

    skip = Skip()
    print("[replay] to end a replay early: N in the viewer window, or Enter in this terminal", flush=True)
    rows = []
    for i, (name, states, acts, start, rec) in enumerate(episodes):
        print(f"[replay] {i + 1}: {name}", flush=True)
        r = replay(env, name, states, acts, start, rec, skip)
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
