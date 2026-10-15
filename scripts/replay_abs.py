"""Do the demos' absolute actions still succeed when replayed open loop?

    python scripts/replay_abs.py --demos ~/Robonuke_Furniture_Bench/demos_resip
    python scripts/replay_abs.py --demos DEMO.pkl --headless

The BC policy trains on ``action.abs``: each pkl's delta actions turned into absolute rot6d
targets by ``furniture_lab.demos.fb_abs_frames`` (the importer's own conversion). This script
puts the ``-FBAbs-ResiP`` env into each demo's first recorded state (arm joints, fingers, part
poses), sends that demo's absolute actions one per step with no feedback, and reports
whether the leg ends up assembled (the env's own success check, soft by default).

That is the ceiling a perfect clone of these demos can reach in this env:

* most demos succeed -> the action interface reproduces the demos; the BC gap is learning.
* most fail -> what the policy is taught does not do what the demos did; look at where the
  EE error grows (per-step lines) and where the leg goes.

Some open-loop drift is physics (contact is chaotic), so a few failures are expected; a
failure rate near the BC policy's is the signal. Successful demos only (``fails/`` skipped).
"""

import argparse
import math
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--demos", required=True, help="a .pkl, or a directory searched recursively (fails/ skipped)")
parser.add_argument("--task", default="Isaac-Furniture-OneLeg-FBAbs-ResiP-v0")
parser.add_argument("--max_demos", type=int, default=None)
parser.add_argument("--hold", type=int, default=20, help="steps holding the last action after the demo ends")
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
from furniture_lab.demos import fb_abs_frames, load_demo  # noqa: E402
from furniture_lab.tasks.assembly.assembly_env_cfg import FBABS_OBS_LAYOUT  # noqa: E402
from furniture_lab.tasks.assembly.mdp.rot6d import rot6d_to_quat  # noqa: E402
from furniture_lab.tasks.assembly.mdp.state import tracker  # noqa: E402

SL, _s = {}, 0
for _name, _w in FBABS_OBS_LAYOUT:
    SL[_name] = slice(_s, _s + _w)
    _s += _w


def demo_paths(root: Path) -> list:
    if root.is_file():
        return [root]
    paths = sorted(p for p in root.rglob("*.pkl") if "fails" not in p.parts and not p.name.startswith("fail"))
    if not paths:
        raise SystemExit(f"no demo .pkl under {root}")
    return paths[: args.max_demos] if args.max_demos else paths


def set_state(env, demo, t: int) -> None:
    """Env 0 into the demo's state at step t; one physics step so the link poses refresh."""
    robot = env.scene["robot"]
    arm = robot.find_joints("panda_joint.*")[0]
    fingers = robot.find_joints("panda_finger_joint.*")[0]
    q = robot.data.default_joint_pos.clone()
    q[:, arm] = torch.tensor(demo.joint_positions[t], device=env.device, dtype=q.dtype)
    q[:, fingers] = float(demo.gripper_width[t]) / 2
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    robot.set_joint_position_target(q)
    origin = env.scene.env_origins[:1]
    for i, name in enumerate(demo.part_names):
        part = env.scene[name]
        pos = torch.tensor(demo.parts_pos[t, i], device=env.device, dtype=torch.float32).unsqueeze(0) + origin
        quat = torch.tensor(demo.parts_quat[t, i], device=env.device, dtype=torch.float32).unsqueeze(0)
        part.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
        part.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(env.physics_dt)
    tracker(env).reset()


def obs(env) -> np.ndarray:
    return env.observation_manager.compute()["policy"][0].detach().cpu().numpy()


def ee_err(a: np.ndarray, b: np.ndarray) -> tuple:
    p = float(np.linalg.norm(a[SL["ee_pos"]] - b[SL["ee_pos"]]))
    qa = rot6d_to_quat(torch.tensor(a[SL["ee_rot6d"]]).reshape(1, 6))
    qb = rot6d_to_quat(torch.tensor(b[SL["ee_rot6d"]]).reshape(1, 6))
    return p, math.degrees(float(quat_error_magnitude(qa, qb)))


def replay(env, path: Path) -> dict:
    demo = load_demo(path)
    with open(path, "rb") as f:
        states, acts = fb_abs_frames(pickle.load(f))
    env.reset()
    set_state(env, demo, 0)
    T = acts.shape[0]
    errs, first = [], None
    for t in range(T + args.hold):
        a = torch.tensor(acts[min(t, T - 1)], device=env.device, dtype=torch.float32).reshape(1, -1)
        env.step(a)
        if first is None and bool(tracker(env).all_assembled()[0]):
            first = t
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
        "name": path.name,
        "steps": T,
        "demo_success": demo.success,
        "success": first is not None,
        "at": first,
        "ee_mm_mean": e[:, 0].mean() * 1000,
        "ee_mm_max": e[:, 0].max() * 1000,
        "ee_deg_max": e[:, 1].max(),
        "part_mm_end": float(np.linalg.norm(parts_now - parts_demo, axis=-1).max()) * 1000,
    }


def main() -> None:
    paths = demo_paths(Path(args.demos).expanduser())
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
    print(f"[replay] {len(paths)} demos, {args.task}, success_mode={cfg.furniture.success_mode}", flush=True)

    rows = []
    for i, path in enumerate(paths):
        print(f"[replay] {i + 1}/{len(paths)} {path}", flush=True)
        r = replay(env, path)
        rows.append(r)
        print(
            f"[replay] {r['name']}: {'SUCCESS at step ' + str(r['at']) if r['success'] else 'FAIL'} "
            f"(demo {r['steps']} steps, recorded success={r['demo_success']}); EE vs demo mean {r['ee_mm_mean']:.1f} mm, "
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
