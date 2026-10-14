"""Is the -FB-v0 interface the demos' interface? Observation and action parity on one demo.

    python scripts/parity_check.py --demo ~/Robonuke_Furniture_Bench/demos/one_leg/<stamp>/<stamp>.pkl --headless
    python scripts/parity_check.py --demo DEMO.pkl --steps 100 --no_resync --headless   # open loop
    python scripts/parity_check.py --demo DEMO.pkl --task Isaac-Furniture-OneLeg-FBAbs-v0 --headless

A BC policy trains on the pkl's 49 observation numbers and 8 action numbers (``-FB-v0``), or
the 61 / 10 of the absolute rot6d interface (``-FBAbs-v0``, ``furniture_lab.demos.fb_abs_frames``,
the same conversion the importer uses); at rollout it gets the env's. This script puts the
task's env into the demo's recorded state and measures both sides of that contract:

1. **Observation parity**: with the env in the demo's state at step t, the env's 49-d
   policy observation against the pkl's, term by term (``ee_pos, ee_quat, ee_pos_vel,
   ee_ori_vel, gripper_width``, and each part's ``pos`` / ``quat``). Poses should agree to
   ~1e-5 (float32); velocities are reconstructed from a still env, so they only agree
   where the demo was still. A quaternion that matches only up to sign is reported as such
   -- that is the kind of mismatch that silently ruins a policy.
2. **Action parity**: the pkl's recorded 8-d action for step t sent through the env's own
   FurnitureBench action term; the EE pose after the step against the demo's step t+1.
   By default the env is re-synced to the demo before every step, so each line is a
   one-step test of the action mapping (expected: < 1 mm, < 1 deg; the demo's EE velocity
   is not restored, which costs a little); ``--no_resync`` runs the recorded actions open
   loop instead and shows the drift.

The verdict lines at the end say what each number means.
"""

import argparse
import math
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--demo", required=True, help="FurnitureBench .pkl from scripts/collect_scripted.py")
parser.add_argument("--task", default="Isaac-Furniture-OneLeg-FB-v0")
parser.add_argument("--steps", type=int, default=60, help="action-parity steps (from the demo's start)")
parser.add_argument("--obs_at", type=str, default="0,50,150", help="demo steps to check observation parity at")
parser.add_argument("--no_resync", action="store_true", help="open-loop replay instead of one-step tests")
parser.add_argument("--print_every", type=int, default=5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_error_magnitude  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

import furniture_lab.tasks  # noqa: E402, F401
from furniture_lab.demos import fb_abs_frames, fb_state, load_demo  # noqa: E402
from furniture_lab.tasks.assembly.assembly_env_cfg import FB_OBS_LAYOUT, FBABS_OBS_LAYOUT  # noqa: E402
from furniture_lab.tasks.assembly.mdp.rot6d import rot6d_to_quat  # noqa: E402
from furniture_lab.tasks.assembly.mdp.state import tracker  # noqa: E402

ABS = "FBAbs" in args.task
LAYOUT = FBABS_OBS_LAYOUT if ABS else FB_OBS_LAYOUT
ROT = "ee_rot6d" if ABS else "ee_quat"
PART_WIDTH = 9 if ABS else 7


def _np(v):
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    return np.asarray(v, dtype=np.float32).reshape(-1)


def raw_frames(path):
    """The pkl's state vectors (T+1, D) and actions (T, A) of the task's interface, as the
    importer builds them."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    if ABS:
        return fb_abs_frames(d)
    states = np.stack([fb_state(o) for o in d["observations"]])
    acts = np.stack([_np(a) for a in d["actions"]])
    return states, acts


def set_state(env, demo, t: int) -> None:
    """Env 0 into the demo's state at step t (arm joints, fingers, part poses), then one
    physics step so PhysX refreshes the link poses the observation reads."""
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


def env_obs(env) -> np.ndarray:
    return env.observation_manager.compute()["policy"][0].detach().cpu().numpy()


def term_slices():
    out, start = {}, 0
    for name, width in LAYOUT:
        out[name] = slice(start, start + width)
        start += width
    return out


def quat_report(a: np.ndarray, b: np.ndarray) -> str:
    direct, flipped = np.abs(a - b).max(), np.abs(a + b).max()
    if flipped < direct and flipped < 1e-3:
        return f"{direct:.2e}  <-- SIGN FLIP (matches -q to {flipped:.1e})"
    return f"{direct:.2e}"


def obs_parity(env, demo, states, t: int, part_names) -> list:
    set_state(env, demo, t)
    ours, theirs = env_obs(env), states[t]
    sl = term_slices()
    lines = [f"[obs] demo step {t}: max |env - pkl| per term"]
    for name in ("ee_pos", "ee_pos_vel", "ee_ori_vel", "gripper_width"):
        lines.append(f"[obs]   {name:14s} {np.abs(ours[sl[name]] - theirs[sl[name]]).max():.2e}")
    rot = (lambda a, b: f"{np.abs(a - b).max():.2e}") if ABS else quat_report
    lines.append(f"[obs]   {ROT:14s} {rot(ours[sl[ROT]], theirs[sl[ROT]])}")
    po, pt = ours[sl["parts_poses"]].reshape(-1, PART_WIDTH), theirs[sl["parts_poses"]].reshape(-1, PART_WIDTH)
    for i, name in enumerate(part_names):
        lines.append(
            f"[obs]   {name:22s} pos {np.abs(po[i, :3] - pt[i, :3]).max():.2e}  rot {rot(po[i, 3:], pt[i, 3:])}"
        )
    return lines


def main() -> None:
    demo = load_demo(args.demo)
    states, acts = raw_frames(args.demo)
    print(f"[parity] {demo.summary()}; pkl states {states.shape}, actions {acts.shape}", flush=True)

    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 1e6
    cfg.terminations.success = None
    cfg.terminations.part_out_of_limits = None
    cfg.apply_friction()
    cfg.apply_gripper()
    env = gym.make(args.task, cfg=cfg).unwrapped
    env.reset()
    if int(env.single_observation_space["policy"].shape[-1]) != states.shape[1]:
        raise SystemExit(f"env observation is {env.single_observation_space['policy'].shape[-1]} wide, pkl {states.shape[1]}")

    # ---------------------------------------------------------------- observations
    for t in (int(s) for s in args.obs_at.split(",")):
        if t < states.shape[0]:
            print("\n".join(obs_parity(env, demo, states, t, demo.part_names)), flush=True)

    # ---------------------------------------------------------------- actions
    sl = term_slices()
    T = min(args.steps, demo.length)
    mode = "open loop" if args.no_resync else "one-step (re-synced before every step)"
    print(f"[act] {mode}: the pkl's action[t] through the FB action term, EE after the step vs the pkl's step t+1", flush=True)
    set_state(env, demo, 0)
    pos_err, rot_err, grip_err = [], [], []
    for t in range(T):
        if not args.no_resync and t > 0:
            set_state(env, demo, t)
        action = torch.tensor(acts[t], device=env.device, dtype=torch.float32).reshape(1, -1)
        env.step(action)
        ours, theirs = env_obs(env), states[t + 1]
        p = float(np.linalg.norm(ours[sl["ee_pos"]] - theirs[sl["ee_pos"]]))
        if ABS:
            qa = rot6d_to_quat(torch.tensor(ours[sl[ROT]]).reshape(1, 6))
            qb = rot6d_to_quat(torch.tensor(theirs[sl[ROT]]).reshape(1, 6))
        else:
            qa = torch.tensor(ours[sl[ROT]][[3, 0, 1, 2]]).reshape(1, 4)  # xyzw -> wxyz
            qb = torch.tensor(theirs[sl[ROT]][[3, 0, 1, 2]]).reshape(1, 4)
        r = float(quat_error_magnitude(qa, qb))
        g = float(abs(ours[sl["gripper_width"]] - theirs[sl["gripper_width"]])[0])
        pos_err.append(p)
        rot_err.append(r)
        grip_err.append(g)
        if t % args.print_every == 0 or t == T - 1:
            moved = float(np.linalg.norm(states[t + 1][sl["ee_pos"]] - states[t][sl["ee_pos"]]))
            print(
                f"[act] step {t:4d}: EE err {p * 1000:6.2f} mm {math.degrees(r):5.2f} deg  width err {g * 1000:5.2f} mm"
                f"   (demo moved {moved * 1000:5.2f} mm, grip cmd {acts[t, 7]:+.0f})",
                flush=True,
            )

    pos_err, rot_err = np.array(pos_err), np.array(rot_err)
    print(
        f"[parity] action {mode}: EE position error mean {pos_err.mean() * 1000:.2f} mm, max {pos_err.max() * 1000:.2f} mm; "
        f"rotation mean {math.degrees(rot_err.mean()):.2f} deg, max {math.degrees(rot_err.max()):.2f} deg over {T} steps",
        flush=True,
    )
    print(
        "[parity] read it as: observation poses ~1e-5 (and, with quaternions, no SIGN FLIP) -> the policy sees the training numbers; "
        "one-step EE errors under ~1 mm / 1 deg -> its actions do what the demos' did; "
        "open-loop drift growing over tens of steps is physics, not the interface.",
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()
    app.close()
