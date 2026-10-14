"""Watch the joint-torque wrench estimate against PhysX's own reaction wrench.

    python scripts/wrench_probe.py                      # viewer: free space, press, twist, lift
    python scripts/wrench_probe.py --headless --csv wrench.csv
    python scripts/wrench_probe.py --noise_tau 0.1 --alpha 0.1 --inertial

The parts are moved off the table (``--keep_parts`` leaves them), terminations are off, and
the hand runs a fixed script with the env's own 7-d action (OSC pose_rel + gripper):

1. ``free``:  zero action -- the estimate should sit at ~0 (noise only);
2. ``move``:  a lateral step -- transients while the arm accelerates (what the plain
   projection cannot tell from contact; ``--inertial`` removes most of it);
3. ``press``: down until the fingertips stop on the table, then keep pushing -- the force
   should settle at a steady +z of a few newtons (the OSC's 90 N/m times the commanded
   offset) and match PhysX's reaction force, with x, y and the torques near 0;
4. ``twist``: keep pushing while the wrist yaws -- a correctly framed estimate keeps its
   +z and near-zero x, y while the hand turns; a wrong frame rotates them;
5. ``lift``:  back up -- the force returns to 0 with the filter's lag.

Each control step prints the filtered estimate, the raw one, PhysX's reference (its joint
reaction wrench through panda_hand, turned into the same external-wrench-on-the-EE, base
frame) and their force difference. ``--csv`` writes the same per step.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Isaac-Furniture-OneLeg-v0")
parser.add_argument("--alpha", type=float, default=None, help="EMA factor (cfg default 0.25; 1 = no filter)")
parser.add_argument("--noise_tau", type=float, default=None, help="joint torque noise std, N m (cfg default 0)")
parser.add_argument("--noise_wrench", type=float, nargs="+", default=None, help="wrench noise std: one, (force torque) or six (cfg default 0)")
parser.add_argument("--coriolis", action="store_true", help="remove the Coriolis torques")
parser.add_argument("--inertial", action="store_true", help="remove M q'' (finite-differenced joint acceleration)")
parser.add_argument("--projection", choices=["pinv", "dynamic"], default=None)
parser.add_argument("--free", type=int, default=40, help="steps standing still")
parser.add_argument("--move", type=str, default="0.0,0.08", help="lateral move dx,dy (m, base frame) before descending")
parser.add_argument("--move_steps", type=int, default=40)
parser.add_argument("--press", type=float, default=-0.6, help="z action while pressing (-1 = 25 mm target offset = ~2 N)")
parser.add_argument("--descend_steps", type=int, default=120, help="at most this many steps to reach the table")
parser.add_argument("--hold", type=int, default=60, help="steps pressing before the twist")
parser.add_argument("--twist", type=float, default=0.3, help="wrist yaw action while twisting (0.1 rad per unit per step)")
parser.add_argument("--twist_steps", type=int, default=80)
parser.add_argument("--lift_steps", type=int, default=40)
parser.add_argument("--keep_parts", action="store_true", help="leave the parts on the table")
parser.add_argument("--csv", type=str, default=None)
parser.add_argument("--every", type=int, default=2, help="print every N control steps")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

import furniture_lab.tasks  # noqa: E402, F401


def fmt(v) -> str:
    return " ".join(f"{x:7.2f}" for x in v.tolist())


def main() -> None:
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 1e6
    cfg.terminations.success = None
    cfg.terminations.part_out_of_limits = None
    fc = cfg.furniture
    if args.alpha is not None:
        fc.wrench_smoothing = args.alpha
    if args.noise_tau is not None:
        fc.wrench_noise_joint_torque = (args.noise_tau,)
    if args.noise_wrench is not None:
        fc.wrench_noise = tuple(args.noise_wrench)
    fc.wrench_coriolis = args.coriolis
    fc.wrench_inertial = args.inertial
    if args.projection is not None:
        fc.wrench_projection = args.projection
    cfg.apply_friction()
    cfg.apply_gripper()
    env = gym.make(args.task, cfg=cfg).unwrapped
    env.reset()
    print(
        f"[probe] alpha {fc.wrench_smoothing}, tau noise {fc.wrench_noise_joint_torque}, wrench noise {fc.wrench_noise}, "
        f"coriolis {fc.wrench_coriolis}, inertial {fc.wrench_inertial}, projection {fc.wrench_projection}",
        flush=True,
    )
    if not args.keep_parts:
        for name in fc.part_names:
            part = env.scene[name]
            pose = part.data.root_pose_w.clone()
            pose[:, 0] += 2.0  # two metres down the table: off the walls, off the hand
            part.write_root_pose_to_sim(pose)
            part.write_root_velocity_to_sim(torch.zeros_like(part.data.root_vel_w))
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)

    writer = None
    if args.csv:
        f = open(args.csv, "w", newline="")
        writer = csv.writer(f)
        writer.writerow(["step", "phase", "ee_x", "ee_y", "ee_z"] + [f"est_{k}" for k in "fx fy fz tx ty tz".split()]
                        + [f"raw_{k}" for k in "fx fy fz tx ty tz".split()] + [f"ref_{k}" for k in "fx fy fz tx ty tz".split()])
    zero = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
    step = 0
    print("        phase    ee z |   est fx   fy   fz   tx   ty   tz |   raw fz |   ref fx   fy   fz   tx   ty   tz | dF", flush=True)

    def run(phase: str, action: torch.Tensor, n: int, stop=None) -> None:
        nonlocal step
        for _ in range(n):
            env.step(action)
            step += 1
            w = env.wrench
            est, raw, ref = w.filtered[0], w.raw[0], w.physx_reference()[0]
            ee = w.ee_pos_b()[0]
            if writer is not None:
                writer.writerow([step, phase, *ee.tolist(), *est.tolist(), *raw.tolist(), *ref.tolist()])
            if step % args.every == 0:
                print(
                    f"{step:5d} {phase:>8} {ee[2]:7.4f} | {fmt(est)} | {raw[2]:7.2f} | {fmt(ref)} | "
                    f"{(est[0:3] - ref[0:3]).norm():5.2f}",
                    flush=True,
                )
            if stop is not None and stop():
                return

    run("free", zero, args.free)

    dx, dy = (float(v) for v in args.move.split(","))
    move = zero.clone()
    # position_scale 0.025 m per unit of action, spread over the move steps
    move[0, 0] = dx / (cfg.actions.arm.position_scale * args.move_steps)
    move[0, 1] = dy / (cfg.actions.arm.position_scale * args.move_steps)
    run("move", move, args.move_steps)
    run("settle", zero, 20)

    down = zero.clone()
    down[0, 2] = args.press
    last_z = [env.wrench.ee_pos_b()[0, 2].item()]
    still = [0]

    def on_table() -> bool:
        z = env.wrench.ee_pos_b()[0, 2].item()
        still[0] = still[0] + 1 if abs(z - last_z[0]) < 2e-4 else 0
        last_z[0] = z
        return still[0] >= 5

    run("descend", down, args.descend_steps, stop=on_table)
    print(f"[probe] fingertips stopped at ee z {last_z[0]:.4f} (contact); pressing with action z {args.press}", flush=True)
    run("press", down, args.hold)
    twist = down.clone()
    twist[0, 5] = args.twist
    run("twist", twist, args.twist_steps)
    up = zero.clone()
    up[0, 2] = -args.press
    run("lift", up, args.lift_steps)
    run("free", zero, 20)

    if writer is not None:
        f.close()
        print(f"[probe] wrote {args.csv}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
