"""First-run smoke test: build the env, step it, and (optionally) seat a leg by teleport.

    python scripts/smoke.py --headless                       # build + 50 zero-action steps
    python scripts/smoke.py --headless --teleport            # + seat leg 4 in the top, check
    python scripts/smoke.py --num_envs 4 --steps 200         # with the viewer

Needs no trained run and no package config: it goes straight to the gym registry. What it
checks: the scene spawns, observation/action sizes, nothing explodes under zero actions, and
with ``--teleport`` whether a leg written straight into its assembled pose (a) stays there
under the SDF contact (no interpenetration push-out) and (b) registers as assembled -- the
test of the asset frames and the assembled-pose numbers.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Furniture-OneLeg-v0")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--teleport", action="store_true", help="seat leg 4 in the top, then check")
parser.add_argument("--hole", type=int, default=0, help="which of the 4 holes to seat it in")
parser.add_argument(
    "--seat", choices=["metadata", "furniturebench"], default="metadata",
    help="seat height: UWLab's metadata.yaml (bolt tip just above the plate) or upstream's number",
)
parser.add_argument(
    "--yaws", type=str, default="0,30,60,90,120,150,180,210,240,270,300,330",
    help="yaw angles (deg) about the screw axis to try; the thread phase is unknown",
)
parser.add_argument("--settle", type=int, default=20, help="steps after the teleport before measuring")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import furniture_lab.tasks  # noqa: E402, F401
from furniture_lab.tasks.assembly.mdp.state import parts_pose_w, tracker  # noqa: E402
from furniture_lab.tasks.furniture.assembly import relative_pose, rotation_diag  # noqa: E402
from furniture_lab.tasks.furniture import catalog  # noqa: E402
from furniture_lab.tasks.furniture.square_table import rel_pose_targets  # noqa: E402


def main() -> None:
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = args.num_envs
    env = gym.make(args.task, cfg=cfg).unwrapped
    obs, _ = env.reset()
    print(f"[smoke] task {args.task}: {env.num_envs} envs, mesh frame {cfg.furniture.mesh_frame}")
    print(f"[smoke] obs policy {tuple(obs['policy'].shape)}  action {env.action_manager.total_action_dim}")

    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    for i in range(args.steps):
        obs, rew, terminated, truncated, extras = env.step(zero)
        if i % 10 == 0 or i == args.steps - 1:
            pos, _ = parts_pose_w(env)
            z = (pos[:, :, 2] - env.scene.env_origins[:, None, 2]).mean(0)
            print(f"[smoke] step {i:4d}  reward {rew.mean().item():.3f}  part z (m) {[f'{v:.4f}' for v in z.tolist()]}")
    assert torch.isfinite(obs["policy"]).all(), "non-finite observation"

    if args.teleport:
        names = cfg.furniture.part_names
        top, leg = env.scene[names[0]], env.scene[names[4]]
        rel = rel_pose_targets([(0, 4)], cfg.furniture.mesh_frame)[0, args.hole].copy()
        if args.seat == "metadata":
            # UWLab's seat: their assembled points coincide; only the height differs from
            # upstream's (bolt tip 0.3 mm above the plate instead of 0.7 mm into its hull)
            meta_top = catalog.metadata(names[0])["assembled_offset"]["pos"]
            meta_leg = catalog.metadata(names[4])["assembled_offset"]["pos"]
            rel[2] = meta_top[2] - meta_leg[2]
        rel_t = torch.tensor(rel, dtype=torch.float32, device=env.device).expand(env.num_envs, -1)
        print(f"[smoke] seat ({args.seat}) in the top's frame: {rel.round(5).tolist()}")
        zero_vel = torch.zeros(env.num_envs, 6, device=env.device)
        results = []
        for yaw_deg in [float(v) for v in args.yaws.split(",")]:
            env.reset()
            for _ in range(5):
                env.step(zero)  # let the parts settle after the reset
            top_pos, top_quat = top.data.root_pos_w.clone(), top.data.root_quat_w.clone()
            seated_pos = top_pos + quat_apply(top_quat, rel_t)
            yaw = torch.full((env.num_envs,), yaw_deg * 3.141592653589793 / 180.0, device=env.device)
            z = torch.zeros_like(yaw)
            # the twist is about the leg's own z (the screw axis): post-multiply
            seated_quat = quat_mul(top_quat, quat_from_euler_xyz(z, z, yaw))
            leg.write_root_pose_to_sim(torch.cat([seated_pos, seated_quat], dim=-1))
            leg.write_root_velocity_to_sim(zero_vel)
            for _ in range(args.settle):
                env.step(zero)
            t = tracker(env)
            drift = (leg.data.root_pos_w - seated_pos).norm(dim=-1)
            # where the leg ended up in the top's frame: the yaw it settled to (the thread
            # phase, if it screwed itself in) and the axis cosines the assembled check reads
            rel_pos, rel_quat = relative_pose(
                top.data.root_pos_w, top.data.root_quat_w, leg.data.root_pos_w, leg.data.root_quat_w
            )
            w, x, y, zq = rel_quat.unbind(-1)
            final_yaw = torch.rad2deg(torch.atan2(2 * (w * zq + x * y), 1 - 2 * (y * y + zq * zq)))
            diag = rotation_diag(rel_quat)
            results.append((yaw_deg, drift.mean().item(), t.assembled[:, 0].float().mean().item()))
            print(
                f"[smoke] yaw {yaw_deg:6.1f} -> {final_yaw.mean().item():7.1f} deg  drift {drift.mean().item():.4f} m "
                f"(max {drift.max().item():.4f})  rel z {rel_pos[:, 2].mean().item():.4f}  "
                f"axis cos {[f'{v:.3f}' for v in diag.mean(0).tolist()]}  assembled {t.assembled[:, 0].tolist()}"
            )
        best = min(results, key=lambda r: r[1])
        print(f"[smoke] least drift at yaw {best[0]:.1f} deg: {best[1]:.4f} m, assembled rate {best[2]:.2f}")
        print("[smoke] read: seated runs that converge to one final yaw = the thread's phase;")
        print("[smoke]       axis cos z ~1 with x/y < 0.94 = seated but turned, so the check's")
        print("[smoke]       orientation rule (all three axes) is what rejects it.")

    env.close()


if __name__ == "__main__":
    main()
    app.close()
