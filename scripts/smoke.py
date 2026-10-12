"""First-run smoke test: build the env, step it, and (optionally) seat a leg by teleport.

    python scripts/smoke.py --headless                       # build + 50 zero-action steps
    python scripts/smoke.py --headless --teleport            # + seat leg 4 in the top, check
    python scripts/smoke.py --num_envs 4 --steps 200         # with the viewer

Needs no trained run and no package config: it goes straight to the gym registry. What it
checks: the scene spawns, observation/action sizes, nothing explodes under zero actions, and
with ``--teleport`` whether a leg written straight into its assembled pose (a) stays there
under the SDF contact (no interpenetration push-out) and (b) registers as assembled -- the
test of the asset frames and the assembled-pose numbers.

The experiments step physics directly (``physics_steps``), not ``env.step``: the env ends an
episode on success, so stepping it would reset the leg back onto the table the moment the
check fires and make a correct seat look like an ejection.
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
parser.add_argument(
    "--slow", action="store_true",
    help="viewer only: hold the placed pose ~2 s before physics runs, render every physics step "
    "(not every 6th), and print the leg's height for the first steps -- to watch an ejection",
)
parser.add_argument(
    "--drop", type=float, default=None, metavar="HEIGHT",
    help="instead of seating: hold the leg aligned HEIGHT m above the seat and release it. Does it "
    "fall to the seat without turning (loose fit, peg-in-hole) or stop on the thread (a real screw)?",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import furniture_lab.tasks  # noqa: E402, F401
from furniture_lab.tasks.assembly.mdp.state import parts_pose_w, tracker  # noqa: E402
from furniture_lab.tasks.furniture.assembly import pairs_assembled, relative_pose, rotation_diag  # noqa: E402
from furniture_lab.tasks.furniture import catalog  # noqa: E402
from furniture_lab.tasks.furniture.square_table import rel_pose_targets  # noqa: E402


def physics_steps(env, n: int) -> None:
    """Advance physics n env-steps WITHOUT env.step(): no terminations, so a leg that counts
    as assembled is not reset out from under the experiment (which is what env.step does --
    success ends the episode, as upstream's done flag does)."""
    render = not args.headless
    for _ in range(n * env.cfg.decimation):
        env.sim.step(render=render)
        env.scene.update(env.physics_dt)


def assembled_now(env) -> torch.Tensor:
    """The raw assembled check on the current poses, not the per-episode latch."""
    t = tracker(env)
    pos, quat = parts_pose_w(env)
    return pairs_assembled(pos, quat, t.pairs, t.targets, t.pos_threshold, t.ori_bound)[:, 0]


def _watch(env, leg, top, seated_pos, steps: int, zero: torch.Tensor, label: str) -> None:
    """Hold the placed pose on screen, then step physics with a per-step height trace."""
    if not args.slow:
        return
    for _ in range(120):
        env.sim.render()  # ~2 s at 60 fps, physics frozen: this is the pose as placed
    for i in range(steps):
        physics_steps(env, 1)
        rel_pos, rel_quat = relative_pose(
            top.data.root_pos_w, top.data.root_quat_w, leg.data.root_pos_w, leg.data.root_quat_w
        )
        print(
            f"[smoke] {label} step {i:3d}: rel z {rel_pos[:, 2].mean().item() * 1000:7.2f} mm  "
            f"upright cos {rotation_diag(rel_quat)[:, 2].mean().item():.3f}  "
            f"|v| {leg.data.root_lin_vel_w.norm(dim=-1).mean().item():.3f} m/s  "
            f"assembled {assembled_now(env).tolist()}"
        )


def main() -> None:
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = args.num_envs
    cfg.furniture.log_resets = True  # say when and why an env resets
    if args.slow:
        cfg.sim.render_interval = 1  # a frame per physics step: 6x slower on screen
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
        if args.drop is not None:
            env.reset()
            for _ in range(5):
                env.step(zero)
            top_pos, top_quat = top.data.root_pos_w.clone(), top.data.root_quat_w.clone()
            above = rel_t.clone()
            above[:, 2] += args.drop
            leg.write_root_pose_to_sim(torch.cat([top_pos + quat_apply(top_quat, above), top_quat], dim=-1))
            leg.write_root_velocity_to_sim(zero_vel)
            print(f"[smoke] drop: leg released {args.drop * 1000:.0f} mm above the seat, upright, twist 0")
            for i in range(args.settle):
                physics_steps(env, 1)
                if i % 10 == 0 or i == args.settle - 1:
                    rel_pos, rel_quat = relative_pose(
                        top.data.root_pos_w, top.data.root_quat_w, leg.data.root_pos_w, leg.data.root_quat_w
                    )
                    w, x, y, zq = rel_quat.unbind(-1)
                    yaw_now = torch.rad2deg(torch.atan2(2 * (w * zq + x * y), 1 - 2 * (y * y + zq * zq)))
                    print(
                        f"[smoke] drop step {i:4d}: height above seat {(rel_pos[:, 2] - rel_t[:, 2]).mean().item() * 1000:7.2f} mm  "
                        f"twist {yaw_now.mean().item():7.1f} deg  upright cos {rotation_diag(rel_quat)[:, 2].mean().item():.3f}  "
                        f"assembled {assembled_now(env).tolist()}"
                    )
            print("[smoke] read: reaches ~0 mm without turning = the fit is loose (no screwing needed);")
            print("[smoke]       stops a few mm up, or turns as it sinks = the thread is doing its job.")
            env.close()
            return
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
            _watch(env, leg, top, seated_pos, min(10, args.settle), zero, f"yaw {yaw_deg:.0f}")
            physics_steps(env, args.settle - (min(10, args.settle) if args.slow else 0))
            now = assembled_now(env)
            drift = (leg.data.root_pos_w - seated_pos).norm(dim=-1)
            # where the leg ended up in the top's frame: the yaw it settled to (the thread
            # phase, if it screwed itself in) and the axis cosines the assembled check reads
            rel_pos, rel_quat = relative_pose(
                top.data.root_pos_w, top.data.root_quat_w, leg.data.root_pos_w, leg.data.root_quat_w
            )
            w, x, y, zq = rel_quat.unbind(-1)
            final_yaw = torch.rad2deg(torch.atan2(2 * (w * zq + x * y), 1 - 2 * (y * y + zq * zq)))
            diag = rotation_diag(rel_quat)
            results.append((yaw_deg, drift.mean().item(), now.float().mean().item()))
            print(
                f"[smoke] yaw {yaw_deg:6.1f} -> {final_yaw.mean().item():7.1f} deg  drift {drift.mean().item():.4f} m "
                f"(max {drift.max().item():.4f})  rel z {rel_pos[:, 2].mean().item():.4f}  "
                f"axis cos {[f'{v:.3f}' for v in diag.mean(0).tolist()]}  assembled {now.tolist()}"
            )
        best = min(results, key=lambda r: r[1])
        print(f"[smoke] least drift at yaw {best[0]:.1f} deg: {best[1]:.4f} m, assembled rate {best[2]:.2f}")
        print("[smoke] read: small drift = the pose is physically consistent at that twist;")
        print("[smoke]       assembled True = upstream's rule (all axes within ~20 deg, position")
        print("[smoke]       within tolerance) accepts it. In the real env success ends the episode.")

    env.close()


if __name__ == "__main__":
    main()
    app.close()
