"""First-run smoke test: build the env, step it, and (optionally) seat a leg by teleport.

    python scripts/smoke.py --headless                       # build + 50 zero-action steps
    python scripts/smoke.py --headless --teleport            # + seat leg 4 in the top, check
    python scripts/smoke.py --num_envs 4 --steps 200         # with the viewer
    python scripts/smoke.py --headless --teleport --out seat.mp4 --num_envs 1 --yaws 0,90
    python scripts/smoke.py --headless --teleport --drop 0.03 --out drop.mp4 --num_envs 1

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
parser.add_argument("--part_friction", type=float, default=None, help="parts' friction (FurnitureSim: 0.15)")
parser.add_argument("--table_friction", type=float, default=None, help="table/obstacle friction (FurnitureSim: 0.10)")
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
    "--contact_offset", type=float, default=None,
    help="override the parts' PhysX contact offset (m; the env uses 0.002). Does the thread's stop "
    "angle move with it? Then the offset is inflating the thread.",
)
parser.add_argument(
    "--rest_offset", type=float, default=None, help="override the parts' rest offset (m; the env uses 0.0)"
)
parser.add_argument(
    "--seat_dz", type=float, default=0.0,
    help="shift the seat height by this much (m), e.g. -0.0005. Through the 9 mm pitch, 0.25 mm "
    "of height is 10 deg of stop angle.",
)
parser.add_argument(
    "--out", type=str, default=None, metavar="FILE.mp4",
    help="record the experiments to an mp4 (works headless): one frame per physics step, with the "
    "placed and measured twist in the bottom-right corner. Needs the package's recorder camera, "
    "so it forces --enable_cameras and clone_in_fabric=False",
)
parser.add_argument("--fps", type=int, default=30, help="playback rate of --out (120 = real time)")
parser.add_argument("--cam", type=str, default="0.95,0.45,0.35", help="camera eye x,y,z (robot-base frame)")
parser.add_argument("--look", type=str, default="0.54,0.0,0.03", help="camera target x,y,z")
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
if args.out:
    args.enable_cameras = True  # a TiledCamera renders only with this
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab.sim.spawners.sensors import PinholeCameraCfg  # noqa: E402

from robonuke_rl_core.recording import VideoWriter, read_camera_rgb  # noqa: E402

CAMERA_KEY = "recorder_cam"

import furniture_lab.tasks  # noqa: E402, F401
from furniture_lab.tasks.assembly.mdp.state import parts_pose_w, tracker  # noqa: E402
from furniture_lab.tasks.furniture.assembly import pairs_assembled, relative_pose, rotation_diag  # noqa: E402
from furniture_lab.tasks.furniture import catalog  # noqa: E402
from furniture_lab.tasks.furniture.square_table import rel_pose_targets  # noqa: E402


class Recorder:
    """Writes one frame per physics step of env 0 to ``--out``, with a caption bottom-right."""

    def __init__(self, env) -> None:
        self.env = env
        self.camera = env.scene[CAMERA_KEY]
        eye = torch.tensor([[float(v) for v in args.cam.split(",")]], device=env.device)
        look = torch.tensor([[float(v) for v in args.look.split(",")]], device=env.device)
        origins = env.scene.env_origins[:1]
        self.camera.set_world_poses_from_view(eye + origins, look + origins, env_ids=[0])
        self.writer = VideoWriter(args.out, fps=args.fps)
        self.caption = ""
        self.frames = 0

    def frame(self) -> None:
        from PIL import Image, ImageDraw

        self.camera.update(self.env.physics_dt)
        rgb = read_camera_rgb(self.camera)[0].numpy()
        image = Image.fromarray(rgb[..., :3])
        if self.caption:
            draw = ImageDraw.Draw(image)
            lines = self.caption.split("\n")
            pad, lh = 6, 14
            box_w = max(len(line) for line in lines) * 7 + 2 * pad
            box_h = len(lines) * lh + 2 * pad
            x0, y0 = image.width - box_w - 8, image.height - box_h - 8
            draw.rectangle([(x0, y0), (x0 + box_w, y0 + box_h)], fill=(0, 0, 0))
            for i, line in enumerate(lines):
                draw.text((x0 + pad, y0 + pad + i * lh), line, fill=(255, 255, 0))
        self.writer.append(torch.from_numpy(__import__("numpy").array(image)))
        self.frames += 1

    def close(self) -> None:
        self.writer.close()
        print(f"[smoke] wrote {args.out}: {self.frames} frames at {args.fps} fps", flush=True)


recorder: "Recorder | None" = None


def measured_twist(env, top, leg) -> tuple:
    """(twist deg, upright cos, rel z m) of the leg in the top's frame, env 0."""
    rel_pos, rel_quat = relative_pose(
        top.data.root_pos_w[:1], top.data.root_quat_w[:1], leg.data.root_pos_w[:1], leg.data.root_quat_w[:1]
    )
    w, x, y, zq = rel_quat.unbind(-1)
    yaw = torch.rad2deg(torch.atan2(2 * (w * zq + x * y), 1 - 2 * (y * y + zq * zq)))
    return yaw.item(), rotation_diag(rel_quat)[0, 2].item(), rel_pos[0, 2].item()


def physics_steps(env, n: int) -> None:
    """Advance physics n env-steps WITHOUT env.step(): no terminations, so a leg that counts
    as assembled is not reset out from under the experiment (which is what env.step does --
    success ends the episode, as upstream's done flag does)."""
    render = not args.headless or recorder is not None
    for _ in range(n * env.cfg.decimation):
        env.sim.step(render=render)
        env.scene.update(env.physics_dt)
        if recorder is not None:
            recorder.frame()


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
        if recorder is not None:
            tw, up, rz = measured_twist(env, top, leg)
            recorder.caption = f"placed twist {label}\nmeasured    {tw:6.1f} deg\nupright cos {up:.3f}  z {rz * 1000:5.1f} mm"
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


def print_physics(env, cfg) -> None:
    """What PhysX actually holds for every part (env 0): mass, inertia, CoM, friction -- the
    USDs carry none of it, the startup events set it, so this is the check that they did."""
    from furniture_lab.tasks.furniture.square_table import PARTS

    print("[physics] part                mass kg (cfg)   inertia diag kg m^2            com m (body)        friction s/d")
    for part in PARTS:
        if part.name not in cfg.furniture.part_names:
            continue
        view = env.scene[part.name].root_physx_view
        m = float(view.get_masses()[0])
        I = view.get_inertias()[0].reshape(3, 3).diagonal().tolist()
        com = view.get_coms()[0, :3].tolist()
        try:
            mat = view.get_material_properties()[0, 0].tolist()  # (static, dynamic, restitution)
            fr = f"{mat[0]:.2f}/{mat[1]:.2f}"
        except Exception:
            fr = "?"
        flag = "" if abs(m - part.mass) < 1e-4 else "   <-- NOT the configured mass"
        print(
            f"[physics] {part.name:20s} {m:7.4f} ({part.mass:.4f})  "
            f"[{I[0]:.2e} {I[1]:.2e} {I[2]:.2e}]  [{com[0]:+.4f} {com[1]:+.4f} {com[2]:+.4f}]  {fr}{flag}"
        )
    sim = env.sim.cfg
    px = sim.physx
    print(
        f"[physics] dt {sim.dt} x decimation {cfg.decimation} = {sim.dt * cfg.decimation:.4f} s/step, "
        f"solver {'TGS' if px.solver_type == 1 else 'PGS'}, bounce thr {px.bounce_threshold_velocity}, "
        f"friction offset {px.friction_offset_threshold}, corr dist {px.friction_correlation_distance}, "
        f"gpu {sim.use_gpu_pipeline if hasattr(sim, 'use_gpu_pipeline') else sim.device}"
    )
    robot = env.scene["robot"]
    rm = robot.root_physx_view.get_masses()[0]
    print(f"[physics] robot link masses kg: {[round(float(v), 3) for v in rm]}  (gravity off on the arm: {cfg.scene.robot.spawn.rigid_props.disable_gravity})")


def main() -> None:
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = args.num_envs
    cfg.furniture.log_resets = True  # say when and why an env resets
    if args.part_friction is not None:
        cfg.furniture.part_friction = args.part_friction
    if args.table_friction is not None:
        cfg.furniture.table_friction = args.table_friction
    cfg.apply_friction()
    if args.contact_offset is not None or args.rest_offset is not None:
        for name in cfg.furniture.part_names:
            props = getattr(cfg.scene, name).spawn.collision_props
            if args.contact_offset is not None:
                props.contact_offset = args.contact_offset
            if args.rest_offset is not None:
                props.rest_offset = args.rest_offset
        print(f"[smoke] parts contact_offset={props.contact_offset} rest_offset={props.rest_offset}")
    if args.slow:
        cfg.sim.render_interval = 1  # a frame per physics step: 6x slower on screen
    if args.out:
        # a manager-based env takes a camera as a scene entry (the package's
        # install_recorder_camera patches a Direct env's _setup_scene, which this env has not)
        cfg.scene.clone_in_fabric = False
        cfg.sim.render_interval = 1
        setattr(
            cfg.scene,
            CAMERA_KEY,
            TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/" + CAMERA_KEY,
                data_types=["rgb"],
                spawn=PinholeCameraCfg(
                    focal_length=24.0, focus_distance=0.5, horizontal_aperture=20.955, clipping_range=(0.05, 20.0)
                ),
                width=640,
                height=480,
            ),
        )
    env = gym.make(args.task, cfg=cfg).unwrapped
    obs, _ = env.reset()
    global recorder
    if args.out:
        recorder = Recorder(env)
    print(f"[smoke] task {args.task}: {env.num_envs} envs, mesh frame {cfg.furniture.mesh_frame}")
    print(f"[smoke] obs policy {tuple(obs['policy'].shape)}  action {env.action_manager.total_action_dim}")
    print_physics(env, cfg)

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
        rel[2] += args.seat_dz
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
                if recorder is not None:
                    tw, up, rz = measured_twist(env, top, leg)
                    recorder.caption = (
                        f"drop from {args.drop * 1000:.0f} mm, placed twist 0.0 deg\nmeasured    {tw:6.1f} deg\n"
                        f"upright cos {up:.3f}  above seat {(rz - rel[2]) * 1000:5.1f} mm"
                    )
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
            if recorder is not None:
                recorder.close()
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
            placed = yaw_deg
            _watch(env, leg, top, seated_pos, min(10, args.settle), zero, f"yaw {yaw_deg:.0f}")
            remaining = args.settle - (min(10, args.settle) if args.slow else 0)
            for _ in range(remaining):
                if recorder is not None:
                    tw, up, rz = measured_twist(env, top, leg)
                    recorder.caption = (
                        f"placed twist {placed:6.1f} deg\nmeasured    {tw:6.1f} deg\n"
                        f"upright cos {up:.3f}  z {rz * 1000:5.1f} mm"
                    )
                physics_steps(env, 1)
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
        seated = [r for r in results if r[1] < 0.002]
        passed = [r for r in seated if r[2] > 0.5]
        print(
            f"[smoke] seated (drift < 2 mm) at {[r[0] for r in seated]} deg; "
            f"assembled by upstream's rule at {[r[0] for r in passed]} deg"
        )
        print("[smoke] read: small drift = the pose is physically consistent at that twist;")
        print("[smoke]       assembled True = upstream's rule (all axes within ~20 deg, position")
        print("[smoke]       within tolerance) accepts it. In the real env success ends the episode.")

    if recorder is not None:
        recorder.close()
    env.close()


if __name__ == "__main__":
    main()
    app.close()
