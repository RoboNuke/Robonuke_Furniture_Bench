"""How closely does the hybrid torque controller track like ResiP's IK + joint PD? A gain search.

    python scripts/hybrid_tracking.py --mode reference --out ref.pt --headless
    python scripts/hybrid_tracking.py --mode hybrid --reference ref.pt --out search.pt --headless \\
        --kpos 565,1000,2000,4000 --krot 28,60,120,250 --dcoef 1.5,2,3,4

``reference`` runs ``Isaac-Furniture-OneLeg-FBAbs-ResiP-v0`` (differential IK into the joint
PD 1000 / 200) on a fixed sequence of absolute EE targets and saves the EE pose every physics
substep. ``hybrid`` runs ``Isaac-Furniture-OneLeg-FBAbs-ResiP-Hybrid-v0`` with every axis on
position control (the selection bits -1, no force target) on the same targets, one candidate
gain set per env -- the grid of ``--kpos`` (K on x, y, z), ``--krot`` (K on the rotations) and
``--dcoef`` (D = dcoef * sqrt(K), Forge's rule is 2) -- and ranks the candidates by how far
their EE path is from the reference's: position and rotation RMS and max over every substep.

The targets are free-space moves from the reset pose, above the parts: steps of 2-3 cm and
15 deg on each axis, held, then smooth sweeps, then a stretch of 10 Hz waypoints moving like a
policy's chunks. The robot always resets to the same joints, so every env starts identically.
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--mode", choices=["reference", "hybrid", "press"], required=True)
parser.add_argument("--press_force", default="5", help="press: comma list of downward force targets on z, N (>= 0)")
parser.add_argument("--press_kf", default="0.1", help="press: comma list of K_f on z; one env per (K_f, target)")
parser.add_argument("--press_start", choices=["contact", "above"], default="contact", help="press: switch to force in contact, or --press_gap above it")
parser.add_argument("--press_gap", type=float, default=0.01, help="press: m above the contact height for --press_start above")
parser.add_argument("--press_steps", type=int, default=60, help="press: 10 Hz steps on force control")
parser.add_argument("--out", required=True, help="where to save the paths (torch .pt)")
parser.add_argument("--reference", default=None, help="hybrid: the reference run's --out")
parser.add_argument("--kpos", default="565", help="hybrid: comma list of K for x, y, z (N/m)")
parser.add_argument("--krot", default="28", help="hybrid: comma list of K for the rotations (N m/rad)")
parser.add_argument("--dcoef", default="2", help="hybrid: comma list of D / sqrt(K)")
parser.add_argument("--top", type=int, default=15, help="hybrid: rows of the ranking to print")
parser.add_argument("--forge_law", action="store_true", help="Forge's law as is: inertial_decoupling and force_feedforward off (K, D in N/m, N s/m)")
parser.add_argument("--dcoef_rot", type=float, default=None, help="hybrid: D / sqrt(K) for the rotations only (default: --dcoef)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import itertools  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import furniture_lab.tasks  # noqa: E402, F401
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

from furniture_lab.tasks.assembly.mdp.observations import _ee_pose_b  # noqa: E402
from furniture_lab.tasks.assembly.mdp.rot6d import quat_to_rot6d  # noqa: E402
from furniture_lab.tasks.assembly.wrench import EE_OFFSET  # noqa: E402

REFERENCE_TASK = "Isaac-Furniture-OneLeg-FBAbs-ResiP-v0"
HYBRID_TASK = "Isaac-Furniture-OneLeg-FBAbs-ResiP-Hybrid-v0"
HZ = 10.0


def target_sequence():
    """``[(dpos (3,), drot_xyz (3,) rad)]`` per 10 Hz step, relative to the start pose."""
    seq = []

    def hold(dp, dr, steps):
        seq.extend([(tuple(dp), tuple(dr))] * steps)

    zero = (0.0, 0.0, 0.0)
    deg = math.pi / 180.0
    hold(zero, zero, 10)
    for axis, size in ((0, 0.02), (1, 0.02), (2, -0.02), (2, 0.03)):
        dp = [0.0, 0.0, 0.0]
        dp[axis] = size
        hold(dp, zero, 15)
        hold(zero, zero, 15)
    for axis in (0, 1, 2):
        dr = [0.0, 0.0, 0.0]
        dr[axis] = 15.0 * deg
        hold(zero, dr, 15)
        hold(zero, zero, 15)
    # smooth sweeps: 8 s
    for k in range(80):
        t = k / HZ
        dp = (0.03 * math.sin(2 * math.pi * t / 3.0), 0.02 * math.sin(2 * math.pi * t / 4.0), 0.015 * math.sin(2 * math.pi * t / 2.0))
        dr = (0.0, 0.0, 25.0 * deg * math.sin(2 * math.pi * t / 4.0))
        seq.append((dp, dr))
    # policy-like waypoints: piecewise-constant 10 Hz targets, each 0-1.5 cm from the last
    gen = torch.Generator().manual_seed(0)
    dp, dr = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    for _ in range(80):
        step = (torch.rand(3, generator=gen) * 2 - 1) * 0.008
        dp = [max(-0.04, min(0.04, a + float(b))) for a, b in zip(dp, step)]
        dr[2] = max(-0.4, min(0.4, dr[2] + float((torch.rand(1, generator=gen) * 2 - 1) * 0.05)))
        seq.append((tuple(dp), tuple(dr)))
    hold(zero, zero, 15)
    return seq


def make_env(task: str, num_envs: int, decouple: bool = True):
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    if hasattr(cfg, "hybrid"):
        cfg.hybrid.inertial_decoupling = decouple
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = 1000.0  # never time out mid-sequence
    if hasattr(cfg.terminations, "success"):
        cfg.terminations.success = None
    cfg.terminations.part_out_of_limits = None
    return gym.make(task, cfg=cfg).unwrapped


def run(env, extra: int):
    """Play the sequence; the EE pose (base frame) after every physics substep."""
    ee = SceneEntityCfg("robot", body_names="panda_hand")
    ee.resolve(env.scene)
    env.reset()
    pos0, quat0 = _ee_pose_b(env, ee, EE_OFFSET)
    pos0, quat0 = pos0.clone(), quat0.clone()
    n = env.num_envs
    positions, quats = [], []
    substep = env.action_manager.apply_action

    def apply_then_record():
        substep()
        # the pose the next substep starts from: after the sim step, read in the next call;
        # recording here is the pose before this substep's physics, which lags by one substep
        # identically in both runs
        p, q = _ee_pose_b(env, ee, EE_OFFSET)
        positions.append(p.clone())
        quats.append(q.clone())

    env.action_manager.apply_action = apply_then_record
    import time

    start = time.time()
    sequence = target_sequence()
    for k, (dp, dr) in enumerate(sequence):
        if k % 50 == 0:
            print(f"[tracking] step {k}/{len(sequence)} ({time.time() - start:.0f} s)", flush=True)
        dp = torch.tensor(dp, device=env.device).expand(n, -1)
        r = torch.tensor(dr, device=env.device).expand(n, -1)
        q = quat_mul(quat_from_euler_xyz(r[:, 0], r[:, 1], r[:, 2]), quat0)  # base-frame rotation
        action = torch.cat((pos0 + dp, quat_to_rot6d(q), -torch.ones(n, 1, device=env.device)), dim=-1)
        if extra:
            axes = extra // 2
            action = torch.cat((action, -torch.ones(n, axes, device=env.device), torch.zeros(n, axes, device=env.device)), dim=-1)
        env.step(action)
    env.action_manager.apply_action = substep
    return torch.stack(positions, dim=1).cpu(), torch.stack(quats, dim=1).cpu()


def quat_angle(a, b):
    """Angle between two wxyz quaternion arrays, rad."""
    dot = (a * b).sum(-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def press():
    """Force control on z against what is under the EE, one env per (K_f, target) pair of the
    grids ``--press_kf`` x ``--press_force`` (magnitudes, N; the target presses down). Each env
    descends on position control until the estimate reads > 3 N (after the first 5 steps), holds
    that height for 1 s, then (``--press_start above``) backs off ``--press_gap`` m and settles,
    and hands z to force control for ``--press_steps`` steps. Every physics substep it records the
    force the EE applies on z by the env's estimate and by PhysX's own joint reaction wrench (the
    truth), and prints per env: the settled mean and peak-to-peak of the true force over the last
    2 s, its peak over the whole force phase, the estimate's settled mean, and the height."""
    kfs = [float(v) for v in args.press_kf.split(",")]
    forces = [float(v) for v in args.press_force.split(",")]
    grid = list(itertools.product(kfs, forces))
    env = make_env(HYBRID_TASK, len(grid), decouple=not args.forge_law)
    if args.forge_law:
        env.action_manager.get_term("hybrid").force_feedforward = False
    term = env.action_manager.get_term("hybrid")
    ee = SceneEntityCfg("robot", body_names="panda_hand")
    ee.resolve(env.scene)
    env.reset()
    pos0, quat0 = _ee_pose_b(env, ee, EE_OFFSET)
    pos0, quat0 = pos0.clone(), quat0.clone()
    n, dev = len(grid), env.device
    kf = torch.tensor([k for k, _ in grid], device=dev)
    fd = -torch.tensor([f for _, f in grid], device=dev)
    process = term.process_actions

    def process_with_kf(actions):  # K_f per env on z, after the term's own split
        process(actions)
        term._force_gains[:, 2] = kf

    term.process_actions = process_with_kf
    rot6d = quat_to_rot6d(quat0)
    grip = -torch.ones(n, 1, device=dev)
    target = pos0.clone()
    touched = torch.zeros(n, dtype=torch.bool, device=dev)
    pos_block = torch.cat((-torch.ones(n, 3, device=dev), torch.zeros(n, 3, device=dev)), dim=-1)
    for k in range(300):
        f = env.wrench.filtered[:, 0:3]
        if k >= 5:
            now = (f.norm(dim=-1) > 3.0) & ~touched
            if bool(now.any()):  # hold the height where the contact began, not below it
                p, _ = _ee_pose_b(env, ee, EE_OFFSET)
                target[now, 2] = p[now, 2]
            touched |= now
        target[:, 2] = torch.where(touched, target[:, 2], target[:, 2] - 0.003)
        env.step(torch.cat((target, rot6d, grip, pos_block), dim=-1))
        if bool(touched.all()):
            break
    if not bool(touched.all()):
        raise RuntimeError(f"[press] envs {(~touched).nonzero().flatten().tolist()} never touched")
    contact_z = target[:, 2].clone()
    if args.press_start == "above":
        target[:, 2] += args.press_gap
    for _ in range(10 if args.press_start == "contact" else 20):  # settle on position control
        env.step(torch.cat((target, rot6d, grip, pos_block), dim=-1))
    bound = float(env.cfg.hybrid.force_target_bounds[2])
    if float(fd.abs().max()) > bound:
        raise ValueError(f"--press_force up to {float(fd.abs().max())} N is past the z bound {bound} N")
    block = pos_block.clone()
    block[:, 2] = 1.0  # z on force
    block[:, 5] = fd / bound
    est, true = [], []
    substep = env.action_manager.apply_action

    def apply_then_record():
        substep()
        est.append(term.applied_force()[:, 2].clone())
        true.append(-env.wrench.physx_reference()[:, 2].clone())

    env.action_manager.apply_action = apply_then_record
    for _ in range(args.press_steps):
        env.step(torch.cat((target, rot6d, grip, block), dim=-1))
    env.action_manager.apply_action = substep
    p, _ = _ee_pose_b(env, ee, EE_OFFSET)
    est, true = torch.stack(est, 1), torch.stack(true, 1)  # (n, substeps)
    tail = 240  # 2 s of 1/120 s substeps
    law = "Forge's K_f (f_d - f)" if args.forge_law else "f_d + K_f (f_d - f), decoupled"
    print(f"[press] {law}; start {args.press_start}; applied f_z, N (negative presses down); {args.press_steps} steps", flush=True)
    print(f"{'K_f':>6} {'f_d':>6} {'true mean':>10} {'true p2p':>9} {'true peak':>10} {'est mean':>9} {'z - contact mm':>15}")
    rows = []
    for i, (k, f) in enumerate(grid):
        t, e = true[i, -tail:], est[i, -tail:]
        ok = bool(torch.isfinite(true[i]).all())
        row = {
            "kf": k, "fd": -f, "true_mean": float(t.mean()), "true_p2p": float(t.max() - t.min()),
            "true_peak": float(true[i].min()), "est_mean": float(e.mean()), "dz_mm": float((p[i, 2] - contact_z[i]) * 1000.0), "finite": ok,
        }
        rows.append(row)
        print(f"{k:6.2f} {-f:6.1f} {row['true_mean']:10.2f} {row['true_p2p']:9.2f} {row['true_peak']:10.2f} {row['est_mean']:9.2f} {row['dz_mm']:15.1f}", flush=True)
    torch.save({"rows": rows, "est": est.cpu(), "true": true.cpu(), "grid": grid, "start": args.press_start}, args.out)


def main():
    if args.mode == "press":
        press()
        return
    if args.mode == "reference":
        env = make_env(REFERENCE_TASK, 1)
        pos, quat = run(env, 0)
        torch.save({"pos": pos[0], "quat": quat[0], "task": REFERENCE_TASK, "steps": len(target_sequence())}, args.out)
        print(f"[tracking] reference: {pos.shape[1]} substeps -> {args.out}", flush=True)
        return
    ref = torch.load(args.reference)
    grid = list(itertools.product(
        [float(v) for v in args.kpos.split(",")], [float(v) for v in args.krot.split(",")], [float(v) for v in args.dcoef.split(",")]
    ))
    env = make_env(HYBRID_TASK, len(grid), decouple=not args.forge_law)
    term = env.action_manager.get_term("hybrid")
    k = torch.tensor([[kp] * 3 + [kr] * 3 for kp, kr, _ in grid], device=env.device)
    c_rot = lambda c: c if args.dcoef_rot is None else args.dcoef_rot
    d = torch.tensor([[c] * 3 + [c_rot(c)] * 3 for _, _, c in grid], device=env.device) * torch.sqrt(k)
    term.stiffness, term.damping = k, d
    pos, quat = run(env, term.action_dim - 10)
    if pos.shape[1] != ref["pos"].shape[0]:
        raise RuntimeError(f"{pos.shape[1]} substeps vs the reference's {ref['pos'].shape[0]}: not the same sequence")
    pos_err = (pos - ref["pos"]).norm(dim=-1) * 1000.0  # mm
    rot_err = quat_angle(quat, ref["quat"].expand_as(quat)) * (180.0 / math.pi)  # deg
    stats = []
    for i, (kp, kr, c) in enumerate(grid):
        pe, re = pos_err[i], rot_err[i]
        bad = not (torch.isfinite(pe).all() and torch.isfinite(re).all())
        stats.append({
            "kpos": kp, "krot": kr, "dcoef": c,
            "pos_rms_mm": float("nan") if bad else float(pe.pow(2).mean().sqrt()),
            "pos_max_mm": float("nan") if bad else float(pe.max()),
            "rot_rms_deg": float("nan") if bad else float(re.pow(2).mean().sqrt()),
            "rot_max_deg": float("nan") if bad else float(re.max()),
        })
    ranked = sorted(stats, key=lambda s: (math.isnan(s["pos_rms_mm"]), s["pos_rms_mm"] + 2.0 * s["rot_rms_deg"]))
    print(f"[tracking] {len(grid)} candidates vs {args.reference} (score: pos RMS mm + 2 x rot RMS deg)")
    print(f"{'K pos':>8} {'K rot':>8} {'D/sqrtK':>8} {'pos rms':>9} {'pos max':>9} {'rot rms':>9} {'rot max':>9}")
    for s in ranked[: args.top]:
        print(f"{s['kpos']:8.0f} {s['krot']:8.0f} {s['dcoef']:8.2f} {s['pos_rms_mm']:9.2f} {s['pos_max_mm']:9.2f} {s['rot_rms_deg']:9.2f} {s['rot_max_deg']:9.2f}")
    torch.save({"stats": stats, "pos": pos, "quat": quat, "reference": args.reference}, args.out)


if __name__ == "__main__":
    import os

    main()
    # Kit's shutdown can hang after a headless run; everything is saved by now
    sys.stdout.flush()
    os._exit(0)
