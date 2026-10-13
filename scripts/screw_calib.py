"""Screw calibration: ONLY the screwing, with everything before it set up by hand.

    python scripts/screw_calib.py --rot_speed 90 --down_speed 0.002 --out screw.mp4 [--headless]
    python scripts/screw_calib.py --rot_speed 90 --pitch 0.004 --press 0.003 --csv screw.csv

Start of the test (teleported, then settled): the table top is in the corner, as the
pre-assembly leaves it (against the front and left obstacles); leg 4 stands upright over
hole ``--hole`` with its bolt ``--seat_dz`` above the seat (it drops onto the thread while
settling); the hand is above the leg, gripper open, already yawed so that (a) the finger
pads are flat on the leg's faces and (b) joint 7 is as close to its -limit as ``--q7_margin``
allows -- the most tightening travel a grasp can have.

Then, driven with absolute EE pose targets (``OneLegReplayAbsCfg``: pose_abs, no success
termination):  approach -> descend onto the leg -> close -> **screw**: yaw decreases at
``--rot_speed`` deg/s (tightening = -world yaw = +joint 7), the TCP stays on the hole axis,
and its height follows ``z_grasp - down_speed * t - press``. ``--down_speed`` is the descent
you want to calibrate (``--pitch`` sets it to pitch * rot_speed / 360, the kinematic ideal
for a thread of that pitch); ``--press`` is a constant extra push. The screw ends when
joint 7 reaches its +limit margin, when the leg counts as assembled, or after
``--screw_time`` s.

Every env step logs (and ``--csv`` saves): time, phase, q7, hand yaw and z, leg yaw and z
(above the seat), leg tilt, leg-hole xy offset, gripper width, assembled. The summary at the
end says how far the hand turned, how far the leg turned (the difference is slip in the
fingers), how far the leg went down, and whether it is assembled. A perfect screw: leg
turn == hand turn, leg z decreasing at pitch * turn, tilt ~0, offset ~0, assembled at the end.
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--hole", type=int, default=0, help="which of the 4 holes (0 = leg 4's)")
parser.add_argument("--seat_dz", type=float, default=0.010, help="m the leg starts above the seat (thread stop ~0.009)")
parser.add_argument("--twist", type=float, default=0.0, help="deg initial twist of the leg about its axis")
parser.add_argument("--top_pos", type=str, default=None, help="x,y of the top's centre (default: the corner)")
parser.add_argument("--hover", type=float, default=0.08, help="m above the grasp height the hand starts")
parser.add_argument("--grasp_dz", type=float, default=0.0, help="TCP height relative to the leg's centre at the grasp")
parser.add_argument("--q7_margin", type=float, default=0.15, help="rad kept inside joint 7's limits")
parser.add_argument("--rot_speed", type=float, default=90.0, help="deg/s of tightening (negative world yaw)")
parser.add_argument("--down_speed", type=float, default=None, help="m/s the height target descends while screwing")
parser.add_argument("--pitch", type=float, default=None, help="thread pitch m: sets down_speed = pitch * rot_speed / 360")
parser.add_argument("--press", type=float, default=0.0, help="m constant extra push below the descending target")
parser.add_argument("--screw_time", type=float, default=10.0, help="s cap on the screw phase")
parser.add_argument("--approach_time", type=float, default=2.0)
parser.add_argument("--descend_time", type=float, default=1.5)
parser.add_argument("--close_time", type=float, default=1.0)
parser.add_argument("--settle", type=int, default=20, help="env steps after the teleport")
parser.add_argument("--part_friction", type=float, default=None)
parser.add_argument("--table_friction", type=float, default=None)
parser.add_argument("--csv", type=str, default=None, help="write the per-step log here")
parser.add_argument("--out", type=str, default=None, help="mp4 of the test")
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--cam", type=str, default="0.75,0.35,0.30")
parser.add_argument("--look", type=str, default="0.55,0.08,0.05")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.out:
    args.enable_cameras = True
if args.down_speed is None:
    args.down_speed = (args.pitch * args.rot_speed / 360.0) if args.pitch else 0.0
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab.sim.spawners.sensors import PinholeCameraCfg  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    combine_frame_transforms,
    quat_apply,
    quat_from_euler_xyz,
    quat_mul,
    subtract_frame_transforms,
)
from robonuke_rl_core.recording import VideoWriter, read_camera_rgb  # noqa: E402

import furniture_lab.tasks  # noqa: E402, F401
from furniture_lab.tasks.assembly.assembly_env import AssemblyEnv  # noqa: E402
from furniture_lab.tasks.assembly.assembly_env_cfg import EE_OFFSET  # noqa: E402
from furniture_lab.tasks.assembly.mdp.state import parts_pose_w, tracker  # noqa: E402
from furniture_lab.tasks.assembly.replay_env_cfg import OneLegReplayAbsCfg  # noqa: E402
from furniture_lab.tasks.furniture import frames  # noqa: E402
from furniture_lab.tasks.furniture.assembly import pairs_assembled, relative_pose, rotation_diag  # noqa: E402
from furniture_lab.tasks.furniture.square_table import PARTS, rel_pose_targets, reset_pose_world  # noqa: E402

CAMERA_KEY = "recorder_cam"
Q7_MIN, Q7_MAX = -2.8973, 2.8973
TOP_HALF_WIDTH = 0.08125


# ------------------------------------------------------------------------------ helpers
class Recorder:
    def __init__(self, env, path: str) -> None:
        self.env = env
        self.camera = env.scene[CAMERA_KEY]
        eye = torch.tensor([[float(v) for v in args.cam.split(",")]], device=env.device)
        look = torch.tensor([[float(v) for v in args.look.split(",")]], device=env.device)
        origins = env.scene.env_origins[:1]
        self.camera.set_world_poses_from_view(eye + origins, look + origins, env_ids=[0])
        self.writer = VideoWriter(path, fps=args.fps)
        self.caption = ""

    def frame(self) -> None:
        from PIL import Image, ImageDraw

        self.camera.update(self.env.physics_dt)
        image = Image.fromarray(read_camera_rgb(self.camera)[0].numpy()[..., :3])
        if self.caption:
            draw = ImageDraw.Draw(image)
            lines = self.caption.split("\n")
            pad, lh = 6, 14
            w = max(len(line) for line in lines) * 7 + 2 * pad
            h = len(lines) * lh + 2 * pad
            x0, y0 = image.width - w - 8, image.height - h - 8
            draw.rectangle([(x0, y0), (x0 + w, y0 + h)], fill=(0, 0, 0))
            for i, line in enumerate(lines):
                draw.text((x0 + pad, y0 + pad + i * lh), line, fill=(255, 255, 0))
        self.writer.append(torch.from_numpy(np.array(image)))

    def close(self) -> None:
        self.writer.close()


def ee_pose_b(env):
    robot = env.scene["robot"]
    body = robot.find_bodies("panda_hand")[0][0]
    pos_w, quat_w = robot.data.body_pos_w[:, body], robot.data.body_quat_w[:, body]
    off = torch.tensor(EE_OFFSET, device=env.device, dtype=pos_w.dtype).expand(env.num_envs, -1)
    pos_w, quat_w = combine_frame_transforms(pos_w, quat_w, off)
    return subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, pos_w, quat_w)


def yaw_of(quat_wxyz: torch.Tensor) -> float:
    """World yaw of the frame's x-axis (rad)."""
    x = quat_apply(quat_wxyz[:1], torch.tensor([[1.0, 0.0, 0.0]], device=quat_wxyz.device))[0]
    return math.atan2(float(x[1]), float(x[0]))


def down_quat(yaw: float, device) -> torch.Tensor:
    """Gripper straight down, hand x-axis at world yaw ``yaw`` (wxyz)."""
    rz = quat_from_euler_xyz(torch.zeros(1, device=device), torch.zeros(1, device=device), torch.tensor([yaw], device=device))
    rx = quat_from_euler_xyz(torch.tensor([math.pi], device=device), torch.zeros(1, device=device), torch.zeros(1, device=device))
    return quat_mul(rz, rx)


def unwrap(prev: float, now: float) -> float:
    d = (now - prev + math.pi) % (2 * math.pi) - math.pi
    return prev + d


def assembled_now(env) -> bool:
    t = tracker(env)
    pos, quat = parts_pose_w(env)
    return bool(pairs_assembled(pos, quat, t.pairs, t.targets, t.pos_threshold, t.ori_bound)[0, 0].item())


# ------------------------------------------------------------------------------ main
def main() -> None:
    cfg = OneLegReplayAbsCfg()
    cfg.scene.num_envs = 1
    cfg.furniture.log_resets = True
    if args.part_friction is not None:
        cfg.furniture.part_friction = args.part_friction
    if args.table_friction is not None:
        cfg.furniture.table_friction = args.table_friction
    cfg.apply_friction()
    if args.out:
        cfg.scene.clone_in_fabric = False
        setattr(
            cfg.scene,
            CAMERA_KEY,
            TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/" + CAMERA_KEY,
                data_types=["rgb"],
                spawn=PinholeCameraCfg(focal_length=24.0, focus_distance=0.5, horizontal_aperture=20.955, clipping_range=(0.05, 20.0)),
                width=640,
                height=480,
            ),
        )
    env = AssemblyEnv(cfg)
    env.reset()
    recorder = Recorder(env, args.out) if args.out else None
    dev = env.device
    dt = env.step_dt
    robot = env.scene["robot"]
    q7_idx = robot.find_joints("panda_joint7")[0][0]
    top, leg = env.scene["square_table_top"], env.scene["square_table_leg4"]
    origin = env.scene.env_origins[:1]

    def q7() -> float:
        return float(robot.data.joint_pos[0, q7_idx])

    def width() -> float:
        return float(robot.data.joint_pos[0, robot.find_joints("panda_finger_joint.*")[0]].sum())

    # -- the hand out of the way first (open, above the reset pose), so the teleport is clean
    pos0, quat0 = ee_pose_b(env)
    hold = torch.cat([pos0, quat0, torch.ones(1, 1, device=dev)], dim=-1)
    for _ in range(5):
        env.step(hold)

    # -- the top in the corner (as the pre-assembly leaves it), reset orientation
    if args.top_pos:
        tx, ty = (float(v) for v in args.top_pos.split(","))
    else:
        # the corner between the front and the left obstacle, 1 cm clear of both (where
        # upstream's push leaves it, give or take that clearance)
        tx = frames.OBSTACLE_FRONT_POS[0] - frames.OBSTACLE_FRONT_SIZE[0] / 2 - TOP_HALF_WIDTH - 0.01
        ty = frames.OBSTACLE_SIDE_POS["obstacle_left"][1] - frames.OBSTACLE_SIDE_SIZE[1] / 2 - TOP_HALF_WIDTH - 0.01
    top_pos = top.data.root_pos_w[:1].clone()
    top_pos[0, 0], top_pos[0, 1] = origin[0, 0] + tx, origin[0, 1] + ty
    # the nominal reset orientation, without the reset's +-15 deg yaw noise: a square top
    # yawed 15 deg reaches 2.5 cm further into the corner and lands inside the obstacle
    _, nominal_quat = reset_pose_world(PARTS[0], cfg.furniture.mesh_frame)
    top_quat = torch.tensor([nominal_quat], dtype=torch.float32, device=dev)
    zero6 = torch.zeros(1, 6, device=dev)
    top.write_root_pose_to_sim(torch.cat([top_pos, top_quat], dim=-1))
    top.write_root_velocity_to_sim(zero6)

    # -- leg 4 upright over the hole, seat_dz above the seat, twisted by --twist
    rel = rel_pose_targets([(0, 4)], cfg.furniture.mesh_frame)[0, args.hole].copy()
    rel[2] += args.seat_dz
    rel_t = torch.tensor(rel, dtype=torch.float32, device=dev).reshape(1, 3)
    twist_q = quat_from_euler_xyz(torch.zeros(1, device=dev), torch.zeros(1, device=dev), torch.tensor([math.radians(args.twist)], device=dev))
    leg_pos = top_pos + quat_apply(top_quat, rel_t)
    leg_quat = quat_mul(top_quat, twist_q)
    leg.write_root_pose_to_sim(torch.cat([leg_pos, leg_quat], dim=-1))
    leg.write_root_velocity_to_sim(zero6)
    for _ in range(args.settle):
        env.step(hold)
    seat_z_w = (top_pos + quat_apply(top_quat, rel_t - torch.tensor([[0, 0, args.seat_dz]], device=dev)))[0, 2]
    hole_w = (top.data.root_pos_w[:1] + quat_apply(top.data.root_quat_w[:1], rel_t))[0]
    hole_b = (hole_w - origin[0]).tolist()  # the robot base is at the env origin

    def leg_state():
        hole_w_ = hole_w
        rel_pos, rel_quat = relative_pose(top.data.root_pos_w[:1], top.data.root_quat_w[:1], leg.data.root_pos_w[:1], leg.data.root_quat_w[:1])
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, rotation_diag(rel_quat)[0, 2].item()))))
        above_seat = (leg.data.root_pos_w[0, 2] - seat_z_w).item()
        off_xy = (leg.data.root_pos_w[0, :2] - hole_w_[:2]).norm().item()
        return yaw_of(leg.data.root_quat_w[:1]), above_seat, tilt, off_xy

    top_b = (top.data.root_pos_w[0] - origin[0]).tolist()
    top_tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(quat_apply(top.data.root_quat_w[:1], torch.tensor([[0.0, 0.0, 1.0]], device=dev))[0, 2])))))
    print(f"[screw] top asked at ({tx:.3f}, {ty:.3f}); after settling it is at ({top_b[0]:.3f}, {top_b[1]:.3f}, {top_b[2]:.4f}), "
          f"tilt {top_tilt:.1f} deg, yaw {math.degrees(yaw_of(top.data.root_quat_w[:1])):+.1f} deg")
    print(f"[screw] hole axis at ({hole_b[0]:.4f}, {hole_b[1]:.4f}) base frame; leg at "
          f"({(leg.data.root_pos_w[0, 0] - origin[0, 0]).item():.4f}, {(leg.data.root_pos_w[0, 1] - origin[0, 1]).item():.4f}, "
          f"{(leg.data.root_pos_w[0, 2] - origin[0, 2]).item():.4f}), {leg_state()[1] * 1000:.1f} mm above the seat, tilt {leg_state()[2]:.1f} deg")
    if abs(top_b[0] - tx) > 0.02 or abs(top_b[1] - ty) > 0.02 or top_tilt > 2.0:
        print("[screw] WARNING: the top moved while settling (teleported into an obstacle?); use --top_pos x,y", flush=True)


    # -- the grasp yaw: pads flat on a face (hand yaw = face yaw + k*90) and joint 7 nearest its -limit
    hand_yaw0 = yaw_of(ee_pose_b(env)[1])
    face_yaw = yaw_of(leg.data.root_quat_w[:1])  # uwlab x: a face normal
    dmax = q7() - (Q7_MIN + args.q7_margin)  # the most the world yaw may grow (+q7 = -yaw)
    d = (face_yaw - hand_yaw0) % (math.pi / 2)
    d += math.floor((dmax - d) / (math.pi / 2)) * (math.pi / 2)
    grasp_yaw = hand_yaw0 + d
    print(f"[screw] hand yaw {math.degrees(hand_yaw0):+.1f} -> grasp yaw {math.degrees(grasp_yaw):+.1f} deg "
          f"(leg face {math.degrees(face_yaw):+.1f}); joint 7 {q7():+.3f} -> {q7() - d:+.3f} rad "
          f"of [{Q7_MIN:+.3f}, {Q7_MAX:+.3f}]: {math.degrees(Q7_MAX - args.q7_margin - (q7() - d)):.0f} deg of tightening available")

    leg_c_z = leg.data.root_pos_w[0, 2].item() - origin[0, 2].item()
    grasp_z = leg_c_z + args.grasp_dz
    rows = []
    t_total = 0.0
    hand_yaw_u, leg_yaw_u = hand_yaw0, yaw_of(leg.data.root_quat_w[:1])
    marks = {}

    def run(phase: str, seconds: float, target_fn, grip: float, stop_fn=None):
        nonlocal t_total, hand_yaw_u, leg_yaw_u
        n = int(round(seconds / dt))
        t0 = t_total
        for i in range(n):
            tp = t_total - t0
            pos, quat = target_fn(tp)
            action = torch.cat([pos.reshape(1, 3), quat.reshape(1, 4), torch.tensor([[grip]], device=dev)], dim=-1)
            env.step(action)
            t_total += dt
            ee_p, ee_q = ee_pose_b(env)
            hand_yaw_u = unwrap(hand_yaw_u, yaw_of(ee_q))
            ly, above, tilt, off = leg_state()
            leg_yaw_u = unwrap(leg_yaw_u, ly)
            asm = assembled_now(env)
            rows.append(dict(t=round(t_total, 3), phase=phase, q7=q7(), hand_yaw_deg=math.degrees(hand_yaw_u),
                             hand_z=float(ee_p[0, 2]), target_z=float(pos[2]), leg_yaw_deg=math.degrees(leg_yaw_u),
                             leg_above_seat_mm=above * 1000, leg_tilt_deg=tilt, leg_hole_offset_mm=off * 1000,
                             gripper_width=width(), assembled=int(asm)))
            if recorder is not None:
                recorder.caption = (f"{phase}  t {t_total:5.2f} s\nq7 {q7():+.2f}  hand yaw {math.degrees(hand_yaw_u):+7.1f}\n"
                                    f"leg yaw {math.degrees(leg_yaw_u):+7.1f}  up {above * 1000:5.1f} mm\n"
                                    f"tilt {tilt:4.1f}  off {off * 1000:4.1f} mm  asm {asm}")
                recorder.frame()
            if i % max(1, int(0.5 / dt)) == 0:
                print(f"[screw] {phase:8s} t {t_total:5.2f}  q7 {q7():+.3f}  hand yaw {math.degrees(hand_yaw_u):+7.1f}  "
                      f"leg yaw {math.degrees(leg_yaw_u):+7.1f}  leg up {above * 1000:5.1f} mm  tilt {tilt:4.1f}  "
                      f"off {off * 1000:4.1f} mm  width {width():.3f}  asm {asm}", flush=True)
            if stop_fn is not None and stop_fn():
                return True
        return False

    hole_xy = (float(hole_b[0]), float(hole_b[1]))
    gq = down_quat(grasp_yaw, dev)

    def above_fn(_):
        return torch.tensor([hole_xy[0], hole_xy[1], grasp_z + args.hover], device=dev), gq

    def descend_fn(tp):
        z = grasp_z + args.hover * max(0.0, 1.0 - tp / args.descend_time)
        return torch.tensor([hole_xy[0], hole_xy[1], z], device=dev), gq

    def grasp_fn(_):
        return torch.tensor([hole_xy[0], hole_xy[1], grasp_z], device=dev), gq

    run("approach", args.approach_time, above_fn, +1.0)
    run("descend", args.descend_time, descend_fn, +1.0)
    run("close", args.close_time, grasp_fn, -1.0)
    marks["grasp"] = (hand_yaw_u, leg_yaw_u, leg_state()[1], q7())
    ee_p, _ = ee_pose_b(env)
    print(f"[screw] after close: width {width():.4f} m (the leg is 0.030; > 0.04 = missed it), TCP "
          f"({ee_p[0, 0]:.4f}, {ee_p[0, 1]:.4f}, {ee_p[0, 2]:.4f}) vs hole ({hole_xy[0]:.4f}, {hole_xy[1]:.4f}), "
          f"leg tilt {leg_state()[2]:.1f} deg, q7 {q7():+.3f}", flush=True)
    if width() > 0.04:
        print("[screw] GRASP MISSED -- the screw that follows means nothing; check the TCP-vs-hole offset above", flush=True)

    omega = math.radians(args.rot_speed)

    def screw_fn(tp):
        z = grasp_z - args.down_speed * tp - args.press
        return torch.tensor([hole_xy[0], hole_xy[1], z], device=dev), down_quat(grasp_yaw - omega * tp, dev)

    asm_streak = [0]
    why = ["time"]

    def screw_stop():
        asm_streak[0] = asm_streak[0] + 1 if assembled_now(env) else 0
        if q7() >= Q7_MAX - args.q7_margin:
            why[0] = "joint 7 at its limit margin"
            return True
        if asm_streak[0] >= 10:  # the FurnitureBench check holding for half a second
            why[0] = "assembled (held 10 steps)"
            return True
        return False

    run("screw", args.screw_time, screw_fn, -1.0, screw_stop)
    print(f"[screw] screw phase ended: {why[0]}", flush=True)
    marks["end"] = (hand_yaw_u, leg_yaw_u, leg_state()[1], q7())

    # -- summary
    (hy0, ly0, up0, q0), (hy1, ly1, up1, q1) = marks["grasp"], marks["end"]
    hand_turn, leg_turn = math.degrees(hy0 - hy1), math.degrees(ly0 - ly1)
    ly, above, tilt, off = leg_state()
    print("[screw] ---------------------------------------------------------------")
    print(f"[screw] rot {args.rot_speed:.0f} deg/s, down {args.down_speed * 1000:.2f} mm/s, press {args.press * 1000:.1f} mm, "
          f"friction parts {cfg.furniture.part_friction} table {cfg.furniture.table_friction}")
    print(f"[screw] hand turned {hand_turn:7.1f} deg (joint 7 {q0:+.3f} -> {q1:+.3f})")
    print(f"[screw] leg  turned {leg_turn:7.1f} deg  -> slip {hand_turn - leg_turn:6.1f} deg")
    print(f"[screw] leg went down {(up0 - up1) * 1000:5.1f} mm: {up0 * 1000:5.1f} -> {up1 * 1000:5.1f} mm above the seat"
          + (f"  ({(up0 - up1) * 1000 / (leg_turn / 360):.2f} mm per leg turn)" if abs(leg_turn) > 30 else ""))
    print(f"[screw] final tilt {tilt:.1f} deg, hole offset {off * 1000:.1f} mm, width {width():.4f}, assembled {assembled_now(env)}, "
          f"ended by: {why[0]}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[screw] wrote {args.csv} ({len(rows)} rows)")
    if recorder is not None:
        recorder.close()
        print(f"[screw] wrote {args.out}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        app.close()
