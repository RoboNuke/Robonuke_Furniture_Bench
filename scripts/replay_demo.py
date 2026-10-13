"""Replay a FurnitureBench sim demo (``.pkl``) in this env, two ways.

    python scripts/replay_demo.py --demo DEMO.pkl --mode ee_track --headless --out ee_track.mp4
    python scripts/replay_demo.py --demo DEMO.pkl --mode actions  --headless --out actions.mp4

The env is reset to the demo's first state (robot joints, gripper, part poses), then:

* ``ee_track`` -- the recorded EE pose of the *next* demo step is the controller's absolute
  target each step (``pose_abs``), gripper from the recorded command. Closed-loop on the
  hand, open-loop on the parts: tests whether this controller and these meshes reproduce
  the demo when driven by the recorded hand path.
* ``actions`` -- the recorded 8-D deltas go through the task's own ``pose_rel`` action, as a
  policy's would: position delta / 0.025, the EE-frame rotation delta re-expressed in the
  base frame and turned into axis-angle / 0.1, gripper sign flipped (FurnitureBench: < 0
  opens; Isaac Lab's binary action: < 0 closes). Fully open-loop; expected to drift.

Per step it prints and (with ``--out``) overlays: EE position/orientation error against the
demo, leg-4 and top pose error, and the assembled flag. A CSV with the same columns is
written next to the mp4. Sim demos are 20 Hz like this env, so one demo step is one env step.

Make a sim demo with FurnitureBench's scripted policy (its Isaac Gym env, Python 3.8)::

    python -m furniture_bench.scripts.collect_data --furniture one_leg --is-sim --scripted \\
        --headless --randomness low --num-demos 1 --out-data-path ~/fb_demos
    # -> ~/fb_demos/one_leg/<date>/<date>.pkl
"""

import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--demo", required=True, help="FurnitureBench .pkl")
parser.add_argument("--mode", choices=["ee_track", "actions"], required=True)
parser.add_argument("--out", type=str, default=None, metavar="FILE.mp4", help="record (headless ok)")
parser.add_argument("--fps", type=int, default=20, help="playback rate; 20 = real time (one frame per env step)")
parser.add_argument("--cam", type=str, default="0.95,0.45,0.35", help="camera eye x,y,z (robot-base frame)")
parser.add_argument("--look", type=str, default="0.54,0.0,0.03", help="camera target x,y,z")
parser.add_argument("--max_steps", type=int, default=None, help="replay only the first N demo steps")
parser.add_argument("--settle", type=int, default=10, help="env steps holding the first pose before the replay")
parser.add_argument("--print_every", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.out:
    args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab.sim.spawners.sensors import PinholeCameraCfg  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    axis_angle_from_quat,
    combine_frame_transforms,
    quat_apply,
    quat_conjugate,
    quat_error_magnitude,
    quat_mul,
    subtract_frame_transforms,
)

from robonuke_rl_core.recording import VideoWriter, read_camera_rgb  # noqa: E402

from furniture_lab.demos import load_demo  # noqa: E402
from furniture_lab.tasks.assembly.assembly_env import AssemblyEnv  # noqa: E402
from furniture_lab.tasks.assembly.assembly_env_cfg import EE_OFFSET  # noqa: E402
from furniture_lab.tasks.assembly.mdp.state import tracker  # noqa: E402
from furniture_lab.tasks.assembly.replay_env_cfg import OneLegReplayAbsCfg, OneLegReplayRelCfg  # noqa: E402

CAMERA_KEY = "recorder_cam"


# ------------------------------------------------------------------------------ recording
class Recorder:
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
        self.writer.append(torch.from_numpy(np.array(image)))
        self.frames += 1

    def close(self) -> None:
        self.writer.close()
        print(f"[replay] wrote {args.out}: {self.frames} frames at {args.fps} fps", flush=True)


# ------------------------------------------------------------------------------ helpers
def ee_pose_b(env):
    """(pos, quat wxyz) of the TCP (panda_hand + EE_OFFSET) in the robot-base frame, env 0."""
    robot = env.scene["robot"]
    body = robot.find_bodies("panda_hand")[0][0]
    pos_w, quat_w = robot.data.body_pos_w[:, body], robot.data.body_quat_w[:, body]
    off = torch.tensor(EE_OFFSET, device=env.device, dtype=pos_w.dtype).expand(env.num_envs, -1)
    pos_w, quat_w = combine_frame_transforms(pos_w, quat_w, off)
    return subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, pos_w, quat_w)


def set_state(env, demo, t: int) -> None:
    """Put env 0 into the demo's state at step t: arm joints, gripper, part poses."""
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
    tracker(env).reset()


def to_t(a, env):
    return torch.tensor(np.asarray(a), device=env.device, dtype=torch.float32).reshape(1, -1)


def gripper_cmd(demo, t: int) -> float:
    """FurnitureBench: < 0 opens. Isaac Lab's binary action: < 0 closes. So negate."""
    return -float(np.sign(demo.act_grip[t]) or 1.0)


def make_action(env, demo, t: int, mode: str) -> torch.Tensor:
    grip = torch.tensor([[gripper_cmd(demo, t)]], device=env.device)
    if mode == "ee_track":
        target_pos = to_t(demo.ee_pos[t + 1], env)
        target_quat = to_t(demo.ee_quat[t + 1], env)
        return torch.cat([target_pos, target_quat, grip], dim=-1)
    # actions: the recorded delta, re-expressed for the pose_rel action
    arm_cfg = env.cfg.actions.arm
    dpos = to_t(demo.act_dpos[t], env) / arm_cfg.position_scale
    q_d = to_t(demo.act_dquat[t], env)  # rotation delta in the EE frame
    _, q_ee = ee_pose_b(env)
    q_base = quat_mul(quat_mul(q_ee, q_d), quat_conjugate(q_ee))  # the same rotation, base frame
    drot = axis_angle_from_quat(q_base) / arm_cfg.orientation_scale
    return torch.cat([dpos, drot, grip], dim=-1)


# ------------------------------------------------------------------------------ main
def main() -> None:
    demo = load_demo(args.demo)
    print(f"[replay] {demo.summary()}")
    T = demo.length if args.max_steps is None else min(demo.length, args.max_steps)

    cfg = (OneLegReplayAbsCfg if args.mode == "ee_track" else OneLegReplayRelCfg)()
    cfg.scene.num_envs = 1
    cfg.furniture.log_resets = True
    if args.out:
        cfg.scene.clone_in_fabric = False
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
    env = AssemblyEnv(cfg)
    env.reset()
    recorder = Recorder(env) if args.out else None

    set_state(env, demo, 0)
    hold = make_action(env, demo, 0, "ee_track") if args.mode == "ee_track" else torch.cat(
        [torch.zeros(1, 6, device=env.device), torch.tensor([[gripper_cmd(demo, 0)]], device=env.device)], dim=-1
    )
    for _ in range(args.settle):
        env.step(hold)
    pos, quat = ee_pose_b(env)
    start_err = (pos[0] - to_t(demo.ee_pos[0], env)[0]).norm().item()
    print(f"[replay] after reset-to-demo: EE position error {start_err * 1000:.1f} mm "
          f"(the demo's robot vs this Panda at the same joints)")

    leg_i, top_i = demo.part_names.index("square_table_leg4"), demo.part_names.index("square_table_top")
    leg, top = env.scene["square_table_leg4"], env.scene["square_table_top"]
    origin = env.scene.env_origins[:1]
    rows = []
    first_assembled = None
    for t in range(T):
        action = make_action(env, demo, t, args.mode)
        env.step(action)
        pos, quat = ee_pose_b(env)
        ee_err = (pos[0] - to_t(demo.ee_pos[t + 1], env)[0]).norm().item()
        ee_rot_err = quat_error_magnitude(quat, to_t(demo.ee_quat[t + 1], env)).item()
        leg_err = (leg.data.root_pos_w[0] - origin[0] - to_t(demo.parts_pos[t + 1, leg_i], env)[0]).norm().item()
        top_err = (top.data.root_pos_w[0] - origin[0] - to_t(demo.parts_pos[t + 1, top_i], env)[0]).norm().item()
        assembled = bool(tracker(env).all_assembled()[0].item())
        if assembled and first_assembled is None:
            first_assembled = t
        rows.append((t, ee_err, ee_rot_err, leg_err, top_err, int(assembled), float(demo.rewards[t])))
        if t % args.print_every == 0 or t == T - 1 or (assembled and first_assembled == t):
            print(
                f"[replay] {args.mode} step {t:4d}/{T}: EE err {ee_err * 1000:6.1f} mm {math.degrees(ee_rot_err):5.1f} deg  "
                f"leg4 err {leg_err * 1000:6.1f} mm  top err {top_err * 1000:6.1f} mm  assembled {assembled}  "
                f"demo reward {demo.rewards[t]:.0f}",
                flush=True,
            )
        if recorder is not None:
            recorder.caption = (
                f"{args.mode}  step {t:4d}/{T}\nEE err {ee_err * 1000:5.1f} mm  {math.degrees(ee_rot_err):4.1f} deg\n"
                f"leg4 err {leg_err * 1000:5.1f} mm\nassembled {assembled}"
            )
            recorder.frame()

    ee = np.array([r[1] for r in rows])
    print(
        f"[replay] done: EE err mean {ee.mean() * 1000:.1f} mm, max {ee.max() * 1000:.1f} mm; "
        f"assembled {'at step ' + str(first_assembled) if first_assembled is not None else 'never'}; "
        f"demo rewarded at steps {np.nonzero(demo.rewards[:T])[0].tolist()}"
    )
    csv_path = Path(args.out).with_suffix(".csv") if args.out else Path(f"replay_{args.mode}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "ee_pos_err_m", "ee_rot_err_rad", "leg4_pos_err_m", "top_pos_err_m", "assembled", "demo_reward"])
        w.writerows(rows)
    print(f"[replay] per-step metrics: {csv_path}")
    if recorder is not None:
        recorder.close()
    env.close()


if __name__ == "__main__":
    main()
    app.close()
