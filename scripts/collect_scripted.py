"""Collect ``one_leg`` demos in this env with FurnitureBench's scripted policy (ported).

    python scripts/collect_scripted.py --headless --num_demos 1 --out_dir ~/fb_demos [--record] [--save_failure]
    # success -> ~/fb_demos/one_leg/<stamp>/<stamp>.pkl  (+ recording.mp4 with --record)
    # failure -> ~/fb_demos/one_leg/fails/fail<n>.pkl     (+ fail<n>.mp4) with --save_failure

The policy (``furniture_lab/scripted_policy.py``) is upstream's state machine, fed this env's
state in upstream's representation and driving the task's own ``pose_rel`` action, so what
it records is what a policy trained here would output: the 8-D FurnitureBench action
``[dpos (m), dquat (xyzw, EE frame), gripper]`` is the delta given to the controller, with
``dpos / 0.025`` and the base-frame axis-angle ``/ 0.1`` as the env action. The pkl has
``observations`` (``robot_state`` dict + ``parts_poses`` in the tag frame, xyzw), ``actions``,
``rewards`` (1 at the step the leg counts as assembled), ``skills``, ``success``, ``furniture``
-- readable by ``furniture_lab.demos.load_demo`` and ``scripts/replay_demo.py``.

A demo ends when the leg is assembled and the hand has lifted clear (success), or at
upstream's ``scripted_timeout`` of 600 steps (failure). With ``--record`` every attempt is
filmed: a success keeps its video beside its pkl, a failure keeps it under ``fails/`` only
with ``--save_failure``.

Not headless, with the viewport window focused: ``k`` makes the current rollout fail now,
``l`` quits immediately (the current attempt is discarded).
"""

import argparse
import math
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out_dir", type=str, default="fb_demos")
parser.add_argument("--num_demos", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--policy", choices=["servo", "scripted"], default="servo",
                    help="servo: closes the loop on the held object (furniture_lab/servo_policy.py); scripted: upstream's hand-only state machine")
parser.add_argument("--lead", type=float, default=10.0, help="servo: deg the hand leads the leg's own yaw while screwing")
parser.add_argument("--pos_speed", type=float, default=0.16, help="servo: m/s cap on the held object's translation")
parser.add_argument("--no_noise", action="store_true", help="disable upstream's target/action noise")
parser.add_argument("--rot_speed", type=float, default=90.0, help="deg/s of wrist rotation while screwing (calibrated: 90)")
parser.add_argument("--wind_speed", type=float, default=180.0, help="deg/s of wrist rotation while winding the open hand back")
parser.add_argument("--insert_lift", type=float, default=0.0, help="m to raise the hold pose above where the leg stopped before opening (0.003 = 3 mm, no press)")
parser.add_argument("--release_dz", type=float, default=0.05, help="servo: m the open hand rises straight up after letting go of the leg, before the regrasp")
parser.add_argument("--insert_floor", type=float, default=0.003, help="servo: m above the seat the leg is never lowered below")
parser.add_argument("--insert_stop", type=float, default=0.0005, help="servo: the leg counts as stopped when it dropped less than this (m) over 3 steps")
parser.add_argument("--insert_rate", type=float, default=0.002, help="m per step the held leg is lowered until it stops (also the press while lowering)")
parser.add_argument("--screw_press", type=float, default=0.005, help="m pressed on the thread per step while screwing (upstream 0.005)")
parser.add_argument("--timeout", type=int, default=600, help="upstream's scripted_timeout for one_leg")
parser.add_argument("--save_failure", action="store_true", help="keep failed attempts (pkl + video) under fails/")
parser.add_argument("--record", action="store_true", help="film every attempt (headless ok)")
parser.add_argument("--fps", type=int, default=None, help="--record playback rate (default: the control rate, real time)")
parser.add_argument("--cam", type=str, default="0.95,0.45,0.35")
parser.add_argument("--look", type=str, default="0.54,0.0,0.03")
parser.add_argument("--print_every", type=int, default=25)
parser.add_argument(
    "--render_every", type=int, default=None,
    help="viewer: render every N physics steps (1/120 s each; default one render per control step: 6 port, 12 resip). "
    "2 = 60 fps. Display only: physics and demos are unchanged",
)
parser.add_argument(
    "--sim", choices=["port", "resip"], default="port",
    help="port: this port's 20 Hz OSC sim; resip: ResiP's FurnitureRLSimEnv -- 10 Hz, differential IK into joint PD "
    "1000/200, friction 0.15/0.10, a 13 N gripper (assembly_env_cfg.apply_resip); the friction/grip flags still override",
)
parser.add_argument(
    "--servo_gain", type=float, default=None,
    help="servo: command = gain x the wanted per-step motion (default 5 for the port's OSC, which closes ~20%% of a "
    "command per step; 1 for resip, whose joint PD closes about all of it)",
)
parser.add_argument("--part_friction", type=float, default=None, help="parts' friction (FurnitureSim: 0.15)")
parser.add_argument("--grip_stiffness", type=float, default=None, help="finger drive N/m (default 600; Isaac Lab's Franka 2000)")
parser.add_argument("--grip_force", type=float, default=None, help="finger effort limit N (default 30; Isaac Lab's Franka 200)")
parser.add_argument("--table_friction", type=float, default=None, help="table/obstacle friction (FurnitureSim: 0.10)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.record:
    args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab.sim.spawners.sensors import PinholeCameraCfg  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    axis_angle_from_quat,
    combine_frame_transforms,
    quat_conjugate,
    quat_mul,
    subtract_frame_transforms,
)

from robonuke_rl_core.recording import VideoWriter, read_camera_rgb  # noqa: E402

from furniture_lab import scripted_policy as sp  # noqa: E402
from furniture_lab import servo_policy as sv  # noqa: E402
from furniture_lab.tasks.assembly.assembly_env import AssemblyEnv  # noqa: E402
from furniture_lab.tasks.assembly.assembly_env_cfg import EE_OFFSET  # noqa: E402
from furniture_lab.tasks.assembly.mdp.state import parts_pose_w, tracker  # noqa: E402
from furniture_lab.tasks.assembly.mdp.terminations import part_out_of_limits  # noqa: E402
from furniture_lab.tasks.furniture.assembly import pairs_assembled  # noqa: E402
from furniture_lab.tasks.assembly.mdp.actions import arm_scales  # noqa: E402
from furniture_lab.tasks.assembly.replay_env_cfg import OneLegReplayRelCfg, OneLegReplayRelResiPCfg  # noqa: E402
from furniture_lab.tasks.furniture import frames  # noqa: E402
from furniture_lab.tasks.furniture.square_table import MESH_FRAMES  # noqa: E402

CAMERA_KEY = "recorder_cam"


# ------------------------------------------------------------------------------ recording
class Recorder:
    def __init__(self, env, path) -> None:
        self.env = env
        self.camera = env.scene[CAMERA_KEY]
        eye = torch.tensor([[float(v) for v in args.cam.split(",")]], device=env.device)
        look = torch.tensor([[float(v) for v in args.look.split(",")]], device=env.device)
        origins = env.scene.env_origins[:1]
        self.camera.set_world_poses_from_view(eye + origins, look + origins, env_ids=[0])
        self.fps = args.fps or max(1, round(1.0 / env.step_dt))
        self.writer = VideoWriter(path, fps=self.fps)
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
        print(f"[collect] wrote {self.writer.path}: {self.frames} frames at {self.fps} fps", flush=True)


class AbortKeys:
    """Keys in the viewport window (click it so it has focus): ``k`` = this rollout fails now,
    ``l`` = quit immediately. No-op when headless."""

    def __init__(self) -> None:
        self.fail = False
        self.quit = False
        self._sub = None
        if args.headless:
            return
        try:
            import carb
            import omni.appwindow

            window = omni.appwindow.get_default_app_window()
            keyboard = window.get_keyboard() if hasattr(window, "get_keyboard") else window.getKeyboard()
            self._input = carb.input.acquire_input_interface()
            self._press = carb.input.KeyboardEventType.KEY_PRESS
            self._sub = self._input.subscribe_to_keyboard_events(keyboard, self._on_key)
            print("[collect] viewport keys: k = this rollout fails now, l = quit", flush=True)
        except Exception as e:  # no window, different carb API ...
            print(f"[collect] viewport keys unavailable: {e}", flush=True)

    def _on_key(self, event, *_args, **_kwargs) -> bool:
        if event.type == self._press:
            name = event.input.name  # carb names keys by their unshifted uppercase letter
            if name == "K":
                self.fail = True
            elif name == "L":
                self.quit = True
        return True

    def new_attempt(self) -> None:
        self.fail = False


def assembled_now(env) -> bool:
    """FurnitureBench's assembled rule on the current poses (the env's tracker is a latch)."""
    tr = tracker(env)
    pos, quat = parts_pose_w(env)
    return bool(pairs_assembled(pos, quat, tr.pairs, tr.targets, tr.pos_threshold, tr.ori_bound)[0, 0].item())


# ------------------------------------------------------------------------------ state out
def ee_pose_b(env):
    robot = env.scene["robot"]
    body = robot.find_bodies("panda_hand")[0][0]
    pos_w, quat_w = robot.data.body_pos_w[:, body], robot.data.body_quat_w[:, body]
    off = torch.tensor(EE_OFFSET, device=env.device, dtype=pos_w.dtype).expand(env.num_envs, -1)
    pos_w, quat_w = combine_frame_transforms(pos_w, quat_w, off)
    return subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, pos_w, quat_w)


def ee_vel_b(env):
    robot = env.scene["robot"]
    body = robot.find_bodies("panda_hand")[0][0]
    from isaaclab.utils.math import quat_apply_inverse

    lin = quat_apply_inverse(robot.data.root_quat_w, robot.data.body_lin_vel_w[:, body])
    ang = quat_apply_inverse(robot.data.root_quat_w, robot.data.body_ang_vel_w[:, body])
    return lin, ang


def read_state(env, part_names, mesh_frame: str):
    """This env's state as :class:`ScriptedState` (tag frame, OBJ mesh frames) and as the
    FurnitureBench observation dict (xyzw) to record."""
    R_mesh = MESH_FRAMES[mesh_frame]
    origin = env.scene.env_origins[0].cpu().numpy()
    pos_b, quat_b = ee_pose_b(env)
    ee_pos = pos_b[0].cpu().numpy().astype(np.float64)
    ee_quat = quat_b[0].cpu().numpy().astype(np.float64)
    robot = env.scene["robot"]
    arm = robot.find_joints("panda_joint.*")[0]
    fingers = robot.find_joints("panda_finger_joint.*")[0]
    width = float(robot.data.joint_pos[0, fingers].sum().item())
    lin, ang = ee_vel_b(env)

    parts_tag, parts_flat = {}, []
    for name in part_names:
        part = env.scene[name]
        p = part.data.root_pos_w[0].cpu().numpy().astype(np.float64) - origin
        q = part.data.root_quat_w[0].cpu().numpy().astype(np.float64)
        T_robot_obj = sp.hom(sp.quat_to_mat(q) @ R_mesh, p)  # USD frame -> OBJ frame
        T_tag = sp.ROBOT_TO_APRIL @ T_robot_obj
        parts_tag[name] = T_tag
        q_tag = sp.mat_to_quat(T_tag[:3, :3])
        parts_flat.append(np.concatenate([T_tag[:3, 3], [q_tag[1], q_tag[2], q_tag[3], q_tag[0]]]))

    obstacles = {"obstacle_front": np.array(frames.OBSTACLE_FRONT_POS)}
    obstacles.update({k: np.array(v) for k, v in frames.OBSTACLE_SIDE_POS.items()})
    state = sp.ScriptedState(
        ee_pos, sp.quat_to_mat(ee_quat), width, parts_tag, obstacles, robot.data.joint_pos[0, arm].cpu().numpy().astype(np.float64)
    )
    obs = {
        "robot_state": {
            "ee_pos": ee_pos.astype(np.float32),
            "ee_quat": np.array([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]], dtype=np.float32),
            "ee_pos_vel": lin[0].cpu().numpy().astype(np.float32),
            "ee_ori_vel": ang[0].cpu().numpy().astype(np.float32),
            "gripper_width": np.array([width], dtype=np.float32),
            "joint_positions": robot.data.joint_pos[0, arm].cpu().numpy().astype(np.float32),
            "joint_velocities": robot.data.joint_vel[0, arm].cpu().numpy().astype(np.float32),
            "joint_torques": robot.data.applied_torque[0, arm].cpu().numpy().astype(np.float32),
        },
        "parts_poses": np.concatenate(parts_flat).astype(np.float32),
    }
    return state, obs


def env_action(env, a: sp.ScriptedAction, ee_quat_b: torch.Tensor) -> torch.Tensor:
    """Upstream's delta -> this env's pose_rel action (+ gripper, sign flipped)."""
    arm_cfg = env.cfg.actions.arm
    pos_scale, rot_scale = arm_scales(arm_cfg)
    dpos = torch.tensor(a.delta_pos, device=env.device, dtype=torch.float32).reshape(1, 3) / pos_scale
    q_d = torch.tensor(a.delta_quat, device=env.device, dtype=torch.float32).reshape(1, 4)
    q_base = quat_mul(quat_mul(ee_quat_b, q_d), quat_conjugate(ee_quat_b))
    drot = axis_angle_from_quat(q_base) / rot_scale
    grip = torch.tensor([[-a.gripper]], device=env.device)  # upstream +1 closes; here < 0 closes
    return torch.cat([dpos, drot, grip], dim=-1)


# ------------------------------------------------------------------------------ main
def main() -> None:
    cfg = OneLegReplayRelResiPCfg() if args.sim == "resip" else OneLegReplayRelCfg()
    cfg.scene.num_envs = 1
    if args.render_every is not None:
        if args.render_every < 1:
            raise SystemExit("--render_every must be >= 1")
        cfg.sim.render_interval = args.render_every
    cfg.furniture.log_resets = True
    if args.part_friction is not None:
        cfg.furniture.part_friction = args.part_friction
    if args.table_friction is not None:
        cfg.furniture.table_friction = args.table_friction
    cfg.apply_friction()
    if args.grip_stiffness is not None:
        cfg.furniture.gripper_stiffness = args.grip_stiffness
    if args.grip_force is not None:
        cfg.furniture.gripper_effort_limit = args.grip_force
    cfg.apply_gripper()
    if args.record:
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
    part_names = list(cfg.furniture.part_names)
    out_root = Path(args.out_dir).expanduser() / "one_leg"
    fails_dir = out_root / "fails"
    saved = 0
    attempt = 0
    # continue the fail<n> numbering across runs
    failures = max([int(f.stem[4:]) for f in fails_dir.glob("fail*.pkl") if f.stem[4:].isdigit()] + [0])
    keys = AbortKeys()
    while saved < args.num_demos:
        attempt += 1
        keys.new_attempt()
        env.reset()
        zero = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
        zero[0, -1] = 1.0  # gripper open
        for _ in range(5):
            env.step(zero)  # let the parts settle after the reset
        tmp_video = out_root / f".attempt{attempt}.mp4"
        recorder = Recorder(env, tmp_video) if args.record else None
        if args.policy == "servo":
            policy = sv.ServoOneLegPolicy(
                seed=args.seed + attempt - 1,
                noise=not args.no_noise,
                pos_step=args.pos_speed * env.step_dt,
                rot_step_deg=args.rot_speed * env.step_dt,
                lead_deg=args.lead,
                press=args.screw_press,
                insert_rate=args.insert_rate,
                insert_floor=args.insert_floor,
                insert_stop=args.insert_stop,
                insert_lift=args.insert_lift,
                release_dz=args.release_dz,
                wind_step_deg=args.wind_speed * env.step_dt,
                gain=args.servo_gain if args.servo_gain is not None else (1.0 if args.sim == "resip" else 5.0),
            )
        else:
            policy = sp.OneLegScriptedPolicy(seed=args.seed + attempt - 1, noise=not args.no_noise)
            policy.leg.yaw_step_deg = args.rot_speed * env.step_dt
            policy.leg.wind_step_deg = args.wind_speed * env.step_dt
            policy.leg.insert_lift = args.insert_lift
            policy.leg.insert_rate = args.insert_rate
        policy.leg.screw_press = args.screw_press
        obs_list, acts, rews, skills = [], [], [], []
        assembled_step = None
        last_state = ""
        t0 = time.time()
        aborted = False
        out_of_limits = False
        for t in range(args.timeout):
            if keys.quit:
                print("[collect] quit by key", flush=True)
                if recorder is not None:
                    recorder.close()
                    tmp_video.unlink(missing_ok=True)
                env.close()
                app.close()
                sys.exit(0)
            if keys.fail:
                aborted = True
                print(f"[collect] step {t:4d}: failed by key", flush=True)
                break
            if bool(part_out_of_limits(env)[0].item()):
                out_of_limits = True
                print(f"[collect] step {t:4d}: a part left the position limits -> failure", flush=True)
                break
            state, obs = read_state(env, part_names, cfg.furniture.mesh_frame)
            a = policy.act(state)
            if a.done:
                break
            if a.state != last_state:
                leg_z = float(env.scene["square_table_leg4"].data.root_pos_w[0, 2]) - float(env.scene.env_origins[0, 2])
                print(f"[collect] step {t:4d}: {a.state}  (leg z {leg_z:.4f}, ee z {state.ee_pos[2]:.4f})", flush=True)
                last_state = a.state
            obs_list.append(obs)
            # upstream's 8-D action, xyzw
            q = a.delta_quat
            acts.append(np.concatenate([a.delta_pos, [q[1], q[2], q[3], q[0]], [a.gripper]]).astype(np.float32))
            _, ee_quat_b = ee_pose_b(env)
            env.step(env_action(env, a, ee_quat_b))
            now_assembled = bool(assembled_now(env))  # the raw check, not the latch
            if a.assembled and assembled_step is None:  # the policy's own check fired
                assembled_step = t
                print(f"[collect] step {t:4d}: assembled", flush=True)
            reward = 1.0 if assembled_step == t else 0.0
            rews.append(reward)
            skills.append(float(a.skill_complete))
            if t % args.print_every == 0:
                print(
                    f"[collect] step {t:4d}  state {a.state:22s}  ee z {state.ee_pos[2]:.3f}  width {state.gripper_width:.3f}",
                    flush=True,
                )
            if recorder is not None:
                recorder.caption = f"scripted  step {t:4d}\n{a.state}\nassembled now {now_assembled}"
                recorder.frame()
        obs_list.append(read_state(env, part_names, cfg.furniture.mesh_frame)[1])  # the final observation
        # the verdict: the leg is assembled NOW, after the hand has lifted clear -- not "was at
        # some step". A leg knocked through the seat pose, or knocked out after seating, fails.
        for _ in range(10):
            env.step(zero)
        final_assembled = bool(assembled_now(env))
        success = assembled_step is not None and final_assembled and not (aborted or out_of_limits)
        if assembled_step is not None and not final_assembled:
            print("[collect] assembled during the episode but NOT at the end: counted as a failure", flush=True)
        print(
            f"[collect] demo {attempt}: {'success' if success else 'FAILED'} after {len(acts)} steps "
            f"({time.time() - t0:.0f} s), assembled at {assembled_step}",
            flush=True,
        )
        if recorder is not None:
            recorder.close()
        data = {
            "observations": obs_list,
            "actions": acts,
            "rewards": rews,
            "skills": skills,
            "success": success,
            "furniture": "one_leg",
            "error": aborted or out_of_limits,
            "error_description": "failed by key" if aborted else ("part out of position limits" if out_of_limits else ""),
        }
        if success:
            stamp = time.strftime("%Y-%m-%d-%H:%M:%S")
            demo_dir = out_root / stamp
            demo_dir.mkdir(parents=True, exist_ok=True)
            with open(demo_dir / f"{stamp}.pkl", "wb") as f:
                pickle.dump(data, f)
            if recorder is not None:
                tmp_video.replace(demo_dir / "recording.mp4")
            print(f"[collect] saved {demo_dir / (stamp + '.pkl')}" + (" + recording.mp4" if recorder else ""), flush=True)
            saved += 1
        else:
            failures += 1
            if args.save_failure:
                fails_dir.mkdir(parents=True, exist_ok=True)
                with open(fails_dir / f"fail{failures}.pkl", "wb") as f:
                    pickle.dump(data, f)
                if recorder is not None:
                    tmp_video.replace(fails_dir / f"fail{failures}.mp4")
                print(f"[collect] saved failure {fails_dir / f'fail{failures}.pkl'}" + (f" + fail{failures}.mp4" if recorder else ""), flush=True)
            elif recorder is not None:
                tmp_video.unlink(missing_ok=True)
        if keys.quit:
            print("[collect] quit by key", flush=True)
            break
        if attempt >= 3 * args.num_demos + 3:
            print("[collect] giving up: too many failures", flush=True)
            break
    env.close()


if __name__ == "__main__":
    main()
    app.close()
