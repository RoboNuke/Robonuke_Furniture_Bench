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
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import furniture_lab.tasks  # noqa: E402, F401
from furniture_lab.tasks.assembly.mdp.state import parts_pose_w, tracker  # noqa: E402
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
        rel = torch.tensor(
            rel_pose_targets([(0, 4)], cfg.furniture.mesh_frame)[0, args.hole],
            dtype=torch.float32, device=env.device,
        ).expand(env.num_envs, -1)
        top_pos, top_quat = top.data.root_pos_w.clone(), top.data.root_quat_w.clone()
        seated_pos = top_pos + quat_apply(top_quat, rel)
        leg.write_root_pose_to_sim(torch.cat([seated_pos, top_quat], dim=-1))
        leg.write_root_velocity_to_sim(torch.zeros(env.num_envs, 6, device=env.device))
        for _ in range(20):
            env.step(zero)
        t = tracker(env)
        drift = (leg.data.root_pos_w - seated_pos).norm(dim=-1)
        print(f"[smoke] teleport hole {args.hole}: assembled {t.all_assembled().tolist()}")
        print(f"[smoke] leg drift after 20 steps (m): {[f'{v:.4f}' for v in drift.tolist()]}")
        print("[smoke] assembled + small drift = frames and seated pose are right;")
        print("[smoke] big drift = the SDF contact rejected the pose (frame or thread phase);")
        print("[smoke] small drift but not assembled = the assembled check's numbers are off.")

    env.close()


if __name__ == "__main__":
    main()
    app.close()
