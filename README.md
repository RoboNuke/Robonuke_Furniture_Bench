# robonuke_furniture

FurnitureBench's square-table assembly (`one_leg`, `square_table`) as a **manager-based
Isaac Lab env**, as a project on [robonuke_rl_core](https://github.com/RoboNuke/robonuke_rl_core)
(made from `robonuke_project_template`: same layout, same entry scripts).

| piece | where |
| --- | --- |
| FurnitureBench's numbers: frames, part reset poses, assembled poses, tolerances | `furniture_lab/tasks/furniture/{frames,square_table}.py` (numpy, tested) |
| the assembled check, batched in torch, and the per-episode latch | `furniture_lab/tasks/furniture/assembly.py` (tested against upstream's logic) |
| Isaac Lab asset configs (parts, fixtures, the effort-controlled Panda) | `furniture_lab/tasks/furniture/assets.py` |
| the env cfg: scene, OSC action, observations, events, reward, terminations | `furniture_lab/tasks/assembly/assembly_env_cfg.py` |
| the env class (publishes `episode/success` for the package) | `furniture_lab/tasks/assembly/assembly_env.py` |
| the MDP terms | `furniture_lab/tasks/assembly/mdp/` |
| task registration: `Isaac-Furniture-OneLeg-v0`, `Isaac-Furniture-SquareTable-v0` | `furniture_lab/tasks/__init__.py` |
| asset catalog: part name -> `assets/` folder, mesh frame, metadata | `furniture_lab/tasks/furniture/catalog.py` |
| OBJ -> USD (SDF collision) and USD inspection | `tools/convert_assets.py`, `tools/check_usd.py` |

The docstring at the top of `assembly_env_cfg.py` lists, item by item, what is copied from
upstream and what is this port's choice.

## Task ids

| id | observation | action | for |
| --- | --- | --- | --- |
| `Isaac-Furniture-OneLeg-v0` | this port's 56-d (base frame, wxyz, + last action) | 7-d: OSC `pose_rel` (scaled) + binary gripper | RL |
| `Isaac-Furniture-OneLeg-FB-v0` | **FurnitureBench's own** 49-d `DEFAULT_STATE_OBS` (`ee_pos, ee_quat xyzw, ee_pos_vel, ee_ori_vel, gripper_width, parts_poses` in the AprilTag frame) | **FurnitureBench's own** 8-d `[dpos, dquat xyzw (EE frame), gripper]`, driving the same OSC + gripper (`mdp.FurnitureBenchAction`) | policies trained on the demo pkls (BC, residual) |
| `Isaac-Furniture-OneLeg-FBAbs-v0` | the FB observation with **every orientation as rot6d** (61-d: `ee_pos, ee_rot6d, ee_pos_vel, ee_ori_vel, gripper_width, parts 5 x [pos, rot6d]`) | 10-d **absolute** target `[goal_pos (base frame), goal_rot6d, gripper]` on the OSC in `pose_abs` mode (`mdp.FurnitureBenchAbsAction`) -- ResiP's `control_mode: pos` | diffusion policies (absolute targets regress better than deltas); the importer derives both columns from the pkls (`furniture_lab.demos.fb_abs_frames`) |
| `Isaac-Furniture-SquareTable-v0` | as the first | as the first | all four legs |

## Setup

```bash
pip install -e .   # editable: the asset catalog resolves assets/ from this checkout
```

1. The package, installed once into the Isaac Lab env (Isaac Lab 0.47.1 / Isaac Sim 5.1 /
   Python 3.11 -- **not** the Python 3.8 Isaac Gym env):

   ```bash
   pip install -e ~/robonuke_rl_core
   ```

2. Part USDs are in the repo (`assets/`, UWLab's files and layout, with their
   `metadata.yaml`). `furniture_lab/tasks/furniture/catalog.py` maps each FurnitureBench
   part name to its folder and records the file's mesh frame (UWLab's meshes are rotated
   relative to FurnitureBench's OBJs; the port corrects for it, and their metadata agrees
   with the corrected assembled pose to 0.35 mm -- `tests/test_catalog.py`). Nothing to set.

   `python tools/check_usd.py assets/SquareLeg/square_leg.usd` shows what a file holds
   (needs `pxr`: `pip install usd-core`). `tools/convert_assets.py` rebuilds USDs from
   FurnitureBench's OBJs with SDF collision if ever needed; such a file is in the OBJ frame,
   so its catalog row would say `frame="obj"`.

3. Run (from the repo root, with the Isaac Lab env active). First the smoke test, which needs
   no trained run and checks the asset frames by seating a leg:

   ```bash
   python scripts/smoke.py --headless --teleport
   ```

   then the package's entry points:

   ```bash
   python scripts/train.py --config configs/experiments/one_leg_ppo.yaml --headless
   python scripts/debug.py --run hur/furniture_bench/<run> --resets --hold_seconds 2
   python scripts/eval.py --run hur/furniture_bench/<run> --eval_config configs/eval/quick.yaml --headless
   ```

## What the env is

* **Scene**, in the robot-base frame (origin at the Panda's base, z up): FurnitureBench's
  table surface at z = -0.015, the three obstacle boxes, and the five square-table parts at
  upstream's from-scratch reset poses. The AprilTag-frame numbers upstream uses are kept
  verbatim and transformed once (`frames.tag_to_world`).
* **Physics**: upstream's PhysX settings; 1/120 s steps, 6 per action (upstream's sim runs
  the policy at 20 Hz, not the 10 Hz its config says -- see the cfg docstring).
* **Action** (7-D): EE delta pose through Isaac Lab's operational-space controller with
  upstream's gains (`kp = [90, 90, 90, 70, 60, 80]`, critically damped, nullspace to the
  reset joints), scaled 2.5 cm / 0.1 rad per unit action (this port's scale), plus a binary
  gripper (`< 0` closes).
* **Observation** (56-D): EE position, quaternion (wxyz), 6-D velocity, gripper width, the
  five parts' poses (7 each, robot-base frame), and the previous action.
* **Reward**: upstream's sparse +1 per newly assembled pair, once per episode. No shaping.
* **Done**: success (every pair assembled), `episode_length_s` (30 s by default), or a part
* **Success** is as strict as the demo collector's verdict: the leg must test assembled
  (FurnitureBench's tolerances) for `furniture.success_hold_steps` (10) consecutive steps
  with the fingers open (`success_release_width`, 5 cm) and the TCP more than
  `success_clearance` (8 cm) from the leg -- it stays seated after the robot lets go. The
  sparse reward pays at that moment. `success_hold_steps: 1, success_require_release: false`
  is upstream's single-step latch.
  fallen off the table (added here).
* **Reset**: upstream's `low` randomization (+-1.5 cm, +-15 deg, collision-free);
  `configs/experiments/one_leg_medium.yaml` sets `medium`.

`task.cfg.*` overrides reach every field of the env cfg, as in any project on the package.

### The end-effector wrench

`env.wrench` (`furniture_lab/tasks/assembly/wrench.py`) estimates the external wrench on
the EE from the joint torques, as the real robot does (`F = (J^T)^+ tau_ext`, base frame,
at FurnitureBench's EE point): the drives' applied torques, optionally minus the Coriolis
and `M q''` terms (the links carry no gravity here), projected through the EE Jacobian.
It runs every physics substep, with Gaussian noise on the joint torques and/or the wrench
and an EMA, like Forge's force sensor; it is zeroed on reset. `env.wrench.filtered` /
`.raw` are `(N, 6)`, force then torque; the observation term `mdp.ee_wrench` is there to
put in a group (none by default, so the task ids' observations are unchanged).

| `furniture.` field | default | meaning |
|---|---|---|
| `wrench_smoothing` | 0.25 | EMA factor per physics substep (1 = unfiltered) |
| `wrench_noise_joint_torque` | (0,) | std of the noise on each joint torque, N m: one value or seven |
| `wrench_noise` | (0,) | std of the noise on the projected wrench: one value, (force, torque), or six |
| `wrench_coriolis` | false | remove PhysX's Coriolis/centrifugal torques |
| `wrench_inertial` | false | remove `M q''` (finite-differenced joint acceleration) |
| `wrench_projection` | pinv | `pinv` (Moore-Penrose) or `dynamic` (`Lambda J M^-1`) |

`scripts/wrench_probe.py` runs a press-and-twist script with the parts off the table and
prints the estimate next to PhysX's joint reaction wrench on `panda_hand`, both as the
external wrench on the EE in the base frame; `--csv` saves it.

### Checking the FB interface against a demo

`scripts/parity_check.py --demo <pkl> --headless` (`--task Isaac-Furniture-OneLeg-FBAbs-v0` for the
absolute interface) puts the task's env into a
demo's recorded state and compares the env's 49-d observation with the pkl's term by term
(flagging quaternions that match only up to sign), then sends the pkl's 8-d actions
through the FB action term one step at a time (re-synced before each) and reports the EE
error against the demo's next step. Run it before training a policy on the demos.

## Replaying FurnitureBench demos

`furniture_lab/demos.py` reads a FurnitureBench `.pkl` into this port's frames (robot-base
frame, `wxyz`, the catalog's mesh frame). `scripts/replay_demo.py` resets the env to the demo's
first state and drives it two ways, recording an mp4 and a per-step CSV (EE and part errors
against the demo, the assembled flag):

```bash
# a demo from FurnitureBench's scripted policy, ported (furniture_lab/scripted_policy.py) and
# run in THIS env; writes FurnitureBench's pkl format and layout
python scripts/collect_scripted.py --headless --num_demos 1 --out_dir ~/fb_demos --record --save_failure
#   success -> ~/fb_demos/one_leg/<stamp>/{<stamp>.pkl, recording.mp4}; failure -> ~/fb_demos/one_leg/fails/fail<n>.{pkl,mp4}
#   success = FurnitureBench's assembled rule fired AND still true at the end, after the hand lifts;
#   a part leaving upstream's position limits ends the attempt as a failure. Not headless, viewport focused:
#   k = this rollout fails now, l = quit.
#   --part_friction / --table_friction override FurnitureSim's 0.15 / 0.10
#   --policy servo (default): closes the loop on the HELD object (furniture_lab/servo_policy.py) -- the top is
#   slid to a corner pose, the leg is carried upright, lowered to contact, released, regrasped square with the
#   wrist wound back, and screwed with the hand leading the leg's own yaw by <= --lead deg; free-hand phases
#   are upstream's states. --policy scripted: upstream's hand-only state machine with this port's regrasp.
#   --rot_speed (deg/s), --pos_speed (m/s), --lead, --screw_press, --insert_rate, --wind_speed tune the servo

python scripts/replay_demo.py --demo ~/fb_demos/one_leg/<d>/<d>.pkl --mode ee_track --headless --out ee_track.mp4
python scripts/replay_demo.py --demo ~/fb_demos/one_leg/<d>/<d>.pkl --mode actions  --headless --out actions.mp4
```

`ee_track` feeds the recorded EE poses as absolute OSC targets (`OneLegReplayAbsCfg`);
`actions` feeds the recorded deltas through the task's own `pose_rel` action
(`OneLegReplayRelCfg`), converted as a policy's output would be. Both cfgs drop the success
termination so a seated leg is not reset mid-replay.

## Not covered

* The package's Forge-family wrappers (hybrid force/position controller, fragile object,
  efficient reset, contact flags, task metrics) key on a `Forge` task name and do not apply
  here. A hybrid action space for this env would be a custom action term in `mdp/`.
* The `6d_rot_mat` orientation wrapper likewise; `ee_quat` is published as a quaternion.
* Camera observations, upstream's `high` randomization table, and the other furniture.
* **Nothing here has run in Isaac Sim yet.** The geometry and the assembled check are tested
  on CPU (`pytest`, 29 tests, incl. a wiring check of the env cfg against stubbed Isaac Lab
  modules); the first run will be the test of the rest.

## Tests

```bash
python -m pytest          # CPU: geometry, assembled check vs upstream, cfg wiring
```
