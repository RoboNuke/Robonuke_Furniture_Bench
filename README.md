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
| `Isaac-Furniture-OneLeg-FBAbs-ResiP-Hybrid-v0` | as `-FBAbs-v0` | 16-d: the abs target + selection 3 + force target 3, on joint torques (`mdp.HybridTorqueAbsAction`, ResiP's sim) | hybrid force/position residuals |
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
  fallen off the table (added here).
* **Success** (`furniture.success_mode`): `soft`, the default, is upstream's single-step latch
  (the first step the check passes, hand anywhere) -- what papers report. `strict` is the demo
  collector's verdict: the leg must test assembled
  (FurnitureBench's tolerances) for `furniture.success_hold_steps` (10) consecutive steps
  with the fingers open (`success_release_width`, 5 cm) and the TCP more than
  `success_clearance` (8 cm) from the leg -- it stays seated after the robot lets go. The
  sparse reward pays at that moment.
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
| `wrench_inertial` | false (true in `-FBAbs-ResiP-Hybrid-v0`) | remove `M q''` (finite-differenced joint acceleration): without it the estimate reads the arm's own acceleration as contact (~1 N in fast free-space moves) |
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

`scripts/replay_abs.py --demos <pkl or dir>` (default task `Isaac-Furniture-OneLeg-FBAbs-ResiP-v0`)
replays each successful demo's absolute rot6d actions (`fb_abs_frames`, what the BC policy
trains on) open loop from the demo's first state and reports whether the leg assembles, with
the EE error against the demo. The success rate is the ceiling a perfect clone reaches.
`--hold` (20) steps of the last action after the demo ends; `--max_demos`; `--print_every` (50).
`--zarr <ResiP processed .zarr>` replays ResiP's own demos instead (each episode's rotation targets first scaled back up by `demos.resip_unscale_rotation`: ResiP's `process_pickles.py` shrinks an episode's rotation deltas by one factor, ~12, so their norm over the episode is 0.35 rad; the factor printed is 20° / the stored plateau): the zarr has no joint positions, so the arm is driven to each demo's first EE pose by the env's IK controller (`--settle`, 30 steps), then the fingers and parts are set; the settle error is printed. ResiP's sim offsets the three obstacles per episode (±2 cm in x and y at low randomness) and records the front one, so the replay moves them to each demo's place (the obstacles are kinematic rigid objects for this; nothing else moves them).

### ResiP's sim

`Isaac-Furniture-OneLeg-ResiP-v0`, `-FB-ResiP-v0` and `-FBAbs-ResiP-v0` are the three interfaces in
ResiP's FurnitureRLSimEnv (`assembly_env_cfg.apply_resip`): a 10 Hz policy (1/120 s physics x 12),
the arm on differential IK (least squares) into joint PD 1000 / 200 instead of the OSC, friction
0.15 parts / 0.10 table, a 600 N/m finger drive capped at 13 N with a 0.2 m/s finger speed limit (`furniture.gripper_velocity_limit`; ResiP's 13 N; a stiffer drive pushed the fingers into the leg, and an emulation of ResiP's slower effort-mode fingers dropped the leg), the
FB action terms' gripper hysteresis (`furniture.gripper_margin` 0.019: the command flips only on a
clear sign change), the three obstacles moved together at every reset by a uniform (x, y)
offset of up to ±2 cm (`furniture.obstacle_offset` 0.02, ResiP's low randomness; 0 elsewhere,
i.e. the fixed poses), and 70 s episodes (their 700-step cap). The demo collector reads where
the obstacles are each episode, so the top is pushed into that episode's corner, and saves
their positions in the pkl (`obstacles_robot`); `replay_abs.py` puts them back (a pkl without
them was recorded with the obstacles at their fixed poses, and replays there). Demos for them:

```bash
python scripts/collect_scripted.py --sim resip --out_dir ~/Robonuke_Furniture_Bench/demos_resip --num_demos 50
```

`--servo_gain` defaults to 1 there (the joint PD closes a command in a step; the OSC's 20 % per
step is why the port's default is 5).

### Hybrid force/position control on joint torques

`Isaac-Furniture-OneLeg-FBAbs-ResiP-Hybrid-v0` (`OneLegFBAbsResiPHybridEnvCfg`) is ResiP's sim and
interface with the arm on joint torques: the arm's drives get no stiffness or damping, and the
action term `mdp.HybridTorqueAbsAction` (`mdp/hybrid.py`) computes, every physics substep,
robonuke_rl_core's hybrid law (`envs/forge/control.py`, what its Forge wrapper runs):

    tau = J^T [ (I - S) Lambda (K e - D v) + S (f_d + K_f (f_d - f)) ] + nullspace

(`inertial_decoupling` and `force_feedforward` off: Forge's law, `(I - S)(K e - D v) + S K_f (f_d - f)`).
The action is `[goal_pos 3, goal_rot6d 6, gripper 1 | selection n | force target n]` (16 with the
default 3 force axes): a selection > 0 hands that axis to force control, a force target in
[-1, 1] is scaled by `force_target_bounds`. `f` is the force the EE applies (minus the joint-torque
estimate of the external wrench, `env.wrench`), `f_d` in the same sense (negative z presses down).
A reset puts every axis back on position control. The fields are the env cfg's `hybrid`
(`task.cfg.hybrid.*` from a project config):

| `hybrid.` field | default | meaning |
|---|---|---|
| `force_axes` | `[1, 1, 1, 0, 0, 0]` | axes the force branch may take; the selection and force blocks are as wide as its sum |
| `force_target_bounds` | `[50, 50, 50, 5, 5, 5]` | the force action in [-1, 1] scales to +/- these, N and N m |
| `default_force_gains` | `[0.1, 0.1, 0.1, 0.01, 0.01, 0.01]` | `K_f` per axis (robonuke_rl_core's default); on x, y, z stable to ~0.3 (below) |
| `stiffness` | `[300, 300, 300, 500, 500, 500]` here (class default Forge's `[565 x3, 28 x3]`) | `K` per axis |
| `damping` | `[60.62 x3, 100.62 x3]` here (class default null: `2 sqrt(K)`) | `D` per axis |
| `kp_null` / `kd_null` | `10` / `6.3246` | nullspace pull toward the reset joints (Forge's) |
| `inertial_decoupling` | `true` | scale the motion branch by the operational-space inertia `Lambda` (Isaac Lab's OSC order: the force axes' motion zeroed before and after `Lambda`); `K`, `D` are then accelerations per error (1/s^2, 1/s); `false`: Forge's motion branch |
| `force_feedforward` | `true` | the force branch commands `f_d + K_f (f_d - f)`: a still EE settles at `f_d` for any `K_f`; `false`: Forge's `K_f (f_d - f)`, which settles at `K_f f_d / (1 + K_f)` |

**The gains track like ResiP's controller.** `scripts/hybrid_tracking.py` plays one sequence of
free-space EE targets (2-3 cm and 15 deg steps, sweeps, 10 Hz policy-like waypoints) through the
`-FBAbs-ResiP-v0` IK + joint PD and through this term (every axis on position), one gain set per
env, and ranks them by EE path distance. Measured:

* Forge's law as is (no decoupling) cannot: at Forge's 565 / 28 the path is 11.4 mm / 7.5 deg RMS
  off, and stiffer, better-damped gains chatter at the substep rate (the light wrist directions
  go unstable at 1/120 s); the best of 64 was 9.6 mm / 9.7 deg RMS.
* With `inertial_decoupling`, K 300 / 500 and D 3.5 / 4.5 sqrt(K) (a dominant time constant near the
  reference's 0.2 s): 0.24 mm RMS (1.4 mm max) and 0.06 deg RMS (0.3 deg max), the roll and pitch
  steps aside. On those the reference itself swings the EE ~19 mm off its fixed position target
  (IK + joint PD), where this holds it within 2.5 mm; the whole sequence is 4.0 mm RMS for that
  reason alone.

```bash
python scripts/hybrid_tracking.py --mode reference --out ref.pt --headless
python scripts/hybrid_tracking.py --mode hybrid --reference ref.pt --out search.pt --headless \
    --kpos 300 --krot 500 --dcoef 3.5 --dcoef_rot 4.5
python scripts/hybrid_tracking.py --mode press --press_kf 0.1,0.2 --press_force 0,1,2,5,8,10 --out press.pt --headless
python scripts/hybrid_tracking.py --mode press --press_start above --press_gap 0.02 --press_kf 0.1,0.2 --press_force 1,2,5,8,10 --out press.pt --headless
```

`--forge_law` runs either mode with both fields off.

**The force branch.** `--mode press` descends onto what is under the EE on position control,
then hands z to force control (in contact, or `--press_start above` from `--press_gap` above it),
one env per (`K_f`, target), and reads the force the EE applies by the estimate and by PhysX's
joint reaction wrench (the truth). Without the feedforward (Forge's law) a -5 N target settled
exactly at `K_f f_d / (1 + K_f)`: -0.45, -2.5, -3.75 N at `K_f` 0.1, 1, 3, so the default
pressed with ~9 % of its target. With it, and the inertial estimate, the true force over the last
2 s of 10 s on force control (N; targets press down):

| `K_f` | targets 0-10 N, in contact | 1-10 N from 0.5 cm and 2 cm above |
|---|---|---|
| 0, 0.1, 0.2, 0.3 | every target within 0.01 N | within 0.01 N |
| 0.5 | -- | rings at 2 N (84 N peak-to-peak) |
| 1 | within 0.02 N | rings at 5 and 10 N |
| >= 1.5 | rings at most targets | -- |

The settled ripple (peak-to-peak) is 0.04-0.55 N either way, steady over the 10 s: it varies from
run to run, not with where force control began (the same in-contact 5 N case gave 0.45 N in one
run and 0.05 N in another).

With the inertial estimate `K_f` also lowers the EE's apparent mass on the force axis by
`1 + K_f`, which is what limits it; the default 0.1 keeps a 5x margin. **The impact is not
damped**: a force axis has no motion damping, so switching to force above the surface hits it
at whatever speed the target built up -- transient peaks of 15-140 N (0.5-2 cm gap) before
settling, ~1.3-1.7x the target when switched in contact. At a 0 N target the EE drifts up a
few mm over 10 s; with nothing to press on, a force axis accelerates the arm away.

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
  here. The hybrid action space is this repo's own term instead (`-FBAbs-ResiP-Hybrid-v0`, above);
  it imports robonuke_rl_core's controller math and action split, so the package must be
  installed.
* The `6d_rot_mat` orientation wrapper likewise; `ee_quat` is published as a quaternion.
* Camera observations, upstream's `high` randomization table, and the other furniture.
* **Nothing here has run in Isaac Sim yet.** The geometry and the assembled check are tested
  on CPU (`pytest`, 29 tests, incl. a wiring check of the env cfg against stubbed Isaac Lab
  modules); the first run will be the test of the rest.

## Tests

```bash
python -m pytest          # CPU: geometry, assembled check vs upstream, cfg wiring
```
