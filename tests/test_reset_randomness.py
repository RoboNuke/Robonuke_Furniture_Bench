"""furniture.randomness: ``port`` keeps the port's uniform layout; FurnitureBench's ``low`` / ``med``
draw upstream's layout exactly (tag-frame offsets and yaws, collision-free with the margin, inside
the position limits) and ``high`` puts the parts at a recorded layout; ResiP's ``kick_low`` /
``kick_med`` / ``kick_high`` put the parts at their nominal poses, kick each once (world-frame xy
force, z torque, for their one 1/60 s simulate()), add the arm-joint noise and the level's
obstacle offset. Fake env, no sim."""

import importlib
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import isaaclab_stub  # noqa: E402

from furniture_lab.tasks.furniture import square_table as st  # noqa: E402

N_ARM = 7


def _quat_mul(a, b):  # wxyz
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack((w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2), dim=-1)


def _quat_from_euler_xyz(r, p, y):  # yaw only here
    return torch.stack((torch.cos(y / 2), torch.zeros_like(y), torch.zeros_like(y), torch.sin(y / 2)), dim=-1)


@pytest.fixture
def events(monkeypatch):
    uninstall = isaaclab_stub.install()
    try:
        mod = importlib.import_module("furniture_lab.tasks.assembly.mdp.events")
        monkeypatch.setattr(mod, "quat_mul", _quat_mul)
        monkeypatch.setattr(mod, "quat_from_euler_xyz", _quat_from_euler_xyz)
        yield mod
    finally:
        uninstall()


class _Rigid:
    def __init__(self, n, pos):
        root = torch.zeros(n, 13)
        root[:, :3] = torch.tensor(pos)
        root[:, 3] = 1.0
        self.data = SimpleNamespace(default_root_state=root)
        self.poses, self.wrenches = [], []
        self.root_physx_view = self  # the kick writes the part's PhysX view directly

    def write_root_pose_to_sim(self, pose, env_ids=None):
        self.poses.append((pose.clone(), env_ids))

    def write_root_velocity_to_sim(self, vel, env_ids=None):
        pass

    def apply_forces_and_torques_at_position(self, force_data, torque_data, position_data, indices, is_global):
        assert position_data is None and is_global and indices.dtype == torch.long and force_data.is_contiguous()
        self.wrenches.append((force_data.clone(), torque_data.clone()))


class _Robot:
    def __init__(self, n):
        self.data = SimpleNamespace(default_joint_pos=torch.linspace(-1.0, 1.0, N_ARM + 2).repeat(n, 1))
        self.state = self.target = None

    def find_joints(self, expr):
        assert expr == "panda_joint.*"
        return list(range(N_ARM)), [f"panda_joint{i + 1}" for i in range(N_ARM)]

    def write_joint_state_to_sim(self, pos, vel, joint_ids=None, env_ids=None):
        self.state = (pos.clone(), vel.clone(), list(joint_ids), env_ids)

    def set_joint_position_target(self, target, joint_ids=None, env_ids=None):
        self.target = (target.clone(), list(joint_ids), env_ids)


def _env(events, n, randomness, obstacle_offset=0.0, physics_dt=1.0 / 120.0):
    scene = {name: _Rigid(n, (0.3 * i, 0.2, 0.0)) for i, name in enumerate(st.PART_NAMES)}  # apart: no collisions
    scene.update({name: _Rigid(n, (0.5, 0.1 * i, 0.03)) for i, name in enumerate(events.OBSTACLE_NAMES)})
    scene["robot"] = _Robot(n)

    class Scene(dict):
        env_origins = torch.zeros(n, 3)

    furniture = SimpleNamespace(part_names=st.PART_NAMES, randomness=randomness, obstacle_offset=obstacle_offset)
    return SimpleNamespace(scene=Scene(scene), num_envs=n, device="cpu", physics_dt=physics_dt,
                           cfg=SimpleNamespace(furniture=furniture), part_kick=None)


def test_the_levels_are_resips(events):
    assert events.resip_level(SimpleNamespace(randomness="port")) is None
    assert events.resip_level(SimpleNamespace(randomness="low")) is None  # FurnitureBench's, no kick
    assert events.resip_level(SimpleNamespace(randomness="kick_med")) is events.RESIP_RANDOMNESS["med"]
    assert events.randomness(SimpleNamespace()) == "port"  # a cfg without the field: port
    assert events.RANDOMNESS_LEVELS == ("port", "low", "med", "high", "kick_low", "kick_med", "kick_high")
    assert events.RESIP_RANDOMNESS["low"] == {"force": 0.2, "torque": 0.007, "obstacle_offset": 0.02, "joint_noise_deg": 5.0}
    assert events.RESIP_RANDOMNESS["med"] == {"force": 0.5, "torque": 0.01, "obstacle_offset": 0.04, "joint_noise_deg": 10.0}
    assert events.RESIP_RANDOMNESS["high"] == {"force": 0.75, "torque": 0.015, "obstacle_offset": 0.06, "joint_noise_deg": 13.0}
    assert events.RESIP_KICK_FORCE_MUL == (25.0, 1.0, 1.0, 1.0, 1.0)
    assert events.RESIP_KICK_TORQUE_MUL == (70.0, 1.0, 1.0, 1.0, 1.0)
    for bad in ("medium", "resip_low"):
        with pytest.raises(ValueError, match="furniture.randomness"):
            events.resip_level(SimpleNamespace(randomness=bad))


def test_port_keeps_the_uniform_layout_and_no_kick(events):
    torch.manual_seed(0)
    env = _env(events, 16, "port")
    ids = torch.arange(16)
    events.randomize_parts_pose(env, ids)
    for name in st.PART_NAMES:
        (pose, got), = env.scene[name].poses
        off = pose[:, :2] - env.scene[name].data.default_root_state[:, :2]
        assert off.abs().max() <= 0.015 + 1e-6 and off.abs().max() > 0.0
    assert env.part_kick is None and env.scene["robot"].state is None


@pytest.mark.parametrize("level", ["low", "med", "high"])
def test_a_kick_level_kicks_the_parts_and_shakes_the_arm(events, level):
    torch.manual_seed(1)
    n = 512
    env = _env(events, n, f"kick_{level}")
    ids = torch.arange(n)
    events.randomize_parts_pose(env, ids)
    lv = events.RESIP_RANDOMNESS[level]
    # the parts are not moved: they stay where reset_scene_to_default put them
    assert all(env.scene[name].poses == [] for name in st.PART_NAMES)
    # the arm: every arm joint within +-noise of its reset value, at rest, its target there too
    robot = env.scene["robot"]
    pos, vel, joints, got = robot.state
    a = math.radians(lv["joint_noise_deg"])
    d = pos - robot.data.default_joint_pos[:, :N_ARM]
    assert joints == list(range(N_ARM)) and torch.equal(got, ids)
    assert d.abs().max() <= a + 1e-6 and d.abs().max() > 0.9 * a and torch.all(vel == 0)
    assert torch.equal(robot.target[0], pos) and robot.target[1] == joints
    # the kick: xy force <= force x mul, z torque <= torque x mul, world frame
    kick = env.part_kick
    assert kick.steps == 2 and torch.all(kick.left == 2)
    f, t = kick.force, kick.torque
    f_max = lv["force"] * torch.tensor(events.RESIP_KICK_FORCE_MUL)
    t_max = lv["torque"] * torch.tensor(events.RESIP_KICK_TORQUE_MUL)
    assert torch.all(f[..., 2] == 0) and torch.all(t[..., :2] == 0)
    mag = f[..., :2].norm(dim=-1)
    assert torch.all(mag <= f_max + 1e-6) and torch.all(mag.max(0).values > 0.95 * f_max)
    assert torch.all(t[..., 2].abs() <= t_max + 1e-6) and torch.all(t[..., 2].abs().max(0).values > 0.95 * t_max)
    angle = torch.atan2(f[..., 1], f[..., 0])
    assert angle.min() < -3.0 and angle.max() > 3.0  # every direction


def test_the_kick_lasts_their_one_simulate(events):
    torch.manual_seed(2)
    env = _env(events, 4, "kick_low")
    events.randomize_parts_pose(env, torch.arange(4))
    top = env.scene["square_table_top"]
    for _ in range(5):
        env.part_kick.apply()
        env.part_kick.write()
    # two physics steps of 1/120 s with the force on (= their 1/60 s), one writing the zeros, then no PhysX write
    assert len(top.wrenches) == 3
    for forces, torques in top.wrenches[:2]:
        assert forces.shape == (4, 3)
        assert torch.equal(forces, env.part_kick.force[:, 0]) and torch.equal(torques, env.part_kick.torque[:, 0])
    assert torch.all(top.wrenches[2][0] == 0) and torch.all(top.wrenches[2][1] == 0)
    assert torch.all(env.part_kick.left == 0) and env.part_kick.writes_due == 0


def test_a_reset_mid_kick_only_kicks_its_own_envs(events):
    torch.manual_seed(3)
    env = _env(events, 4, "kick_low")
    events.randomize_parts_pose(env, torch.arange(4))
    env.part_kick.apply()
    env.part_kick.apply()
    events.randomize_parts_pose(env, torch.tensor([2]))  # env 2 resets again
    env.part_kick.apply()
    env.part_kick.write()
    forces = env.scene["square_table_leg4"].wrenches[-1][0]
    assert torch.all(forces[[0, 1, 3]] == 0) and torch.any(forces[2] != 0)
    assert env.part_kick.writes_due == 2  # the re-arm restarted the countdown


def test_the_kick_needs_a_physics_step_that_divides_it(events):
    env = _env(events, 2, "kick_low", physics_dt=1.0 / 50.0)
    with pytest.raises(ValueError, match="kick duration"):
        events.randomize_parts_pose(env, torch.arange(2))


def test_a_level_sets_the_obstacle_offset(events):
    torch.manual_seed(4)
    n = 256
    env = _env(events, n, "kick_med", obstacle_offset=0.0)  # the field is ignored under a level
    events.randomize_obstacles(env, torch.arange(n))
    (pose, _), = env.scene["obstacle_front"].poses
    off = pose[:, :2] - env.scene["obstacle_front"].data.default_root_state[:, :2]
    assert off.abs().max() <= 0.04 + 1e-6 and off.abs().max() > 0.035
    env = _env(events, n, "port", obstacle_offset=0.0)
    events.randomize_obstacles(env, torch.arange(n))
    assert env.scene["obstacle_front"].poses == []


# ------------------------------------------------------------- FurnitureBench's own levels
def _fb_env(events, n, randomness, high_layout=0, obstacle_offset=0.05):
    """The fake env with every part's default pose at its real nominal reset pose."""
    env = _env(events, n, randomness, obstacle_offset=obstacle_offset)
    for p in st.PARTS:
        pos, quat = st.reset_pose_world(p)
        root = env.scene[p.name].data.default_root_state
        root[:, :3] = torch.tensor(pos)
        root[:, 3:7] = torch.tensor(quat)
    env.cfg.furniture.mesh_frame = st.mesh_frame()
    env.cfg.furniture.high_layout = high_layout
    return env


def _tag_draw(env):
    """The written world poses -> upstream's draw: each part's tag-frame (x, y) and yaw about the tag z."""
    from furniture_lab.tasks.furniture.frames import TAG_ROT_W

    R = torch.tensor(TAG_ROT_W, dtype=torch.float32)
    xy, yaw = [], []
    for p in st.PARTS:
        (pose, _), = env.scene[p.name].poses
        nominal = env.scene[p.name].data.default_root_state
        d_tag = (pose[:, :3] - nominal[:, :3]) @ R  # R^T d, row-wise
        assert d_tag[:, 2].abs().max() < 1e-6  # planar
        xy.append(torch.tensor(p.reset_pos_tag[:2]) + d_tag[:, :2])
        conj = nominal[:, 3:7] * torch.tensor([1.0, -1.0, -1.0, -1.0])
        rel = _quat_mul(pose[:, 3:7], conj)  # a rotation about the world z
        assert rel[:, 1:3].abs().max() < 1e-5
        yaw.append(2 * torch.atan2(rel[:, 3], rel[:, 0]) * float(TAG_ROT_W[2][2]))  # tag z = -world z
    return torch.stack(xy, 1), torch.stack(yaw, 1)  # (n, P, 2), (n, P)


def _upstream_collides(p1, y1, l1, p2, y2, l2, margin=0.01):
    """furniture_bench Part.is_collision, line for line (rotated corners -> axis-aligned bounds)."""
    def bounds(p, yaw, ln):
        c, s = math.cos(yaw), math.sin(yaw)
        corners = [(-ln[0] / 2, ln[1] / 2), (ln[0] / 2, ln[1] / 2), (-ln[0] / 2, -ln[1] / 2), (ln[0] / 2, -ln[1] / 2)]
        xs = [c * x - s * y for x, y in corners]
        ys = [s * x + c * y for x, y in corners]
        return p[0] + min(xs), p[0] + max(xs), p[1] + min(ys), p[1] + max(ys)

    a, b = bounds(p1, y1, l1), bounds(p2, y2, l2)
    if a[0] > b[1] + margin or a[1] < b[0] - margin:
        return False
    if a[2] > b[3] + margin or a[3] < b[2] - margin:
        return False
    return True


UPSTREAM_LEN = {"square_table_top": (0.1625, 0.1625), "leg": (0.03 + 0.02, 0.0875)}


def test_the_furniturebench_constants_are_upstreams():
    assert st.RANDOMIZE_INIT_POSE == {"low": (0.015, 15.0), "med": (0.05, 45.0)}
    assert st.PARTS_POS_LIM == ((-0.21, 0.21), (0.07, 0.37)) and st.COLLISION_MARGIN == 0.01
    assert st.RANDOMIZE_MAX_TRIAL == 300000
    for p in st.PARTS:
        assert st.RESET_LEN[p.name] == UPSTREAM_LEN.get(p.name, UPSTREAM_LEN["leg"])
        assert len(st.HIGH_RAND_RESET[p.name]) == 3


@pytest.mark.parametrize("level", ["low", "med"])
def test_a_furniturebench_level_draws_upstreams_layout(events, level):
    torch.manual_seed(5)
    n = 256 if level == "low" else 64
    env = _fb_env(events, n, level)
    events.randomize_parts_pose(env, torch.arange(n))
    events.randomize_obstacles(env, torch.arange(n))
    pos_range, rot = st.RANDOMIZE_INIT_POSE[level]
    xy, yaw = _tag_draw(env)
    nominal = torch.tensor([p.reset_pos_tag[:2] for p in st.PARTS])
    off = xy - nominal
    assert off.abs().max() <= pos_range + 1e-5 and off.abs().max() > 0.9 * pos_range
    assert yaw.abs().max() <= math.radians(rot) + 1e-5 and yaw.abs().max() > 0.9 * math.radians(rot)
    (x_lo, x_hi), (y_lo, y_hi) = st.PARTS_POS_LIM
    assert torch.all(xy[..., 0] >= x_lo - 1e-6) and torch.all(xy[..., 0] <= x_hi + 1e-6)
    assert torch.all(xy[..., 1] >= y_lo - 1e-6) and torch.all(xy[..., 1] <= y_hi + 1e-6)
    # the legs sit on the y limit: they only move away from the robot (tag +y)
    assert torch.all(off[:, 1:, 1] >= -1e-6) and off[:, 1:, 1].max() > 0.9 * pos_range
    lens = [UPSTREAM_LEN.get(p.name, UPSTREAM_LEN["leg"]) for p in st.PARTS]
    for e in range(n):
        for i in range(5):
            for j in range(i + 1, 5):
                assert not _upstream_collides(xy[e, i].tolist(), float(yaw[e, i]), lens[i], xy[e, j].tolist(), float(yaw[e, j]), lens[j])
    # as in FurnitureSimEnv: the obstacles fixed (obstacle_offset unused), the arm at its reset joints, no kick
    assert all(env.scene[k].poses == [] for k in events.OBSTACLE_NAMES)
    assert env.scene["robot"].state is None and env.part_kick is None


def test_furniturebench_high_is_a_recorded_layout(events):
    env = _fb_env(events, 8, "high", high_layout=1)
    events.randomize_parts_pose(env, torch.arange(8))
    for p in st.PARTS:
        (pose, _), = env.scene[p.name].poses
        pos, quat = st.tag_pose_world(*st.HIGH_RAND_RESET[p.name][1], st.mesh_frame())
        q = torch.tensor(quat) / torch.tensor(quat).norm()
        assert torch.allclose(pose[:, :3], torch.tensor(pos, dtype=torch.float32).expand(8, 3), atol=1e-6)
        assert torch.allclose(pose[:, 3:7], q.float().expand(8, 4), atol=1e-6)
    torch.manual_seed(6)
    env = _fb_env(events, 64, "high", high_layout=-1)
    events.randomize_parts_pose(env, torch.arange(64))
    (pose, _), = env.scene["square_table_top"].poses
    assert len({tuple(r) for r in pose[:, :3].round(decimals=5).tolist()}) == 3  # all three layouts drawn
    with pytest.raises(ValueError, match="high_layout"):
        events.randomize_parts_pose(_fb_env(events, 2, "high", high_layout=3), torch.arange(2))


def test_no_valid_furniturebench_draw_raises(events, monkeypatch):
    monkeypatch.setattr(st, "RANDOMIZE_MAX_TRIAL", 256)
    env = _fb_env(events, 4, "low")
    with pytest.raises(RuntimeError, match="no valid draw"):
        events.fb_layout(env, torch.arange(4), 0.5, 0.0)  # +-0.5 m: never inside the limits


def test_the_effort_gripper_sends_its_command_as_joint_effort(monkeypatch):
    uninstall = isaaclab_stub.install()
    try:
        g = importlib.import_module("furniture_lab.tasks.assembly.mdp.gripper")
        sent = {}
        term = g.BinaryJointEffortAction.__new__(g.BinaryJointEffortAction)
        term._asset = SimpleNamespace(set_joint_effort_target=lambda target, joint_ids=None: sent.update(t=target, ids=joint_ids))
        term._processed_actions, term._joint_ids, term.device = torch.tensor([[13.0, 13.0], [-13.0, -13.0]]), [7, 8], "cpu"
        term.apply_actions()
        # the ids reach the articulation as a device tensor (a list would be copied to the device every substep)
        assert torch.equal(sent["t"], term._processed_actions) and torch.equal(sent["ids"], torch.tensor([7, 8]))
        assert sent["ids"] is term._joint_index()  # built once
        # the travel cap: every finger joint's limits to [0, upper]
        written = {}
        robot = SimpleNamespace(data=SimpleNamespace(joint_pos_limits=torch.tensor([[[0.0, 0.04], [0.0, 0.04]]] * 3)),
                                write_joint_position_limit_to_sim=lambda lim, joint_ids=None: written.update(lim=lim, ids=joint_ids))
        g.set_finger_limits(SimpleNamespace(scene={"robot": robot}), None, 0.0325,
                            SimpleNamespace(name="robot", joint_ids=[0, 1]))
        assert torch.allclose(written["lim"], torch.tensor([[[0.0, 0.0325], [0.0, 0.0325]]] * 3)) and written["ids"] == [0, 1]
        with pytest.raises(ValueError, match="unresolved"):  # the event manager did not resolve it: all joints
            g.set_finger_limits(SimpleNamespace(scene={"robot": robot}), None, 0.0325, SimpleNamespace(name="robot", joint_ids=slice(None)))
    finally:
        uninstall()
