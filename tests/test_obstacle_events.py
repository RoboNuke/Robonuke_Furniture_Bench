"""randomize_obstacles: one offset per env for all three obstacles, in [-r, r] in x and y."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import isaaclab_stub  # noqa: E402


@pytest.fixture
def events():
    uninstall = isaaclab_stub.install()
    try:
        yield importlib.import_module("furniture_lab.tasks.assembly.mdp.events")
    finally:
        uninstall()


class _Obstacle:
    def __init__(self, n, pos):
        root = torch.zeros(n, 13)
        root[:, :3] = torch.tensor(pos)
        root[:, 3] = 1.0
        self.data = SimpleNamespace(default_root_state=root)
        self.written = None

    def write_root_pose_to_sim(self, pose, env_ids=None):
        self.written = (pose.clone(), env_ids)


def _env(events, n, r):
    scene = {name: _Obstacle(n, (0.5 + i, 0.1 * i, 0.03)) for i, name in enumerate(events.OBSTACLE_NAMES)}
    origins = torch.arange(n, dtype=torch.float32)[:, None].repeat(1, 3) * 2.5

    class Scene(dict):
        env_origins = origins

    s = Scene(scene)
    return SimpleNamespace(scene=s, num_envs=n, device="cpu", cfg=SimpleNamespace(furniture=SimpleNamespace(obstacle_offset=r)))


def test_zero_offset_leaves_obstacles_alone(events):
    env = _env(events, 4, 0.0)
    events.randomize_obstacles(env, torch.arange(4))
    assert all(env.scene[k].written is None for k in events.OBSTACLE_NAMES)


def test_offset_is_shared_bounded_and_planar(events):
    torch.manual_seed(0)
    n, r = 64, 0.02
    env = _env(events, n, r)
    ids = torch.arange(n)
    events.randomize_obstacles(env, ids)
    offs = []
    for k in events.OBSTACLE_NAMES:
        pose, got_ids = env.scene[k].written
        assert torch.equal(got_ids, ids)
        default = env.scene[k].data.default_root_state
        offs.append(pose[:, :3] - default[:, :3] - env.scene.env_origins)
        assert torch.equal(pose[:, 3:7], default[:, 3:7])
    for o in offs[1:]:
        torch.testing.assert_close(o, offs[0])  # all three move together
    assert offs[0][:, :2].abs().max() <= r and offs[0][:, 2].abs().max() == 0
    assert offs[0][:, :2].abs().max() > 0.5 * r  # actually spread
