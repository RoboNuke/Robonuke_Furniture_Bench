"""The hybrid torque term's config and action layout (``mdp/hybrid.py``), on CPU with the
stubbed Isaac Lab modules. The torque law itself is robonuke_rl_core's (``envs/forge/control.py``,
tested there against Factory's); what the sim does with it is checked by
``scripts/hybrid_tracking.py``."""

import importlib
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import isaaclab_stub  # noqa: E402


@pytest.fixture
def hybrid():
    uninstall = isaaclab_stub.install()
    try:
        yield importlib.import_module("furniture_lab.tasks.assembly.mdp.hybrid")
    finally:
        uninstall()


def test_the_layout_is_the_abs_target_then_selection_then_force(hybrid):
    from robonuke_rl_core.envs.interface import ActionLayout

    layout = ActionLayout(hybrid.HybridTorqueCfg().controller_cfg())
    assert layout.action_dim == 16
    assert layout.pose_slice == slice(0, 10)
    assert layout.selection_indices == [10, 11, 12]
    assert layout.force_slice == slice(13, 16)
    z_only = hybrid.HybridTorqueCfg()
    z_only.force_axes = [0, 0, 1, 0, 0, 0]
    assert ActionLayout(z_only.controller_cfg()).action_dim == 12


def test_the_split_reads_position_from_a_negative_selection(hybrid):
    """A selection of -1 (the actor's position bit) is S = 0; +1 is force with its target."""
    from robonuke_rl_core.envs.interface import ActionInterface

    cfg = hybrid.HybridTorqueCfg()
    cfg.force_target_bounds = [10.0, 10.0, 10.0, 5.0, 5.0, 5.0]
    interface = ActionInterface(cfg.controller_cfg())
    action = torch.zeros(1, 16)
    action[0, 10:13] = torch.tensor([-1.0, 1.0, -1.0])
    action[0, 13:16] = torch.tensor([0.5, -0.5, 1.0])
    pose, selection, force_target, _, force_gains = interface.split(action)
    assert pose.shape == (1, 10)
    assert selection.tolist() == [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
    assert force_target[0, 0:3].tolist() == [5.0, -5.0, 10.0]
    assert force_gains[0, 0:3].tolist() == pytest.approx([0.1, 0.1, 0.1])


def test_the_decoupling_and_the_feedforward_are_on_by_default(hybrid):
    cfg = hybrid.HybridTorqueCfg()
    assert cfg.inertial_decoupling is True and cfg.force_feedforward is True


def test_damping_defaults_to_forges_rule(hybrid):
    cfg = hybrid.HybridTorqueCfg()
    k, d = cfg.gains("cpu")
    assert torch.allclose(d, 2.0 * torch.sqrt(k))
    cfg.damping = [1.0] * 6
    assert cfg.gains("cpu")[1].tolist() == [1.0] * 6


@pytest.mark.parametrize(
    "field, value",
    [("stiffness", [1.0] * 5), ("stiffness", [-1.0] + [1.0] * 5), ("damping", [1.0] * 7), ("kp_null", -1.0),
     ("force_target_bounds", [0.0] * 6), ("force_axes", [0] * 6)],
)
def test_validate_refuses(hybrid, field, value):
    cfg = hybrid.HybridTorqueCfg()
    setattr(cfg, field, value)
    with pytest.raises(ValueError):
        cfg.validate()


def test_a_force_axis_error_does_not_leak_through_the_inertia(hybrid):
    """The decoupled branch zeroes the force axes' motion before Lambda: a big pose error on a
    force-controlled axis must not reach the other axes through Lambda's off-diagonal terms (it
    did, and the arm leapt ~1 m when z went to force control while pressing)."""
    lam = torch.eye(6).unsqueeze(0) + 0.5 * torch.ones(1, 6, 6)  # strongly coupled
    selection = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    force = torch.tensor([[0.0, 0.0, -2.0, 0.0, 0.0, 0.0]])
    small = torch.zeros(1, 6)
    huge_z = small.clone()
    huge_z[0, 2] = 1.0e4
    a = hybrid.decoupled_wrench(selection, small, lam, force)
    b = hybrid.decoupled_wrench(selection, huge_z, lam, force)
    assert torch.equal(a, b)
    assert b[0, 2] == -2.0  # the force axis is the force branch's alone


def test_with_no_force_axis_it_is_plain_inertial_decoupling(hybrid):
    lam = torch.randn(2, 6, 6)
    motion = torch.randn(2, 6)
    out = hybrid.decoupled_wrench(torch.zeros(2, 6), motion, lam, torch.randn(2, 6))
    assert torch.allclose(out, (lam @ motion.unsqueeze(-1)).squeeze(-1))
