"""The joint-torque wrench estimator's math, with no sim: the projection recovers a wrench
that produced the torques, the filter is Forge's EMA, the noise fields parse."""

import pytest
import torch

from furniture_lab.tasks.assembly.wrench import _std, ema, project_torques


def random_arm(n: int = 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    jac = torch.randn((n, 6, 7), generator=g)
    a = torch.randn((n, 7, 7), generator=g)
    mass = a @ a.transpose(1, 2) + 7 * torch.eye(7)  # SPD
    wrench = torch.randn((n, 6), generator=g)
    return jac, mass, wrench


@pytest.mark.parametrize("dynamic", [False, True])
def test_the_projection_recovers_the_wrench_behind_the_torques(dynamic):
    jac, mass, wrench = random_arm()
    tau_ext = (jac.transpose(1, 2) @ wrench.unsqueeze(-1)).squeeze(-1)  # J^T F
    out = project_torques(jac, tau_ext, mass if dynamic else None)
    assert out.shape == (4, 6)
    assert torch.allclose(out, wrench, atol=1e-4)


def test_the_sign_convention_pressing_down_reads_an_upward_force():
    # a one-joint "arm" whose joint torque pushes the EE down: tau_J = J^T F_robot_on_env,
    # the external wrench is the reaction, +z
    jac = torch.zeros((1, 6, 7))
    jac[0, 2, 0] = 1.0  # joint 0 moves the EE along z
    tau_j = torch.tensor([[-3.0, 0, 0, 0, 0, 0, 0]])  # pushing down with 3 N
    assert project_torques(jac, -tau_j)[0, 2].item() == pytest.approx(3.0)


def test_ema_is_forges_and_reaches_a_constant():
    state = torch.zeros(2)
    for _ in range(40):
        state = ema(state, torch.tensor([4.0, -2.0]), 0.25)
    assert torch.allclose(state, torch.tensor([4.0, -2.0]), atol=1e-3)
    assert ema(torch.zeros(1), torch.ones(1), 0.25).item() == pytest.approx(0.25)


def test_noise_fields_parse_one_two_or_full_width():
    assert _std((0.1,), 7, "cpu", "joint torque").tolist() == [pytest.approx(0.1)] * 7
    assert _std(0.2, 6, "cpu", "wrench").tolist() == [pytest.approx(0.2)] * 6
    assert _std((1.0, 0.1), 6, "cpu", "wrench").tolist() == pytest.approx([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])
    assert _std(tuple(range(7)), 7, "cpu", "joint torque").tolist() == pytest.approx(list(range(7)))
    with pytest.raises(ValueError, match="joint torque"):
        _std((0.1, 0.2), 7, "cpu", "joint torque")
    with pytest.raises(ValueError, match="non-negative"):
        _std((-1.0,), 6, "cpu", "wrench")
