"""``SubstepGraph`` on CUDA with fake PhysX views (no Isaac Sim): the warm-up substeps run eagerly, the
capture happens once, and every replay fetches the reads, replays the torch work and sends the writes --
to the values the eager region gives for the same inputs. Run with ``pytest -m gpu tests/GPU``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from furniture_lab.substep_graph import SubstepGraph

pytestmark = pytest.mark.gpu


class FakeView:
    def __init__(self, n: int, device) -> None:
        self.pos = torch.randn(n, 7, device=device)
        self.jac = torch.randn(n, 6, 7, device=device)
        self.targets = None
        self.calls = 0

    def get_dof_positions(self):
        self.calls += 1
        return self.pos.clone()  # a new tensor per call, as the tensor API

    def get_jacobians(self):
        return self.jac.clone()

    def set_dof_position_targets(self, data, indices):
        self.targets = data.clone()
        self.indices = indices


class FakeAsset:
    def __init__(self, view, device):
        self.root_physx_view = view
        self.data = SimpleNamespace()  # no lazy buffers to invalidate
        self.target = torch.zeros(view.pos.shape[0], 7, device=device)
        self.indices = torch.arange(view.pos.shape[0], device=device)


def eager_target(view: FakeView) -> torch.Tensor:
    pos, jac = view.pos, view.jac
    step = torch.linalg.solve(jac @ jac.transpose(1, 2) + 1e-4 * torch.eye(6, device=pos.device), (jac @ pos.unsqueeze(-1)))
    return pos + (jac.transpose(1, 2) @ step).squeeze(-1) * 0.01


def test_replays_match_the_eager_substep():
    if not torch.cuda.is_available():
        pytest.fail("this GPU test needs CUDA")
    device = torch.device("cuda")
    torch.manual_seed(0)
    view = FakeView(8, device)
    asset = FakeAsset(view, device)
    sim_view = SimpleNamespace()

    def region():
        pos = view.get_dof_positions()
        jac = view.get_jacobians()
        # the sync-free solve the real substep makes (cholesky_ex + cholesky_solve), into a fixed buffer
        gram = jac @ jac.transpose(1, 2) + 1e-4 * torch.eye(6, device=device)
        factor, info = torch.linalg.cholesky_ex(gram)
        y = torch.linalg.solve_triangular(factor, jac @ pos.unsqueeze(-1), upper=False)
        step = torch.linalg.solve_triangular(factor.transpose(1, 2), y, upper=True)
        asset.target.copy_(pos + (jac.transpose(1, 2) @ step).squeeze(-1) * 0.01)
        view.set_dof_position_targets(asset.target, asset.indices)

    flag = [True]
    g = SubstepGraph(region, {"robot": asset}, sim_view, lambda: (flag[0],), warmup=2)
    for i in range(6):
        view.pos.copy_(torch.randn_like(view.pos))
        view.jac.copy_(torch.randn_like(view.jac))
        g.run()
        torch.cuda.synchronize()
        assert g.captured == (i >= 2)
        assert torch.allclose(view.targets, eager_target(view), atol=1e-5), i
    assert view.calls == 6  # one read per substep, the capture's included (its fetch)
    # a Python-level choice changing after the capture is caught
    flag[0] = False
    with pytest.raises(RuntimeError, match="structure changed"):
        g.run()
