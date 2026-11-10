"""furniture_lab.linalg: the Cholesky-based pseudo-inverse applications equal torch's pinv on full-rank
Jacobians, stay bounded where pinv blows up, zero a failed row instead of spreading NaN, and make no
host synchronization on CUDA (the reason they exist)."""

import pytest
import torch

from furniture_lab.linalg import damped_pinv_apply, damped_pinv_t_apply, spd_solve


def jacobians(n: int = 8, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn((n, 6, 7), generator=g)


def test_undamped_matches_pinv_on_full_rank_jacobians():
    jac = jacobians()
    g = torch.Generator().manual_seed(1)
    dx = torch.randn((8, 6), generator=g)
    tau = torch.randn((8, 7), generator=g)
    want_dq = (torch.linalg.pinv(jac) @ dx.unsqueeze(-1)).squeeze(-1)
    want_f = (torch.linalg.pinv(jac.transpose(1, 2)) @ tau.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(damped_pinv_apply(jac, dx), want_dq, atol=1e-5)
    assert torch.allclose(damped_pinv_t_apply(jac, tau), want_f, atol=1e-5)
    assert damped_pinv_apply(jac, dx).dtype == dx.dtype


def test_small_damping_is_a_tiny_relative_change_on_a_poorly_conditioned_jacobian():
    # singular values 1 .. 0.01 (a Franka's near a stretched pose): damping 1e-4 changes the smallest
    # direction by 1e-8 / 1e-4 = 1e-4 relative, the others less
    g = torch.Generator().manual_seed(2)
    u, _ = torch.linalg.qr(torch.randn((6, 6), generator=g))
    v, _ = torch.linalg.qr(torch.randn((7, 7), generator=g))
    s = torch.tensor([1.0, 0.6, 0.3, 0.1, 0.03, 0.01])
    jac = (u @ torch.diag(s) @ v[:, :6].T).unsqueeze(0).to(torch.float64)
    dx = torch.randn((1, 6), generator=g).to(torch.float64)
    exact = damped_pinv_apply(jac, dx, 0.0)
    damped = damped_pinv_apply(jac, dx, 1e-4)
    assert (damped - exact).norm() / exact.norm() < 2e-4
    assert torch.allclose(exact, (torch.linalg.pinv(jac) @ dx.unsqueeze(-1)).squeeze(-1), atol=1e-8)


def test_a_singular_jacobian_never_yields_nan():
    jac = jacobians(2)
    jac[:, 5, :] = jac[:, 4, :]  # rank 5: J J^T singular
    dx = torch.ones((2, 6))
    damped = damped_pinv_apply(jac, dx, 1e-2)
    assert torch.isfinite(damped).all() and damped.norm() < 1e3
    # undamped: the factorization either fails (-> a zero step) or rounds through; never NaN either way
    assert torch.isfinite(damped_pinv_apply(jac, dx, 0.0)).all()


def test_spd_solve_zeroes_only_the_failed_rows():
    a = torch.eye(3).repeat(2, 1, 1)
    a[1] = 0.0  # not positive definite
    b = torch.ones((2, 3, 1))
    x = spd_solve(a, b)
    assert torch.equal(x[0], b[0]) and torch.equal(x[1], torch.zeros((3, 1)))


def test_shape_errors_are_loud():
    jac = jacobians(1)
    with pytest.raises(ValueError, match="x must be"):
        damped_pinv_apply(jac, torch.zeros((1, 7)))
    with pytest.raises(ValueError, match="tau must be"):
        damped_pinv_t_apply(jac, torch.zeros((1, 6)))
    with pytest.raises(ValueError, match="damping"):
        damped_pinv_apply(jac, torch.zeros((1, 6)), -1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the host-sync check needs a CUDA device")
def test_no_host_synchronization_on_cuda():
    import warnings

    jac = jacobians().cuda()
    dx, tau = torch.randn((8, 6), device="cuda"), torch.randn((8, 7), device="cuda")
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode(1)
    try:
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            damped_pinv_apply(jac, dx, 1e-4)
            damped_pinv_t_apply(jac, tau, 1e-4)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    syncs = [str(w.message) for w in seen if "synchroniz" in str(w.message).lower()]
    assert syncs == [], syncs
