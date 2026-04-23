"""Tests for test-time adaptation (``src/geohead/adaptation/test_time.py``).

Specifications under test live in ``docs/design.md`` §4.3 (inner loop)
and §8.1–§8.3 (test-time adaptation).
"""

from __future__ import annotations

import pytest
import torch

from geohead.adaptation.test_time import (
    geo_adapt,
    inner_rule_adapt,
    ridge_adapt,
)
from geohead.losses.head_reg import head_regularizer, second_moment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _toy_problem(n: int = 30, p: int = 5, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, p, generator=g)
    beta_true = torch.randn(p, generator=g)
    y = z @ beta_true + 0.01 * torch.randn(n, generator=g)
    beta_0 = torch.randn(p, generator=g)
    return z, y, beta_0, beta_true


def _inner_loss(
    z: torch.Tensor,
    y: torch.Tensor,
    beta: torch.Tensor,
    beta_0: torch.Tensor,
    sigma: torch.Tensor,
    lambda_h: float,
    eps: float,
) -> torch.Tensor:
    return ((z @ beta - y) ** 2).mean() + lambda_h * head_regularizer(
        beta, beta_0, sigma, epsilon=eps
    )


# ---------------------------------------------------------------------------
# ridge_adapt
# ---------------------------------------------------------------------------


def test_ridge_adapt_shape() -> None:
    z, y, beta_0, _ = _toy_problem(n=30, p=5)
    beta = ridge_adapt(z, y, beta_0, lambda_=0.1)
    assert beta.shape == (5,)
    assert beta.dtype == z.dtype


def test_ridge_adapt_lambda_zero_matches_ols() -> None:
    z, y, beta_0, _ = _toy_problem(n=50, p=5)
    beta = ridge_adapt(z, y, beta_0, lambda_=0.0)
    beta_ols = torch.linalg.lstsq(z, y).solution
    torch.testing.assert_close(beta, beta_ols, rtol=1e-4, atol=1e-4)


def test_ridge_adapt_lambda_large_approaches_beta_0() -> None:
    z, y, beta_0, _ = _toy_problem(n=30, p=5)
    beta = ridge_adapt(z, y, beta_0, lambda_=1e8)
    torch.testing.assert_close(beta, beta_0, rtol=1e-4, atol=1e-4)


def test_ridge_adapt_scalar_analytic() -> None:
    """Hand-checked 1-feature problem.

    With ``z = [[1], [2], [3]]`` and ``y = [2, 4, 6]`` (exact ``y = 2 x``)::

        Z^T Z / n = 14 / 3
        Z^T y / n = 28 / 3
        beta = (28/3 + lambda * beta_0) / (14/3 + lambda)
    """
    z = torch.tensor([[1.0], [2.0], [3.0]])
    y = torch.tensor([2.0, 4.0, 6.0])
    beta_0 = torch.tensor([1.5])

    for lam in (0.0, 0.1, 1.0, 10.0):
        expected = (28.0 / 3.0 + lam * 1.5) / (14.0 / 3.0 + lam)
        beta = ridge_adapt(z, y, beta_0, lambda_=lam)
        assert beta.shape == (1,)
        assert abs(beta.item() - expected) < 1e-5


def test_ridge_adapt_shape_validation() -> None:
    z = torch.randn(10, 5)
    y = torch.randn(10)
    beta_0 = torch.randn(5)
    with pytest.raises(ValueError):
        ridge_adapt(z.flatten(), y, beta_0, 0.1)  # z not 2D
    with pytest.raises(ValueError):
        ridge_adapt(z, y.unsqueeze(-1), beta_0, 0.1)  # y not 1D
    with pytest.raises(ValueError):
        ridge_adapt(z, y[:-1], beta_0, 0.1)  # y / z length mismatch
    with pytest.raises(ValueError):
        ridge_adapt(z, y, beta_0[:-1], 0.1)  # p mismatch
    with pytest.raises(ValueError):
        ridge_adapt(z, y, beta_0, lambda_=-0.1)  # negative lambda


# ---------------------------------------------------------------------------
# geo_adapt
# ---------------------------------------------------------------------------


def test_geo_adapt_shape() -> None:
    z, y, beta_0, _ = _toy_problem(n=30, p=5)
    sigma = second_moment(z)
    beta = geo_adapt(z, y, beta_0, sigma, lambda_=0.1)
    assert beta.shape == (5,)


def test_geo_adapt_identity_sigma_reduces_to_ridge() -> None:
    """With ``sigma = I`` and ``eps = 0``, ``M = I`` and geo equals ridge."""
    z, y, beta_0, _ = _toy_problem(n=30, p=5)
    sigma = torch.eye(5)
    beta_geo = geo_adapt(z, y, beta_0, sigma, lambda_=0.3, eps=0.0)
    beta_ridge = ridge_adapt(z, y, beta_0, lambda_=0.3)
    torch.testing.assert_close(beta_geo, beta_ridge, rtol=1e-5, atol=1e-5)


def test_geo_adapt_large_lambda_approaches_beta_0() -> None:
    z, y, beta_0, _ = _toy_problem(n=30, p=5)
    sigma = second_moment(z)
    beta = geo_adapt(z, y, beta_0, sigma, lambda_=1e10, eps=1e-3)
    torch.testing.assert_close(beta, beta_0, rtol=1e-3, atol=1e-3)


def test_geo_adapt_kkt_gradient_vanishes() -> None:
    """Closed-form solution must be a stationary point of the geo loss."""
    torch.manual_seed(0)
    z, y, beta_0, _ = _toy_problem(n=20, p=4, seed=1)
    a = torch.randn(4, 4)
    sigma = a @ a.T
    lam = 0.5
    eps = 1e-3

    beta_closed = geo_adapt(z, y, beta_0, sigma, lambda_=lam, eps=eps)

    beta = beta_closed.clone().requires_grad_(True)
    loss = _inner_loss(z, y, beta, beta_0, sigma, lambda_h=lam, eps=eps)
    (grad,) = torch.autograd.grad(loss, beta)
    assert grad.norm().item() < 1e-4


def test_geo_adapt_shape_validation() -> None:
    z, y, beta_0, _ = _toy_problem(n=20, p=5)
    sigma = second_moment(z)
    with pytest.raises(ValueError):
        geo_adapt(z, y, beta_0, sigma[:, :-1], 0.1)  # sigma not square
    with pytest.raises(ValueError):
        geo_adapt(z, y, beta_0, torch.randn(4, 4), 0.1)  # wrong sigma size
    with pytest.raises(ValueError):
        geo_adapt(z, y, beta_0, sigma.flatten(), 0.1)  # sigma not 2D
    with pytest.raises(ValueError):
        geo_adapt(z, y, beta_0, sigma, lambda_=-0.1)  # negative lambda
    with pytest.raises(ValueError):
        geo_adapt(z, y, beta_0, sigma, lambda_=0.1, eps=-1e-3)  # negative eps


# ---------------------------------------------------------------------------
# inner_rule_adapt
# ---------------------------------------------------------------------------


def test_inner_rule_adapt_shape() -> None:
    z, y, beta_0, _ = _toy_problem(n=30, p=5)
    sigma = second_moment(z)
    beta = inner_rule_adapt(
        z, y, beta_0, sigma, lambda_h=0.1, eta=0.01, steps=5
    )
    assert beta.shape == (5,)
    # Default test-time call returns a detached tensor.
    assert not beta.requires_grad


def test_inner_rule_adapt_zero_steps_returns_beta_0() -> None:
    z, y, beta_0, _ = _toy_problem()
    sigma = second_moment(z)
    beta = inner_rule_adapt(
        z, y, beta_0, sigma, lambda_h=0.1, eta=0.01, steps=0
    )
    torch.testing.assert_close(beta, beta_0)


def test_inner_rule_adapt_converges_to_geo() -> None:
    """Many small GD steps converge to the closed-form geo solution."""
    torch.manual_seed(0)
    z, y, beta_0, _ = _toy_problem(n=40, p=4, seed=2)
    sigma = second_moment(z)
    lam = 0.1
    eps = 1e-3

    beta_geo = geo_adapt(z, y, beta_0, sigma, lambda_=lam, eps=eps)
    beta_gd = inner_rule_adapt(
        z, y, beta_0, sigma, lambda_h=lam, eta=0.05, steps=2000, eps=eps
    )
    torch.testing.assert_close(beta_gd, beta_geo, rtol=5e-3, atol=5e-3)


def test_inner_rule_adapt_loss_decreases_with_more_steps() -> None:
    z, y, beta_0, _ = _toy_problem(n=40, p=5, seed=3)
    sigma = second_moment(z)
    lam = 0.1
    eps = 1e-4
    eta = 0.02

    loss_at_0 = float(
        _inner_loss(z, y, beta_0, beta_0, sigma, lambda_h=lam, eps=eps).item()
    )
    prev = loss_at_0
    for k in (1, 2, 5, 10, 20):
        beta_k = inner_rule_adapt(
            z, y, beta_0, sigma, lambda_h=lam, eta=eta, steps=k, eps=eps
        )
        loss_k = float(
            _inner_loss(z, y, beta_k, beta_0, sigma, lambda_h=lam, eps=eps).item()
        )
        assert loss_k < loss_at_0, f"k={k}: {loss_k} not below baseline {loss_at_0}"
        assert loss_k <= prev + 1e-8, f"k={k}: loss increased from {prev} to {loss_k}"
        prev = loss_k


def test_inner_rule_adapt_manual_gradient_agrees() -> None:
    """Autograd gradient on the inner loss should equal the analytic one."""
    torch.manual_seed(0)
    z, y, beta_0, _ = _toy_problem(n=25, p=5, seed=4)
    sigma = second_moment(z)
    eps = 1e-3
    lam = 0.2
    n = z.shape[0]
    p = z.shape[1]

    beta = beta_0.clone().requires_grad_(True)
    loss = _inner_loss(z, y, beta, beta_0, sigma, lambda_h=lam, eps=eps)
    (grad_auto,) = torch.autograd.grad(loss, beta)

    m_mat = sigma + eps * torch.eye(p)
    grad_manual = (2.0 / n) * (z.T @ (z @ beta.detach() - y)) + 2.0 * lam * (
        m_mat @ (beta.detach() - beta_0)
    )
    torch.testing.assert_close(grad_auto, grad_manual, rtol=1e-5, atol=1e-5)


def test_inner_rule_adapt_create_graph_is_differentiable() -> None:
    """With ``create_graph=True`` the final beta is differentiable w.r.t. beta_0."""
    z, y, beta_0, _ = _toy_problem(n=20, p=4, seed=5)
    beta_0 = beta_0.clone().requires_grad_(True)
    sigma = second_moment(z)

    beta_final = inner_rule_adapt(
        z,
        y,
        beta_0,
        sigma,
        lambda_h=0.1,
        eta=0.01,
        steps=3,
        create_graph=True,
    )
    # The returned tensor carries a graph.
    assert beta_final.requires_grad

    (beta_final.pow(2).sum()).backward()
    assert beta_0.grad is not None
    assert beta_0.grad.shape == beta_0.shape
    assert torch.isfinite(beta_0.grad).all()
    assert beta_0.grad.norm().item() > 0.0


def test_inner_rule_adapt_validation() -> None:
    z, y, beta_0, _ = _toy_problem(n=20, p=5)
    sigma = second_moment(z)
    with pytest.raises(ValueError):
        inner_rule_adapt(z, y, beta_0, sigma, lambda_h=-0.1, eta=0.01, steps=1)
    with pytest.raises(ValueError):
        inner_rule_adapt(z, y, beta_0, sigma, lambda_h=0.1, eta=0.0, steps=1)
    with pytest.raises(ValueError):
        inner_rule_adapt(z, y, beta_0, sigma, lambda_h=0.1, eta=0.01, steps=-1)
    with pytest.raises(ValueError):
        inner_rule_adapt(z, y, beta_0, sigma, lambda_h=0.1, eta=0.01, steps=1, eps=-1)


# ---------------------------------------------------------------------------
# preconditioned inner rule
# ---------------------------------------------------------------------------


def test_inner_rule_preconditioned_default_is_vanilla() -> None:
    """Omitting ``preconditioned`` must reproduce the original update."""
    torch.manual_seed(0)
    z, y, beta_0, _ = _toy_problem(n=40, p=6, seed=10)
    sigma = second_moment(z)

    vanilla = inner_rule_adapt(
        z, y, beta_0, sigma, lambda_h=0.1, eta=0.01, steps=3
    )
    explicit_off = inner_rule_adapt(
        z, y, beta_0, sigma, lambda_h=0.1, eta=0.01, steps=3,
        preconditioned=False,
    )
    assert torch.allclose(vanilla, explicit_off)


def test_inner_rule_preconditioned_step_one_reaches_ols() -> None:
    """One preconditioned step with ``η=1/2``, ``λ_h=0`` and ``ε→0``
    must land on the OLS solution (derivation in the function docstring).
    """
    torch.manual_seed(0)
    z, y, beta_0, _ = _toy_problem(n=60, p=5, seed=11)
    sigma = second_moment(z)

    # β_OLS = (ZᵀZ)⁻¹ Zᵀy — full-rank since n >> p.
    beta_ols = torch.linalg.solve(z.T @ z, z.T @ y)

    beta_prec = inner_rule_adapt(
        z, y, beta_0, sigma,
        lambda_h=0.0, eta=0.5, steps=1,
        eps=1e-10, preconditioned=True,
    )
    assert torch.allclose(beta_prec, beta_ols, atol=1e-5, rtol=1e-4)


def test_inner_rule_preconditioned_stable_under_feature_scale() -> None:
    """Vanilla GD diverges when ``||z||`` grows; the preconditioned
    variant stays bounded (scale invariance property)."""
    torch.manual_seed(0)
    z, y, beta_0, _ = _toy_problem(n=40, p=4, seed=12)
    scale = 100.0
    z_big = z * scale
    sigma_big = second_moment(z_big)

    vanilla_big = inner_rule_adapt(
        z_big, y, beta_0, sigma_big,
        lambda_h=0.1, eta=0.1, steps=5,
    )
    prec_big = inner_rule_adapt(
        z_big, y, beta_0, sigma_big,
        lambda_h=0.1, eta=0.1, steps=5,
        preconditioned=True,
    )

    # Vanilla GD explodes — residual blows up.
    vanilla_res = (z_big @ vanilla_big - y).abs().max().item()
    prec_res = (z_big @ prec_big - y).abs().max().item()
    assert vanilla_res > 1e3, (
        "vanilla GD should diverge at this feature scale — "
        f"got max residual {vanilla_res}"
    )
    # Preconditioned must be comparable to β_0's fit (no blow-up).
    base_res = (z_big @ beta_0 - y).abs().max().item()
    assert prec_res < base_res * 2 + 1.0, (
        "preconditioned inner rule should stay bounded; "
        f"got max residual {prec_res} vs base {base_res}"
    )


def test_inner_rule_preconditioned_scale_invariance() -> None:
    """Predictions ``Zβ̂`` must be invariant under ``Z → cZ, β_0 → β_0/c``.

    Uniformly rescaling the encoder output should not change the
    prediction of the adapted model: the preconditioner exactly
    absorbs the scale into the metric.
    """
    torch.manual_seed(0)
    z, y, beta_0, _ = _toy_problem(n=50, p=4, seed=13)
    sigma = second_moment(z)
    c = 7.0
    z_c = z * c
    beta_0_c = beta_0 / c
    sigma_c = second_moment(z_c)

    b1 = inner_rule_adapt(
        z, y, beta_0, sigma,
        lambda_h=0.1, eta=0.1, steps=3,
        preconditioned=True,
    )
    b2 = inner_rule_adapt(
        z_c, y, beta_0_c, sigma_c,
        lambda_h=0.1, eta=0.1, steps=3,
        preconditioned=True,
    )
    assert torch.allclose(z @ b1, z_c @ b2, atol=1e-4, rtol=1e-4)


def test_inner_rule_preconditioned_create_graph() -> None:
    """``preconditioned=True`` must remain differentiable for the outer
    meta-loop (used by the bilevel trainer)."""
    torch.manual_seed(0)
    z, y, beta_0, _ = _toy_problem(n=40, p=5, seed=14)
    beta_0 = beta_0.detach().clone().requires_grad_(True)
    sigma = second_moment(z)

    beta_final = inner_rule_adapt(
        z, y, beta_0, sigma,
        lambda_h=0.1, eta=0.05, steps=3,
        create_graph=True, preconditioned=True,
    )
    assert beta_final.requires_grad

    beta_final.pow(2).sum().backward()
    assert beta_0.grad is not None
    assert torch.isfinite(beta_0.grad).all()
    assert beta_0.grad.norm().item() > 0.0


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


def test_all_methods_improve_mse_over_beta_0() -> None:
    """On a well-posed toy problem, all 3 methods should reduce MSE vs beta_0."""
    torch.manual_seed(0)
    z, y, beta_0, _ = _toy_problem(n=80, p=5, seed=6)
    sigma = second_moment(z)

    mse_0 = float(((z @ beta_0 - y) ** 2).mean().item())
    b_ridge = ridge_adapt(z, y, beta_0, lambda_=0.1)
    b_geo = geo_adapt(z, y, beta_0, sigma, lambda_=0.1)
    b_inner = inner_rule_adapt(
        z, y, beta_0, sigma, lambda_h=0.1, eta=0.1, steps=50
    )

    mse_ridge = float(((z @ b_ridge - y) ** 2).mean().item())
    mse_geo = float(((z @ b_geo - y) ** 2).mean().item())
    mse_inner = float(((z @ b_inner - y) ** 2).mean().item())

    assert mse_ridge < mse_0
    assert mse_geo < mse_0
    assert mse_inner < mse_0
