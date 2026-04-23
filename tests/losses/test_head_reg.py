"""Tests for the geometry-aware head regularizer (``src/geohead/losses/head_reg.py``).

Specifications under test live in ``docs/design.md`` sec. 4.3 and sec. 8.2.
"""

from __future__ import annotations

import pytest
import torch

from geohead.losses.head_reg import head_regularizer, second_moment

# ---------------------------------------------------------------------------
# head_regularizer
# ---------------------------------------------------------------------------


def test_output_is_scalar() -> None:
    beta = torch.randn(8)
    beta_0 = torch.randn(8)
    sigma = torch.eye(8)
    out = head_regularizer(beta, beta_0, sigma, epsilon=1e-6)
    assert out.ndim == 0


def test_zero_when_beta_equals_beta_0() -> None:
    beta = torch.randn(16)
    sigma = torch.randn(16, 16)
    sigma = sigma @ sigma.T  # PSD
    out = head_regularizer(beta, beta.clone(), sigma, epsilon=1e-3)
    assert out.item() == pytest.approx(0.0, abs=1e-7)


def test_reduces_to_ridge_when_sigma_zero() -> None:
    """With Sigma_hat = 0 the regularizer becomes ``epsilon * ||beta - beta_0||^2``."""
    beta = torch.tensor([1.0, -2.0, 0.5, 0.0])
    beta_0 = torch.tensor([0.0, 0.0, 0.0, 0.0])
    sigma = torch.zeros(4, 4)
    out = head_regularizer(beta, beta_0, sigma, epsilon=2.0)
    expected = 2.0 * (beta - beta_0).pow(2).sum().item()
    assert out.item() == pytest.approx(expected, abs=1e-6)


def test_non_negative_for_psd_sigma() -> None:
    """(beta - beta_0)^T (Sigma + eps I) (beta - beta_0) >= 0 for PSD Sigma and eps>=0."""
    gen = torch.Generator().manual_seed(0)
    for _ in range(20):
        p = 12
        beta = torch.randn(p, generator=gen)
        beta_0 = torch.randn(p, generator=gen)
        a = torch.randn(p, p, generator=gen)
        sigma = a @ a.T  # PSD
        out = head_regularizer(beta, beta_0, sigma, epsilon=1e-4)
        assert out.item() >= -1e-9


def test_gradient_matches_analytic_expression() -> None:
    """grad_beta L = 2 (Sigma + eps I) (beta - beta_0).

    grad_beta_0 L = -2 (Sigma + eps I) (beta - beta_0) = -grad_beta L.
    """
    p = 6
    gen = torch.Generator().manual_seed(1)
    beta = torch.randn(p, generator=gen, requires_grad=True)
    beta_0 = torch.randn(p, generator=gen, requires_grad=True)
    sigma = torch.randn(p, p, generator=gen)
    sigma = sigma @ sigma.T
    eps = 0.01

    out = head_regularizer(beta, beta_0, sigma, epsilon=eps)
    out.backward()

    diff = (beta - beta_0).detach()
    expected_beta_grad = 2.0 * (sigma @ diff + eps * diff)
    assert torch.allclose(beta.grad, expected_beta_grad, atol=1e-5)
    assert torch.allclose(beta_0.grad, -expected_beta_grad, atol=1e-5)


def test_analytic_reference_diagonal() -> None:
    """Hand-calculated reference.

    beta = [1, 0], beta_0 = [0, 0], Sigma = diag(2, 3), eps = 1.
    (Sigma + I) = diag(3, 4).  d^T (Sigma + I) d = 1 * 3 + 0 * 4 = 3.
    """
    beta = torch.tensor([1.0, 0.0])
    beta_0 = torch.tensor([0.0, 0.0])
    sigma = torch.diag(torch.tensor([2.0, 3.0]))
    out = head_regularizer(beta, beta_0, sigma, epsilon=1.0)
    assert out.item() == pytest.approx(3.0, abs=1e-6)


def test_shape_validation() -> None:
    with pytest.raises(ValueError, match="1-D"):
        head_regularizer(torch.randn(2, 3), torch.randn(6), torch.eye(6))
    with pytest.raises(ValueError, match="share shape"):
        head_regularizer(torch.randn(5), torch.randn(6), torch.eye(6))
    with pytest.raises(ValueError, match=r"sigma_hat must be"):
        head_regularizer(torch.randn(6), torch.randn(6), torch.eye(5))


def test_epsilon_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        head_regularizer(torch.randn(4), torch.randn(4), torch.eye(4), epsilon=-0.1)


# ---------------------------------------------------------------------------
# second_moment
# ---------------------------------------------------------------------------


def test_second_moment_shape_and_symmetry() -> None:
    z = torch.randn(30, 8)
    sigma = second_moment(z)
    assert sigma.shape == (8, 8)
    assert torch.allclose(sigma, sigma.T, atol=1e-6)


def test_second_moment_matches_definition() -> None:
    z = torch.randn(50, 6)
    expected = z.T @ z / z.shape[0]
    expected = 0.5 * (expected + expected.T)
    assert torch.allclose(second_moment(z), expected, atol=1e-7)


def test_second_moment_is_psd() -> None:
    z = torch.randn(20, 5)
    sigma = second_moment(z)
    eigs = torch.linalg.eigvalsh(sigma)
    assert eigs.min().item() >= -1e-6


def test_second_moment_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        second_moment(torch.randn(5))


def test_second_moment_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        second_moment(torch.empty(0, 4))


def test_head_regularizer_integrates_with_second_moment() -> None:
    """End-to-end: estimate Sigma_hat from a batch Z, use it in head_regularizer."""
    gen = torch.Generator().manual_seed(2)
    z = torch.randn(64, 10, generator=gen)
    beta = torch.randn(10, generator=gen)
    beta_0 = torch.randn(10, generator=gen)
    sigma_hat = second_moment(z)
    out = head_regularizer(beta, beta_0, sigma_hat, epsilon=1e-4)
    # Should equal mean of (z @ (beta - beta_0))^2 plus eps ||diff||^2.
    diff = beta - beta_0
    expected = (z @ diff).pow(2).mean() + 1e-4 * diff.pow(2).sum()
    assert out.item() == pytest.approx(expected.item(), rel=1e-5, abs=1e-6)
