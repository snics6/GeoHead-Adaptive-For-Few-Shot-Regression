"""Tests for ``src/geohead/evaluation/metrics.py``.

Specifications under test live in ``docs/design.md`` §9.1–§9.2.
"""

from __future__ import annotations

import pytest
import torch

from geohead.evaluation.metrics import (
    evaluate_head,
    head_correction_geo,
    head_correction_l2,
    query_mae,
    query_mse,
)
from geohead.losses.head_reg import head_regularizer, second_moment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_case(n: int = 40, p: int = 8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, p, generator=g)
    beta_0 = torch.randn(p, generator=g)
    y = z @ beta_0 + 0.1 * torch.randn(n, generator=g)
    return z, y, beta_0


# ---------------------------------------------------------------------------
# query_mse / query_mae
# ---------------------------------------------------------------------------


def test_query_mse_zero_at_perfect_fit() -> None:
    z, _, beta_0 = _make_case()
    y_hat = z @ beta_0
    mse = query_mse(z, y_hat, beta_0)
    torch.testing.assert_close(mse, torch.zeros(()))


def test_query_mae_zero_at_perfect_fit() -> None:
    z, _, beta_0 = _make_case()
    y_hat = z @ beta_0
    mae = query_mae(z, y_hat, beta_0)
    torch.testing.assert_close(mae, torch.zeros(()))


def test_query_mse_matches_reference_formula() -> None:
    z, y, beta_0 = _make_case()
    beta = beta_0 + 0.05 * torch.randn_like(beta_0)
    mse = query_mse(z, y, beta)
    ref = ((z @ beta - y) ** 2).mean()
    torch.testing.assert_close(mse, ref)


def test_query_mae_matches_reference_formula() -> None:
    z, y, beta_0 = _make_case()
    beta = beta_0 + 0.05 * torch.randn_like(beta_0)
    mae = query_mae(z, y, beta)
    ref = (z @ beta - y).abs().mean()
    torch.testing.assert_close(mae, ref)


def test_query_mse_shape_validation() -> None:
    z, y, beta = _make_case(n=10, p=4)
    with pytest.raises(ValueError):
        query_mse(z[0], y, beta)  # z 1-D
    with pytest.raises(ValueError):
        query_mse(z, y.unsqueeze(-1), beta)  # y 2-D
    with pytest.raises(ValueError):
        query_mse(z, y, beta.unsqueeze(-1))  # beta 2-D
    with pytest.raises(ValueError):
        query_mse(z, y[:5], beta)  # n mismatch
    with pytest.raises(ValueError):
        query_mse(z, y, beta[:2])  # p mismatch
    with pytest.raises(ValueError):
        query_mse(z[:0], y[:0], beta)  # empty


# ---------------------------------------------------------------------------
# head_correction_l2 / head_correction_geo
# ---------------------------------------------------------------------------


def test_head_correction_l2_zero_at_identity() -> None:
    _, _, beta_0 = _make_case()
    val = head_correction_l2(beta_0, beta_0)
    torch.testing.assert_close(val, torch.zeros(()))


def test_head_correction_l2_matches_vector_norm() -> None:
    _, _, beta_0 = _make_case()
    beta = beta_0 + torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    val = head_correction_l2(beta, beta_0)
    torch.testing.assert_close(val, torch.tensor(1.0))


def test_head_correction_geo_zero_at_identity() -> None:
    z, _, beta_0 = _make_case()
    sigma = second_moment(z)
    val = head_correction_geo(beta_0, beta_0, sigma, eps=1e-6)
    torch.testing.assert_close(val, torch.zeros(()))


def test_head_correction_geo_matches_head_regularizer() -> None:
    """By construction both use the quadratic form ``(β-β₀)ᵀ(Σ+εI)(β-β₀)``."""
    z, _, beta_0 = _make_case()
    sigma = second_moment(z)
    beta = beta_0 + 0.3 * torch.randn_like(beta_0)
    val = head_correction_geo(beta, beta_0, sigma, eps=1e-4)
    ref = head_regularizer(beta, beta_0, sigma, epsilon=1e-4)
    torch.testing.assert_close(val, ref)


def test_head_correction_geo_psd_for_psd_sigma() -> None:
    """Σ̂ = (1/n) ZᵀZ is PSD ⇒ the quadratic form is non-negative."""
    z, _, beta_0 = _make_case(n=50, p=6, seed=7)
    sigma = second_moment(z)
    for s in range(10):
        beta = beta_0 + torch.randn(beta_0.shape[0], generator=torch.Generator().manual_seed(s))
        assert head_correction_geo(beta, beta_0, sigma, eps=1e-6).item() >= 0.0


def test_head_correction_geo_shape_validation() -> None:
    z, _, beta_0 = _make_case(n=10, p=4)
    sigma = second_moment(z)
    with pytest.raises(ValueError):
        head_correction_geo(beta_0[:2], beta_0, sigma)
    with pytest.raises(ValueError):
        head_correction_geo(beta_0, beta_0, sigma[:2, :2])
    with pytest.raises(ValueError):
        head_correction_geo(beta_0, beta_0, sigma, eps=-1e-6)


# ---------------------------------------------------------------------------
# evaluate_head convenience wrapper
# ---------------------------------------------------------------------------


def test_evaluate_head_returns_all_four_metrics_as_floats() -> None:
    z, y, beta_0 = _make_case()
    sigma = second_moment(z)
    beta = beta_0 + 0.1 * torch.randn_like(beta_0)
    out = evaluate_head(z, y, beta, beta_0, sigma, eps=1e-6)
    assert set(out.keys()) == {"mse", "mae", "delta_l2", "delta_geo"}
    for k, v in out.items():
        assert isinstance(v, float), f"{k} must be a Python float, got {type(v)}"


def test_evaluate_head_identity_case() -> None:
    """With β = β_0, both correction metrics are exactly 0.0."""
    z, y, beta_0 = _make_case()
    sigma = second_moment(z)
    out = evaluate_head(z, y, beta_0, beta_0, sigma)
    assert out["delta_l2"] == pytest.approx(0.0, abs=1e-7)
    assert out["delta_geo"] == pytest.approx(0.0, abs=1e-7)
    # MSE / MAE on the original labels are non-negative and finite.
    assert out["mse"] >= 0.0
    assert out["mae"] >= 0.0
