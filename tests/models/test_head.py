"""Tests for the linear regression head (``src/geohead/models/head.py``).

Specification: ``docs/design.md`` sec. 7.
"""

from __future__ import annotations

import pytest
import torch

from geohead.models.head import LinearHead


def test_output_shape_is_one_dimensional() -> None:
    head = LinearHead(p=32)
    z = torch.randn(8, 32)
    y = head(z)
    assert y.shape == (8,)


def test_has_no_bias_and_correct_param_count() -> None:
    head = LinearHead(p=32)
    params = list(head.parameters())
    # Exactly one parameter tensor (the weight), total elements == p.
    assert len(params) == 1
    assert params[0].numel() == 32
    for name, _ in head.named_parameters():
        assert "bias" not in name


def test_beta_property_shape_and_identity() -> None:
    head = LinearHead(p=32)
    assert head.beta.shape == (32,)
    # The beta property must expose the same underlying storage as linear.weight.
    assert head.beta.data_ptr() == head.linear.weight.data_ptr()


def test_beta_in_place_edit_affects_forward() -> None:
    """Mutating ``beta`` (same storage as the weight) changes the prediction."""
    head = LinearHead(p=4)
    z = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    with torch.no_grad():
        head.beta.zero_()
    assert head(z).item() == 0.0
    with torch.no_grad():
        head.beta.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert head(z).item() == pytest.approx(1.0)


def test_forward_matches_manual_inner_product() -> None:
    head = LinearHead(p=5)
    z = torch.randn(16, 5)
    y_module = head(z)
    y_manual = z @ head.beta
    assert torch.allclose(y_module, y_manual, atol=1e-6)


def test_gradient_flows() -> None:
    head = LinearHead(p=32)
    z = torch.randn(8, 32)
    loss = head(z).pow(2).sum()
    loss.backward()
    assert head.linear.weight.grad is not None
    assert torch.isfinite(head.linear.weight.grad).all()


def test_forward_rejects_wrong_shape() -> None:
    head = LinearHead(p=32)
    with pytest.raises(ValueError):
        head(torch.randn(32))  # 1D
    with pytest.raises(ValueError):
        head(torch.randn(8, 33))  # wrong feature dim
