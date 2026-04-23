"""Tests for the MLP encoder (``src/geohead/models/encoder.py``).

Specification: ``docs/design.md`` sec. 7.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from geohead.models.encoder import MLPEncoder


def test_output_shape_matches_p() -> None:
    enc = MLPEncoder(d_x=16, hidden=(64, 64), p=32)
    x = torch.randn(8, 16)
    z = enc(x)
    assert z.shape == (8, 32)


def test_layer_composition_linear_relu_layernorm() -> None:
    """Each hidden block is Linear -> ReLU -> LayerNorm, output layer is bare Linear."""
    enc = MLPEncoder(d_x=16, hidden=(64, 64), p=32)
    modules = list(enc.net)
    # 2 hidden blocks (3 modules each) + 1 final Linear = 7 modules.
    assert len(modules) == 7
    assert isinstance(modules[0], nn.Linear)
    assert isinstance(modules[1], nn.ReLU)
    assert isinstance(modules[2], nn.LayerNorm)
    assert isinstance(modules[3], nn.Linear)
    assert isinstance(modules[4], nn.ReLU)
    assert isinstance(modules[5], nn.LayerNorm)
    assert isinstance(modules[6], nn.Linear)  # output, no activation / no norm


def test_final_layer_has_no_activation_or_norm() -> None:
    enc = MLPEncoder(d_x=16, hidden=(64, 64), p=32)
    assert isinstance(enc.net[-1], nn.Linear)
    # Feeding a large positive input, the output must be allowed to be negative
    # somewhere (which cannot happen after ReLU): verifies there is no final ReLU.
    x = torch.randn(64, 16) * 5.0
    z = enc(x)
    assert (z < 0).any()


def test_parameter_count() -> None:
    d_x, p = 16, 32
    h1, h2 = 64, 64
    enc = MLPEncoder(d_x=d_x, hidden=(h1, h2), p=p)
    # Three Linear layers with bias + two LayerNorms (each with weight and bias).
    expected = (
        (d_x * h1 + h1)
        + (h1 * h2 + h2)
        + (h2 * p + p)
        + (2 * h1)  # LN1 affine (gamma, beta)
        + (2 * h2)  # LN2 affine (gamma, beta)
    )
    actual = sum(t.numel() for t in enc.parameters())
    assert actual == expected


def test_forward_is_deterministic_given_init() -> None:
    torch.manual_seed(0)
    enc1 = MLPEncoder()
    torch.manual_seed(0)
    enc2 = MLPEncoder()
    x = torch.randn(4, 16)
    assert torch.allclose(enc1(x), enc2(x))


def test_layernorm_produces_standardised_hidden_activations() -> None:
    """Activations right after each LayerNorm should have per-sample mean 0 and
    unit variance (LN default)."""
    enc = MLPEncoder(d_x=16, hidden=(64, 64), p=32)
    x = torch.randn(32, 16)

    captured: list[torch.Tensor] = []

    def hook(_mod, _inp, out):
        captured.append(out.detach())

    handles = [m.register_forward_hook(hook) for m in enc.net if isinstance(m, nn.LayerNorm)]
    try:
        _ = enc(x)
    finally:
        for h in handles:
            h.remove()

    assert len(captured) == 2
    for h_out in captured:
        # Normalisation is along the feature dimension (last dim), per sample.
        means = h_out.mean(dim=-1)
        stds = h_out.std(dim=-1, unbiased=False)
        assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)
        assert torch.allclose(stds, torch.ones_like(stds), atol=1e-4)


def test_gradient_flows_through_all_params() -> None:
    enc = MLPEncoder()
    x = torch.randn(8, enc.d_x)
    loss = enc(x).pow(2).sum()
    loss.backward()
    for name, param in enc.named_parameters():
        assert param.grad is not None, f"no grad for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite grad for {name}"


def test_forward_rejects_wrong_shape() -> None:
    enc = MLPEncoder(d_x=16, hidden=(64, 64), p=32)
    with pytest.raises(ValueError):
        enc(torch.randn(16))  # 1D
    with pytest.raises(ValueError):
        enc(torch.randn(8, 17))  # wrong d_x


def test_custom_hidden_sizes() -> None:
    enc = MLPEncoder(d_x=8, hidden=(32, 16, 8), p=4)
    x = torch.randn(5, 8)
    z = enc(x)
    assert z.shape == (5, 4)
    # 3 hidden blocks (3 modules each) + 1 output Linear = 10 modules.
    assert len(list(enc.net)) == 10


def test_empty_hidden_rejected() -> None:
    with pytest.raises(ValueError):
        MLPEncoder(d_x=16, hidden=(), p=32)
