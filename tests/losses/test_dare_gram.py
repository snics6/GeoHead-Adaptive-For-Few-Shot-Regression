"""Tests for the DARE-GRAM regularizer (``src/geohead/losses/dare_gram.py``).

Specifications under test live in ``docs/design.md`` sec. 5.
"""

from __future__ import annotations

import math

import pytest
import torch

from geohead.losses.dare_gram import (
    DareGramInfo,
    _gram_pinv_eigendecomp,
    _pinv_from_eig,
    dare_gram_regularizer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_features(b: int, p: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(b, p, generator=gen)


# ---------------------------------------------------------------------------
# Pseudo-inverse helper
# ---------------------------------------------------------------------------


def test_eigendecomp_sorted_descending() -> None:
    z = _random_features(64, 16, seed=0)
    _V, lam, _k = _gram_pinv_eigendecomp(z, threshold=0.99, eps=1e-6)
    # Descending order (non-strict for ties).
    diffs = lam[:-1] - lam[1:]
    assert (diffs >= -1e-6).all()


def test_eigendecomp_eigenvalues_match_gram() -> None:
    z = _random_features(64, 8, seed=1)
    V, lam, _k = _gram_pinv_eigendecomp(z, threshold=0.99, eps=1e-6)
    G = z.T @ z
    reconstructed = V @ torch.diag(lam) @ V.T
    assert torch.allclose(G, reconstructed, atol=1e-4)


def test_eigendecomp_k_reaches_threshold() -> None:
    z = _random_features(64, 16, seed=2)
    _V, lam, k = _gram_pinv_eigendecomp(z, threshold=0.99, eps=1e-6)
    total = lam.sum()
    cum_k = lam[:k].sum()
    cum_km1 = lam[: k - 1].sum() if k > 1 else torch.tensor(0.0)
    assert (cum_k / total).item() > 0.99
    if k > 1:
        # k is minimal: with k-1 components, we should NOT exceed threshold.
        assert (cum_km1 / total).item() <= 0.99


def test_eigendecomp_handles_batch_smaller_than_feature_dim() -> None:
    b, p = 4, 16
    z = _random_features(b, p, seed=3)
    V, lam, k = _gram_pinv_eigendecomp(z, threshold=0.99, eps=1e-6)
    assert V.shape == (p, p)
    assert lam.shape == (p,)
    assert 1 <= k <= p
    # With b < p, at most b eigenvalues of Z^T Z are non-zero.
    assert (lam[b:] <= 1e-8).all()


def test_pinv_matches_torch_linalg_pinv_when_k_full() -> None:
    z = _random_features(64, 8, seed=4)
    V, lam, _k = _gram_pinv_eigendecomp(z, threshold=0.99, eps=1e-6)
    G_pinv_ours = _pinv_from_eig(V, lam, k=lam.shape[0], eps=1e-12)
    G = z.T @ z
    G_pinv_ref = torch.linalg.pinv(G)
    assert torch.allclose(G_pinv_ours, G_pinv_ref, atol=1e-3)


# ---------------------------------------------------------------------------
# Identity / symmetry properties
# ---------------------------------------------------------------------------


def test_identical_features_give_zero_loss() -> None:
    z = _random_features(64, 16, seed=5)
    out, info = dare_gram_regularizer(z, z.clone(), return_components=True)
    assert out.item() == pytest.approx(0.0, abs=1e-5)
    assert info.L_cos.item() == pytest.approx(0.0, abs=1e-5)
    assert info.L_scale.item() == pytest.approx(0.0, abs=1e-5)
    assert info.k_s == info.k_t == info.k


def test_row_permutation_invariance() -> None:
    z1 = _random_features(64, 16, seed=6)
    perm = torch.randperm(z1.shape[0], generator=torch.Generator().manual_seed(999))
    z2 = z1[perm]
    out_ab = dare_gram_regularizer(z1, z2)
    assert out_ab.item() == pytest.approx(0.0, abs=1e-5)


def test_cosine_is_scale_invariant() -> None:
    """Scaling Z_s by a positive constant rescales G_s^+ by the same constant
    per column, so column-wise cosines are unchanged (L_cos invariant)."""
    z1 = _random_features(64, 16, seed=7)
    z2 = _random_features(64, 16, seed=8)
    _, info_unit = dare_gram_regularizer(z1, z2, return_components=True)
    _, info_scaled = dare_gram_regularizer(3.0 * z1, z2, return_components=True)
    assert info_scaled.L_cos.item() == pytest.approx(info_unit.L_cos.item(), rel=1e-4, abs=1e-5)


def test_scale_loss_responds_to_rescaling() -> None:
    z1 = _random_features(64, 16, seed=9)
    _, info_same = dare_gram_regularizer(z1, z1.clone(), return_components=True)
    _, info_diff = dare_gram_regularizer(2.0 * z1, z1.clone(), return_components=True)
    assert info_same.L_scale.item() == pytest.approx(0.0, abs=1e-5)
    assert info_diff.L_scale.item() > 0.0


# ---------------------------------------------------------------------------
# Loss composition and autograd
# ---------------------------------------------------------------------------


def test_weighting_linearity() -> None:
    z1 = _random_features(64, 16, seed=10)
    z2 = _random_features(64, 16, seed=11)
    _, info = dare_gram_regularizer(z1, z2, return_components=True)
    expected = 0.02 * info.L_cos + 3e-4 * info.L_scale
    out = dare_gram_regularizer(
        z1, z2, alpha_cos=0.02, gamma_scale=3e-4
    )
    assert out.item() == pytest.approx(expected.item(), rel=1e-5, abs=1e-8)


def test_output_is_scalar() -> None:
    z1 = _random_features(32, 16, seed=12)
    z2 = _random_features(40, 16, seed=13)
    out = dare_gram_regularizer(z1, z2)
    assert out.ndim == 0


def test_gradient_flows() -> None:
    z1 = _random_features(32, 16, seed=14).requires_grad_(True)
    z2 = _random_features(40, 16, seed=15).requires_grad_(True)
    out = dare_gram_regularizer(z1, z2)
    out.backward()
    assert z1.grad is not None
    assert z2.grad is not None
    assert torch.isfinite(z1.grad).all()
    assert torch.isfinite(z2.grad).all()
    # Gradient should not be trivially zero for generic inputs.
    assert z1.grad.abs().sum().item() > 0.0
    assert z2.grad.abs().sum().item() > 0.0


def test_info_is_detached() -> None:
    z1 = _random_features(32, 16, seed=16).requires_grad_(True)
    z2 = _random_features(32, 16, seed=17).requires_grad_(True)
    _, info = dare_gram_regularizer(z1, z2, return_components=True)
    assert not info.L_cos.requires_grad
    assert not info.L_scale.requires_grad


# ---------------------------------------------------------------------------
# Shape and threshold sanity
# ---------------------------------------------------------------------------


def test_small_batch_smaller_than_p() -> None:
    z1 = _random_features(8, 32, seed=18)
    z2 = _random_features(10, 32, seed=19)
    out = dare_gram_regularizer(z1, z2)
    assert torch.isfinite(out)


def test_threshold_controls_k() -> None:
    z = _random_features(64, 16, seed=20)
    _, _, k_low = _gram_pinv_eigendecomp(z, threshold=0.5, eps=1e-6)
    _, _, k_high = _gram_pinv_eigendecomp(z, threshold=0.99, eps=1e-6)
    assert k_low <= k_high


def test_raises_on_dimension_mismatch() -> None:
    z1 = _random_features(16, 16, seed=21)
    z2 = _random_features(16, 20, seed=22)
    with pytest.raises(ValueError, match="feature dimension"):
        dare_gram_regularizer(z1, z2)


def test_raises_on_non_2d() -> None:
    z1 = torch.randn(8)
    z2 = torch.randn(16, 8)
    with pytest.raises(ValueError, match="2D"):
        dare_gram_regularizer(z1, z2)


# ---------------------------------------------------------------------------
# Analytic reference on a tiny hand-crafted example
# ---------------------------------------------------------------------------


def test_analytic_reference_diagonal() -> None:
    """With diagonal Z_s and Z_t sharing the same eigenbasis, L_cos=0 and
    L_scale = || lambda_s - lambda_t ||_2 on the common top-k subspace.

    Take p=2 rows=[2,0]/[0,1] vs [3,0]/[0,1]: lambdas are [4,1] vs [9,1].
    At threshold=0.99, both need k=2.  Expected L_scale = sqrt(25 + 0) = 5.
    """
    z_s = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    z_t = torch.tensor([[3.0, 0.0], [0.0, 1.0]])
    _, info = dare_gram_regularizer(
        z_s, z_t, threshold=0.99, return_components=True
    )
    assert info.k == 2
    assert info.L_cos.item() == pytest.approx(0.0, abs=1e-5)
    assert info.L_scale.item() == pytest.approx(5.0, abs=1e-5)


def test_analytic_reference_rotated_target() -> None:
    """Rotating Z_t changes its eigenbasis so columns of G_t^+ no longer
    align with those of G_s^+, producing a non-zero L_cos."""
    z_s = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    theta = math.radians(30.0)
    R = torch.tensor(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    z_t = z_s @ R
    _, info = dare_gram_regularizer(
        z_s, z_t, threshold=0.99, return_components=True
    )
    # Rotation preserves Z^T Z spectrum, so L_scale should still be 0.
    assert info.L_scale.item() == pytest.approx(0.0, abs=1e-5)
    # But the cosine alignment of G^+ columns must be degraded.
    assert info.L_cos.item() > 0.05
