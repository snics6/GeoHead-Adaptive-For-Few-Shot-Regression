"""Tests for the baseline trainer (``src/geohead/training/baseline.py``).

Specifications under test live in ``docs/design.md`` §8.1 and §10.
"""

from __future__ import annotations

import pytest
import torch

from geohead.adaptation.test_time import ridge_adapt
from geohead.data.toy import ToyConfig, build_toy_dataset
from geohead.models.encoder import MLPEncoder
from geohead.models.head import LinearHead
from geohead.training.baseline import (
    BaselineConfig,
    baseline_train,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tiny_corpora(n_per: int = 200, d_x: int = 6, seed: int = 0):
    """Three synthetic linear-GT corpora large enough for DARE-GRAM SVD."""
    g = torch.Generator().manual_seed(seed)
    true_w = torch.randn(d_x, generator=g)
    corpora: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for k, name in enumerate(("D1", "D2", "D3")):
        gk = torch.Generator().manual_seed(seed + k + 1)
        x = torch.randn(n_per, d_x, generator=gk)
        y = x @ true_w + 0.01 * torch.randn(n_per, generator=gk)
        corpora[name] = (x, y)
    return corpora


def _make_models(d_x: int = 6, p: int = 8, seed: int = 0):
    torch.manual_seed(seed)
    encoder = MLPEncoder(d_x=d_x, hidden=(16,), p=p)
    head = LinearHead(p=p)
    return encoder, head


# ---------------------------------------------------------------------------
# BaselineConfig validation
# ---------------------------------------------------------------------------


def test_baseline_config_validation() -> None:
    # Zero outer_steps is permitted (no-op run that still records nothing).
    BaselineConfig(outer_steps=0)
    with pytest.raises(ValueError):
        BaselineConfig(outer_steps=-1)
    with pytest.raises(ValueError):
        BaselineConfig(lr=0)
    with pytest.raises(ValueError):
        BaselineConfig(batch_size_source=0)
    with pytest.raises(ValueError):
        BaselineConfig(batch_size_target=-1)
    with pytest.raises(ValueError):
        BaselineConfig(alpha_cos=-0.1)
    with pytest.raises(ValueError):
        BaselineConfig(gamma_scale=-1e-4)
    with pytest.raises(ValueError):
        BaselineConfig(threshold=0.0)
    with pytest.raises(ValueError):
        BaselineConfig(threshold=1.1)
    with pytest.raises(ValueError):
        BaselineConfig(dare_eps=0.0)
    with pytest.raises(ValueError):
        BaselineConfig(weight_decay=-1e-3)
    with pytest.raises(ValueError):
        BaselineConfig(log_every=0)


# ---------------------------------------------------------------------------
# baseline_train: basic behaviour
# ---------------------------------------------------------------------------


def test_baseline_train_updates_params_in_place() -> None:
    encoder, head = _make_models(d_x=6, p=8, seed=0)
    corpora = _tiny_corpora(n_per=150, d_x=6)
    before_enc = [p.detach().clone() for p in encoder.parameters()]
    before_head = head.beta.detach().clone()

    cfg = BaselineConfig(
        outer_steps=20,
        lr=1e-2,
        batch_size_source=32,
        batch_size_target=32,
        log_every=5,
    )
    g = torch.Generator().manual_seed(0)
    baseline_train(encoder, head, corpora, config=cfg, generator=g)

    assert not torch.allclose(head.beta.detach(), before_head)
    after_enc = [p.detach() for p in encoder.parameters()]
    assert any(not torch.allclose(a, b) for a, b in zip(after_enc, before_enc))


def test_baseline_train_history_lengths() -> None:
    encoder, head = _make_models()
    corpora = _tiny_corpora(n_per=150, d_x=6)
    cfg = BaselineConfig(outer_steps=25, lr=1e-2, log_every=10)
    g = torch.Generator().manual_seed(0)
    hist = baseline_train(encoder, head, corpora, config=cfg, generator=g)

    # Logged at steps {10, 20} plus the final step 25 -> 3 records.
    assert hist.step == [10, 20, 25]
    for attr in ("total_loss", "src_loss", "cos_loss", "scale_loss", "pair"):
        assert len(getattr(hist, attr)) == len(hist.step)
    for i, j in hist.pair:
        assert i in corpora and j in corpora and i != j


def test_baseline_train_zero_steps_noop() -> None:
    encoder, head = _make_models(d_x=6, p=8, seed=0)
    corpora = _tiny_corpora(n_per=100, d_x=6)
    before_enc = [p.detach().clone() for p in encoder.parameters()]
    before_head = head.beta.detach().clone()

    cfg = BaselineConfig(outer_steps=0)
    hist = baseline_train(encoder, head, corpora, config=cfg)

    for a, b in zip(encoder.parameters(), before_enc):
        torch.testing.assert_close(a.detach(), b)
    torch.testing.assert_close(head.beta.detach(), before_head)
    assert hist.step == []


def test_baseline_train_total_loss_decreases() -> None:
    encoder, head = _make_models(d_x=6, p=8, seed=0)
    corpora = _tiny_corpora(n_per=300, d_x=6, seed=1)
    cfg = BaselineConfig(
        outer_steps=400,
        lr=1e-2,
        batch_size_source=48,
        batch_size_target=48,
        log_every=50,
    )
    g = torch.Generator().manual_seed(0)
    hist = baseline_train(encoder, head, corpora, config=cfg, generator=g)

    # Average of early and late logs to smooth out single-step noise.
    early = sum(hist.src_loss[:2]) / 2
    late = sum(hist.src_loss[-2:]) / 2
    assert late < 0.5 * early, f"src_loss did not decrease enough: {early} -> {late}"


def test_baseline_train_deterministic_given_generator() -> None:
    corpora = _tiny_corpora(n_per=150, d_x=6, seed=3)

    enc1, head1 = _make_models(d_x=6, p=8, seed=0)
    enc2, head2 = _make_models(d_x=6, p=8, seed=0)

    cfg = BaselineConfig(outer_steps=30, lr=1e-2, log_every=10)
    g1 = torch.Generator().manual_seed(111)
    g2 = torch.Generator().manual_seed(111)

    baseline_train(enc1, head1, corpora, config=cfg, generator=g1)
    baseline_train(enc2, head2, corpora, config=cfg, generator=g2)

    torch.testing.assert_close(head1.beta.detach(), head2.beta.detach())
    for p1, p2 in zip(enc1.parameters(), enc2.parameters()):
        torch.testing.assert_close(p1.detach(), p2.detach())


def test_baseline_train_rejects_too_few_corpora() -> None:
    encoder, head = _make_models()
    corpora = _tiny_corpora()
    single = {"D1": corpora["D1"]}
    with pytest.raises(ValueError):
        baseline_train(encoder, head, single)


def test_baseline_train_rejects_bad_shapes() -> None:
    encoder, head = _make_models()
    bad = {
        "D1": (torch.randn(50, 6), torch.randn(50, 1)),  # Y is 2-D
        "D2": (torch.randn(50, 6), torch.randn(50)),
    }
    with pytest.raises(ValueError):
        baseline_train(encoder, head, bad, config=BaselineConfig(outer_steps=1))


# ---------------------------------------------------------------------------
# baseline_train: integration with the toy dataset
# ---------------------------------------------------------------------------


def test_baseline_train_runs_on_toy_dataset() -> None:
    cfg_data = ToyConfig(seed=0)
    ds = build_toy_dataset(cfg_data, n_train_per_corpus=500)

    encoder = MLPEncoder(d_x=cfg_data.d_x, hidden=(64, 64), p=32)
    head = LinearHead(p=32)
    initial_head = head.beta.detach().clone()

    cfg = BaselineConfig(
        outer_steps=30,
        lr=1e-3,
        batch_size_source=64,
        batch_size_target=64,
        log_every=10,
    )
    g = torch.Generator().manual_seed(0)
    hist = baseline_train(encoder, head, ds.train, config=cfg, generator=g)

    assert len(hist.step) == 3
    assert not torch.allclose(head.beta.detach(), initial_head)
    # All pairs drawn from the training corpora only.
    for i, j in hist.pair:
        assert i in {"D1", "D2", "D3"} and j in {"D1", "D2", "D3"}


def test_baseline_train_does_not_touch_test_corpora() -> None:
    cfg_data = ToyConfig(seed=0)
    ds = build_toy_dataset(cfg_data, n_train_per_corpus=300)

    encoder = MLPEncoder(d_x=cfg_data.d_x, hidden=(32,), p=16)
    head = LinearHead(p=16)

    cfg = BaselineConfig(outer_steps=5, lr=1e-3, log_every=1)
    g = torch.Generator().manual_seed(0)
    hist = baseline_train(encoder, head, ds.train, config=cfg, generator=g)

    for i, j in hist.pair:
        assert i not in {"T1", "T2"}
        assert j not in {"T1", "T2"}


# ---------------------------------------------------------------------------
# Integration with test-time adaptation (§7)
# ---------------------------------------------------------------------------


def test_baseline_output_is_usable_for_test_time_ridge_adapt() -> None:
    """Pipeline-level contract: the post-baseline ``(encoder, head.beta)``
    can be piped through the test-time adaptation API without shape or
    dtype errors, and produces a non-trivial update distinct from β_0.

    Whether ``ridge_adapt`` actually *improves* MSE over β_0 is a separate
    empirical claim covered by ``tests/adaptation/test_test_time.py``
    (see §7 of ``docs/design.md``); that claim depends on representation
    quality and is not a contract of :func:`baseline_train`.
    """
    cfg_data = ToyConfig(seed=0)
    ds = build_toy_dataset(cfg_data, n_train_per_corpus=400)

    encoder = MLPEncoder(d_x=cfg_data.d_x, hidden=(32,), p=16)
    head = LinearHead(p=16)

    cfg = BaselineConfig(
        outer_steps=50,
        lr=1e-3,
        batch_size_source=64,
        batch_size_target=64,
        log_every=25,
    )
    g = torch.Generator().manual_seed(0)
    baseline_train(encoder, head, ds.train, config=cfg, generator=g)

    beta_0 = head.beta.detach().clone()
    encoder.eval()

    # ``ds.test[name]`` is ``{"support": (X, Y), "query": (X, Y)}``
    # (see :func:`build_toy_dataset`), so the split is already performed.
    x_support, y_support = ds.test["T1"]["support"]
    x_query, y_query = ds.test["T1"]["query"]

    k_shot = 32
    x_support = x_support[:k_shot]
    y_support = y_support[:k_shot]

    with torch.no_grad():
        z_support = encoder(x_support)
        z_query = encoder(x_query)

    beta_hat = ridge_adapt(z_support, y_support, beta_0=beta_0, lambda_=1e-2)

    # Shape / dtype contract.
    assert beta_hat.shape == beta_0.shape
    assert beta_hat.dtype == beta_0.dtype
    # ``ridge_adapt`` must not no-op at a strictly positive ``lambda_`` on
    # non-degenerate features: β̂ should move away from β_0.
    assert not torch.allclose(beta_hat, beta_0)

    with torch.no_grad():
        mse_beta0 = ((z_query @ beta_0 - y_query) ** 2).mean().item()
        mse_adapt = ((z_query @ beta_hat - y_query) ** 2).mean().item()

    # Finite-valued forward-pass contract.
    assert torch.isfinite(torch.tensor([mse_beta0, mse_adapt])).all()
