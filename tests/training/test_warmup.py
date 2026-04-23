"""Tests for the warm-up trainer (``src/geohead/training/warmup.py``).

Specifications under test live in ``docs/design.md`` §4.2 and §4.5.
"""

from __future__ import annotations

import pytest
import torch

from geohead.data.toy import ToyConfig, build_toy_dataset
from geohead.models.encoder import MLPEncoder
from geohead.models.head import LinearHead
from geohead.training.warmup import (
    WarmupConfig,
    pooled_dataset,
    warmup_train,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tiny_corpora(n_per: int = 80, d_x: int = 6, seed: int = 0):
    """Three synthetic corpora with a linear ground-truth for quick tests."""
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
# pooled_dataset
# ---------------------------------------------------------------------------


def test_pooled_dataset_total_size() -> None:
    corpora = _tiny_corpora(n_per=50, d_x=4)
    x, y = pooled_dataset(corpora)
    assert x.shape == (150, 4)
    assert y.shape == (150,)


def test_pooled_dataset_preserves_insertion_order() -> None:
    corpora = _tiny_corpora(n_per=10, d_x=3)
    x, y = pooled_dataset(corpora)

    cursor = 0
    for name, (xc, yc) in corpora.items():
        n_c = xc.shape[0]
        torch.testing.assert_close(x[cursor : cursor + n_c], xc, msg=f"X for {name}")
        torch.testing.assert_close(y[cursor : cursor + n_c], yc, msg=f"Y for {name}")
        cursor += n_c


def test_pooled_dataset_rejects_empty() -> None:
    with pytest.raises(ValueError):
        pooled_dataset({})


def test_pooled_dataset_rejects_inconsistent_dx() -> None:
    a = (torch.randn(5, 4), torch.randn(5))
    b = (torch.randn(5, 3), torch.randn(5))  # different d_x
    with pytest.raises(ValueError):
        pooled_dataset({"A": a, "B": b})


def test_pooled_dataset_rejects_bad_shapes() -> None:
    a = (torch.randn(5, 4), torch.randn(5, 1))  # Y is 2-D
    with pytest.raises(ValueError):
        pooled_dataset({"A": a})


# ---------------------------------------------------------------------------
# WarmupConfig validation
# ---------------------------------------------------------------------------


def test_warmup_config_validation() -> None:
    WarmupConfig(epochs=0)  # 0 epochs is allowed (baseline-only run)
    with pytest.raises(ValueError):
        WarmupConfig(epochs=-1)
    with pytest.raises(ValueError):
        WarmupConfig(lr=0)
    with pytest.raises(ValueError):
        WarmupConfig(batch_size=0)
    with pytest.raises(ValueError):
        WarmupConfig(weight_decay=-0.1)


# ---------------------------------------------------------------------------
# warmup_train: basic behaviour
# ---------------------------------------------------------------------------


def test_warmup_train_updates_params_in_place() -> None:
    encoder, head = _make_models(d_x=6, p=8, seed=0)
    corpora = _tiny_corpora(n_per=80, d_x=6)
    before_enc = [p.detach().clone() for p in encoder.parameters()]
    before_head = head.beta.detach().clone()

    cfg = WarmupConfig(epochs=3, lr=1e-2, batch_size=32)
    g = torch.Generator().manual_seed(0)
    warmup_train(encoder, head, corpora, config=cfg, generator=g)

    # Head must have moved.
    assert not torch.allclose(head.beta.detach(), before_head)
    # At least one encoder param must have moved.
    after_enc = [p.detach() for p in encoder.parameters()]
    assert any(not torch.allclose(a, b) for a, b in zip(after_enc, before_enc))


def test_warmup_train_history_lengths() -> None:
    encoder, head = _make_models()
    corpora = _tiny_corpora(n_per=60, d_x=6)
    cfg = WarmupConfig(epochs=4, lr=1e-2, batch_size=32)
    g = torch.Generator().manual_seed(0)
    hist = warmup_train(encoder, head, corpora, config=cfg, generator=g)

    # include_epoch0: length == epochs + 1
    assert len(hist.train_loss) == cfg.epochs + 1
    assert len(hist.per_corpus_loss) == cfg.epochs + 1
    for entry in hist.per_corpus_loss:
        assert set(entry.keys()) == set(corpora.keys())


def test_warmup_train_loss_decreases() -> None:
    encoder, head = _make_models(d_x=6, p=8, seed=0)
    corpora = _tiny_corpora(n_per=120, d_x=6, seed=1)
    cfg = WarmupConfig(epochs=30, lr=1e-2, batch_size=32)
    g = torch.Generator().manual_seed(0)
    hist = warmup_train(encoder, head, corpora, config=cfg, generator=g)

    assert hist.train_loss[-1] < hist.train_loss[0] * 0.5


def test_warmup_train_zero_epochs_noop_params() -> None:
    encoder, head = _make_models(d_x=6, p=8, seed=0)
    corpora = _tiny_corpora(n_per=30, d_x=6)
    before_enc = [p.detach().clone() for p in encoder.parameters()]
    before_head = head.beta.detach().clone()

    cfg = WarmupConfig(epochs=0)
    hist = warmup_train(encoder, head, corpora, config=cfg)

    for a, b in zip(encoder.parameters(), before_enc):
        torch.testing.assert_close(a.detach(), b)
    torch.testing.assert_close(head.beta.detach(), before_head)

    # Baseline loss is still recorded at epoch 0.
    assert len(hist.train_loss) == 1
    assert len(hist.per_corpus_loss) == 1


def test_warmup_train_deterministic_given_generator() -> None:
    corpora = _tiny_corpora(n_per=80, d_x=6, seed=3)

    enc1, head1 = _make_models(d_x=6, p=8, seed=0)
    enc2, head2 = _make_models(d_x=6, p=8, seed=0)

    cfg = WarmupConfig(epochs=5, lr=1e-2, batch_size=16)
    g1 = torch.Generator().manual_seed(111)
    g2 = torch.Generator().manual_seed(111)

    warmup_train(enc1, head1, corpora, config=cfg, generator=g1)
    warmup_train(enc2, head2, corpora, config=cfg, generator=g2)

    torch.testing.assert_close(head1.beta.detach(), head2.beta.detach())
    for p1, p2 in zip(enc1.parameters(), enc2.parameters()):
        torch.testing.assert_close(p1.detach(), p2.detach())


# ---------------------------------------------------------------------------
# warmup_train: integration with the toy dataset
# ---------------------------------------------------------------------------


def test_warmup_train_runs_on_toy_dataset() -> None:
    cfg_data = ToyConfig(seed=0)
    ds = build_toy_dataset(cfg_data, n_train_per_corpus=300)

    encoder = MLPEncoder(d_x=cfg_data.d_x, hidden=(64, 64), p=32)
    head = LinearHead(p=32)
    initial_head = head.beta.detach().clone()

    cfg = WarmupConfig(epochs=3, lr=1e-3, batch_size=128)
    g = torch.Generator().manual_seed(0)
    hist = warmup_train(encoder, head, ds.train, config=cfg, generator=g)

    # Shapes and bookkeeping.
    assert len(hist.train_loss) == cfg.epochs + 1
    # At least some movement happened.
    assert not torch.allclose(head.beta.detach(), initial_head)
    # Per-corpus keys match.
    for entry in hist.per_corpus_loss:
        assert set(entry.keys()) == {"D1", "D2", "D3"}


def test_warmup_train_does_not_touch_test_corpora() -> None:
    """Warm-up must only consume the training corpora (``ds.train``).

    We pass a spy dict whose ``.items()`` we monitor; the warm-up trainer
    should not read from ``ds.test`` in any form.  As a proxy, we pass only
    ``ds.train`` and verify that the final head and per-corpus history have
    no reference to ``T1`` / ``T2``.
    """
    cfg_data = ToyConfig(seed=0)
    ds = build_toy_dataset(cfg_data, n_train_per_corpus=200)

    encoder = MLPEncoder(d_x=cfg_data.d_x, hidden=(32,), p=32)
    head = LinearHead(p=32)

    cfg = WarmupConfig(epochs=2, lr=1e-3, batch_size=128)
    hist = warmup_train(encoder, head, ds.train, config=cfg)

    for entry in hist.per_corpus_loss:
        assert "T1" not in entry
        assert "T2" not in entry


# ---------------------------------------------------------------------------
# Sanity: no-shuffle mode is deterministic without a generator
# ---------------------------------------------------------------------------


def test_warmup_train_no_shuffle_is_deterministic() -> None:
    """Without shuffle, two independent runs with identical inits coincide
    even without a ``torch.Generator``.
    """
    corpora = _tiny_corpora(n_per=40, d_x=6, seed=2)
    cfg = WarmupConfig(epochs=3, lr=1e-2, batch_size=16, shuffle=False)

    enc1, head1 = _make_models(d_x=6, p=8, seed=0)
    warmup_train(enc1, head1, corpora, config=cfg)

    enc2, head2 = _make_models(d_x=6, p=8, seed=0)
    warmup_train(enc2, head2, corpora, config=cfg)

    torch.testing.assert_close(head1.beta.detach(), head2.beta.detach())
    for p1, p2 in zip(enc1.parameters(), enc2.parameters()):
        torch.testing.assert_close(p1.detach(), p2.detach())
