"""Tests for the B1 trainer (``src/geohead/training/b1.py``).

Specifications under test live in ``docs/design.md`` §8.2.
"""

from __future__ import annotations

import pytest
import torch

from geohead.data.toy import ToyConfig, build_toy_dataset
from geohead.models.encoder import MLPEncoder
from geohead.models.head import LinearHead
from geohead.training.b1 import B1Config, b1_train


def _tiny_corpora(n_per: int = 200, d_x: int = 6, seed: int = 0):
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
# B1Config validation
# ---------------------------------------------------------------------------


def test_b1_config_validation() -> None:
    B1Config(outer_steps=0)
    with pytest.raises(ValueError):
        B1Config(outer_steps=-1)
    with pytest.raises(ValueError):
        B1Config(lr=0)
    with pytest.raises(ValueError):
        B1Config(weight_decay=-1e-3)
    with pytest.raises(ValueError):
        B1Config(support_size=0)
    with pytest.raises(ValueError):
        B1Config(query_size=-1)
    with pytest.raises(ValueError):
        B1Config(batch_source_size=0)
    with pytest.raises(ValueError):
        B1Config(batch_target_size=-1)
    with pytest.raises(ValueError):
        B1Config(log_every=0)


# ---------------------------------------------------------------------------
# b1_train: basic behaviour
# ---------------------------------------------------------------------------


def test_b1_train_updates_params_in_place() -> None:
    encoder, head = _make_models(d_x=6, p=8, seed=0)
    corpora = _tiny_corpora(n_per=150, d_x=6)
    before_enc = [p.detach().clone() for p in encoder.parameters()]
    before_head = head.beta.detach().clone()

    cfg = B1Config(
        outer_steps=20,
        lr=1e-2,
        support_size=16,
        query_size=32,
        batch_source_size=32,
        batch_target_size=32,
        log_every=5,
    )
    g = torch.Generator().manual_seed(0)
    b1_train(encoder, head, corpora, config=cfg, generator=g)

    assert not torch.allclose(head.beta.detach(), before_head)
    after_enc = [p.detach() for p in encoder.parameters()]
    assert any(not torch.allclose(a, b) for a, b in zip(after_enc, before_enc))


def test_b1_train_history_lengths() -> None:
    encoder, head = _make_models()
    corpora = _tiny_corpora(n_per=150, d_x=6)
    cfg = B1Config(
        outer_steps=25,
        lr=1e-2,
        support_size=16,
        query_size=32,
        batch_source_size=32,
        batch_target_size=32,
        log_every=10,
    )
    g = torch.Generator().manual_seed(0)
    hist = b1_train(encoder, head, corpora, config=cfg, generator=g)

    assert hist.step == [10, 20, 25]
    for attr in ("total_loss", "pair"):
        assert len(getattr(hist, attr)) == len(hist.step)
    for i, j in hist.pair:
        assert i in corpora and j in corpora and i != j


def test_b1_train_zero_steps_noop() -> None:
    encoder, head = _make_models(d_x=6, p=8, seed=0)
    corpora = _tiny_corpora(n_per=100, d_x=6)
    before_enc = [p.detach().clone() for p in encoder.parameters()]
    before_head = head.beta.detach().clone()

    hist = b1_train(encoder, head, corpora, config=B1Config(outer_steps=0))

    for a, b in zip(encoder.parameters(), before_enc):
        torch.testing.assert_close(a.detach(), b)
    torch.testing.assert_close(head.beta.detach(), before_head)
    assert hist.step == []


def test_b1_train_total_loss_decreases() -> None:
    encoder, head = _make_models(d_x=6, p=8, seed=0)
    corpora = _tiny_corpora(n_per=300, d_x=6, seed=1)
    cfg = B1Config(
        outer_steps=400,
        lr=1e-2,
        support_size=24,
        query_size=48,
        batch_source_size=48,
        batch_target_size=48,
        log_every=50,
    )
    g = torch.Generator().manual_seed(0)
    hist = b1_train(encoder, head, corpora, config=cfg, generator=g)

    early = sum(hist.total_loss[:2]) / 2
    late = sum(hist.total_loss[-2:]) / 2
    assert late < 0.5 * early, (
        f"B1 total_loss did not decrease enough: {early} -> {late}"
    )


def test_b1_train_deterministic_given_generator() -> None:
    corpora = _tiny_corpora(n_per=150, d_x=6, seed=3)

    enc1, head1 = _make_models(d_x=6, p=8, seed=0)
    enc2, head2 = _make_models(d_x=6, p=8, seed=0)

    cfg = B1Config(outer_steps=30, lr=1e-2, log_every=10)
    g1 = torch.Generator().manual_seed(111)
    g2 = torch.Generator().manual_seed(111)

    b1_train(enc1, head1, corpora, config=cfg, generator=g1)
    b1_train(enc2, head2, corpora, config=cfg, generator=g2)

    torch.testing.assert_close(head1.beta.detach(), head2.beta.detach())
    for p1, p2 in zip(enc1.parameters(), enc2.parameters()):
        torch.testing.assert_close(p1.detach(), p2.detach())


def test_b1_train_rejects_too_few_corpora() -> None:
    encoder, head = _make_models()
    corpora = _tiny_corpora()
    single = {"D1": corpora["D1"]}
    with pytest.raises(ValueError):
        b1_train(encoder, head, single)


def test_b1_train_runs_on_toy_dataset() -> None:
    cfg_data = ToyConfig(seed=0)
    ds = build_toy_dataset(cfg_data, n_train_per_corpus=500)

    encoder = MLPEncoder(d_x=cfg_data.d_x, hidden=(64, 64), p=32)
    head = LinearHead(p=32)
    initial_head = head.beta.detach().clone()

    cfg = B1Config(
        outer_steps=30,
        lr=1e-3,
        support_size=32,
        query_size=64,
        batch_source_size=64,
        batch_target_size=64,
        log_every=10,
    )
    g = torch.Generator().manual_seed(0)
    hist = b1_train(encoder, head, ds.train, config=cfg, generator=g)

    assert len(hist.step) == 3
    assert not torch.allclose(head.beta.detach(), initial_head)
    for i, j in hist.pair:
        assert i in {"D1", "D2", "D3"} and j in {"D1", "D2", "D3"}
