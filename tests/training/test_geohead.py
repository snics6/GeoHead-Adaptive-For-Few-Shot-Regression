"""Tests for the GeoHead bilevel meta-trainer
(``src/geohead/training/geohead.py``).

Specifications under test live in ``docs/design.md`` §4.3–§4.5 and §10.
"""

from __future__ import annotations

import pytest
import torch

from geohead.adaptation.test_time import inner_rule_adapt
from geohead.data.episode import EpisodeSizes
from geohead.data.toy import ToyConfig, build_toy_dataset
from geohead.losses.head_reg import second_moment
from geohead.models.encoder import MLPEncoder
from geohead.models.head import LinearHead
from geohead.training.geohead import (
    GeoHeadConfig,
    geohead_train,
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
# GeoHeadConfig validation
# ---------------------------------------------------------------------------


def test_geohead_config_defaults_and_episode_sizes() -> None:
    cfg = GeoHeadConfig()
    assert cfg.inner_steps == 5
    assert cfg.outer_steps == 10_000
    assert cfg.lambda_D == 1.0
    sizes = cfg.episode_sizes()
    assert isinstance(sizes, EpisodeSizes)
    assert sizes.support == cfg.support_size
    assert sizes.query == cfg.query_size
    assert sizes.batch_source == cfg.batch_source_size
    assert sizes.batch_target == cfg.batch_target_size


def test_geohead_config_validation() -> None:
    # outer_steps == 0 is allowed (no-op run).
    GeoHeadConfig(outer_steps=0)
    # inner_steps == 0 is allowed (β' = β_0; §4.3 degenerate case).
    GeoHeadConfig(inner_steps=0)

    with pytest.raises(ValueError):
        GeoHeadConfig(outer_steps=-1)
    with pytest.raises(ValueError):
        GeoHeadConfig(outer_lr=0)
    with pytest.raises(ValueError):
        GeoHeadConfig(weight_decay=-1e-3)
    with pytest.raises(ValueError):
        GeoHeadConfig(inner_steps=-1)
    with pytest.raises(ValueError):
        GeoHeadConfig(inner_lr=0)
    with pytest.raises(ValueError):
        GeoHeadConfig(lambda_h=-0.1)
    with pytest.raises(ValueError):
        GeoHeadConfig(head_reg_eps=-1e-6)
    with pytest.raises(ValueError):
        GeoHeadConfig(lambda_D=-1.0)
    with pytest.raises(ValueError):
        GeoHeadConfig(alpha_cos=-0.1)
    with pytest.raises(ValueError):
        GeoHeadConfig(gamma_scale=-1e-4)
    with pytest.raises(ValueError):
        GeoHeadConfig(threshold=0.0)
    with pytest.raises(ValueError):
        GeoHeadConfig(threshold=1.1)
    with pytest.raises(ValueError):
        GeoHeadConfig(dare_eps=0.0)
    with pytest.raises(ValueError):
        GeoHeadConfig(support_size=0)
    with pytest.raises(ValueError):
        GeoHeadConfig(query_size=-1)
    with pytest.raises(ValueError):
        GeoHeadConfig(batch_source_size=0)
    with pytest.raises(ValueError):
        GeoHeadConfig(batch_target_size=0)
    with pytest.raises(ValueError):
        GeoHeadConfig(log_every=0)


# ---------------------------------------------------------------------------
# geohead_train: basic behaviour
# ---------------------------------------------------------------------------


def test_geohead_train_updates_params_in_place() -> None:
    encoder, head = _make_models(d_x=6, p=8, seed=0)
    corpora = _tiny_corpora(n_per=200, d_x=6)
    before_enc = [p.detach().clone() for p in encoder.parameters()]
    before_head = head.beta.detach().clone()

    cfg = GeoHeadConfig(
        outer_steps=20,
        outer_lr=1e-2,
        inner_steps=3,
        inner_lr=0.05,
        support_size=16,
        query_size=32,
        batch_source_size=32,
        batch_target_size=32,
        log_every=5,
    )
    g = torch.Generator().manual_seed(0)
    geohead_train(encoder, head, corpora, config=cfg, generator=g)

    assert not torch.allclose(head.beta.detach(), before_head)
    after_enc = [p.detach() for p in encoder.parameters()]
    assert any(not torch.allclose(a, b) for a, b in zip(after_enc, before_enc))


def test_geohead_train_history_lengths() -> None:
    encoder, head = _make_models()
    corpora = _tiny_corpora(n_per=200, d_x=6)
    cfg = GeoHeadConfig(
        outer_steps=25,
        outer_lr=1e-2,
        inner_steps=2,
        support_size=16,
        query_size=32,
        batch_source_size=32,
        batch_target_size=32,
        log_every=10,
    )
    g = torch.Generator().manual_seed(0)
    hist = geohead_train(encoder, head, corpora, config=cfg, generator=g)

    # Logged at steps {10, 20} plus the final step 25 -> 3 records.
    assert hist.step == [10, 20, 25]
    for attr in (
        "total_loss",
        "qry_loss",
        "src_loss",
        "cos_loss",
        "scale_loss",
        "inner_delta_norm",
        "pair",
    ):
        assert len(getattr(hist, attr)) == len(hist.step)
    for i, j in hist.pair:
        assert i in corpora and j in corpora and i != j
    # Inner update is non-trivial (β' ≠ β_0 for positive inner_steps,
    # non-degenerate features).
    assert all(d >= 0.0 for d in hist.inner_delta_norm)


def test_geohead_train_zero_outer_steps_noop() -> None:
    encoder, head = _make_models(d_x=6, p=8, seed=0)
    corpora = _tiny_corpora(n_per=100, d_x=6)
    before_enc = [p.detach().clone() for p in encoder.parameters()]
    before_head = head.beta.detach().clone()

    cfg = GeoHeadConfig(outer_steps=0)
    hist = geohead_train(encoder, head, corpora, config=cfg)

    for a, b in zip(encoder.parameters(), before_enc):
        torch.testing.assert_close(a.detach(), b)
    torch.testing.assert_close(head.beta.detach(), before_head)
    assert hist.step == []


def test_geohead_train_inner_steps_zero_still_trains() -> None:
    """With ``inner_steps=0``, β' = β_0, so the outer loss reduces to the
    §8.1 objective evaluated on ``(Q, B_i, B_j)`` plus a redundant query
    term.  Training should still progress without errors.
    """
    encoder, head = _make_models(d_x=6, p=8, seed=0)
    corpora = _tiny_corpora(n_per=200, d_x=6)
    before_head = head.beta.detach().clone()

    cfg = GeoHeadConfig(
        outer_steps=15,
        outer_lr=1e-2,
        inner_steps=0,
        support_size=16,
        query_size=32,
        batch_source_size=32,
        batch_target_size=32,
        log_every=5,
    )
    g = torch.Generator().manual_seed(0)
    hist = geohead_train(encoder, head, corpora, config=cfg, generator=g)

    assert not torch.allclose(head.beta.detach(), before_head)
    # With K=0 the delta β' - β_0 is exactly zero.
    for d in hist.inner_delta_norm:
        assert d == pytest.approx(0.0, abs=1e-6)


def test_geohead_train_query_loss_decreases() -> None:
    encoder, head = _make_models(d_x=6, p=8, seed=0)
    corpora = _tiny_corpora(n_per=400, d_x=6, seed=1)
    cfg = GeoHeadConfig(
        outer_steps=400,
        outer_lr=1e-2,
        inner_steps=3,
        inner_lr=0.1,
        lambda_h=0.1,
        lambda_D=1.0,
        support_size=32,
        query_size=48,
        batch_source_size=48,
        batch_target_size=48,
        log_every=50,
    )
    g = torch.Generator().manual_seed(0)
    hist = geohead_train(encoder, head, corpora, config=cfg, generator=g)

    # Average of early and late logs to smooth single-step noise.
    early = sum(hist.qry_loss[:2]) / 2
    late = sum(hist.qry_loss[-2:]) / 2
    assert late < 0.5 * early, f"qry_loss did not decrease enough: {early} -> {late}"


def test_geohead_train_deterministic_given_generator() -> None:
    corpora = _tiny_corpora(n_per=200, d_x=6, seed=3)

    enc1, head1 = _make_models(d_x=6, p=8, seed=0)
    enc2, head2 = _make_models(d_x=6, p=8, seed=0)

    cfg = GeoHeadConfig(
        outer_steps=20,
        outer_lr=1e-2,
        inner_steps=3,
        support_size=16,
        query_size=32,
        batch_source_size=32,
        batch_target_size=32,
        log_every=10,
    )
    g1 = torch.Generator().manual_seed(111)
    g2 = torch.Generator().manual_seed(111)

    geohead_train(enc1, head1, corpora, config=cfg, generator=g1)
    geohead_train(enc2, head2, corpora, config=cfg, generator=g2)

    torch.testing.assert_close(head1.beta.detach(), head2.beta.detach())
    for p1, p2 in zip(enc1.parameters(), enc2.parameters()):
        torch.testing.assert_close(p1.detach(), p2.detach())


def test_geohead_train_rejects_too_few_corpora() -> None:
    encoder, head = _make_models()
    corpora = _tiny_corpora()
    single = {"D1": corpora["D1"]}
    with pytest.raises(ValueError):
        geohead_train(encoder, head, single)


def test_geohead_train_rejects_bad_shapes() -> None:
    encoder, head = _make_models()
    bad = {
        "D1": (torch.randn(50, 6), torch.randn(50, 1)),  # Y is 2-D
        "D2": (torch.randn(50, 6), torch.randn(50)),
    }
    with pytest.raises(ValueError):
        geohead_train(encoder, head, bad, config=GeoHeadConfig(outer_steps=1))


# ---------------------------------------------------------------------------
# Gradient flow: β_0 must receive gradient through BOTH the inner loop
# (via β') and the direct L_src term.
# ---------------------------------------------------------------------------


def test_geohead_train_gradient_flows_to_both_encoder_and_head() -> None:
    """One outer step must populate gradients on every parameter.

    The outer loss ``L_qry + λ_D (L_src + α L_cos + γ L_scale)`` depends
    on every encoder parameter (via ``z_S, z_Q, z_{B_i}, z_{B_j}``) and
    on the head weight (via ``β_0`` in both the inner loop and the
    explicit ``z_{B_i} @ β_0`` term).  A single backward therefore sets
    ``.grad`` on every parameter to a non-zero, finite tensor.
    """
    encoder, head = _make_models(d_x=6, p=8, seed=0)
    corpora = _tiny_corpora(n_per=200, d_x=6)

    cfg = GeoHeadConfig(
        outer_steps=1,
        outer_lr=1e-3,
        inner_steps=2,
        inner_lr=0.05,
        support_size=16,
        query_size=32,
        batch_source_size=32,
        batch_target_size=32,
        log_every=1,
    )
    g = torch.Generator().manual_seed(0)
    geohead_train(encoder, head, corpora, config=cfg, generator=g)

    # After one Adam step, the optimizer has cleared grads (set_to_none).
    # Re-run a manual forward-backward on a fresh episode to inspect grads.
    from geohead.data.episode import sample_episode, sample_random_pair
    from geohead.losses.dare_gram import dare_gram_regularizer

    g2 = torch.Generator().manual_seed(42)
    i, j = sample_random_pair(list(corpora.keys()), generator=g2)
    ep = sample_episode(corpora, i, j, cfg.episode_sizes(), generator=g2)

    z_s = encoder(ep.support_x)
    z_q = encoder(ep.query_x)
    z_bi = encoder(ep.batch_source_x)
    z_bj = encoder(ep.batch_target_x)

    sigma_s = second_moment(z_s)
    beta_0 = head.beta
    beta_prime = inner_rule_adapt(
        z_s,
        ep.support_y,
        beta_0=beta_0,
        sigma=sigma_s,
        lambda_h=cfg.lambda_h,
        eta=cfg.inner_lr,
        steps=cfg.inner_steps,
        eps=cfg.head_reg_eps,
        create_graph=True,
    )
    loss_qry = ((z_q @ beta_prime - ep.query_y) ** 2).mean()
    loss_src = ((z_bi @ beta_0 - ep.batch_source_y) ** 2).mean()
    loss_reg = dare_gram_regularizer(
        z_bi,
        z_bj,
        alpha_cos=cfg.alpha_cos,
        gamma_scale=cfg.gamma_scale,
        threshold=cfg.threshold,
        eps=cfg.dare_eps,
    )
    loss = loss_qry + cfg.lambda_D * (loss_src + loss_reg)

    for p in encoder.parameters():
        if p.grad is not None:
            p.grad.zero_()
    for p in head.parameters():
        if p.grad is not None:
            p.grad.zero_()
    loss.backward()

    for p in encoder.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
        assert p.grad.abs().sum() > 0
    for p in head.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
        assert p.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Train/test consistency: the inner rule used during training is
# identical to the ``inner`` test-time adapter (§8.3).
# ---------------------------------------------------------------------------


def test_geohead_training_inner_rule_matches_test_time_inner_rule() -> None:
    """Given the same ``(θ, β_0)`` and the same support ``(z_S, y_S)``,
    calling :func:`inner_rule_adapt` with ``create_graph=True`` (training
    use) and ``create_graph=False`` (test-time use) must produce the
    *same* ``β'`` up to autograd detachment.  This is a contract of the
    design (``design.md`` §8.3: "training の inner rule をそのまま適用").
    """
    encoder, head = _make_models(d_x=6, p=8, seed=0)
    corpora = _tiny_corpora(n_per=200, d_x=6)

    # Do a few outer steps so (θ, β_0) are not at initialisation.
    cfg = GeoHeadConfig(
        outer_steps=10,
        outer_lr=1e-2,
        inner_steps=3,
        inner_lr=0.05,
        support_size=16,
        query_size=32,
        batch_source_size=32,
        batch_target_size=32,
        log_every=5,
    )
    g = torch.Generator().manual_seed(0)
    geohead_train(encoder, head, corpora, config=cfg, generator=g)

    encoder.eval()
    # Draw a support set from one corpus.
    x_s, y_s = corpora["D1"]
    x_s = x_s[:24]
    y_s = y_s[:24]
    with torch.no_grad():
        z_s = encoder(x_s)
        sigma_s = second_moment(z_s)

    beta_0 = head.beta.detach().clone()

    beta_train = inner_rule_adapt(
        z_s,
        y_s,
        beta_0=beta_0.clone().requires_grad_(True),
        sigma=sigma_s,
        lambda_h=cfg.lambda_h,
        eta=cfg.inner_lr,
        steps=cfg.inner_steps,
        eps=cfg.head_reg_eps,
        create_graph=True,
    ).detach()

    beta_test = inner_rule_adapt(
        z_s,
        y_s,
        beta_0=beta_0,
        sigma=sigma_s,
        lambda_h=cfg.lambda_h,
        eta=cfg.inner_lr,
        steps=cfg.inner_steps,
        eps=cfg.head_reg_eps,
        create_graph=False,
    )

    torch.testing.assert_close(beta_train, beta_test, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Integration with the toy dataset
# ---------------------------------------------------------------------------


def test_geohead_train_runs_on_toy_dataset() -> None:
    cfg_data = ToyConfig(seed=0)
    ds = build_toy_dataset(cfg_data, n_train_per_corpus=500)

    encoder = MLPEncoder(d_x=cfg_data.d_x, hidden=(64, 64), p=32)
    head = LinearHead(p=32)
    initial_head = head.beta.detach().clone()

    cfg = GeoHeadConfig(
        outer_steps=20,
        outer_lr=1e-3,
        inner_steps=3,
        inner_lr=0.1,
        support_size=32,
        query_size=48,
        batch_source_size=48,
        batch_target_size=48,
        log_every=10,
    )
    g = torch.Generator().manual_seed(0)
    hist = geohead_train(encoder, head, ds.train, config=cfg, generator=g)

    assert len(hist.step) == 2  # steps 10, 20
    assert not torch.allclose(head.beta.detach(), initial_head)


def test_geohead_train_does_not_touch_test_corpora() -> None:
    cfg_data = ToyConfig(seed=0)
    ds = build_toy_dataset(cfg_data, n_train_per_corpus=300)

    encoder = MLPEncoder(d_x=cfg_data.d_x, hidden=(32,), p=16)
    head = LinearHead(p=16)

    cfg = GeoHeadConfig(
        outer_steps=5,
        outer_lr=1e-3,
        inner_steps=2,
        support_size=16,
        query_size=24,
        batch_source_size=24,
        batch_target_size=24,
        log_every=1,
    )
    g = torch.Generator().manual_seed(0)
    hist = geohead_train(encoder, head, ds.train, config=cfg, generator=g)

    for i, j in hist.pair:
        assert i not in {"T1", "T2"}
        assert j not in {"T1", "T2"}
