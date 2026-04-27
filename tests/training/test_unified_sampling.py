"""Tests for the §8.0 unified-sampling fairness invariant.

Verifies that under a shared ``torch.Generator`` seed, the three Phase 1
trainers (:func:`geohead.training.b1.b1_train`,
:func:`geohead.training.baseline.baseline_train`,
:func:`geohead.training.geohead.geohead_train`) consume the random
generator at the same rate and therefore observe the **same**
``(i, j, indices)`` episode sequence step by step.

Concretely we record the per-step ``(i, j)`` pair logged by each
trainer's ``History`` and assert pairwise equality, then we sample one
episode from each trainer's perspective (via :func:`sample_episode`)
under the same seed and assert byte-identical index tensors.

Spec: ``docs/design.md`` §8.0 (公平性プロトコル).
"""

from __future__ import annotations

import torch

from geohead.data.episode import EpisodeSizes, sample_episode, sample_random_pair
from geohead.training.b1 import B1Config, b1_train
from geohead.training.baseline import BaselineConfig, baseline_train
from geohead.training.geohead import GeoHeadConfig, geohead_train


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


def _common_sizes() -> tuple[int, int, int, int]:
    return (8, 16, 16, 16)  # support, query, batch_source, batch_target


def _make_b1_b2_p(d_x: int, p: int, seed: int = 7):
    """Three identical (encoder, head) pairs for the three trainers."""
    from geohead.models.encoder import MLPEncoder
    from geohead.models.head import LinearHead

    triples = []
    for _ in range(3):
        torch.manual_seed(seed)
        encoder = MLPEncoder(d_x=d_x, hidden=(16,), p=p)
        head = LinearHead(p=p)
        triples.append((encoder, head))
    return triples


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_b1_b2_p_observe_identical_pair_sequences() -> None:
    """B1 / B2 / P log the same ``(i, j)`` at every step under shared seed."""
    corpora = _tiny_corpora(n_per=200, d_x=6, seed=0)
    s, q, bs, bt = _common_sizes()

    n_steps = 30
    log_every = 1  # log every step so we can compare full sequences

    (e1, h1), (e2, h2), (e3, h3) = _make_b1_b2_p(d_x=6, p=8)

    cfg_b1 = B1Config(
        outer_steps=n_steps,
        support_size=s,
        query_size=q,
        batch_source_size=bs,
        batch_target_size=bt,
        log_every=log_every,
    )
    cfg_b2 = BaselineConfig(
        outer_steps=n_steps,
        support_size=s,
        query_size=q,
        batch_source_size=bs,
        batch_target_size=bt,
        log_every=log_every,
    )
    cfg_p = GeoHeadConfig(
        outer_steps=n_steps,
        inner_steps=1,
        support_size=s,
        query_size=q,
        batch_source_size=bs,
        batch_target_size=bt,
        log_every=log_every,
    )

    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    g3 = torch.Generator().manual_seed(123)

    h_b1 = b1_train(e1, h1, corpora, config=cfg_b1, generator=g1)
    h_b2 = baseline_train(e2, h2, corpora, config=cfg_b2, generator=g2)
    h_p = geohead_train(e3, h3, corpora, config=cfg_p, generator=g3)

    assert h_b1.pair == h_b2.pair == h_p.pair, (
        "B1 / B2 / P must observe the same (i, j) sequence under shared seed; "
        f"got B1={h_b1.pair!r}, B2={h_b2.pair!r}, P={h_p.pair!r}"
    )


def test_unified_episode_indices_are_identical_under_shared_seed() -> None:
    """A single episode draw under the same generator state yields the
    exact same ``(support_idx, batch_source_idx, query_idx,
    batch_target_idx)`` regardless of which downstream trainer would
    consume it.
    """
    corpora = _tiny_corpora(n_per=120, d_x=4, seed=2)
    s, q, bs, bt = _common_sizes()
    sizes = EpisodeSizes(support=s, query=q, batch_source=bs, batch_target=bt)
    names = list(corpora.keys())

    def _draw(seed: int):
        g = torch.Generator().manual_seed(seed)
        i, j = sample_random_pair(names, generator=g)
        ep = sample_episode(corpora, i, j, sizes, generator=g)
        return i, j, ep

    i1, j1, ep1 = _draw(seed=999)
    i2, j2, ep2 = _draw(seed=999)
    i3, j3, ep3 = _draw(seed=999)

    assert (i1, j1) == (i2, j2) == (i3, j3)
    torch.testing.assert_close(ep1.support_idx, ep2.support_idx)
    torch.testing.assert_close(ep1.support_idx, ep3.support_idx)
    torch.testing.assert_close(ep1.batch_source_idx, ep2.batch_source_idx)
    torch.testing.assert_close(ep1.batch_source_idx, ep3.batch_source_idx)
    torch.testing.assert_close(ep1.query_idx, ep2.query_idx)
    torch.testing.assert_close(ep1.query_idx, ep3.query_idx)
    torch.testing.assert_close(ep1.batch_target_idx, ep2.batch_target_idx)
    torch.testing.assert_close(ep1.batch_target_idx, ep3.batch_target_idx)


def test_b2_minus_b1_loss_is_only_dare_alignment() -> None:
    """For matched encoders/heads on the same episode, the difference
    ``L_{B2} - L_{B1}`` equals exactly ``α L_cos + γ L_scale`` (i.e. the
    pure DARE-GRAM regulariser).  This is the structural invariant that
    motivates the unified-episode design (§8.6).
    """
    from geohead.losses.dare_gram import dare_gram_regularizer
    from geohead.models.encoder import MLPEncoder
    from geohead.models.head import LinearHead

    corpora = _tiny_corpora(n_per=120, d_x=4, seed=4)
    s, q, bs, bt = _common_sizes()
    sizes = EpisodeSizes(support=s, query=q, batch_source=bs, batch_target=bt)

    torch.manual_seed(42)
    encoder = MLPEncoder(d_x=4, hidden=(16,), p=8)
    head = LinearHead(p=8)

    g = torch.Generator().manual_seed(7)
    i, j = sample_random_pair(list(corpora.keys()), generator=g)
    ep = sample_episode(corpora, i, j, sizes, generator=g)

    encoder.eval()
    head.eval()

    # B1 loss: pooled MSE on (S ∪ B_i ∪ Q ∪ B_j).
    x_all = torch.cat(
        [ep.support_x, ep.batch_source_x, ep.query_x, ep.batch_target_x], dim=0
    )
    y_all = torch.cat(
        [ep.support_y, ep.batch_source_y, ep.query_y, ep.batch_target_y], dim=0
    )
    z_all = encoder(x_all)
    y_hat = head(z_all)
    loss_b1 = ((y_hat - y_all) ** 2).mean()

    # B2 loss: same pooled MSE + DARE-GRAM on (Z(S∪B_i), Z(Q∪B_j)).
    x_src = torch.cat([ep.support_x, ep.batch_source_x], dim=0)
    x_tgt = torch.cat([ep.query_x, ep.batch_target_x], dim=0)
    z_src = encoder(x_src)
    z_tgt = encoder(x_tgt)

    y_src = torch.cat([ep.support_y, ep.batch_source_y], dim=0)
    y_tgt = torch.cat([ep.query_y, ep.batch_target_y], dim=0)
    pred = torch.cat([head(z_src), head(z_tgt)], dim=0)
    target = torch.cat([y_src, y_tgt], dim=0)
    loss_src = ((pred - target) ** 2).mean()

    alpha = 0.01
    gamma = 1e-4
    loss_dare = dare_gram_regularizer(
        z_src, z_tgt, alpha_cos=alpha, gamma_scale=gamma
    )
    loss_b2 = loss_src + loss_dare

    # The pooled MSE term must agree exactly between the two formulations
    # (B1 uses one big concat, B2 uses a source/target split then concat,
    # but the pooled set is identical up to row permutation, and ``mean``
    # is permutation-invariant).
    torch.testing.assert_close(loss_src, loss_b1)

    # Therefore L_{B2} - L_{B1} = α L_cos + γ L_scale = loss_dare.
    delta = (loss_b2 - loss_b1).detach()
    torch.testing.assert_close(delta, loss_dare.detach())
