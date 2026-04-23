"""Tests for the episode sampler (``src/geohead/data/episode.py``).

Specifications under test live in ``docs/design.md`` sec. 4.4 and 4.6.
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch

from geohead.data.episode import (
    EpisodeSizes,
    sample_episode,
    sample_random_pair,
)
from geohead.data.toy import ToyConfig, build_toy_dataset


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_corpora(n: int = 200, d_x: int = 5, seed: int = 0):
    """Build a small synthetic set of corpora with deterministic entries.

    Each corpus uses distinct X values so we can trace provenance cleanly.
    """
    g = torch.Generator().manual_seed(seed)
    corpora: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for k, name in enumerate(("D1", "D2", "D3")):
        base = 1000.0 * (k + 1)
        x = base + torch.arange(n * d_x, dtype=torch.float32).reshape(n, d_x) + \
            0.01 * torch.randn(n, d_x, generator=g)
        y = base + torch.arange(n, dtype=torch.float32)
        corpora[name] = (x, y)
    return corpora


# ---------------------------------------------------------------------------
# sample_random_pair
# ---------------------------------------------------------------------------


def test_sample_random_pair_always_distinct() -> None:
    g = torch.Generator().manual_seed(42)
    names = ["D1", "D2", "D3"]
    for _ in range(200):
        i, j = sample_random_pair(names, generator=g)
        assert i in names and j in names
        assert i != j


def test_sample_random_pair_deterministic() -> None:
    names = ["D1", "D2", "D3"]
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    pairs_1 = [sample_random_pair(names, generator=g1) for _ in range(50)]
    pairs_2 = [sample_random_pair(names, generator=g2) for _ in range(50)]
    assert pairs_1 == pairs_2


def test_sample_random_pair_covers_all_orderings() -> None:
    names = ["D1", "D2", "D3"]
    g = torch.Generator().manual_seed(0)
    counter: Counter[tuple[str, str]] = Counter()
    n_trials = 6000
    for _ in range(n_trials):
        counter[sample_random_pair(names, generator=g)] += 1

    # 3 * 2 = 6 off-diagonal pairs, expected ~1000 each.
    assert len(counter) == 6
    for pair, count in counter.items():
        i, j = pair
        assert i != j
        # Loose uniformity check: within 25% of the mean.
        assert 750 <= count <= 1250, (pair, count)


def test_sample_random_pair_too_few_names() -> None:
    with pytest.raises(ValueError):
        sample_random_pair(["only_one"])


# ---------------------------------------------------------------------------
# sample_episode: shapes and provenance
# ---------------------------------------------------------------------------


def test_sample_episode_shapes() -> None:
    corpora = _make_corpora(n=200, d_x=5)
    sizes = EpisodeSizes(support=16, query=24, batch_source=32, batch_target=40)
    g = torch.Generator().manual_seed(0)
    ep = sample_episode(corpora, "D1", "D2", sizes=sizes, generator=g)

    assert ep.i == "D1" and ep.j == "D2"

    assert ep.support_x.shape == (16, 5)
    assert ep.support_y.shape == (16,)
    assert ep.support_idx.shape == (16,)

    assert ep.query_x.shape == (24, 5)
    assert ep.query_y.shape == (24,)
    assert ep.query_idx.shape == (24,)

    assert ep.batch_source_x.shape == (32, 5)
    assert ep.batch_source_y.shape == (32,)
    assert ep.batch_source_idx.shape == (32,)

    assert ep.batch_target_x.shape == (40, 5)
    assert ep.batch_target_y.shape == (40,)
    assert ep.batch_target_idx.shape == (40,)


def test_sample_episode_provenance_matches_indices() -> None:
    corpora = _make_corpora(n=200, d_x=5)
    g = torch.Generator().manual_seed(123)
    ep = sample_episode(corpora, "D1", "D3", generator=g)

    x_i, y_i = corpora["D1"]
    x_j, y_j = corpora["D3"]

    torch.testing.assert_close(ep.support_x, x_i[ep.support_idx])
    torch.testing.assert_close(ep.support_y, y_i[ep.support_idx])
    torch.testing.assert_close(ep.batch_source_x, x_i[ep.batch_source_idx])
    torch.testing.assert_close(ep.batch_source_y, y_i[ep.batch_source_idx])

    torch.testing.assert_close(ep.query_x, x_j[ep.query_idx])
    torch.testing.assert_close(ep.query_y, y_j[ep.query_idx])
    torch.testing.assert_close(ep.batch_target_x, x_j[ep.batch_target_idx])
    torch.testing.assert_close(ep.batch_target_y, y_j[ep.batch_target_idx])


def test_sample_episode_support_and_batch_source_are_disjoint() -> None:
    corpora = _make_corpora(n=200, d_x=5)
    g = torch.Generator().manual_seed(456)
    sizes = EpisodeSizes(support=32, query=32, batch_source=64, batch_target=64)
    ep = sample_episode(corpora, "D1", "D2", sizes=sizes, generator=g)

    s_idx = set(ep.support_idx.tolist())
    b_idx = set(ep.batch_source_idx.tolist())
    assert len(s_idx) == sizes.support  # no self-duplicates
    assert len(b_idx) == sizes.batch_source
    assert s_idx.isdisjoint(b_idx)


def test_sample_episode_query_and_batch_target_are_disjoint() -> None:
    corpora = _make_corpora(n=200, d_x=5)
    g = torch.Generator().manual_seed(789)
    sizes = EpisodeSizes(support=32, query=32, batch_source=64, batch_target=64)
    ep = sample_episode(corpora, "D1", "D2", sizes=sizes, generator=g)

    q_idx = set(ep.query_idx.tolist())
    bt_idx = set(ep.batch_target_idx.tolist())
    assert len(q_idx) == sizes.query
    assert len(bt_idx) == sizes.batch_target
    assert q_idx.isdisjoint(bt_idx)


# ---------------------------------------------------------------------------
# sample_episode: determinism and randomness
# ---------------------------------------------------------------------------


def test_sample_episode_deterministic_given_generator() -> None:
    corpora = _make_corpora(n=200, d_x=5)
    g1 = torch.Generator().manual_seed(2024)
    g2 = torch.Generator().manual_seed(2024)
    ep1 = sample_episode(corpora, "D1", "D2", generator=g1)
    ep2 = sample_episode(corpora, "D1", "D2", generator=g2)

    torch.testing.assert_close(ep1.support_idx, ep2.support_idx)
    torch.testing.assert_close(ep1.query_idx, ep2.query_idx)
    torch.testing.assert_close(ep1.batch_source_idx, ep2.batch_source_idx)
    torch.testing.assert_close(ep1.batch_target_idx, ep2.batch_target_idx)


def test_sample_episode_different_seeds_differ() -> None:
    corpora = _make_corpora(n=200, d_x=5)
    g1 = torch.Generator().manual_seed(1)
    g2 = torch.Generator().manual_seed(2)
    ep1 = sample_episode(corpora, "D1", "D2", generator=g1)
    ep2 = sample_episode(corpora, "D1", "D2", generator=g2)
    # Index sets should overlap only partially; at minimum the full sequences
    # must not be identical.
    assert not torch.equal(ep1.support_idx, ep2.support_idx)


# ---------------------------------------------------------------------------
# sample_episode: validation
# ---------------------------------------------------------------------------


def test_sample_episode_rejects_same_corpus() -> None:
    corpora = _make_corpora(n=200, d_x=5)
    with pytest.raises(ValueError):
        sample_episode(corpora, "D1", "D1")


def test_sample_episode_rejects_unknown_corpus() -> None:
    corpora = _make_corpora(n=200, d_x=5)
    with pytest.raises(KeyError):
        sample_episode(corpora, "D1", "DOES_NOT_EXIST")
    with pytest.raises(KeyError):
        sample_episode(corpora, "DOES_NOT_EXIST", "D2")


def test_sample_episode_rejects_oversized_disjoint_draw() -> None:
    corpora = _make_corpora(n=50, d_x=5)  # small corpora
    sizes = EpisodeSizes(support=30, query=10, batch_source=30, batch_target=10)
    # |S| + |B_i| = 60 > 50
    with pytest.raises(ValueError):
        sample_episode(corpora, "D1", "D2", sizes=sizes)


def test_episode_sizes_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        EpisodeSizes(support=0)
    with pytest.raises(ValueError):
        EpisodeSizes(query=-1)


# ---------------------------------------------------------------------------
# Integration with the toy dataset
# ---------------------------------------------------------------------------


def test_sample_episode_works_on_toy_dataset() -> None:
    cfg = ToyConfig(seed=0)
    ds = build_toy_dataset(cfg, n_train_per_corpus=500)
    g = torch.Generator().manual_seed(0)

    i, j = sample_random_pair(list(ds.train.keys()), generator=g)
    ep = sample_episode(ds.train, i, j, generator=g)

    assert ep.i == i and ep.j == j
    assert ep.support_x.shape[1] == cfg.d_x
    assert ep.query_x.shape[1] == cfg.d_x
    # Label consistency via index lookup into the original corpus.
    torch.testing.assert_close(ep.support_y, ds.train[i][1][ep.support_idx])
    torch.testing.assert_close(ep.query_y, ds.train[j][1][ep.query_idx])
