"""Episode sampler for meta-learning over training corpora.

Implements the episode construction specified in ``docs/design.md`` §4.4.

Given the labeled training corpora ``{D_1, D_2, D_3}``, an episode is built as:

* draw an ordered pair ``(i, j)`` with ``i != j``;
* draw a support set ``S`` (size ``|S|``) from ``D_i``;
* draw a query set  ``Q`` (size ``|Q|``) from ``D_j``;
* draw a DARE-GRAM source batch ``B_i`` (size ``|B_i|``) from ``D_i``,
  **disjoint** from ``S`` (§4.6);
* draw a DARE-GRAM target batch ``B_j`` (size ``|B_j|``) from ``D_j``,
  **disjoint** from ``Q``.

All sampling is without replacement and deterministic given a
``torch.Generator``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "EpisodeSizes",
    "EpisodeBatch",
    "sample_random_pair",
    "sample_episode",
]


@dataclass(frozen=True)
class EpisodeSizes:
    """Per-episode sample sizes.

    Defaults follow ``docs/design.md`` §10 (meta-training).
    """

    support: int = 32
    query: int = 64
    batch_source: int = 64
    batch_target: int = 64

    def __post_init__(self) -> None:
        for name in ("support", "query", "batch_source", "batch_target"):
            v = getattr(self, name)
            if not isinstance(v, int) or v <= 0:
                raise ValueError(f"{name} must be a positive int, got {v!r}")


@dataclass
class EpisodeBatch:
    """One meta-learning episode.

    Indices are kept to verify disjointness, debug provenance, and trace
    reproducibility.  Tensors ``*_x`` are gathered via ``*_idx`` from the
    corresponding corpus, so ``x == corpora[name][0][idx]`` exactly.
    """

    i: str
    j: str

    support_x: Tensor
    support_y: Tensor
    support_idx: Tensor

    query_x: Tensor
    query_y: Tensor
    query_idx: Tensor

    batch_source_x: Tensor
    batch_source_y: Tensor
    batch_source_idx: Tensor

    batch_target_x: Tensor
    batch_target_y: Tensor
    batch_target_idx: Tensor


def sample_random_pair(
    corpus_names: Sequence[str],
    generator: torch.Generator | None = None,
) -> tuple[str, str]:
    """Uniformly sample an ordered pair ``(i, j)`` with ``i != j``.

    The distribution is uniform over the ``|corpus_names| * (|corpus_names| - 1)``
    ordered off-diagonal pairs.
    """
    if len(corpus_names) < 2:
        raise ValueError(
            f"need at least 2 corpora to form a pair, got {len(corpus_names)}"
        )

    names = list(corpus_names)
    n = len(names)
    # Flat index over ordered pairs with i != j: total n * (n - 1).
    total = n * (n - 1)
    flat = int(torch.randint(0, total, (1,), generator=generator).item())
    i_idx = flat // (n - 1)
    off = flat % (n - 1)
    j_idx = off if off < i_idx else off + 1
    return names[i_idx], names[j_idx]


def _sample_disjoint_indices(
    n_total: int,
    size_a: int,
    size_b: int,
    generator: torch.Generator | None,
) -> tuple[Tensor, Tensor]:
    """Draw two disjoint index sets of sizes ``size_a`` and ``size_b``
    from ``range(n_total)`` uniformly without replacement.
    """
    if size_a + size_b > n_total:
        raise ValueError(
            f"cannot draw disjoint ({size_a} + {size_b}) > {n_total} samples"
        )
    perm = torch.randperm(n_total, generator=generator)
    idx_a = perm[:size_a]
    idx_b = perm[size_a : size_a + size_b]
    return idx_a, idx_b


def sample_episode(
    corpora: Mapping[str, tuple[Tensor, Tensor]],
    i: str,
    j: str,
    sizes: EpisodeSizes = EpisodeSizes(),
    generator: torch.Generator | None = None,
) -> EpisodeBatch:
    """Sample one meta-learning episode with source ``i`` and target ``j``.

    The support ``S`` and DARE-GRAM source batch ``B_i`` are drawn disjointly
    from ``corpora[i]``; the query ``Q`` and DARE-GRAM target batch ``B_j``
    are drawn disjointly from ``corpora[j]``.

    Parameters
    ----------
    corpora:
        Mapping from corpus name to ``(X, Y)``.  ``X`` has shape ``(n, d_x)``
        and ``Y`` has shape ``(n,)``.
    i, j:
        Source and target corpus names; must differ and both be keys of
        ``corpora``.
    sizes:
        Per-episode sample sizes.
    generator:
        ``torch.Generator`` controlling all random draws (sample indices
        only; ``(i, j)`` must be chosen by the caller).
    """
    if i == j:
        raise ValueError(f"source and target corpora must differ, got i == j == {i!r}")
    if i not in corpora:
        raise KeyError(f"unknown corpus {i!r}; available: {list(corpora)}")
    if j not in corpora:
        raise KeyError(f"unknown corpus {j!r}; available: {list(corpora)}")

    x_i, y_i = corpora[i]
    x_j, y_j = corpora[j]

    if x_i.ndim != 2 or y_i.ndim != 1 or x_i.shape[0] != y_i.shape[0]:
        raise ValueError(
            f"corpus {i!r} must have X of shape (n, d_x) and Y of shape (n,); "
            f"got {tuple(x_i.shape)} and {tuple(y_i.shape)}"
        )
    if x_j.ndim != 2 or y_j.ndim != 1 or x_j.shape[0] != y_j.shape[0]:
        raise ValueError(
            f"corpus {j!r} must have X of shape (n, d_x) and Y of shape (n,); "
            f"got {tuple(x_j.shape)} and {tuple(y_j.shape)}"
        )

    n_i = x_i.shape[0]
    n_j = x_j.shape[0]

    support_idx, batch_source_idx = _sample_disjoint_indices(
        n_i, sizes.support, sizes.batch_source, generator
    )
    query_idx, batch_target_idx = _sample_disjoint_indices(
        n_j, sizes.query, sizes.batch_target, generator
    )

    return EpisodeBatch(
        i=i,
        j=j,
        support_x=x_i.index_select(0, support_idx),
        support_y=y_i.index_select(0, support_idx),
        support_idx=support_idx,
        query_x=x_j.index_select(0, query_idx),
        query_y=y_j.index_select(0, query_idx),
        query_idx=query_idx,
        batch_source_x=x_i.index_select(0, batch_source_idx),
        batch_source_y=y_i.index_select(0, batch_source_idx),
        batch_source_idx=batch_source_idx,
        batch_target_x=x_j.index_select(0, batch_target_idx),
        batch_target_y=y_j.index_select(0, batch_target_idx),
        batch_target_idx=batch_target_idx,
    )
