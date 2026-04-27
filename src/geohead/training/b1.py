"""B1 trainer: source-only supervised MSE on the unified episode (§8.2).

Implements the ``B1`` learner of ``docs/design.md`` §8.2.

At every outer step we draw the **same unified episode** that B2 and P
draw — an ordered pair ``(i, j)`` with ``i != j`` plus the four splits
``(S_i, B_i, Q_j, B_j)`` from :func:`geohead.data.episode.sample_episode`.
B1 then pools the labels of *all four* splits and minimises a single
supervised MSE:

    L_{B1}(t) = (1/N) Σ_{(x,y) ∈ S_i ∪ B_i ∪ Q_j ∪ B_j}
                       (β^T φ_θ(x) - y)^2,        N = 224.

This matches B2's pooled MSE term exactly, so

    L_{B2}(t) - L_{B1}(t) = α_cos · L_cos + γ_scale · L_scale,

i.e. the **B1 vs B2 gap is precisely the DARE-GRAM regulariser**
(§8 fairness protocol).  Provided callers feed all three learners the
same ``torch.Generator`` seed for Phase 1, the raw episode index
sequence is bit-identical across B1 / B2 / P.

Notes
-----
* ``b1_train`` modifies the provided ``encoder`` and ``head`` **in place**.
* The function does *not* run warm-up; callers should restore the shared
  warm-up checkpoint before calling so that the three learners start from
  the same ``(θ, β_0)``.
* No DARE-GRAM, no inner loop, no support/query distinction in the
  *loss*.  The four splits are merely the data slots of the shared
  unified-episode sampler — B1 happens to use them all symmetrically.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from geohead.data.episode import EpisodeSizes, sample_episode, sample_random_pair
from geohead.models.head import LinearHead

__all__ = [
    "B1Config",
    "B1History",
    "b1_train",
]


# ---------------------------------------------------------------------------
# Configuration and history
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class B1Config:
    """Hyperparameters for the B1 (source-only) trainer (§8.2).

    Defaults mirror :class:`geohead.training.baseline.BaselineConfig` so
    that B1, B2, and P share the same step count and per-step batch
    structure.
    """

    outer_steps: int = 5_000
    lr: float = 1e-3
    weight_decay: float = 0.0

    # Per-episode sample sizes — must match B2 / P to keep the unified
    # episode sampler's RNG consumption identical across learners.
    support_size: int = 32
    query_size: int = 64
    batch_source_size: int = 64
    batch_target_size: int = 64

    log_every: int = 100

    def __post_init__(self) -> None:
        if self.outer_steps < 0:
            raise ValueError(f"outer_steps must be >= 0, got {self.outer_steps}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        for name in (
            "support_size",
            "query_size",
            "batch_source_size",
            "batch_target_size",
        ):
            v = getattr(self, name)
            if not isinstance(v, int) or v <= 0:
                raise ValueError(f"{name} must be a positive int, got {v!r}")
        if self.log_every <= 0:
            raise ValueError(f"log_every must be > 0, got {self.log_every}")

    def episode_sizes(self) -> EpisodeSizes:
        return EpisodeSizes(
            support=self.support_size,
            query=self.query_size,
            batch_source=self.batch_source_size,
            batch_target=self.batch_target_size,
        )


@dataclass
class B1History:
    """Diagnostics recorded every ``log_every`` steps (plus the final step).

    ``total_loss[t]`` equals the pooled MSE on the four-split episode.
    There is no DARE term, so we keep this single scalar plus the
    sampled ``pair[t]`` for provenance.
    """

    step: list[int] = field(default_factory=list)
    total_loss: list[float] = field(default_factory=list)
    pair: list[tuple[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------


def b1_train(
    encoder: nn.Module,
    head: LinearHead,
    corpora: Mapping[str, tuple[Tensor, Tensor]],
    config: B1Config = B1Config(),
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> B1History:
    """Run the B1 source-only training loop on the unified episode (§8.2).

    Each step samples one episode via :func:`sample_episode` (the same
    helper used by B2 and P), then minimises the pooled MSE on the
    concatenation of all four labelled splits.  A single Adam step
    updates ``(θ, β)``.

    Parameters
    ----------
    encoder:
        Representation network ``φ_θ : R^{d_x} → R^p``.
    head:
        ``LinearHead`` with output dimension 1 (no bias).
    corpora:
        Labeled *training* corpora ``{D_1, ..., D_C}`` (``C >= 2``).
        Must not include the held-out test corpora ``T_k``.
    config:
        B1 hyperparameters (§8 / §10).
    generator:
        ``torch.Generator`` controlling pair selection and episode
        sampling.  When B1 / B2 / P are seeded identically with the same
        generator state, the raw episode index sequence is bit-identical.
    device:
        Target device; defaults to the encoder's current device.
    """
    if len(corpora) < 2:
        raise ValueError(
            f"b1_train requires >= 2 training corpora, got {len(corpora)}"
        )

    if device is None:
        try:
            device = next(encoder.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)

    encoder.to(device)
    head.to(device)

    corpora_dev: dict[str, tuple[Tensor, Tensor]] = {}
    for name, (x, y) in corpora.items():
        if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
            raise ValueError(
                f"corpus {name!r} must have X of shape (n, d_x) and Y of shape (n,); "
                f"got {tuple(x.shape)} and {tuple(y.shape)}"
            )
        corpora_dev[name] = (x.to(device), y.to(device))

    corpus_names = list(corpora_dev.keys())
    sizes = config.episode_sizes()

    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(
        params, lr=config.lr, weight_decay=config.weight_decay
    )

    history = B1History()

    encoder.train()
    head.train()

    for step in range(1, config.outer_steps + 1):
        i, j = sample_random_pair(corpus_names, generator=generator)
        ep = sample_episode(corpora_dev, i, j, sizes, generator=generator)

        # Pool features and labels across all four splits (§8.2).
        x_all = torch.cat(
            [ep.support_x, ep.batch_source_x, ep.query_x, ep.batch_target_x],
            dim=0,
        )
        y_all = torch.cat(
            [ep.support_y, ep.batch_source_y, ep.query_y, ep.batch_target_y],
            dim=0,
        )
        z_all = encoder(x_all)
        y_hat = head(z_all)
        loss = ((y_hat - y_all) ** 2).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % config.log_every == 0 or step == config.outer_steps:
            history.step.append(step)
            history.total_loss.append(float(loss.detach().item()))
            history.pair.append((i, j))

    return history
