"""B2 trainer: pooled supervised MSE + DARE-GRAM on the unified episode (§8.3).

Implements the ``B2`` learner of ``docs/design.md`` §8.3.

At every outer step we draw the **same unified episode** that B1 and P
draw — an ordered pair ``(i, j)`` with ``i != j`` plus the four splits
``(S_i, B_i, Q_j, B_j)`` from :func:`geohead.data.episode.sample_episode`.
B2 then minimises

    L_{B2}(t) = L_{B1}(t)
              + α_cos · L_cos(Z_{S_i ∪ B_i}, Z_{Q_j ∪ B_j})
              + γ_scale · L_scale(...).

* ``L_{B1}`` is the **identical pooled MSE** computed by
  :func:`geohead.training.b1.b1_train` over all 224 labelled samples in
  the four splits.
* The DARE-GRAM regulariser is provided by
  :func:`geohead.losses.dare_gram.dare_gram_regularizer` and aligns the
  inverse Gram of the source side ``S_i ∪ B_i`` with that of the target
  side ``Q_j ∪ B_j`` (96 vs. 128 features).

Provided callers feed all three learners the same ``torch.Generator``
seed for Phase 1, the raw episode index sequence is bit-identical
across B1 / B2 / P, so

    L_{B2}(t) - L_{B1}(t) = α L_cos + γ L_scale

isolates the DARE-GRAM contribution exactly (§8 fairness protocol).

Notes
-----
* ``baseline_train`` modifies the provided ``encoder`` and ``head`` **in
  place**.
* Test corpora ``{T_1, T_2}`` must not be passed in; the loop only sees
  the training corpora.
* Determinism flows from an optional ``torch.Generator`` driving both
  :func:`sample_random_pair` and :func:`sample_episode`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from geohead.data.episode import EpisodeSizes, sample_episode, sample_random_pair
from geohead.losses.dare_gram import dare_gram_regularizer
from geohead.models.head import LinearHead

__all__ = [
    "BaselineConfig",
    "BaselineHistory",
    "baseline_train",
]


# ---------------------------------------------------------------------------
# Configuration and history
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineConfig:
    """Hyperparameters for the B2 trainer (§8.3).

    Defaults follow ``docs/design.md`` §10 for the meta-training budget.
    Per §8.3 the total objective is

        L_pooled-MSE + α_cos L_cos + γ_scale L_scale

    where ``L_pooled-MSE`` is taken over all four splits of the unified
    episode (= ``L_{B1}``), and the DARE alignment is computed between
    the source side ``S_i ∪ B_i`` and the target side ``Q_j ∪ B_j``.
    No extra ``λ_D`` factor; ``α_cos`` / ``γ_scale`` fully determine the
    regulariser magnitude.

    The four ``*_size`` fields control the unified-episode batch and
    **must match** the B1 / GeoHead configs to keep the per-step RNG
    consumption identical across learners (§8.0).
    """

    outer_steps: int = 10_000
    lr: float = 1e-3

    # Per-episode sample sizes for the unified sampler.  Defaults match
    # GeoHeadConfig so all three learners produce the same episode
    # sequence under a shared generator.
    support_size: int = 32
    query_size: int = 64
    batch_source_size: int = 64
    batch_target_size: int = 64

    alpha_cos: float = 0.01
    gamma_scale: float = 1e-4
    threshold: float = 0.99
    dare_eps: float = 1e-6
    weight_decay: float = 0.0
    log_every: int = 100

    def __post_init__(self) -> None:
        if self.outer_steps < 0:
            raise ValueError(f"outer_steps must be >= 0, got {self.outer_steps}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        for name in (
            "support_size",
            "query_size",
            "batch_source_size",
            "batch_target_size",
        ):
            v = getattr(self, name)
            if not isinstance(v, int) or v <= 0:
                raise ValueError(f"{name} must be a positive int, got {v!r}")
        if self.alpha_cos < 0:
            raise ValueError(f"alpha_cos must be >= 0, got {self.alpha_cos}")
        if self.gamma_scale < 0:
            raise ValueError(f"gamma_scale must be >= 0, got {self.gamma_scale}")
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {self.threshold}")
        if self.dare_eps <= 0:
            raise ValueError(f"dare_eps must be > 0, got {self.dare_eps}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
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
class BaselineHistory:
    """Diagnostics recorded at every ``log_every`` steps (plus the final step).

    ``total_loss[t]`` is the full objective, ``src_loss[t]`` is the
    pooled MSE on the four-split episode ``S_i ∪ B_i ∪ Q_j ∪ B_j``
    (= ``L_{B1}``), and ``cos_loss[t]`` / ``scale_loss[t]`` are the
    *unweighted* ``L_cos`` / ``L_scale`` components (weights ``α_cos``,
    ``γ_scale`` from the config).  ``pair[t]`` is the ``(i, j)`` sampled
    at that step.
    """

    step: list[int] = field(default_factory=list)
    total_loss: list[float] = field(default_factory=list)
    src_loss: list[float] = field(default_factory=list)
    cos_loss: list[float] = field(default_factory=list)
    scale_loss: list[float] = field(default_factory=list)
    pair: list[tuple[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main baseline routine
# ---------------------------------------------------------------------------


def baseline_train(
    encoder: nn.Module,
    head: LinearHead,
    corpora: Mapping[str, tuple[Tensor, Tensor]],
    config: BaselineConfig = BaselineConfig(),
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> BaselineHistory:
    """Run the §8.3 B2 training loop on the unified episode.

    Each step samples one episode via :func:`sample_episode` (the same
    helper used by B1 and P), then minimises

        L_pooled-MSE(S ∪ B_i ∪ Q ∪ B_j)
            + α_cos · L_cos(Z_{S∪B_i}, Z_{Q∪B_j})
            + γ_scale · L_scale(...).

    A single Adam step updates ``(θ, β)``.

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
        B2 hyperparameters (§10).
    generator:
        ``torch.Generator`` controlling pair selection and episode
        sampling.  When B1 / B2 / P share the same seed the raw
        episode index sequence is bit-identical (§8.0).
    device:
        Target device; defaults to the encoder's current device.

    Returns
    -------
    BaselineHistory:
        Per-step diagnostics, recorded every ``config.log_every`` steps
        and at the final step.
    """
    if len(corpora) < 2:
        raise ValueError(
            f"baseline_train requires >= 2 training corpora, got {len(corpora)}"
        )

    if device is None:
        try:
            device = next(encoder.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)

    encoder.to(device)
    head.to(device)

    # Move corpora to device once; toy scale (~15k rows total) fits easily.
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

    history = BaselineHistory()

    encoder.train()
    head.train()

    for step in range(1, config.outer_steps + 1):
        i, j = sample_random_pair(corpus_names, generator=generator)
        ep = sample_episode(corpora_dev, i, j, sizes, generator=generator)

        # Source side = S_i ∪ B_i (96 samples, with labels).
        # Target side = Q_j ∪ B_j (128 samples, labels also used in
        #               pooled MSE; features used in DARE).
        x_src = torch.cat([ep.support_x, ep.batch_source_x], dim=0)
        y_src = torch.cat([ep.support_y, ep.batch_source_y], dim=0)
        x_tgt = torch.cat([ep.query_x, ep.batch_target_x], dim=0)
        y_tgt = torch.cat([ep.query_y, ep.batch_target_y], dim=0)

        z_src = encoder(x_src)
        z_tgt = encoder(x_tgt)

        # Pooled MSE on all 224 labelled samples — matches B1 exactly so
        # that L_{B2} - L_{B1} = α L_cos + γ L_scale (§8 fairness).
        pred = torch.cat([head(z_src), head(z_tgt)], dim=0)
        target = torch.cat([y_src, y_tgt], dim=0)
        loss_src = ((pred - target) ** 2).mean()

        # DARE-GRAM regulariser between source-side and target-side
        # features.  ``info`` carries the unweighted components for
        # logging.
        loss_dare, info = dare_gram_regularizer(
            z_src,
            z_tgt,
            alpha_cos=config.alpha_cos,
            gamma_scale=config.gamma_scale,
            threshold=config.threshold,
            eps=config.dare_eps,
            return_components=True,
        )

        loss_total = loss_src + loss_dare

        optimizer.zero_grad(set_to_none=True)
        loss_total.backward()
        optimizer.step()

        if step % config.log_every == 0 or step == config.outer_steps:
            history.step.append(step)
            history.total_loss.append(float(loss_total.detach().item()))
            history.src_loss.append(float(loss_src.detach().item()))
            history.cos_loss.append(float(info.L_cos.item()))
            history.scale_loss.append(float(info.L_scale.item()))
            history.pair.append((i, j))

    return history
