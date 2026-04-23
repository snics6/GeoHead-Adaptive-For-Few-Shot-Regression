"""Baseline trainer: multi-source supervised MSE + DARE-GRAM regularizer.

Implements the ``DARE+ridge`` baseline of ``docs/design.md`` §8.1:

1. Start from fresh ``(encoder, head)`` (optionally warmed up externally
   by :func:`geohead.training.warmup.warmup_train` for an ablation arm).
2. At every outer step draw an ordered pair ``(i, j)`` with ``i != j``
   uniformly from the training corpora, then one labeled batch ``B_i``
   from ``D_i`` and one labeled batch ``B_j`` from ``D_j``.
3. Minimise the strict §8.1 objective

       L_src(B_i ∪ B_j) + α_cos · L_cos(Z_s, Z_t) + γ_scale · L_scale(Z_s, Z_t)

   jointly over ``(θ, β)`` with Adam.  The DARE-GRAM regularizer is
   provided by :func:`geohead.losses.dare_gram.dare_gram_regularizer`.
4. After training, ``head.beta`` is used as ``β_0`` for test-time
   adaptation (§7).

Notes
-----
* ``baseline_train`` modifies the provided ``encoder`` and ``head`` **in
  place**.
* Test corpora ``{T_1, T_2}`` must not be passed in; the loop only sees
  the training corpora.
* Determinism flows from an optional ``torch.Generator`` that drives
  both :func:`sample_random_pair` and :func:`sample_dare_pair`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from geohead.data.episode import sample_dare_pair, sample_random_pair
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
    """Hyperparameters for the DARE+ridge baseline (§8.1).

    Defaults follow ``docs/design.md`` §10 for the meta-training budget.
    Per §8.1 the total objective is ``L_src + α_cos L_cos + γ_scale L_scale``
    (no extra ``λ_D`` factor; the α and γ weights fully determine the
    regulariser magnitude).
    """

    outer_steps: int = 10_000
    lr: float = 1e-3
    batch_size_source: int = 64
    batch_size_target: int = 64
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
        if self.batch_size_source <= 0:
            raise ValueError(
                f"batch_size_source must be > 0, got {self.batch_size_source}"
            )
        if self.batch_size_target <= 0:
            raise ValueError(
                f"batch_size_target must be > 0, got {self.batch_size_target}"
            )
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


@dataclass
class BaselineHistory:
    """Diagnostics recorded at every ``log_every`` steps (plus the final step).

    ``total_loss[t]`` is the full objective, ``src_loss[t]`` is the pooled
    MSE on ``B_i ∪ B_j``, and ``cos_loss[t]`` / ``scale_loss[t]`` are the
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
    """Run the §8.1 baseline training loop.

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
        Baseline hyperparameters (§10).
    generator:
        ``torch.Generator`` controlling pair selection and batch sampling.
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

    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(
        params, lr=config.lr, weight_decay=config.weight_decay
    )

    history = BaselineHistory()

    encoder.train()
    head.train()

    for step in range(1, config.outer_steps + 1):
        i, j = sample_random_pair(corpus_names, generator=generator)
        batch = sample_dare_pair(
            corpora_dev,
            i,
            j,
            source_size=config.batch_size_source,
            target_size=config.batch_size_target,
            generator=generator,
        )

        z_i = encoder(batch.source_x)
        z_j = encoder(batch.target_x)

        # L_src: pooled MSE over B_i ∪ B_j (§8.1).
        y_hat_i = head(z_i)
        y_hat_j = head(z_j)
        pred = torch.cat([y_hat_i, y_hat_j], dim=0)
        target = torch.cat([batch.source_y, batch.target_y], dim=0)
        loss_src = ((pred - target) ** 2).mean()

        # DARE-GRAM regularizer: α·L_cos + γ·L_scale (info returns the
        # unweighted components for logging).
        loss_dare, info = dare_gram_regularizer(
            z_i,
            z_j,
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
