"""Warm-up trainer: pooled supervised MSE to initialize ``(θ, β_0)``.

Implements ``docs/design.md`` §4.2 and §4.5 (warm-up phase):

1. Pool the labeled training corpora ``D_1, D_2, D_3``.
2. Train encoder ``θ`` and a single linear head ``β`` with standard
   supervised MSE regression (Adam, 20 epochs, lr=1e-3 by default).
3. Carry over: ``β_0 := β``, ``θ := θ`` as the initialisation for the
   bilevel phase (§4.3–§4.5).

Notes
-----
* ``warmup_train`` modifies the provided ``encoder`` and ``head`` **in place**.
  After the call, ``head.beta`` is ready to be used as the meta-initial
  head ``β_0``.
* All randomness flows from an optional ``torch.Generator`` so identical
  runs reproduce exactly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from geohead.models.head import LinearHead

__all__ = [
    "WarmupConfig",
    "WarmupHistory",
    "pooled_dataset",
    "warmup_train",
]


# ---------------------------------------------------------------------------
# Configuration and history
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WarmupConfig:
    """Hyperparameters for the warm-up phase.

    Defaults follow ``docs/design.md`` §10.  ``batch_size`` is not fixed by
    the design doc; 256 is a sensible default for the toy setup
    (pooled size ≈ 15 000 → ~60 steps / epoch).
    """

    epochs: int = 20
    lr: float = 1e-3
    batch_size: int = 256
    weight_decay: float = 0.0
    shuffle: bool = True

    def __post_init__(self) -> None:
        if self.epochs < 0:
            raise ValueError(f"epochs must be >= 0, got {self.epochs}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")


@dataclass
class WarmupHistory:
    """History of the warm-up run.

    ``train_loss[0]`` is the baseline pooled MSE *before* any update;
    ``train_loss[epoch]`` is the mean pooled MSE over the mini-batches of
    that epoch.  Same convention for ``per_corpus_loss`` (epoch 0 entry is
    measured pre-training; subsequent entries are measured at the end of
    each epoch with ``torch.no_grad``).
    """

    train_loss: list[float] = field(default_factory=list)
    per_corpus_loss: list[dict[str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Data pooling
# ---------------------------------------------------------------------------


def pooled_dataset(
    corpora: Mapping[str, tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor]:
    """Concatenate ``(X, Y)`` from all corpora in insertion order.

    Parameters
    ----------
    corpora:
        Mapping from corpus name to ``(X, Y)``.  ``X`` has shape
        ``(n_c, d_x)`` and ``Y`` has shape ``(n_c,)``.

    Returns
    -------
    X_pooled, Y_pooled:
        Concatenated tensors of shapes ``(sum_c n_c, d_x)`` and
        ``(sum_c n_c,)``.
    """
    if len(corpora) == 0:
        raise ValueError("corpora must be non-empty")

    xs: list[Tensor] = []
    ys: list[Tensor] = []
    d_x: int | None = None
    for name, (x, y) in corpora.items():
        if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
            raise ValueError(
                f"corpus {name!r} must have X of shape (n, d_x) and Y of shape (n,); "
                f"got {tuple(x.shape)} and {tuple(y.shape)}"
            )
        if d_x is None:
            d_x = x.shape[1]
        elif x.shape[1] != d_x:
            raise ValueError(
                f"inconsistent d_x across corpora: {d_x} vs {x.shape[1]} for {name!r}"
            )
        xs.append(x)
        ys.append(y)

    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _forward_mse(
    encoder: nn.Module, head: LinearHead, x: Tensor, y: Tensor
) -> Tensor:
    """MSE of ``head(encoder(x))`` vs ``y``; differentiable."""
    z = encoder(x)
    y_hat = head(z)
    return ((y_hat - y) ** 2).mean()


@torch.no_grad()
def _eval_per_corpus(
    encoder: nn.Module,
    head: LinearHead,
    corpora: Mapping[str, tuple[Tensor, Tensor]],
    device: torch.device,
) -> dict[str, float]:
    was_training_enc = encoder.training
    was_training_head = head.training
    encoder.eval()
    head.eval()
    try:
        out: dict[str, float] = {}
        for name, (x, y) in corpora.items():
            x = x.to(device)
            y = y.to(device)
            out[name] = float(_forward_mse(encoder, head, x, y).item())
        return out
    finally:
        encoder.train(was_training_enc)
        head.train(was_training_head)


# ---------------------------------------------------------------------------
# Main warm-up routine
# ---------------------------------------------------------------------------


def warmup_train(
    encoder: nn.Module,
    head: LinearHead,
    corpora: Mapping[str, tuple[Tensor, Tensor]],
    config: WarmupConfig = WarmupConfig(),
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> WarmupHistory:
    """Run the warm-up phase (pooled supervised MSE training).

    The ``encoder`` and ``head`` are updated **in place**.  After the call,
    ``head.beta`` can be used as the meta-initial head ``β_0`` for the
    bilevel phase (§4.5).

    Parameters
    ----------
    encoder:
        Representation network ``φ_θ : R^{d_x} → R^p``.
    head:
        ``LinearHead`` with output dimension 1 (no bias).
    corpora:
        Labeled training corpora ``{D_1, ..., D_C}``.
    config:
        Warm-up hyperparameters.
    generator:
        ``torch.Generator`` controlling mini-batch shuffling (reproducibility).
    device:
        Target device; defaults to the encoder's current device.

    Returns
    -------
    WarmupHistory:
        ``train_loss[0]`` is the pre-training baseline pooled MSE.
        ``train_loss[epoch]`` (``epoch >= 1``) is the mean mini-batch MSE
        during that epoch.  ``per_corpus_loss`` is evaluated in no-grad at
        the corresponding points.
    """
    if device is None:
        try:
            device = next(encoder.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)

    encoder.to(device)
    head.to(device)

    x_pool, y_pool = pooled_dataset(corpora)
    x_pool = x_pool.to(device)
    y_pool = y_pool.to(device)
    n = x_pool.shape[0]

    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(
        params, lr=config.lr, weight_decay=config.weight_decay
    )

    history = WarmupHistory()

    history.train_loss.append(
        float(_forward_mse(encoder, head, x_pool, y_pool).detach().item())
    )
    history.per_corpus_loss.append(_eval_per_corpus(encoder, head, corpora, device))

    for _epoch in range(config.epochs):
        if config.shuffle:
            perm = torch.randperm(n, generator=generator, device=x_pool.device)
        else:
            perm = torch.arange(n, device=x_pool.device)

        running_loss = 0.0
        n_batches = 0
        encoder.train()
        head.train()
        for start in range(0, n, config.batch_size):
            idx = perm[start : start + config.batch_size]
            xb = x_pool.index_select(0, idx)
            yb = y_pool.index_select(0, idx)

            loss = _forward_mse(encoder, head, xb, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().item())
            n_batches += 1

        epoch_loss = running_loss / max(n_batches, 1)
        history.train_loss.append(epoch_loss)
        history.per_corpus_loss.append(
            _eval_per_corpus(encoder, head, corpora, device)
        )

    return history
