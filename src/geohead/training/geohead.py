"""GeoHead bilevel meta-trainer (head-only ANIL + DARE-GRAM).

Implements the proposed method of ``docs/design.md`` §4.3–§4.5:

1. Start from ``(θ, β_0)`` initialised by :func:`warmup_train` (or from
   scratch for an ablation arm).
2. At every outer step draw an ordered pair ``(i, j)`` with ``i != j``
   uniformly from the training corpora, then sample one episode
   ``(S, Q, B_i, B_j)`` via :func:`sample_episode`.
3. **Inner loop** (head-only ANIL, §4.3).  Starting from the current
   ``β_0 = head.beta``, run ``K`` steps of gradient descent on

       L_inner(β) = (1/|S|) ||Z_S β - y_S||^2
                     + λ_h (β - β_0)^T (Σ̂_S + ε I) (β - β_0)

   to obtain ``β' = β^{(K)}``.  The step is implemented by
   :func:`geohead.adaptation.test_time.inner_rule_adapt` with
   ``create_graph=True`` so that ``β'`` remains differentiable w.r.t.
   ``(θ, β_0)``.  This is strictly the *same* rule used at test time by
   the ``inner`` adaptation method (§8.3), which guarantees train/test
   consistency.
4. **Outer loss** (§4.4):

       L_outer = L_qry(Q; β', θ) + λ_D (L_src(B_i; β_0, θ) + α L_cos + γ L_scale)

   * ``L_qry`` is the query MSE using the *adapted* head ``β'``.
   * ``L_src`` is the source-corpus MSE using the *meta-initial* head
     ``β_0`` — this keeps the direct supervisory signal on ``β_0``
     exactly as in the §8.1 baseline, providing a stable anchor for the
     head that is being meta-learned.
   * ``α L_cos + γ L_scale`` is the DARE-GRAM regularizer on
     ``(Z_{B_i}, Z_{B_j})``.
5. Backprop ``L_outer`` through the unrolled inner loop and take one
   Adam step on ``(θ, β_0)``.

Notes
-----
* ``geohead_train`` modifies the provided ``encoder`` and ``head`` **in
  place**.
* Test corpora ``{T_1, T_2}`` must not be passed in; the loop only sees
  the training corpora.
* The inner loop uses plain SGD (no Adam state) — consistent with the
  test-time ``inner_rule_adapt``.
* Determinism flows from an optional ``torch.Generator`` that drives
  both :func:`sample_random_pair` and :func:`sample_episode`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from geohead.adaptation.test_time import inner_rule_adapt
from geohead.data.episode import EpisodeSizes, sample_episode, sample_random_pair
from geohead.losses.dare_gram import dare_gram_regularizer
from geohead.losses.head_reg import second_moment
from geohead.models.head import LinearHead

__all__ = [
    "GeoHeadConfig",
    "GeoHeadHistory",
    "geohead_train",
]


# ---------------------------------------------------------------------------
# Configuration and history
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeoHeadConfig:
    """Hyperparameters for the GeoHead bilevel meta-trainer (§4.3–§4.5).

    Defaults follow ``docs/design.md`` §10 (meta-training).
    """

    # Outer optimisation
    outer_steps: int = 10_000
    outer_lr: float = 1e-3
    weight_decay: float = 0.0

    # Inner loop (head-only ANIL)
    inner_steps: int = 5
    inner_lr: float = 0.1
    lambda_h: float = 0.1
    head_reg_eps: float = 1e-6

    # DARE-GRAM weights and SVD truncation
    lambda_D: float = 1.0
    alpha_cos: float = 0.01
    gamma_scale: float = 1e-4
    threshold: float = 0.99
    dare_eps: float = 1e-6

    # Per-episode sample sizes (forwarded to EpisodeSizes)
    support_size: int = 32
    query_size: int = 64
    batch_source_size: int = 64
    batch_target_size: int = 64

    # Logging
    log_every: int = 100

    def __post_init__(self) -> None:
        if self.outer_steps < 0:
            raise ValueError(f"outer_steps must be >= 0, got {self.outer_steps}")
        if self.outer_lr <= 0:
            raise ValueError(f"outer_lr must be > 0, got {self.outer_lr}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.inner_steps < 0:
            raise ValueError(f"inner_steps must be >= 0, got {self.inner_steps}")
        if self.inner_lr <= 0:
            raise ValueError(f"inner_lr must be > 0, got {self.inner_lr}")
        if self.lambda_h < 0:
            raise ValueError(f"lambda_h must be >= 0, got {self.lambda_h}")
        if self.head_reg_eps < 0:
            raise ValueError(f"head_reg_eps must be >= 0, got {self.head_reg_eps}")
        if self.lambda_D < 0:
            raise ValueError(f"lambda_D must be >= 0, got {self.lambda_D}")
        if self.alpha_cos < 0:
            raise ValueError(f"alpha_cos must be >= 0, got {self.alpha_cos}")
        if self.gamma_scale < 0:
            raise ValueError(f"gamma_scale must be >= 0, got {self.gamma_scale}")
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {self.threshold}")
        if self.dare_eps <= 0:
            raise ValueError(f"dare_eps must be > 0, got {self.dare_eps}")
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
        """Return the :class:`EpisodeSizes` induced by this config."""
        return EpisodeSizes(
            support=self.support_size,
            query=self.query_size,
            batch_source=self.batch_source_size,
            batch_target=self.batch_target_size,
        )


@dataclass
class GeoHeadHistory:
    """Diagnostics recorded at every ``log_every`` steps (plus the final step).

    ``total_loss[t]`` is the full outer objective ``L_outer``,
    ``qry_loss[t]`` is the MSE on ``Q`` using the adapted head ``β'``,
    ``src_loss[t]`` is the MSE on ``B_i`` using ``β_0``, and
    ``cos_loss[t]`` / ``scale_loss[t]`` are the *unweighted*
    ``L_cos`` / ``L_scale`` components.  ``inner_delta_norm[t]`` is the
    Euclidean norm ``||β' - β_0||`` at that step — a handy scalar to
    track how aggressive the inner update is over training.  ``pair[t]``
    records the episode pair ``(i, j)``.
    """

    step: list[int] = field(default_factory=list)
    total_loss: list[float] = field(default_factory=list)
    qry_loss: list[float] = field(default_factory=list)
    src_loss: list[float] = field(default_factory=list)
    cos_loss: list[float] = field(default_factory=list)
    scale_loss: list[float] = field(default_factory=list)
    inner_delta_norm: list[float] = field(default_factory=list)
    pair: list[tuple[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------


def geohead_train(
    encoder: nn.Module,
    head: LinearHead,
    corpora: Mapping[str, tuple[Tensor, Tensor]],
    config: GeoHeadConfig = GeoHeadConfig(),
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> GeoHeadHistory:
    """Run the GeoHead bilevel meta-training loop (§4.3–§4.5).

    Parameters
    ----------
    encoder:
        Representation network ``φ_θ : R^{d_x} → R^p``.  Typically
        initialised by :func:`geohead.training.warmup.warmup_train`.
    head:
        ``LinearHead`` whose weight is treated as the meta-initial head
        ``β_0``.  Also typically initialised by warm-up.
    corpora:
        Labeled *training* corpora ``{D_1, ..., D_C}`` (``C >= 2``).
        Must not include the held-out test corpora ``T_k``.
    config:
        Bilevel hyperparameters (§10).
    generator:
        ``torch.Generator`` controlling pair selection and episode
        sampling.  Inner-loop GD is deterministic given its inputs.
    device:
        Target device; defaults to the encoder's current device.

    Returns
    -------
    GeoHeadHistory:
        Per-step diagnostics, recorded every ``config.log_every`` steps
        and at the final step.
    """
    if len(corpora) < 2:
        raise ValueError(
            f"geohead_train requires >= 2 training corpora, got {len(corpora)}"
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
        params, lr=config.outer_lr, weight_decay=config.weight_decay
    )

    history = GeoHeadHistory()

    encoder.train()
    head.train()

    for step in range(1, config.outer_steps + 1):
        i, j = sample_random_pair(corpus_names, generator=generator)
        ep = sample_episode(corpora_dev, i, j, sizes, generator=generator)

        # --- Forward features (all attached to θ) ---
        z_s = encoder(ep.support_x)
        z_q = encoder(ep.query_x)
        z_bi = encoder(ep.batch_source_x)
        z_bj = encoder(ep.batch_target_x)

        # --- Inner loop: head-only ANIL, K-step GD from β_0 ---
        # sigma_s is kept attached to θ so that the outer gradient flows
        # back through L_head_reg (§4.3 definition).
        sigma_s = second_moment(z_s)
        beta_0 = head.beta  # 1-D view of head.linear.weight

        beta_prime = inner_rule_adapt(
            z_s,
            ep.support_y,
            beta_0=beta_0,
            sigma=sigma_s,
            lambda_h=config.lambda_h,
            eta=config.inner_lr,
            steps=config.inner_steps,
            eps=config.head_reg_eps,
            create_graph=True,
        )

        # --- Outer losses (§4.4) ---
        # L_qry: query MSE with adapted head β'
        y_hat_q = z_q @ beta_prime
        loss_qry = ((y_hat_q - ep.query_y) ** 2).mean()

        # L_src: source-corpus MSE with meta-initial head β_0 (stable
        # direct supervisory signal on β_0, aligning with §8.1 baseline).
        y_hat_bi = z_bi @ beta_0
        loss_src = ((y_hat_bi - ep.batch_source_y) ** 2).mean()

        # DARE-GRAM regularizer on (Z_{B_i}, Z_{B_j}); info carries the
        # unweighted L_cos, L_scale for logging.
        loss_dare_reg, info = dare_gram_regularizer(
            z_bi,
            z_bj,
            alpha_cos=config.alpha_cos,
            gamma_scale=config.gamma_scale,
            threshold=config.threshold,
            eps=config.dare_eps,
            return_components=True,
        )

        loss_total = loss_qry + config.lambda_D * (loss_src + loss_dare_reg)

        # Snapshot diagnostics BEFORE the outer update; otherwise the
        # in-place Adam step on ``head.linear.weight`` would shift the
        # view ``beta_0`` and break the ``||β' - β_0||`` interpretation
        # (β' is a detached-values clone of the pre-step weight, while
        # ``beta_0`` is a live view of the post-step weight).
        should_log = step % config.log_every == 0 or step == config.outer_steps
        if should_log:
            with torch.no_grad():
                delta_norm = float((beta_prime - beta_0).detach().norm().item())
            log_total = float(loss_total.detach().item())
            log_qry = float(loss_qry.detach().item())
            log_src = float(loss_src.detach().item())
            log_cos = float(info.L_cos.item())
            log_scale = float(info.L_scale.item())

        optimizer.zero_grad(set_to_none=True)
        loss_total.backward()
        optimizer.step()

        if should_log:
            history.step.append(step)
            history.total_loss.append(log_total)
            history.qry_loss.append(log_qry)
            history.src_loss.append(log_src)
            history.cos_loss.append(log_cos)
            history.scale_loss.append(log_scale)
            history.inner_delta_norm.append(delta_norm)
            history.pair.append((i, j))

    return history
