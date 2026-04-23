"""Evaluation runner: 4-method × test-corpus × k-shot × seed matrix.

Implements the evaluation protocol of ``docs/design.md`` §9.3:

* For each test corpus ``T_k ∈ {T_1, T_2}`` (structure
  ``{"support": (X, Y), "query": (X, Y)}`` produced by
  :func:`build_toy_dataset`).
* For each ``k`` in ``config.k_shots`` (default ``(1, 3, 5, 10, 20)``).
* For each ``seed`` in ``range(config.n_seeds)`` (default 20).
* For each adaptation method in ``config.methods`` (default
  ``("none", "ridge", "geo", "inner")``):

  * ``none``: no adaptation, ``β̂ = β_0``;
  * ``ridge``: :func:`ridge_adapt`;
  * ``geo``: :func:`geo_adapt`;
  * ``inner``: :func:`inner_rule_adapt` with ``create_graph=False``
    (identical rule to the training inner loop, §8.3).

Fair-comparison invariant
-------------------------
For a fixed ``(corpus, k_shot, seed)`` triple, the **same** support
sub-sample is shown to every method.  This isolates method effects from
sampling noise and is the raison d'être of the nested-loop order below
(outer loop over ``(corpus, k_shot, seed)``, inner loop over methods).

Output
------
``evaluate_model`` returns a list of ``EvalRecord`` dicts (long format)
with keys::

    {
        "corpus":    str,     # e.g. "T1"
        "k_shot":    int,
        "seed":      int,
        "method":    str,
        "mse":       float,
        "mae":       float,
        "delta_l2":  float,
        "delta_geo": float,
    }

The helpers :func:`aggregate` (mean ± 95 % CI per ``(corpus, k_shot,
method)``) and :func:`to_pandas` (lazy pandas export) make downstream
analysis straightforward.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from geohead.adaptation.test_time import geo_adapt, inner_rule_adapt, ridge_adapt
from geohead.evaluation.metrics import evaluate_head
from geohead.losses.head_reg import second_moment
from geohead.models.head import LinearHead

__all__ = [
    "EvalConfig",
    "EvalRecord",
    "evaluate_model",
    "aggregate",
    "to_pandas",
]


EvalRecord = dict[str, Any]

_VALID_METHODS = ("none", "ridge", "geo", "inner")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalConfig:
    """Hyperparameters for :func:`evaluate_model`.

    Defaults follow ``docs/design.md`` §10 (test-time section) except
    for ``ridge_lambda`` / ``geo_lambda``, which are not in the doc;
    ``0.1`` is a reasonable default that matches the scale of
    ``lambda_h`` for the inner-rule method and keeps the three
    adaptation methods broadly comparable.
    """

    k_shots: tuple[int, ...] = (1, 3, 5, 10, 20)
    n_seeds: int = 20
    methods: tuple[str, ...] = ("none", "ridge", "geo", "inner")

    # Adaptation hyperparameters
    ridge_lambda: float = 0.1
    geo_lambda: float = 0.1
    inner_lambda_h: float = 0.1
    inner_lr: float = 0.1
    inner_steps: int = 5
    head_reg_eps: float = 1e-6
    # When ``True`` the ``inner`` method uses the damped natural-gradient
    # preconditioner ``(Σ_hat + ε I)^{-1}`` — should match
    # :attr:`GeoHeadConfig.preconditioned_inner` when evaluating a
    # GeoHead model for train/test parity (§8.3).
    inner_preconditioned: bool = False

    # Reproducibility.  A ``support`` sub-sample at (corpus=c, k_shot=k,
    # seed=s) is drawn from Generator(seed_base + seed) — this is the
    # ONLY source of randomness in ``evaluate_model``.
    seed_base: int = 0

    def __post_init__(self) -> None:
        if not self.k_shots:
            raise ValueError("k_shots must be a non-empty tuple")
        for k in self.k_shots:
            if not isinstance(k, int) or k <= 0:
                raise ValueError(f"every k_shot must be a positive int, got {k!r}")
        if not isinstance(self.n_seeds, int) or self.n_seeds <= 0:
            raise ValueError(f"n_seeds must be a positive int, got {self.n_seeds!r}")
        if not self.methods:
            raise ValueError("methods must be a non-empty tuple")
        for m in self.methods:
            if m not in _VALID_METHODS:
                raise ValueError(
                    f"unknown method {m!r}; supported: {_VALID_METHODS}"
                )
        for name in ("ridge_lambda", "geo_lambda", "inner_lambda_h"):
            v = getattr(self, name)
            if v < 0:
                raise ValueError(f"{name} must be >= 0, got {v}")
        if self.inner_lr <= 0:
            raise ValueError(f"inner_lr must be > 0, got {self.inner_lr}")
        if self.inner_steps < 0:
            raise ValueError(f"inner_steps must be >= 0, got {self.inner_steps}")
        if self.head_reg_eps < 0:
            raise ValueError(f"head_reg_eps must be >= 0, got {self.head_reg_eps}")


# ---------------------------------------------------------------------------
# Core method dispatch
# ---------------------------------------------------------------------------


def _adapt(
    method: str,
    z_sup: Tensor,
    y_sup: Tensor,
    beta_0: Tensor,
    sigma_hat: Tensor,
    config: EvalConfig,
) -> Tensor:
    """Dispatch to the right test-time adaptation rule.

    The ``sigma_hat`` argument is shared across methods inside one
    ``(corpus, k_shot, seed)`` triple, keeping fair-comparison semantics.
    """
    if method == "none":
        return beta_0.detach().clone()
    if method == "ridge":
        return ridge_adapt(z_sup, y_sup, beta_0, lambda_=config.ridge_lambda)
    if method == "geo":
        return geo_adapt(
            z_sup, y_sup, beta_0,
            sigma=sigma_hat, lambda_=config.geo_lambda, eps=config.head_reg_eps,
        )
    if method == "inner":
        return inner_rule_adapt(
            z_sup, y_sup, beta_0,
            sigma=sigma_hat,
            lambda_h=config.inner_lambda_h,
            eta=config.inner_lr,
            steps=config.inner_steps,
            eps=config.head_reg_eps,
            create_graph=False,
            preconditioned=config.inner_preconditioned,
        )
    raise ValueError(f"unknown method {method!r}")  # unreachable given _VALID_METHODS


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def evaluate_model(
    encoder: nn.Module,
    head: LinearHead,
    test_corpora: Mapping[str, Mapping[str, tuple[Tensor, Tensor]]],
    config: EvalConfig = EvalConfig(),
    device: torch.device | str | None = None,
) -> list[EvalRecord]:
    """Run the full evaluation matrix described in §9.3.

    Parameters
    ----------
    encoder:
        Trained representation network ``φ_θ``.  Will be set to ``eval()``
        and wrapped in ``torch.no_grad()`` throughout.
    head:
        Trained ``LinearHead``; its current ``head.beta`` is used as
        ``β_0`` for every ``(corpus, k_shot, seed, method)`` cell.
    test_corpora:
        Held-out test corpora ``{T_1, T_2, ...}`` with the nested
        structure ``{name: {"support": (X, Y), "query": (X, Y)}}``.
        The *support pool* is sub-sampled ``n_seeds`` times per
        ``k_shot``; the *query pool* is fixed.
    config:
        Hyperparameters (k-shots, seeds, adaptation params).
    device:
        Target device; defaults to the encoder's current device.

    Returns
    -------
    list[EvalRecord]:
        One record per ``(corpus, k_shot, seed, method)`` tuple.  Total
        length: ``|corpora| · |k_shots| · n_seeds · |methods|``.
    """
    if not test_corpora:
        raise ValueError("test_corpora must contain at least one corpus")
    for name, bundle in test_corpora.items():
        if not isinstance(bundle, Mapping) or "support" not in bundle or "query" not in bundle:
            raise ValueError(
                f"corpus {name!r} must be a mapping with keys 'support' and 'query'"
            )
        sx, sy = bundle["support"]
        qx, qy = bundle["query"]
        if sx.ndim != 2 or sy.ndim != 1 or sx.shape[0] != sy.shape[0]:
            raise ValueError(
                f"corpus {name!r} support must have X (n, d_x) and Y (n,); "
                f"got {tuple(sx.shape)} and {tuple(sy.shape)}"
            )
        if qx.ndim != 2 or qy.ndim != 1 or qx.shape[0] != qy.shape[0]:
            raise ValueError(
                f"corpus {name!r} query must have X (n, d_x) and Y (n,); "
                f"got {tuple(qx.shape)} and {tuple(qy.shape)}"
            )

    if device is None:
        try:
            device = next(encoder.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)

    encoder.to(device)
    head.to(device)
    encoder.eval()
    head.eval()

    beta_0 = head.beta.detach().clone().to(device)

    records: list[EvalRecord] = []

    # NOTE: we deliberately do NOT wrap this loop in ``torch.no_grad()``.
    # ``inner_rule_adapt(create_graph=False)`` runs K steps of
    # :func:`torch.autograd.grad` internally; disabling gradient tracking
    # on the outer forward would break that inner autograd.  Instead we
    # detach every tensor that feeds into metrics (``z_query`` below)
    # and rely on :func:`evaluate_head` to wrap its own ``no_grad()``
    # around the final float conversions.
    for corpus_name, bundle in test_corpora.items():
        x_sup_pool, y_sup_pool = bundle["support"]
        x_qry_pool, y_qry_pool = bundle["query"]
        x_sup_pool = x_sup_pool.to(device)
        y_sup_pool = y_sup_pool.to(device)
        x_qry_pool = x_qry_pool.to(device)
        y_qry_pool = y_qry_pool.to(device)

        # Query features are identical for every (k_shot, seed, method);
        # they are only read (never backpropagated through), so detach
        # once to free the encoder graph for this forward pass.
        with torch.no_grad():
            z_query = encoder(x_qry_pool)

        n_sup_pool = x_sup_pool.shape[0]
        for k in config.k_shots:
            if k > n_sup_pool:
                raise ValueError(
                    f"corpus {corpus_name!r} support pool has {n_sup_pool} rows; "
                    f"cannot draw k_shot={k}"
                )

            for seed_offset in range(config.n_seeds):
                # Single generator drives the support sub-sample for
                # this (corpus, k, seed); ALL methods share it, which
                # is the fair-comparison invariant.
                gen = torch.Generator(device="cpu").manual_seed(
                    config.seed_base + seed_offset
                )
                idx = torch.randperm(n_sup_pool, generator=gen)[:k].to(device)
                x_sup = x_sup_pool.index_select(0, idx)
                y_sup = y_sup_pool.index_select(0, idx)

                # Support features must carry a live graph so the
                # ``inner`` method's internal ``autograd.grad`` works;
                # ``ridge`` / ``geo`` / ``none`` ignore the graph.
                z_sup = encoder(x_sup)
                sigma_hat = second_moment(z_sup)
                # Detach for downstream methods — we never backprop into
                # θ here, and detaching avoids retaining the encoder
                # graph once the support is processed.
                z_sup_det = z_sup.detach()
                sigma_det = sigma_hat.detach()

                for method in config.methods:
                    beta_hat = _adapt(
                        method, z_sup_det, y_sup, beta_0, sigma_det, config
                    )
                    metrics = evaluate_head(
                        z_query,
                        y_qry_pool,
                        beta_hat.detach(),
                        beta_0,
                        sigma_det,
                        eps=config.head_reg_eps,
                    )
                    records.append(
                        {
                            "corpus": corpus_name,
                            "k_shot": int(k),
                            "seed": int(seed_offset),
                            "method": method,
                            **metrics,
                        }
                    )

    return records


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


@dataclass
class _AggAccumulator:
    values: list[float] = field(default_factory=list)

    def add(self, v: float) -> None:
        self.values.append(v)

    def summarise(self) -> dict[str, float]:
        n = len(self.values)
        mean = sum(self.values) / n
        if n >= 2:
            var = sum((v - mean) ** 2 for v in self.values) / (n - 1)
            sem = math.sqrt(var / n)
            ci95 = 1.96 * sem
        else:
            sem = 0.0
            ci95 = 0.0
        return {"mean": mean, "sem": sem, "ci95": ci95, "n": n}


def aggregate(
    records: Sequence[EvalRecord],
    metrics: Sequence[str] = ("mse", "mae", "delta_l2", "delta_geo"),
) -> list[EvalRecord]:
    """Collapse seeds into ``(corpus, k_shot, method)`` summaries.

    For every metric in ``metrics`` the returned row has three fields:
    ``{metric}_mean``, ``{metric}_sem``, and ``{metric}_ci95`` (the
    half-width of a 95 % normal CI around the mean).  ``seed_count`` is
    the number of seeds aggregated.
    """
    groups: dict[tuple[str, int, str], dict[str, _AggAccumulator]] = {}
    for rec in records:
        key = (rec["corpus"], int(rec["k_shot"]), rec["method"])
        slot = groups.setdefault(key, {m: _AggAccumulator() for m in metrics})
        for m in metrics:
            slot[m].add(float(rec[m]))

    out: list[EvalRecord] = []
    for (corpus, k_shot, method), slot in sorted(groups.items()):
        row: EvalRecord = {"corpus": corpus, "k_shot": k_shot, "method": method}
        for m in metrics:
            summary = slot[m].summarise()
            row[f"{m}_mean"] = summary["mean"]
            row[f"{m}_sem"] = summary["sem"]
            row[f"{m}_ci95"] = summary["ci95"]
        row["seed_count"] = slot[metrics[0]].summarise()["n"]
        out.append(row)
    return out


def to_pandas(records: Sequence[EvalRecord]):
    """Lazy pandas import; return a ``DataFrame`` from ``records``.

    Raises :class:`ImportError` with a helpful message if pandas is not
    installed (pandas is *not* a hard dependency of this project).
    """
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only without pandas
        raise ImportError(
            "to_pandas requires pandas; install with `pip install pandas` "
            "or stay on list[dict] records."
        ) from exc
    return pd.DataFrame.from_records(list(records))
