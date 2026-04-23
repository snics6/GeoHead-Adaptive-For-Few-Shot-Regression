"""Evaluation metrics for the few-shot regression setting.

Implements the primary and auxiliary metrics listed in ``docs/design.md``
§9.1–§9.2:

* :func:`query_mse`, :func:`query_mae` — primary target metrics
  ``(1/|Q|) Σ (β^T z - y)^2`` and ``(1/|Q|) Σ |β^T z - y|``.
* :func:`head_correction_l2` — Euclidean head-correction magnitude
  ``||β - β_0||_2``.
* :func:`head_correction_geo` — geometry-aware head-correction magnitude
  ``(β - β_0)^T (Σ̂_t + ε I) (β - β_0)`` (the same quadratic form that
  underlies :func:`geohead.losses.head_reg.head_regularizer`).
* :func:`evaluate_head` — convenience wrapper returning all four
  diagnostics as plain Python ``float``\\s (detached, CPU-side), suitable
  for logging.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

__all__ = [
    "query_mse",
    "query_mae",
    "head_correction_l2",
    "head_correction_geo",
    "evaluate_head",
]


def _validate_zy_beta(z_query: Tensor, y_query: Tensor, beta: Tensor) -> tuple[int, int]:
    if z_query.ndim != 2:
        raise ValueError(f"z_query must be 2-D (n, p); got {tuple(z_query.shape)}")
    if y_query.ndim != 1:
        raise ValueError(f"y_query must be 1-D (n,); got {tuple(y_query.shape)}")
    if beta.ndim != 1:
        raise ValueError(f"beta must be 1-D (p,); got {tuple(beta.shape)}")
    n, p = z_query.shape
    if n == 0:
        raise ValueError("z_query must have at least one row")
    if y_query.shape[0] != n:
        raise ValueError(f"y_query length {y_query.shape[0]} does not match z_query rows {n}")
    if beta.shape[0] != p:
        raise ValueError(f"beta length {beta.shape[0]} does not match z_query columns {p}")
    return n, p


def query_mse(z_query: Tensor, y_query: Tensor, beta: Tensor) -> Tensor:
    """Target query MSE ``(1/|Q|) Σ (β^T z - y)^2`` (scalar tensor)."""
    _validate_zy_beta(z_query, y_query, beta)
    return ((z_query @ beta - y_query) ** 2).mean()


def query_mae(z_query: Tensor, y_query: Tensor, beta: Tensor) -> Tensor:
    """Target query MAE ``(1/|Q|) Σ |β^T z - y|`` (scalar tensor)."""
    _validate_zy_beta(z_query, y_query, beta)
    return (z_query @ beta - y_query).abs().mean()


def head_correction_l2(beta: Tensor, beta_0: Tensor) -> Tensor:
    """Euclidean head-correction magnitude ``||β - β_0||_2`` (scalar)."""
    if beta.ndim != 1 or beta_0.ndim != 1:
        raise ValueError(
            f"beta / beta_0 must be 1-D; got {tuple(beta.shape)} and {tuple(beta_0.shape)}"
        )
    if beta.shape != beta_0.shape:
        raise ValueError(
            f"beta and beta_0 must share shape; got {tuple(beta.shape)} vs {tuple(beta_0.shape)}"
        )
    return torch.linalg.vector_norm(beta - beta_0)


def head_correction_geo(
    beta: Tensor,
    beta_0: Tensor,
    sigma_hat: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Geometry-aware head-correction ``(β - β_0)^T (Σ̂ + ε I) (β - β_0)``.

    This is the §9.2 auxiliary metric and matches the quadratic form
    evaluated by :func:`geohead.losses.head_reg.head_regularizer` on the
    test support.  ``Σ̂`` is typically the target-support second moment.
    """
    if beta.ndim != 1 or beta_0.ndim != 1:
        raise ValueError(
            f"beta / beta_0 must be 1-D; got {tuple(beta.shape)} and {tuple(beta_0.shape)}"
        )
    if beta.shape != beta_0.shape:
        raise ValueError(
            f"beta and beta_0 must share shape; got {tuple(beta.shape)} vs {tuple(beta_0.shape)}"
        )
    p = beta.shape[0]
    if sigma_hat.ndim != 2 or sigma_hat.shape != (p, p):
        raise ValueError(
            f"sigma_hat must be ({p}, {p}); got {tuple(sigma_hat.shape)}"
        )
    if eps < 0:
        raise ValueError(f"eps must be non-negative; got {eps}")
    diff = beta - beta_0
    return diff @ (sigma_hat @ diff + eps * diff)


def evaluate_head(
    z_query: Tensor,
    y_query: Tensor,
    beta: Tensor,
    beta_0: Tensor,
    sigma_hat: Tensor,
    eps: float = 1e-6,
) -> dict[str, Any]:
    """Compute all §9.1–§9.2 metrics for one ``β`` vs ``β_0`` comparison.

    Returns a dict of plain Python floats::

        {
            "mse":       query_mse(z_query, y_query, β),
            "mae":       query_mae(z_query, y_query, β),
            "delta_l2":  ||β - β_0||_2,
            "delta_geo": (β - β_0)^T (Σ̂ + εI) (β - β_0),
        }
    """
    with torch.no_grad():
        return {
            "mse": float(query_mse(z_query, y_query, beta).item()),
            "mae": float(query_mae(z_query, y_query, beta).item()),
            "delta_l2": float(head_correction_l2(beta, beta_0).item()),
            "delta_geo": float(head_correction_geo(beta, beta_0, sigma_hat, eps=eps).item()),
        }
