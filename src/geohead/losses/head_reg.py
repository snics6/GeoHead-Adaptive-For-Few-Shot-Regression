"""Geometry-aware head regularization.

Implements ``docs/design.md`` sec. 4.3 (inner loop) and sec. 8.2 (test-time
adaptation ``geo``):

    L_head-reg(beta; beta_0, Sigma_hat) = (beta - beta_0)^T (Sigma_hat + eps I) (beta - beta_0).

The regularizer penalises deviations from the meta-initial head ``beta_0``
using the *geometry* of the support batch (or any representative feature
batch), expressed through the empirical second moment matrix
``Sigma_hat = (1/n) sum_i z_i z_i^T``.  Compared with a plain ridge
``||beta - beta_0||^2``, it gives more budget for deviations along
feature directions that carry little support mass.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["head_regularizer", "second_moment"]


def head_regularizer(
    beta: Tensor,
    beta_0: Tensor,
    sigma_hat: Tensor,
    epsilon: float = 1e-6,
) -> Tensor:
    """Return the scalar ``(beta - beta_0)^T (Sigma_hat + eps I) (beta - beta_0)``.

    Parameters
    ----------
    beta, beta_0: 1-D tensors of shape ``(p,)``.
    sigma_hat: 2-D tensor of shape ``(p, p)`` (expected PSD; symmetry
        not enforced).
    epsilon: non-negative floor on the quadratic form for numerical
        stability when ``Sigma_hat`` is (near-)singular.

    Returns
    -------
    Scalar tensor, differentiable w.r.t. both ``beta`` and ``beta_0``
    (and ``sigma_hat`` when it itself carries gradients).
    """
    if beta.ndim != 1 or beta_0.ndim != 1:
        raise ValueError(
            f"beta / beta_0 must be 1-D; got shapes {tuple(beta.shape)} and {tuple(beta_0.shape)}"
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
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative; got {epsilon}")

    diff = beta - beta_0
    # (Sigma_hat + eps I) @ diff computed without materialising the full
    # p x p identity.
    sigma_diff = sigma_hat @ diff + epsilon * diff
    return diff @ sigma_diff


def second_moment(z: Tensor) -> Tensor:
    """Empirical uncentered second moment ``Sigma_hat = (1/n) Z^T Z``.

    Parameters
    ----------
    z: 2-D tensor of shape ``(n, p)`` (at least one sample).

    Returns
    -------
    Symmetric PSD tensor of shape ``(p, p)``.
    """
    if z.ndim != 2:
        raise ValueError(f"z must be 2-D, got shape {tuple(z.shape)}")
    n = z.shape[0]
    if n == 0:
        raise ValueError("second_moment requires at least one sample")
    sigma = z.T @ z / n
    # Symmetrise against numerical noise while preserving autograd.
    return 0.5 * (sigma + sigma.T)
