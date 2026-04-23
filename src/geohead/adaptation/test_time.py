"""Test-time few-shot head adaptation methods.

Implements the three adaptation rules defined in ``docs/design.md``
§8.1–§8.3:

* :func:`ridge_adapt` — closed-form ridge (Baseline 1 ``DARE+ridge``)::

      beta_hat = argmin_beta (1/n) ||Z beta - y||^2 + lambda ||beta - beta_0||^2

* :func:`geo_adapt` — closed-form geometry-aware ridge (Baseline 2
  ``DARE+geo``)::

      beta_hat = argmin_beta (1/n) ||Z beta - y||^2
                  + lambda (beta - beta_0)^T (Sigma_hat + eps I) (beta - beta_0)

* :func:`inner_rule_adapt` — ``K``-step gradient descent on the inner
  loss of §4.3 (Proposed; identical rule used by the bilevel inner loop)::

      L_inner(beta) = (1/n) ||Z beta - y||^2
                       + lambda_h (beta - beta_0)^T (Sigma_hat + eps I) (beta - beta_0)
      beta^{(k+1)} = beta^{(k)} - eta * grad_beta L_inner(beta^{(k)})

All three operate at the **feature level**: the caller passes
``z = phi_theta(x)`` as a ``(n, p)`` tensor.  ``sigma`` is the empirical
second moment ``Sigma_hat`` (use :func:`geohead.losses.head_reg.second_moment`
to compute it from a reference feature batch).  ``eps * I`` is added
internally for numerical stability, mirroring
:func:`geohead.losses.head_reg.head_regularizer`.

The ``inner_rule_adapt`` function exposes a ``create_graph`` flag so that
the same rule can be embedded differentiably in the outer meta-learning
loop (§4.4).  With ``create_graph=False`` (default, test time) the
returned tensor is detached; with ``create_graph=True`` the graph
connecting ``beta_final`` to ``beta_0`` (and any upstream parameters) is
preserved and can be backpropagated through.
"""

from __future__ import annotations

import torch
from torch import Tensor

from geohead.losses.head_reg import head_regularizer

__all__ = [
    "ridge_adapt",
    "geo_adapt",
    "inner_rule_adapt",
]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_zy_beta(z: Tensor, y: Tensor, beta_0: Tensor) -> tuple[int, int]:
    if z.ndim != 2:
        raise ValueError(f"z must be 2D (n, p); got shape {tuple(z.shape)}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D (n,); got shape {tuple(y.shape)}")
    if beta_0.ndim != 1:
        raise ValueError(f"beta_0 must be 1D (p,); got shape {tuple(beta_0.shape)}")
    n, p = z.shape
    if n == 0:
        raise ValueError("z must have at least one row (n >= 1)")
    if y.shape[0] != n:
        raise ValueError(f"y length {y.shape[0]} does not match z rows {n}")
    if beta_0.shape[0] != p:
        raise ValueError(
            f"beta_0 length {beta_0.shape[0]} does not match z columns {p}"
        )
    return n, p


def _validate_sigma(sigma: Tensor, p: int) -> None:
    if sigma.ndim != 2 or sigma.shape[0] != p or sigma.shape[1] != p:
        raise ValueError(
            f"sigma must be a ({p}, {p}) square matrix; got shape {tuple(sigma.shape)}"
        )


# ---------------------------------------------------------------------------
# Closed-form methods
# ---------------------------------------------------------------------------


def ridge_adapt(
    z: Tensor,
    y: Tensor,
    beta_0: Tensor,
    lambda_: float,
) -> Tensor:
    """Closed-form ridge head adaptation (Baseline 1, §8.1).

    Solves
    ``(Z^T Z / n + lambda I) beta = Z^T y / n + lambda beta_0``
    using :func:`torch.linalg.solve` for numerical stability.
    """
    n, p = _validate_zy_beta(z, y, beta_0)
    if lambda_ < 0:
        raise ValueError(f"lambda_ must be non-negative; got {lambda_}")

    eye = torch.eye(p, dtype=z.dtype, device=z.device)
    a_mat = (z.T @ z) / n + lambda_ * eye
    b_vec = (z.T @ y) / n + lambda_ * beta_0
    return torch.linalg.solve(a_mat, b_vec)


def geo_adapt(
    z: Tensor,
    y: Tensor,
    beta_0: Tensor,
    sigma: Tensor,
    lambda_: float,
    eps: float = 1e-6,
) -> Tensor:
    """Closed-form geometry-aware ridge head adaptation (Baseline 2, §8.2).

    With ``M := sigma + eps * I``, solves
    ``(Z^T Z / n + lambda M) beta = Z^T y / n + lambda M beta_0``.
    """
    n, p = _validate_zy_beta(z, y, beta_0)
    _validate_sigma(sigma, p)
    if lambda_ < 0:
        raise ValueError(f"lambda_ must be non-negative; got {lambda_}")
    if eps < 0:
        raise ValueError(f"eps must be non-negative; got {eps}")

    eye = torch.eye(p, dtype=z.dtype, device=z.device)
    m_mat = sigma + eps * eye
    a_mat = (z.T @ z) / n + lambda_ * m_mat
    b_vec = (z.T @ y) / n + lambda_ * (m_mat @ beta_0)
    return torch.linalg.solve(a_mat, b_vec)


# ---------------------------------------------------------------------------
# Gradient-descent inner rule
# ---------------------------------------------------------------------------


def inner_rule_adapt(
    z: Tensor,
    y: Tensor,
    beta_0: Tensor,
    sigma: Tensor,
    lambda_h: float,
    eta: float,
    steps: int,
    eps: float = 1e-6,
    *,
    create_graph: bool = False,
) -> Tensor:
    """``K``-step gradient descent on the inner loss (§4.3).

    Per step::

        pred = z @ beta
        loss = (1/n) * sum (pred - y)^2
             + lambda_h * (beta - beta_0)^T (sigma + eps I) (beta - beta_0)
        beta := beta - eta * grad_beta loss

    with ``beta^{(0)} = beta_0`` and ``K = steps``.

    Parameters
    ----------
    z, y, beta_0, sigma:
        See module docstring.
    lambda_h, eta, eps:
        Inner-loss regularisation strength, step size, and covariance
        floor.  Non-negative.
    steps:
        Number of gradient-descent steps (``K`` in the design doc).
        ``steps == 0`` returns ``beta_0.clone()``.
    create_graph:
        If ``True`` the computation graph through all ``K`` inner steps
        is retained so the outer meta-loss can be differentiated w.r.t.
        ``beta_0`` (and any parameters feeding ``z``, ``sigma``).  The
        caller must ensure ``beta_0`` (and relevant inputs) require
        gradients.  If ``False`` (default, test time) the iterate is
        detached between steps and the returned tensor is detached.
    """
    _, p = _validate_zy_beta(z, y, beta_0)
    _validate_sigma(sigma, p)
    if lambda_h < 0:
        raise ValueError(f"lambda_h must be non-negative; got {lambda_h}")
    if eta <= 0:
        raise ValueError(f"eta must be positive; got {eta}")
    if steps < 0:
        raise ValueError(f"steps must be non-negative; got {steps}")
    if eps < 0:
        raise ValueError(f"eps must be non-negative; got {eps}")

    if steps == 0:
        return beta_0.clone() if create_graph else beta_0.detach().clone()

    if create_graph:
        # Trust the caller: beta_0 should already be part of the outer
        # computation graph (e.g., a meta-parameter).
        beta = beta_0
    else:
        beta = beta_0.detach().clone().requires_grad_(True)

    for _ in range(steps):
        pred = z @ beta
        loss_mse = ((pred - y) ** 2).mean()
        loss_reg = head_regularizer(beta, beta_0, sigma, epsilon=eps)
        loss = loss_mse + lambda_h * loss_reg
        grad = torch.autograd.grad(loss, beta, create_graph=create_graph)[0]
        beta_next = beta - eta * grad
        if create_graph:
            beta = beta_next
        else:
            beta = beta_next.detach().requires_grad_(True)

    return beta if create_graph else beta.detach()
