"""Exact reproduction of the DARE-GRAM regularizer.

Implements the inverse-Gram alignment loss from Nejjar et al. (2023),
"DARE-GRAM: Unsupervised Domain Adaptation Regression by Aligning Inverse
Gram Matrices" (CVPR 2024 workshops / arXiv:2303.10396), following the
derivations in section 3.3-3.5 of the paper and section 5 of
``docs/design.md``.

Only the *regularizer* part

    R(Z_s, Z_t) = alpha_cos * L_cos(Z_s, Z_t) + gamma_scale * L_scale(Z_s, Z_t)

is provided here.  The supervised term ``L_src`` (source MSE) is external
because (a) the DARE-GRAM paper composes ``L_src + R`` while
(b) the meta-learning outer loop of GeoHead composes ``L_qry + lambda_D * R``
(``docs/design.md`` sec. 4.4).  Keeping ``R`` as a pure regularizer
accommodates both uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["DareGramInfo", "dare_gram_regularizer"]


@dataclass(frozen=True)
class DareGramInfo:
    """Diagnostic components of a DARE-GRAM evaluation.

    All tensor fields are detached scalars (safe to log / plot).
    """

    L_cos: Tensor
    L_scale: Tensor
    k_s: int
    k_t: int
    k: int


# ---------------------------------------------------------------------------
# Pseudo-inverse Gram via SVD
# ---------------------------------------------------------------------------


def _gram_pinv_eigendecomp(
    z: Tensor, threshold: float, eps: float
) -> tuple[Tensor, Tensor, int]:
    """Eigen-decomposition of ``Z^T Z`` via SVD of ``Z``.

    Parameters
    ----------
    z: Tensor of shape ``(b, p)``.
    threshold: cumulative-variance ratio used to choose the number of
        components ``k`` (``docs/design.md`` eq. 5.1, e.g. 0.99).
    eps: numerical floor (unused inside this helper; caller applies it on
        inversion).

    Returns
    -------
    (V, lambdas, k) where

    - ``V``: orthogonal matrix of shape ``(p, p)`` whose columns are the
      right singular vectors of ``Z`` (eigenvectors of ``G = Z^T Z``),
      ordered so that ``lambdas`` is descending.
    - ``lambdas``: tensor of shape ``(p,)`` with the eigenvalues of ``G``
      in descending order.  Zero-padded if ``b < p``.
    - ``k``: smallest integer such that the cumulative variance ratio
      strictly exceeds ``threshold`` (at least 1, at most ``p``).
    """
    if z.ndim != 2:
        raise ValueError(f"dare_gram expects 2D features, got shape {tuple(z.shape)}")
    _b, p = z.shape

    # SVD with full_matrices=True so Vh is (p, p); U is not needed.
    _u, s, vh = torch.linalg.svd(z, full_matrices=True)
    # torch.linalg.svd returns s sorted in descending order.

    lambdas = z.new_zeros(p)
    lambdas[: s.shape[0]] = s * s

    V = vh.T.contiguous()  # (p, p): columns are eigenvectors of G.

    # Cumulative variance ratio (detach for the discrete argmax).
    with torch.no_grad():
        total = lambdas.sum()
        cum = torch.cumsum(lambdas, dim=0)
        # Guarantee total > 0; if the batch is all-zero, fall back to k=1.
        if total.item() <= 0.0:
            k = 1
        else:
            ratio = cum / total.clamp_min(eps)
            mask = ratio > threshold
            if bool(mask.any()):
                k = int(mask.int().argmax().item()) + 1
            else:
                k = p

    return V, lambdas, k


def _pinv_from_eig(V: Tensor, lambdas: Tensor, k: int, eps: float) -> Tensor:
    """Build ``G^+ = V diag(1/max(lambda_i, eps), ..., 0, ...) V^T``.

    Only the top-``k`` eigenvalues are inverted; positions ``[k:]`` are
    exactly zero.  Differentiable w.r.t. ``V`` and ``lambdas``.
    """
    p = lambdas.shape[0]
    inv = lambdas.new_zeros(p)
    inv[:k] = 1.0 / torch.clamp(lambdas[:k], min=eps)
    # V diag(inv) V^T, computed without materialising diag.
    return (V * inv.unsqueeze(0)) @ V.T


# ---------------------------------------------------------------------------
# Main regularizer
# ---------------------------------------------------------------------------


def dare_gram_regularizer(
    z_source: Tensor,
    z_target: Tensor,
    *,
    alpha_cos: float = 0.01,
    gamma_scale: float = 1e-4,
    threshold: float = 0.99,
    eps: float = 1e-6,
    return_components: bool = False,
) -> Tensor | tuple[Tensor, DareGramInfo]:
    """DARE-GRAM regularizer ``alpha_cos * L_cos + gamma_scale * L_scale``.

    Parameters
    ----------
    z_source, z_target: feature batches of shape ``(b_s, p)`` and
        ``(b_t, p)`` respectively.  Must share the feature dimension ``p``.
    alpha_cos, gamma_scale: loss weights (``docs/design.md`` tab. 10).
    threshold: cumulative-variance threshold ``T`` used to pick ``k_d``.
    eps: numerical floor on inverted eigenvalues and on cosine denominators.
    return_components: if True, also return a :class:`DareGramInfo`.

    Returns
    -------
    Scalar tensor (autograd-connected to ``z_source`` and ``z_target``),
    optionally paired with a :class:`DareGramInfo`.

    Notes
    -----
    * Formulas follow ``docs/design.md`` sec. 5.  In particular, the common
      number of principal components is ``k = max(k_s, k_t)`` so that both
      domains meet the cumulative-variance threshold simultaneously.
    * ``torch.linalg.svd`` is differentiable but requires distinct
      singular values for numerically stable gradients; this is generic
      for continuous feature distributions.
    """
    if z_source.ndim != 2 or z_target.ndim != 2:
        raise ValueError(
            f"expected 2D tensors, got shapes {tuple(z_source.shape)} and {tuple(z_target.shape)}"
        )
    if z_source.shape[1] != z_target.shape[1]:
        raise ValueError(
            "z_source and z_target must share the feature dimension; "
            f"got {z_source.shape[1]} vs {z_target.shape[1]}"
        )

    V_s, lam_s, k_s = _gram_pinv_eigendecomp(z_source, threshold, eps)
    V_t, lam_t, k_t = _gram_pinv_eigendecomp(z_target, threshold, eps)
    k = max(k_s, k_t)

    G_s_pinv = _pinv_from_eig(V_s, lam_s, k, eps)
    G_t_pinv = _pinv_from_eig(V_t, lam_t, k, eps)

    # ---------- L_cos: column-wise cosine similarity of G^+. ----------
    # Using F.normalize (x / max(||x||, eps)) avoids the scale-dependent
    # bias of a 'dot / (|a|*|b| + eps)' formulation: when the true norms
    # exceed eps (generic case), division is exact and cos(a, a) == 1.
    G_s_unit = F.normalize(G_s_pinv, p=2, dim=0, eps=eps)  # (p, p)
    G_t_unit = F.normalize(G_t_pinv, p=2, dim=0, eps=eps)  # (p, p)
    cos = (G_s_unit * G_t_unit).sum(dim=0)                 # (p,)
    L_cos = (1.0 - cos).abs().sum()

    # ---------- L_scale: L2 distance of top-k eigenvalues. ----------
    L_scale = (lam_s[:k] - lam_t[:k]).norm()

    total = alpha_cos * L_cos + gamma_scale * L_scale

    if return_components:
        info = DareGramInfo(
            L_cos=L_cos.detach(),
            L_scale=L_scale.detach(),
            k_s=k_s,
            k_t=k_t,
            k=k,
        )
        return total, info
    return total
