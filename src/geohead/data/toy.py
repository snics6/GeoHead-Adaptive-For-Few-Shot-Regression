"""Toy data generator for GeoHead experiments.

Implements the protocol specified in ``docs/design.md`` section 6.

The toy setup consists of:

* A frozen oracle encoder ``phi_star : R^{d_x} -> R^{p_star}`` shared by all
  domains.  This plays the role of the "true" representation.
* Three labeled training corpora ``D1, D2, D3`` with moderate covariate and
  head shifts (used during meta-training).
* Two held-out test corpora ``T1`` (interpolation) and ``T2`` (extrapolation)
  that are never touched during training and are used only for test-time
  few-shot adaptation.

For each domain ``d`` with spec ``(mu_d, Sigma_d, beta_d^*)``::

    x ~ N(mu_d, Sigma_d),   y = beta_d^{*T} phi_star(x) + eps,   eps ~ N(0, sigma^2).

All randomness flows from a single integer seed to guarantee reproducibility.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

__all__ = [
    "DomainSpec",
    "ToyConfig",
    "ToyDataset",
    "build_domain_specs",
    "build_phi_star",
    "build_toy_dataset",
    "sample_domain",
]


# ---------------------------------------------------------------------------
# Configuration and containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToyConfig:
    """Hyperparameters of the toy data protocol.

    Defaults follow ``docs/design.md`` sections 6 and 10.
    """

    d_x: int = 16
    p_star: int = 32
    phi_hidden: tuple[int, ...] = (64, 32)

    noise_sigma: float = 0.1

    mu_norm: float = 0.5
    cov_log_std: float = 0.3
    head_shift_train: float = 0.4

    mu_norm_extrap: float = 1.0
    cov_log_std_extrap: float = 0.6
    head_shift_extrap: float = 0.8

    seed: int = 0


@dataclass(frozen=True)
class DomainSpec:
    """Ground-truth distribution parameters for a single domain."""

    name: str
    mu: Tensor          # shape (d_x,)
    cov_factor: Tensor  # shape (d_x, d_x); Sigma = cov_factor @ cov_factor.T
    beta_star: Tensor   # shape (p_star,)

    @property
    def cov(self) -> Tensor:
        """Covariance matrix ``Sigma = L L^T``."""
        return self.cov_factor @ self.cov_factor.T


@dataclass
class ToyDataset:
    """Full toy dataset bundle returned by :func:`build_toy_dataset`."""

    config: ToyConfig
    phi_star: nn.Module
    specs: Mapping[str, DomainSpec]
    train: Mapping[str, tuple[Tensor, Tensor]]
    test: Mapping[str, Mapping[str, tuple[Tensor, Tensor]]]


# ---------------------------------------------------------------------------
# Oracle encoder phi_star
# ---------------------------------------------------------------------------


def build_phi_star(cfg: ToyConfig, seed: int | None = None) -> nn.Module:
    """Build the frozen oracle MLP ``phi_star``.

    Architecture: ``d_x -> phi_hidden[0] -> ... -> phi_hidden[-1] -> p_star``,
    with ReLU between linear layers (no activation after the final layer).
    Parameters are initialised deterministically from ``cfg.seed`` (or the
    override ``seed``) and frozen with ``requires_grad=False``.
    """
    actual_seed = cfg.seed if seed is None else seed
    gen = torch.Generator().manual_seed(int(actual_seed))

    dims: list[int] = [cfg.d_x, *cfg.phi_hidden, cfg.p_star]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        lin = nn.Linear(dims[i], dims[i + 1])
        bound = 1.0 / (dims[i] ** 0.5)
        with torch.no_grad():
            lin.weight.uniform_(-bound, bound, generator=gen)
            lin.bias.uniform_(-bound, bound, generator=gen)
        layers.append(lin)
        if i < len(dims) - 2:
            layers.append(nn.ReLU())

    net = nn.Sequential(*layers)
    for p in net.parameters():
        p.requires_grad_(False)
    net.eval()
    return net


# ---------------------------------------------------------------------------
# Random building blocks for domain specs
# ---------------------------------------------------------------------------


def _random_unit_vector(d: int, generator: torch.Generator) -> Tensor:
    v = torch.randn(d, generator=generator)
    return v / v.norm()


def _random_orthogonal(d: int, generator: torch.Generator) -> Tensor:
    """Uniform random orthogonal matrix via QR (Haar measure)."""
    a = torch.randn(d, d, generator=generator)
    q, r = torch.linalg.qr(a)
    # Fix sign so the distribution is Haar on O(d).
    q = q * torch.sign(torch.diag(r))
    return q


def _cov_factor(d: int, log_std: float, generator: torch.Generator) -> Tensor:
    """Sample a PD covariance and return its Cholesky factor ``L``.

    ``Sigma = R diag(exp(xi)) R^T``, ``xi ~ N(0, log_std^2)``, ``R ~ Haar(O(d))``.
    """
    r = _random_orthogonal(d, generator)
    log_eig = torch.randn(d, generator=generator) * log_std
    s = torch.exp(log_eig)
    cov = r @ torch.diag(s) @ r.T
    cov = 0.5 * (cov + cov.T)
    return torch.linalg.cholesky(cov)


def _head_directions(
    p: int, n_deltas: int, generator: torch.Generator
) -> tuple[Tensor, list[Tensor]]:
    """Sample ``beta_base ~ N(0, I_p)`` plus ``n_deltas`` unit vectors that are
    mutually orthogonal and orthogonal to ``beta_base``.
    """
    full = torch.randn(p, n_deltas + 1, generator=generator)
    beta_base = full[:, 0].clone()

    u = beta_base / beta_base.norm()
    r = full[:, 1:]
    # Remove the beta_base component, then orthonormalise the residuals.
    r = r - u[:, None] * (u @ r)
    q, _ = torch.linalg.qr(r)
    # Re-project and re-normalise for numerical safety.
    q = q - u[:, None] * (u @ q)
    q = q / q.norm(dim=0, keepdim=True)

    deltas = [q[:, i].contiguous() for i in range(n_deltas)]
    return beta_base, deltas


# ---------------------------------------------------------------------------
# Domain specs
# ---------------------------------------------------------------------------


def build_domain_specs(cfg: ToyConfig) -> dict[str, DomainSpec]:
    """Construct deterministic domain specs for ``D1, D2, D3, T1, T2``.

    Design rules (``docs/design.md`` sec. 6.3):

    * ``D1``: ``mu=0``, ``Sigma=I``, ``beta* = beta_base``.
    * ``D2, D3``: shifted mean (``||mu|| = cfg.mu_norm``), random covariance
      with log-eigenvalue std ``cfg.cov_log_std``, ``beta* = beta_base + 0.4 * delta_i``.
    * ``T1`` (interp): mean and covariance are the midpoint of ``D2, D3``,
      ``beta* = beta_base + 0.4 * (0.5 * delta_1 + 0.5 * delta_2)``.
    * ``T2`` (extrap): new random direction, larger mean shift and covariance
      spread, ``beta* = beta_base + 0.8 * delta_4``.
    """
    gen = torch.Generator().manual_seed(int(cfg.seed) + 101)

    beta_base, deltas = _head_directions(cfg.p_star, n_deltas=4, generator=gen)
    delta_1, delta_2, _delta_3, delta_4 = deltas

    mu_D1 = torch.zeros(cfg.d_x)
    mu_D2 = cfg.mu_norm * _random_unit_vector(cfg.d_x, gen)
    mu_D3 = cfg.mu_norm * _random_unit_vector(cfg.d_x, gen)
    mu_T1 = 0.5 * (mu_D2 + mu_D3)
    mu_T2 = cfg.mu_norm_extrap * _random_unit_vector(cfg.d_x, gen)

    L_D1 = torch.eye(cfg.d_x)
    L_D2 = _cov_factor(cfg.d_x, cfg.cov_log_std, gen)
    L_D3 = _cov_factor(cfg.d_x, cfg.cov_log_std, gen)
    cov_T1 = 0.5 * (L_D2 @ L_D2.T + L_D3 @ L_D3.T)
    cov_T1 = 0.5 * (cov_T1 + cov_T1.T)
    L_T1 = torch.linalg.cholesky(cov_T1)
    L_T2 = _cov_factor(cfg.d_x, cfg.cov_log_std_extrap, gen)

    a = cfg.head_shift_train
    b = cfg.head_shift_extrap

    return {
        "D1": DomainSpec("D1", mu_D1, L_D1, beta_base.clone()),
        "D2": DomainSpec("D2", mu_D2, L_D2, beta_base + a * delta_1),
        "D3": DomainSpec("D3", mu_D3, L_D3, beta_base + a * delta_2),
        "T1": DomainSpec(
            "T1", mu_T1, L_T1, beta_base + a * (0.5 * delta_1 + 0.5 * delta_2)
        ),
        "T2": DomainSpec("T2", mu_T2, L_T2, beta_base + b * delta_4),
    }


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_domain(
    phi_star: nn.Module,
    spec: DomainSpec,
    n: int,
    noise_sigma: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Draw ``n`` i.i.d. samples ``(x, y)`` from the domain described by
    ``spec``.

    ``x ~ N(mu, L L^T)`` and ``y = beta*^T phi_star(x) + eps``,
    with ``eps ~ N(0, noise_sigma^2)``.
    """
    gen = torch.Generator().manual_seed(int(seed))
    d_x = spec.mu.shape[0]
    z = torch.randn(n, d_x, generator=gen)
    x = spec.mu[None, :] + z @ spec.cov_factor.T
    with torch.no_grad():
        phi = phi_star(x)
    eps = torch.randn(n, generator=gen) * noise_sigma
    y = phi @ spec.beta_star + eps
    return x, y


# ---------------------------------------------------------------------------
# Full dataset builder
# ---------------------------------------------------------------------------


def build_toy_dataset(
    cfg: ToyConfig | None = None,
    n_train_per_corpus: int = 5000,
    n_test_support: int = 200,
    n_test_query: int = 1000,
) -> ToyDataset:
    """Build the complete toy dataset (training + held-out test corpora).

    Sample counts default to those listed in ``docs/design.md`` sec. 6.4.
    All randomness is derived from ``cfg.seed`` so calling this function
    with the same config produces bit-identical data.
    """
    if cfg is None:
        cfg = ToyConfig()

    phi_star = build_phi_star(cfg)
    specs = build_domain_specs(cfg)

    # Per-domain seeds, all derived from cfg.seed.
    base = int(cfg.seed) * 10_000 + 1

    train: dict[str, tuple[Tensor, Tensor]] = {}
    for i, name in enumerate(("D1", "D2", "D3")):
        train[name] = sample_domain(
            phi_star, specs[name], n_train_per_corpus, cfg.noise_sigma, seed=base + i
        )

    test: dict[str, dict[str, tuple[Tensor, Tensor]]] = {}
    for k, name in enumerate(("T1", "T2")):
        support_seed = base + 1_000 + 2 * k
        query_seed = base + 1_000 + 2 * k + 1
        support = sample_domain(
            phi_star, specs[name], n_test_support, cfg.noise_sigma, seed=support_seed
        )
        query = sample_domain(
            phi_star, specs[name], n_test_query, cfg.noise_sigma, seed=query_seed
        )
        test[name] = {"support": support, "query": query}

    return ToyDataset(
        config=cfg,
        phi_star=phi_star,
        specs=specs,
        train=train,
        test=test,
    )
