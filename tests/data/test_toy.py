"""Tests for the toy data generator (``src/geohead/data/toy.py``).

Specifications under test live in ``docs/design.md`` sec. 6.
"""

from __future__ import annotations

import pytest
import torch

from geohead.data.toy import (
    ToyConfig,
    build_domain_specs,
    build_phi_star,
    build_toy_dataset,
    sample_domain,
)

# ---------------------------------------------------------------------------
# phi_star
# ---------------------------------------------------------------------------


def test_phi_star_output_shape_and_frozen() -> None:
    cfg = ToyConfig()
    phi = build_phi_star(cfg)
    assert all(not p.requires_grad for p in phi.parameters())
    x = torch.randn(8, cfg.d_x)
    z = phi(x)
    assert z.shape == (8, cfg.p_star)


def test_phi_star_is_deterministic() -> None:
    cfg = ToyConfig(seed=3)
    phi_a = build_phi_star(cfg)
    phi_b = build_phi_star(cfg)
    x = torch.randn(5, cfg.d_x)
    assert torch.allclose(phi_a(x), phi_b(x))


def test_phi_star_seed_differs() -> None:
    x = torch.randn(5, ToyConfig().d_x)
    phi_a = build_phi_star(ToyConfig(seed=1))
    phi_b = build_phi_star(ToyConfig(seed=2))
    assert not torch.allclose(phi_a(x), phi_b(x))


# ---------------------------------------------------------------------------
# Domain specs
# ---------------------------------------------------------------------------


def test_specs_contain_all_domains() -> None:
    specs = build_domain_specs(ToyConfig())
    assert set(specs) == {"D1", "D2", "D3", "T1", "T2"}


def test_D1_is_canonical() -> None:
    cfg = ToyConfig()
    d1 = build_domain_specs(cfg)["D1"]
    assert torch.equal(d1.mu, torch.zeros(cfg.d_x))
    assert torch.equal(d1.cov_factor, torch.eye(cfg.d_x))


def test_training_mu_norms() -> None:
    cfg = ToyConfig()
    specs = build_domain_specs(cfg)
    for name in ("D2", "D3"):
        assert specs[name].mu.norm().item() == pytest.approx(cfg.mu_norm, abs=1e-6)


def test_all_covariances_are_pd() -> None:
    specs = build_domain_specs(ToyConfig())
    for name, spec in specs.items():
        eigs = torch.linalg.eigvalsh(spec.cov)
        assert eigs.min().item() > 0, f"{name} covariance is not PD (min eig {eigs.min()})"


def test_delta_directions_are_orthogonal() -> None:
    """Recovered beta-shift directions for D2 and D3 should be orthogonal unit vectors."""
    cfg = ToyConfig()
    specs = build_domain_specs(cfg)
    a = cfg.head_shift_train
    d1 = (specs["D2"].beta_star - specs["D1"].beta_star) / a
    d2 = (specs["D3"].beta_star - specs["D1"].beta_star) / a
    assert d1.norm().item() == pytest.approx(1.0, abs=1e-5)
    assert d2.norm().item() == pytest.approx(1.0, abs=1e-5)
    assert abs(d1.dot(d2).item()) < 1e-5


def test_T2_head_shift_exceeds_T1() -> None:
    cfg = ToyConfig()
    specs = build_domain_specs(cfg)
    shift_T1 = (specs["T1"].beta_star - specs["D1"].beta_star).norm().item()
    shift_T2 = (specs["T2"].beta_star - specs["D1"].beta_star).norm().item()
    assert shift_T2 > shift_T1


def test_T1_beta_is_interpolation() -> None:
    cfg = ToyConfig()
    specs = build_domain_specs(cfg)
    expected = specs["D1"].beta_star + 0.5 * (
        (specs["D2"].beta_star - specs["D1"].beta_star)
        + (specs["D3"].beta_star - specs["D1"].beta_star)
    )
    assert torch.allclose(specs["T1"].beta_star, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# sample_domain
# ---------------------------------------------------------------------------


def test_sample_domain_shapes() -> None:
    cfg = ToyConfig()
    phi = build_phi_star(cfg)
    spec = build_domain_specs(cfg)["D2"]
    x, y = sample_domain(phi, spec, n=17, noise_sigma=0.1, seed=123)
    assert x.shape == (17, cfg.d_x)
    assert y.shape == (17,)


def test_sample_domain_is_deterministic() -> None:
    cfg = ToyConfig()
    phi = build_phi_star(cfg)
    spec = build_domain_specs(cfg)["D2"]
    x1, y1 = sample_domain(phi, spec, n=64, noise_sigma=0.1, seed=42)
    x2, y2 = sample_domain(phi, spec, n=64, noise_sigma=0.1, seed=42)
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


def test_sample_mean_and_cov_match_spec() -> None:
    """Empirical moments of a large sample should match the ground-truth spec."""
    cfg = ToyConfig()
    phi = build_phi_star(cfg)
    spec = build_domain_specs(cfg)["D2"]
    n = 50_000
    x, _ = sample_domain(phi, spec, n=n, noise_sigma=0.0, seed=7)
    emp_mu = x.mean(dim=0)
    emp_cov = torch.cov(x.T)
    assert (emp_mu - spec.mu).norm().item() < 0.05
    err = (emp_cov - spec.cov).norm() / spec.cov.norm()
    assert err.item() < 0.05


def test_ols_recovers_beta_star_without_noise() -> None:
    """With zero observation noise, OLS on (phi_star(x), y) should recover beta*."""
    cfg = ToyConfig()
    phi = build_phi_star(cfg)
    spec = build_domain_specs(cfg)["D3"]
    x, y = sample_domain(phi, spec, n=2_000, noise_sigma=0.0, seed=3)
    with torch.no_grad():
        z = phi(x)
    beta_hat = torch.linalg.pinv(z) @ y
    assert (beta_hat - spec.beta_star).norm().item() < 1e-3


# ---------------------------------------------------------------------------
# Full dataset
# ---------------------------------------------------------------------------


def test_dataset_sizes() -> None:
    ds = build_toy_dataset(
        n_train_per_corpus=500,
        n_test_support=50,
        n_test_query=100,
    )
    for name in ("D1", "D2", "D3"):
        x, y = ds.train[name]
        assert x.shape == (500, ds.config.d_x)
        assert y.shape == (500,)
    for name in ("T1", "T2"):
        xs, ys = ds.test[name]["support"]
        xq, yq = ds.test[name]["query"]
        assert xs.shape == (50, ds.config.d_x)
        assert ys.shape == (50,)
        assert xq.shape == (100, ds.config.d_x)
        assert yq.shape == (100,)


def test_dataset_is_deterministic() -> None:
    kwargs = {"n_train_per_corpus": 100, "n_test_support": 10, "n_test_query": 20}
    ds1 = build_toy_dataset(cfg=ToyConfig(seed=42), **kwargs)
    ds2 = build_toy_dataset(cfg=ToyConfig(seed=42), **kwargs)
    for name in ("D1", "D2", "D3"):
        assert torch.equal(ds1.train[name][0], ds2.train[name][0])
        assert torch.equal(ds1.train[name][1], ds2.train[name][1])
    for name in ("T1", "T2"):
        for split in ("support", "query"):
            assert torch.equal(ds1.test[name][split][0], ds2.test[name][split][0])
            assert torch.equal(ds1.test[name][split][1], ds2.test[name][split][1])


def test_dataset_changes_with_seed() -> None:
    kwargs = {"n_train_per_corpus": 100, "n_test_support": 10, "n_test_query": 20}
    ds1 = build_toy_dataset(cfg=ToyConfig(seed=1), **kwargs)
    ds2 = build_toy_dataset(cfg=ToyConfig(seed=2), **kwargs)
    # D1 shares mu=0, Sigma=I regardless of seed, but X samples are independent.
    assert not torch.equal(ds1.train["D1"][0], ds2.train["D1"][0])
