"""Tests for ``src/geohead/evaluation/runner.py``.

Specifications under test live in ``docs/design.md`` §9.3.
"""

from __future__ import annotations

import pytest
import torch

from geohead.data.toy import ToyConfig, build_toy_dataset
from geohead.evaluation.runner import (
    EvalConfig,
    aggregate,
    evaluate_model,
    to_pandas,
)
from geohead.models.encoder import MLPEncoder
from geohead.models.head import LinearHead


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tiny_test_corpora(
    n_support: int = 40,
    n_query: int = 120,
    d_x: int = 6,
    seed: int = 0,
):
    """Two toy test corpora with the ``{"support": ..., "query": ...}``
    structure used by :func:`build_toy_dataset`.
    """
    g = torch.Generator().manual_seed(seed)
    true_w = torch.randn(d_x, generator=g)
    out: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    for k, name in enumerate(("T1", "T2")):
        gs = torch.Generator().manual_seed(seed + 100 * (k + 1))
        gq = torch.Generator().manual_seed(seed + 100 * (k + 1) + 1)
        x_s = torch.randn(n_support, d_x, generator=gs)
        y_s = x_s @ true_w + 0.05 * torch.randn(n_support, generator=gs)
        x_q = torch.randn(n_query, d_x, generator=gq)
        y_q = x_q @ true_w + 0.05 * torch.randn(n_query, generator=gq)
        out[name] = {"support": (x_s, y_s), "query": (x_q, y_q)}
    return out


def _make_models(d_x: int = 6, p: int = 8, seed: int = 0):
    torch.manual_seed(seed)
    encoder = MLPEncoder(d_x=d_x, hidden=(16,), p=p)
    head = LinearHead(p=p)
    return encoder, head


# ---------------------------------------------------------------------------
# EvalConfig validation
# ---------------------------------------------------------------------------


def test_eval_config_defaults() -> None:
    cfg = EvalConfig()
    assert cfg.k_shots == (1, 3, 5, 10, 20)
    assert cfg.n_seeds == 20
    assert cfg.methods == ("none", "ridge", "geo", "inner")
    assert cfg.inner_steps == 5
    assert cfg.inner_lr == 0.1


def test_eval_config_validation() -> None:
    with pytest.raises(ValueError):
        EvalConfig(k_shots=())
    with pytest.raises(ValueError):
        EvalConfig(k_shots=(0, 1, 2))
    with pytest.raises(ValueError):
        EvalConfig(k_shots=(1, -3))
    with pytest.raises(ValueError):
        EvalConfig(n_seeds=0)
    with pytest.raises(ValueError):
        EvalConfig(methods=())
    with pytest.raises(ValueError):
        EvalConfig(methods=("bogus",))
    with pytest.raises(ValueError):
        EvalConfig(ridge_lambda=-0.1)
    with pytest.raises(ValueError):
        EvalConfig(geo_lambda=-0.1)
    with pytest.raises(ValueError):
        EvalConfig(inner_lambda_h=-0.1)
    with pytest.raises(ValueError):
        EvalConfig(inner_lr=0)
    with pytest.raises(ValueError):
        EvalConfig(inner_steps=-1)
    with pytest.raises(ValueError):
        EvalConfig(head_reg_eps=-1e-6)


# ---------------------------------------------------------------------------
# evaluate_model: core contract
# ---------------------------------------------------------------------------


def test_evaluate_model_record_cardinality_and_keys() -> None:
    corpora = _tiny_test_corpora(n_support=40)
    encoder, head = _make_models()
    cfg = EvalConfig(k_shots=(1, 3, 5), n_seeds=4)
    records = evaluate_model(encoder, head, corpora, config=cfg)

    # |corpora| * |k_shots| * n_seeds * |methods| = 2 * 3 * 4 * 4 = 96
    assert len(records) == 2 * 3 * 4 * 4

    required = {
        "corpus",
        "k_shot",
        "seed",
        "method",
        "mse",
        "mae",
        "delta_l2",
        "delta_geo",
    }
    for rec in records:
        assert required.issubset(rec.keys())
        assert rec["corpus"] in corpora
        assert rec["k_shot"] in cfg.k_shots
        assert 0 <= rec["seed"] < cfg.n_seeds
        assert rec["method"] in cfg.methods


def test_evaluate_model_covers_full_grid() -> None:
    corpora = _tiny_test_corpora(n_support=40)
    encoder, head = _make_models()
    cfg = EvalConfig(k_shots=(1, 5), n_seeds=3, methods=("none", "ridge", "geo", "inner"))
    records = evaluate_model(encoder, head, corpora, config=cfg)

    keys = {(r["corpus"], r["k_shot"], r["seed"], r["method"]) for r in records}
    expected = {
        (c, k, s, m)
        for c in corpora
        for k in cfg.k_shots
        for s in range(cfg.n_seeds)
        for m in cfg.methods
    }
    assert keys == expected


def test_evaluate_model_none_method_has_zero_correction() -> None:
    """``method='none'`` means β̂ = β_0, so both correction metrics must
    be exactly 0.0."""
    corpora = _tiny_test_corpora(n_support=30)
    encoder, head = _make_models()
    cfg = EvalConfig(k_shots=(1, 5), n_seeds=2)
    records = evaluate_model(encoder, head, corpora, config=cfg)

    for rec in records:
        if rec["method"] == "none":
            assert rec["delta_l2"] == pytest.approx(0.0, abs=1e-7)
            assert rec["delta_geo"] == pytest.approx(0.0, abs=1e-7)


def test_evaluate_model_all_methods_see_same_support_sample() -> None:
    """Fair-comparison invariant (§9.3): for a fixed ``(corpus, k, seed)``
    triple, every method's query MSE must be computed from the *same*
    support sub-sample.

    For ``method='none'`` the query MSE does not depend on the support,
    but for the three adaptation methods it does.  The ``none`` MSE is
    constant across seeds (fixed β_0, fixed query), while each of the
    three adapted methods varies with seed — which is the expected
    signature of seeded support sub-sampling.
    """
    corpora = _tiny_test_corpora(n_support=40)
    encoder, head = _make_models()
    cfg = EvalConfig(
        k_shots=(5,),
        n_seeds=5,
        methods=("none", "ridge", "geo", "inner"),
    )
    records = evaluate_model(encoder, head, corpora, config=cfg)

    # Group by (corpus, method), collect MSEs per seed.
    groups: dict[tuple[str, str], list[float]] = {}
    for rec in records:
        groups.setdefault((rec["corpus"], rec["method"]), []).append(rec["mse"])

    for (corpus, method), mses in groups.items():
        if method == "none":
            # All seeds identical (β_0 is fixed, query is fixed).
            for v in mses[1:]:
                assert v == pytest.approx(mses[0], rel=1e-6), (
                    f"none MSE varies with seed on {corpus}: {mses}"
                )
        else:
            # Should vary with seed.
            spread = max(mses) - min(mses)
            assert spread > 0, (
                f"{method} MSE does not vary with seed on {corpus}: {mses}"
            )


def test_evaluate_model_deterministic() -> None:
    corpora = _tiny_test_corpora(n_support=30)
    enc1, head1 = _make_models()
    enc2, head2 = _make_models()

    cfg = EvalConfig(k_shots=(1, 3), n_seeds=3, seed_base=42)
    rec1 = evaluate_model(enc1, head1, corpora, config=cfg)
    rec2 = evaluate_model(enc2, head2, corpora, config=cfg)

    assert len(rec1) == len(rec2)
    for a, b in zip(rec1, rec2):
        assert a["corpus"] == b["corpus"]
        assert a["k_shot"] == b["k_shot"]
        assert a["seed"] == b["seed"]
        assert a["method"] == b["method"]
        for key in ("mse", "mae", "delta_l2", "delta_geo"):
            assert a[key] == pytest.approx(b[key], rel=1e-7, abs=1e-10)


def test_evaluate_model_seed_base_changes_results() -> None:
    corpora = _tiny_test_corpora(n_support=30)
    encoder, head = _make_models()

    cfg_a = EvalConfig(k_shots=(3,), n_seeds=2, seed_base=0)
    cfg_b = EvalConfig(k_shots=(3,), n_seeds=2, seed_base=1000)
    rec_a = evaluate_model(encoder, head, corpora, config=cfg_a)
    rec_b = evaluate_model(encoder, head, corpora, config=cfg_b)

    # Adapted-method MSEs must change when the support sub-sample changes.
    for a, b in zip(rec_a, rec_b):
        if a["method"] != "none":
            assert a["mse"] != b["mse"]


def test_evaluate_model_subset_methods() -> None:
    corpora = _tiny_test_corpora(n_support=30)
    encoder, head = _make_models()
    cfg = EvalConfig(k_shots=(3,), n_seeds=2, methods=("none", "inner"))
    records = evaluate_model(encoder, head, corpora, config=cfg)
    assert len(records) == 2 * 1 * 2 * 2
    assert {r["method"] for r in records} == {"none", "inner"}


def test_evaluate_model_rejects_k_shot_above_pool() -> None:
    corpora = _tiny_test_corpora(n_support=10)
    encoder, head = _make_models()
    cfg = EvalConfig(k_shots=(1, 20), n_seeds=1)
    with pytest.raises(ValueError):
        evaluate_model(encoder, head, corpora, config=cfg)


def test_evaluate_model_rejects_malformed_corpus_bundles() -> None:
    encoder, head = _make_models()
    with pytest.raises(ValueError):
        evaluate_model(encoder, head, {})  # empty
    # Missing "query" key
    with pytest.raises(ValueError):
        evaluate_model(
            encoder,
            head,
            {"T1": {"support": (torch.randn(5, 6), torch.randn(5))}},
        )
    # Non-matching shapes inside support
    with pytest.raises(ValueError):
        evaluate_model(
            encoder,
            head,
            {
                "T1": {
                    "support": (torch.randn(5, 6), torch.randn(5, 1)),
                    "query": (torch.randn(10, 6), torch.randn(10)),
                }
            },
        )


# ---------------------------------------------------------------------------
# Integration with the toy dataset
# ---------------------------------------------------------------------------


def test_evaluate_model_runs_on_toy_dataset() -> None:
    cfg_data = ToyConfig(seed=0)
    ds = build_toy_dataset(cfg_data, n_train_per_corpus=200)

    encoder = MLPEncoder(d_x=cfg_data.d_x, hidden=(32,), p=16)
    head = LinearHead(p=16)

    cfg = EvalConfig(k_shots=(1, 5, 10), n_seeds=4, seed_base=0)
    records = evaluate_model(encoder, head, ds.test, config=cfg)

    # 2 corpora * 3 k_shots * 4 seeds * 4 methods = 96
    assert len(records) == 96
    assert {r["corpus"] for r in records} == set(ds.test.keys())


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregate_cardinality_and_summary_fields() -> None:
    corpora = _tiny_test_corpora(n_support=30)
    encoder, head = _make_models()
    cfg = EvalConfig(k_shots=(1, 5), n_seeds=5)
    records = evaluate_model(encoder, head, corpora, config=cfg)
    agg = aggregate(records)

    # |corpora| * |k_shots| * |methods| = 2 * 2 * 4 = 16
    assert len(agg) == 16
    expected_keys = {
        "corpus", "k_shot", "method",
        "mse_mean", "mse_sem", "mse_ci95",
        "mae_mean", "mae_sem", "mae_ci95",
        "delta_l2_mean", "delta_l2_sem", "delta_l2_ci95",
        "delta_geo_mean", "delta_geo_sem", "delta_geo_ci95",
        "seed_count",
    }
    for row in agg:
        assert expected_keys.issubset(row.keys())
        assert row["seed_count"] == cfg.n_seeds
        # SEM / CI must be non-negative.
        for metric in ("mse", "mae", "delta_l2", "delta_geo"):
            assert row[f"{metric}_sem"] >= 0
            assert row[f"{metric}_ci95"] >= 0


def test_aggregate_none_method_has_zero_std_on_mse() -> None:
    """``method='none'`` is deterministic across seeds, so the SEM / CI
    of its MSE must be exactly zero.
    """
    corpora = _tiny_test_corpora(n_support=30)
    encoder, head = _make_models()
    cfg = EvalConfig(k_shots=(1, 5), n_seeds=4)
    records = evaluate_model(encoder, head, corpora, config=cfg)
    agg = aggregate(records)
    for row in agg:
        if row["method"] == "none":
            assert row["mse_sem"] == pytest.approx(0.0, abs=1e-8)
            assert row["mse_ci95"] == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# to_pandas (lazy pandas)
# ---------------------------------------------------------------------------


def test_to_pandas_roundtrip_if_available() -> None:
    pd = pytest.importorskip("pandas")
    corpora = _tiny_test_corpora(n_support=30)
    encoder, head = _make_models()
    cfg = EvalConfig(k_shots=(1,), n_seeds=2)
    records = evaluate_model(encoder, head, corpora, config=cfg)

    df = to_pandas(records)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(records)
    assert set(df.columns) >= {
        "corpus", "k_shot", "seed", "method", "mse", "mae", "delta_l2", "delta_geo"
    }
