"""Unit tests for :mod:`geohead.experiments.main` (M4 main experiment).

All tests use a **tiny** schedule so the full suite still finishes in a
few seconds on CPU.  They verify:

* :class:`M4Config` defaults match §12 M4 (5 000 outer steps, 3 train
  seeds, 20 eval seeds).
* Validation of ``n_train_seeds``, ``learners``, ``encoder_*``.
* :func:`run_main_experiment` creates the expected artefact layout with
  one ``run_i/`` sub-folder per train seed.
* Record cardinality = ``|learners| × 2 × |k_shots| × n_seeds × |methods| × n_train_seeds``.
* Every record carries a ``train_seed`` field ∈ ``range(n_train_seeds)``.
* Aggregated CI is computed over ``n_train_seeds × n_seeds`` samples.
* Determinism: same config → identical records.
* Different ``master_seed`` → records differ.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from geohead.evaluation.runner import EvalConfig
from geohead.experiments.main import (
    M4Config,
    M4Result,
    run_main_experiment,
)
from geohead.experiments.sanity import LEARNERS
from geohead.training.baseline import BaselineConfig
from geohead.training.geohead import GeoHeadConfig
from geohead.training.warmup import WarmupConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_config(
    n_train_seeds: int = 2,
    learners: tuple[str, ...] = LEARNERS,
    master_seed: int = 0,
) -> M4Config:
    """~2-3 s schedule suitable for unit tests."""
    return M4Config(
        n_train_per_corpus=200,
        n_test_support=40,
        n_test_query=80,
        encoder_hidden=(32, 32),
        encoder_p=16,
        warmup=WarmupConfig(epochs=1, batch_size=128, lr=1e-3),
        baseline=BaselineConfig(
            outer_steps=5,
            support_size=8,
            query_size=16,
            batch_source_size=16,
            batch_target_size=16,
            log_every=5,
        ),
        geohead=GeoHeadConfig(
            outer_steps=5,
            inner_steps=1,
            support_size=8,
            query_size=16,
            batch_source_size=16,
            batch_target_size=16,
            log_every=5,
        ),
        eval=EvalConfig(k_shots=(1, 4), n_seeds=2),
        learners=learners,
        n_train_seeds=n_train_seeds,
        device="cpu",
        master_seed=master_seed,
    )


# ---------------------------------------------------------------------------
# Config defaults & validation
# ---------------------------------------------------------------------------


def test_m4_config_defaults_match_proposal() -> None:
    c = M4Config()
    assert c.learners == LEARNERS
    assert c.warmup.epochs == 30
    # M4-specific: 5 000 outer steps, 20 eval seeds, 3 train seeds.
    assert c.baseline.outer_steps == 5000
    assert c.geohead.outer_steps == 5000
    assert c.eval.k_shots == (1, 4, 8, 16, 24)
    assert c.eval.n_seeds == 20
    assert c.n_train_seeds == 3
    # Shared with v8 SanityConfig.
    assert c.encoder_hidden == (128, 128)
    assert c.encoder_p == 32
    # Preconditioned inner rule must be on for both train and eval.
    assert c.geohead.preconditioned_inner is True
    assert c.eval.inner_preconditioned is True


def test_m4_config_rejects_unknown_learner() -> None:
    with pytest.raises(ValueError, match="unknown learner"):
        M4Config(learners=("B1", "BOGUS"))


def test_m4_config_rejects_duplicate_learner() -> None:
    with pytest.raises(ValueError, match="duplicate learner"):
        M4Config(learners=("B1", "B1"))


def test_m4_config_rejects_zero_n_train_seeds() -> None:
    with pytest.raises(ValueError, match="n_train_seeds"):
        M4Config(n_train_seeds=0)


def test_m4_config_rejects_negative_n_train_seeds() -> None:
    with pytest.raises(ValueError, match="n_train_seeds"):
        M4Config(n_train_seeds=-1)


def test_m4_config_to_sanity_config_roundtrip() -> None:
    c = _tiny_config(master_seed=7)
    sc = c.to_sanity_config()
    assert sc.master_seed == 7
    assert sc.learners == c.learners
    assert sc.encoder_p == c.encoder_p
    assert sc.baseline.outer_steps == c.baseline.outer_steps
    # Deep copy: the SanityConfig's sub-dataclasses are distinct instances.
    assert sc.toy is not c.toy
    assert sc.baseline is not c.baseline
    assert sc.geohead is not c.geohead
    assert sc.eval is not c.eval

    sc2 = c.to_sanity_config(master_seed=42)
    assert sc2.master_seed == 42


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------


def test_run_main_experiment_creates_expected_layout(tmp_path: Path) -> None:
    cfg = _tiny_config(n_train_seeds=2)
    result = run_main_experiment(cfg, output_dir=tmp_path)

    assert isinstance(result, M4Result)
    assert result.output_dir == tmp_path
    # Top-level files
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "records.jsonl").exists()
    assert (tmp_path / "aggregated.csv").exists()
    assert (tmp_path / "summary.md").exists()
    assert (tmp_path / "comparison.md").exists()
    # One history folder per train seed
    for i in range(cfg.n_train_seeds):
        run_dir = tmp_path / f"run_{i}"
        assert run_dir.is_dir()
        assert (run_dir / "config.json").exists()
        assert (run_dir / "history" / "warmup.json").exists()
        assert (run_dir / "history" / "b1.json").exists()
        assert (run_dir / "history" / "baseline.json").exists()
        assert (run_dir / "history" / "geohead.json").exists()
    # At least one plot written
    pngs = sorted((tmp_path / "plots").glob("*.png"))
    assert pngs, "expected at least one sample-efficiency plot"


def test_run_main_experiment_record_cardinality(tmp_path: Path) -> None:
    cfg = _tiny_config(n_train_seeds=3)
    result = run_main_experiment(cfg, output_dir=tmp_path)

    expected = (
        len(cfg.learners)
        * 2  # T1, T2
        * len(cfg.eval.k_shots)
        * cfg.eval.n_seeds
        * len(cfg.eval.methods)
        * cfg.n_train_seeds
    )
    assert len(result.records) == expected

    required = {
        "learner", "corpus", "k_shot", "seed", "method",
        "train_seed",
        "mse", "mae", "delta_l2", "delta_geo",
    }
    train_seeds_seen = set()
    for r in result.records:
        assert required.issubset(r.keys())
        assert r["train_seed"] in range(cfg.n_train_seeds)
        train_seeds_seen.add(r["train_seed"])
    assert train_seeds_seen == set(range(cfg.n_train_seeds))


def test_run_main_experiment_aggregation_uses_all_samples(tmp_path: Path) -> None:
    cfg = _tiny_config(n_train_seeds=2)
    result = run_main_experiment(cfg, output_dir=tmp_path)

    # Every aggregated cell should be backed by n_train_seeds × n_seeds
    # samples (aggregate() silently collapses train_seed because it is
    # not in the group-by key).
    expected_n = cfg.n_train_seeds * cfg.eval.n_seeds
    for row in result.aggregated:
        assert row["seed_count"] == expected_n

    # Cardinality of the aggregated table.
    expected_cells = (
        len(cfg.learners) * 2 * len(cfg.eval.k_shots) * len(cfg.eval.methods)
    )
    assert len(result.aggregated) == expected_cells


def test_run_main_experiment_records_jsonl_has_train_seed(tmp_path: Path) -> None:
    cfg = _tiny_config(n_train_seeds=2)
    run_main_experiment(cfg, output_dir=tmp_path)
    lines = (tmp_path / "records.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(l) for l in lines]
    assert rows
    assert all("train_seed" in r for r in rows)
    assert {r["train_seed"] for r in rows} == set(range(cfg.n_train_seeds))


def test_run_main_experiment_deterministic(tmp_path: Path) -> None:
    cfg = _tiny_config(master_seed=42, n_train_seeds=2)
    r1 = run_main_experiment(cfg, output_dir=tmp_path / "a")
    r2 = run_main_experiment(cfg, output_dir=tmp_path / "b")

    assert len(r1.records) == len(r2.records)
    for a, b in zip(r1.records, r2.records):
        assert a["learner"] == b["learner"]
        assert a["corpus"] == b["corpus"]
        assert a["k_shot"] == b["k_shot"]
        assert a["seed"] == b["seed"]
        assert a["method"] == b["method"]
        assert a["train_seed"] == b["train_seed"]
        for key in ("mse", "mae", "delta_l2", "delta_geo"):
            assert a[key] == pytest.approx(b[key], rel=0, abs=1e-6)


def test_run_main_experiment_different_seeds_differ(tmp_path: Path) -> None:
    cfg0 = _tiny_config(master_seed=0, n_train_seeds=2)
    cfg1 = _tiny_config(master_seed=1, n_train_seeds=2)
    r0 = run_main_experiment(cfg0, output_dir=tmp_path / "a")
    r1 = run_main_experiment(cfg1, output_dir=tmp_path / "b")
    diffs = [abs(a["mse"] - b["mse"]) for a, b in zip(r0.records, r1.records)]
    assert max(diffs) > 1e-6


def test_run_main_experiment_subset_learners(tmp_path: Path) -> None:
    cfg = _tiny_config(learners=("B1", "P"), n_train_seeds=2)
    result = run_main_experiment(cfg, output_dir=tmp_path)
    assert {r["learner"] for r in result.records} == {"B1", "P"}
    # B2 history must not be written for any run.
    for i in range(cfg.n_train_seeds):
        assert not (tmp_path / f"run_{i}" / "history" / "baseline.json").exists()


def test_run_main_experiment_fair_comparison_invariant(tmp_path: Path) -> None:
    """For every record, the ``none`` method's head correction must be 0."""
    cfg = _tiny_config(n_train_seeds=2)
    result = run_main_experiment(cfg, output_dir=tmp_path)
    for r in result.records:
        if r["method"] == "none":
            assert r["delta_l2"] == pytest.approx(0.0, abs=1e-8)
            assert r["delta_geo"] == pytest.approx(0.0, abs=1e-8)


def test_run_main_experiment_config_json_roundtrip(tmp_path: Path) -> None:
    cfg = _tiny_config(master_seed=13, n_train_seeds=2)
    result = run_main_experiment(cfg, output_dir=tmp_path)
    raw = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert raw["master_seed"] == 13
    assert raw["n_train_seeds"] == 2
    assert raw["learners"] == list(cfg.learners)
    # Per-run config was written too.
    for i in range(cfg.n_train_seeds):
        sub = json.loads(
            (tmp_path / f"run_{i}" / "config.json").read_text(encoding="utf-8")
        )
        assert sub["master_seed"] == 13 + i * 1_000_000
    # Result points to the same M4Config instance.
    assert dataclasses.asdict(result.config)["n_train_seeds"] == 2


def test_run_main_experiment_summary_mentions_ci(tmp_path: Path) -> None:
    cfg = _tiny_config(n_train_seeds=2)
    run_main_experiment(cfg, output_dir=tmp_path)
    text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "M4" in text
    assert "n_train_seeds" in text
    assert "95 % CI" in text
    for learner in cfg.learners:
        assert learner in text
    for corpus in ("T1", "T2"):
        assert corpus in text


def test_run_main_experiment_comparison_md_structure(tmp_path: Path) -> None:
    """`comparison.md` must contain all 5 headline sections."""
    cfg = _tiny_config(n_train_seeds=2)
    run_main_experiment(cfg, output_dir=tmp_path)
    text = (tmp_path / "comparison.md").read_text(encoding="utf-8")
    # Section headings
    assert "## 1. Headline" in text
    assert "## 2. Relative improvement" in text
    assert "## 3. Method ranking" in text
    assert "## 4. Full MSE tables" in text
    assert "## 5. Head-correction sanity" in text
    # Every learner and every method must be referenced somewhere.
    for ll in cfg.learners:
        assert ll in text
    for m in cfg.eval.methods:
        assert f"`{m}`" in text
    # Relative-improvement sign formatting ("+X.Y%" or "-X.Y%").
    import re
    assert re.search(r"[+\-]\d+\.\d%", text), \
        "expected signed percentage improvement in comparison.md"


def test_run_main_experiment_unified_axes_shared_across_learners(
    tmp_path: Path,
) -> None:
    """Every sample-efficiency plot for one corpus should be saved — the
    unified-axes helper must not crash on degenerate (min==max) cells.
    """
    cfg = _tiny_config(n_train_seeds=2)
    run_main_experiment(cfg, output_dir=tmp_path)
    pngs = {p.name for p in (tmp_path / "plots").glob("*.png")}
    # Per-learner files: 2 corpora × 3 learners = 6.
    for corpus in ("T1", "T2"):
        for ll in cfg.learners:
            assert f"sample_efficiency_{corpus}_{ll}.png" in pngs
            assert f"head_correction_{corpus}_{ll}.png" in pngs
    # Per-method files: 2 corpora × 4 methods = 8.
    for corpus in ("T1", "T2"):
        for m in cfg.eval.methods:
            assert f"sample_efficiency_{corpus}_by_{m}.png" in pngs


def test_compute_unified_axes_returns_nontrivial_ranges(tmp_path: Path) -> None:
    """The axis-range helper produces positive-width (lo < hi) ranges."""
    from geohead.experiments.main import _compute_unified_axes

    cfg = _tiny_config(n_train_seeds=2)
    result = run_main_experiment(cfg, output_dir=tmp_path)
    mse_ylim, delta_xlim = _compute_unified_axes(result.aggregated)
    assert set(mse_ylim.keys()) == {"T1", "T2"}
    for corpus, (lo, hi) in mse_ylim.items():
        assert 0 < lo < hi, f"{corpus} mse_ylim invalid: ({lo}, {hi})"
    for corpus, (lo, hi) in delta_xlim.items():
        assert 0 == lo < hi, f"{corpus} delta_xlim invalid: ({lo}, {hi})"


def test_run_main_experiment_n_train_seeds_1_degrades_to_single_run(
    tmp_path: Path,
) -> None:
    """With n_train_seeds=1 the result should be statistically comparable
    to a plain sanity-check run with the same seed (records are reordered
    but come from identical underlying computations).
    """
    cfg = _tiny_config(n_train_seeds=1, master_seed=3)
    result = run_main_experiment(cfg, output_dir=tmp_path)
    assert result.config.n_train_seeds == 1
    # All records must carry train_seed=0.
    assert all(r["train_seed"] == 0 for r in result.records)
    # One run_0 folder, no run_1.
    assert (tmp_path / "run_0").is_dir()
    assert not (tmp_path / "run_1").exists()
