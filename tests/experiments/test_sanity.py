"""Unit tests for :mod:`geohead.experiments.sanity`.

Every non-trivial test uses a **tiny** schedule so the full suite still
runs in a couple of seconds on CPU.  The tests verify:

* :class:`SanityConfig` validation (unknown / duplicate learners).
* :func:`run_sanity_check` creates the expected artefact layout.
* Record cardinality matches ``|learners| · |corpora| · |k_shots| · n_seeds · |methods|``.
* Determinism under a fixed ``master_seed``.
* A learner subset (e.g. ``("B1",)``) yields records only for that
  learner.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch

from geohead.evaluation.runner import EvalConfig
from geohead.experiments.sanity import (
    LEARNERS,
    SanityConfig,
    SanityResult,
    run_sanity_check,
)
from geohead.training.baseline import BaselineConfig
from geohead.training.geohead import GeoHeadConfig
from geohead.training.warmup import WarmupConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_config(
    learners: tuple[str, ...] = LEARNERS,
    master_seed: int = 0,
) -> SanityConfig:
    """A ~1 s schedule suitable for unit tests."""
    return SanityConfig(
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
        device="cpu",
        master_seed=master_seed,
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_sanity_config_defaults_match_proposal() -> None:
    c = SanityConfig()
    assert c.learners == LEARNERS
    assert c.warmup.epochs == 30
    assert c.baseline.outer_steps == 1500
    assert c.geohead.outer_steps == 1500
    assert c.eval.k_shots == (1, 4, 8, 16, 24)
    assert c.eval.n_seeds == 5
    assert c.encoder_hidden == (128, 128)
    assert c.encoder_p == 32


def test_sanity_config_rejects_unknown_learner() -> None:
    with pytest.raises(ValueError, match="unknown learner"):
        SanityConfig(learners=("B1", "BOGUS"))


def test_sanity_config_rejects_duplicate_learner() -> None:
    with pytest.raises(ValueError, match="duplicate learner"):
        SanityConfig(learners=("B1", "B1"))


def test_sanity_config_rejects_empty_hidden() -> None:
    with pytest.raises(ValueError, match="encoder_hidden"):
        SanityConfig(encoder_hidden=())


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------


def test_run_sanity_check_creates_all_artefacts(tmp_path: Path) -> None:
    cfg = _tiny_config()
    result = run_sanity_check(cfg, output_dir=tmp_path)

    assert isinstance(result, SanityResult)
    assert result.output_dir == tmp_path
    # Core files
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "records.jsonl").exists()
    assert (tmp_path / "aggregated.csv").exists()
    assert (tmp_path / "summary.md").exists()
    # Histories
    assert (tmp_path / "history" / "warmup.json").exists()
    assert (tmp_path / "history" / "b1.json").exists()
    assert (tmp_path / "history" / "baseline.json").exists()
    assert (tmp_path / "history" / "geohead.json").exists()
    # Plots: at least one per corpus × learner, plus per-method view
    pngs = sorted((tmp_path / "plots").glob("*.png"))
    assert len(pngs) > 0


def test_run_sanity_check_record_cardinality(tmp_path: Path) -> None:
    cfg = _tiny_config()
    result = run_sanity_check(cfg, output_dir=tmp_path)

    expected = (
        len(cfg.learners)
        * 2  # T1, T2
        * len(cfg.eval.k_shots)
        * cfg.eval.n_seeds
        * len(cfg.eval.methods)
    )
    assert len(result.records) == expected

    # Every record must carry the six canonical keys plus "learner".
    required = {"learner", "corpus", "k_shot", "seed", "method", "mse",
                "mae", "delta_l2", "delta_geo"}
    for r in result.records:
        assert required.issubset(r.keys())
        assert r["learner"] in cfg.learners


def test_run_sanity_check_records_jsonl_roundtrips(tmp_path: Path) -> None:
    cfg = _tiny_config()
    result = run_sanity_check(cfg, output_dir=tmp_path)

    lines = (tmp_path / "records.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(result.records)
    reloaded = [json.loads(line) for line in lines]
    # Same learner set
    assert {r["learner"] for r in reloaded} == set(cfg.learners)


def test_run_sanity_check_deterministic_same_seed(tmp_path: Path) -> None:
    cfg = _tiny_config(master_seed=42)
    r1 = run_sanity_check(cfg, output_dir=tmp_path / "a")
    r2 = run_sanity_check(cfg, output_dir=tmp_path / "b")

    assert len(r1.records) == len(r2.records)
    # Records are appended in a fixed nested-loop order so element-wise
    # comparison is well defined.
    for a, b in zip(r1.records, r2.records):
        assert a["learner"] == b["learner"]
        assert a["corpus"] == b["corpus"]
        assert a["k_shot"] == b["k_shot"]
        assert a["seed"] == b["seed"]
        assert a["method"] == b["method"]
        for key in ("mse", "mae", "delta_l2", "delta_geo"):
            assert a[key] == pytest.approx(b[key], rel=0, abs=1e-6)


def test_run_sanity_check_different_seeds_differ(tmp_path: Path) -> None:
    cfg0 = _tiny_config(master_seed=0)
    cfg1 = _tiny_config(master_seed=1)
    r0 = run_sanity_check(cfg0, output_dir=tmp_path / "a")
    r1 = run_sanity_check(cfg1, output_dir=tmp_path / "b")
    # At least one record should differ materially.
    diffs = [
        abs(a["mse"] - b["mse"]) for a, b in zip(r0.records, r1.records)
    ]
    assert max(diffs) > 1e-6


def test_run_sanity_check_subset_learners(tmp_path: Path) -> None:
    cfg = _tiny_config(learners=("B1",))
    result = run_sanity_check(cfg, output_dir=tmp_path)
    assert {r["learner"] for r in result.records} == {"B1"}
    # B2/P histories should NOT be written if their learners are not asked.
    assert not (tmp_path / "history" / "baseline.json").exists()
    assert not (tmp_path / "history" / "geohead.json").exists()
    assert (tmp_path / "history" / "b1.json").exists()


def test_run_sanity_check_aggregated_has_expected_keys(tmp_path: Path) -> None:
    cfg = _tiny_config()
    result = run_sanity_check(cfg, output_dir=tmp_path)
    required = {
        "learner", "corpus", "k_shot", "method",
        "mse_mean", "mse_sem", "mse_ci95",
        "mae_mean", "mae_sem", "mae_ci95",
        "delta_l2_mean", "delta_l2_sem", "delta_l2_ci95",
        "delta_geo_mean", "delta_geo_sem", "delta_geo_ci95",
        "seed_count",
    }
    for row in result.aggregated:
        assert required.issubset(row.keys())
    # Cardinality of aggregated table
    expected = (
        len(cfg.learners) * 2 * len(cfg.eval.k_shots) * len(cfg.eval.methods)
    )
    assert len(result.aggregated) == expected


def test_run_sanity_check_config_json_roundtrip(tmp_path: Path) -> None:
    cfg = _tiny_config(master_seed=7)
    result = run_sanity_check(cfg, output_dir=tmp_path)
    raw = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert raw["master_seed"] == 7
    assert raw["learners"] == list(cfg.learners)
    # Nested dataclass dump
    assert raw["warmup"]["epochs"] == cfg.warmup.epochs
    assert raw["baseline"]["outer_steps"] == cfg.baseline.outer_steps
    assert raw["geohead"]["outer_steps"] == cfg.geohead.outer_steps

    # Sanity: result object still points to the same config instance.
    assert dataclasses.asdict(result.config)["master_seed"] == 7


def test_run_sanity_check_summary_mentions_learners(tmp_path: Path) -> None:
    cfg = _tiny_config()
    run_sanity_check(cfg, output_dir=tmp_path)
    text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    # Every learner should appear in the markdown table(s).
    for learner in cfg.learners:
        assert learner in text
    # Each corpus too.
    for corpus in ("T1", "T2"):
        assert corpus in text


def test_run_sanity_check_fair_comparison_invariant(tmp_path: Path) -> None:
    """For fixed (learner, corpus, k, seed), the ``none`` method's
    head correction must be exactly 0 — this doubles as a sanity check
    that every method is evaluated against the same β₀ within a cell.
    """
    cfg = _tiny_config()
    result = run_sanity_check(cfg, output_dir=tmp_path)
    for r in result.records:
        if r["method"] == "none":
            assert r["delta_l2"] == pytest.approx(0.0, abs=1e-8)
            assert r["delta_geo"] == pytest.approx(0.0, abs=1e-8)


def test_run_sanity_check_handles_cpu_device_string(tmp_path: Path) -> None:
    cfg = dataclasses.replace(_tiny_config(), device="cpu")
    # Make sure torch doesn't choke on the string form.
    result = run_sanity_check(cfg, output_dir=tmp_path)
    assert len(result.records) > 0


def test_run_sanity_check_warmup_history_logged(tmp_path: Path) -> None:
    cfg = _tiny_config()
    result = run_sanity_check(cfg, output_dir=tmp_path)
    # Epoch 0 baseline + epochs post-training
    assert len(result.warmup_history.train_loss) == cfg.warmup.epochs + 1
