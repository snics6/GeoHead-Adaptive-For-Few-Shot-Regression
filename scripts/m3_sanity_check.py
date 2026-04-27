"""M3 sanity-check driver (CLI wrapper around :func:`run_sanity_check`).

Usage
-----
::

    # defaults: 1500 outer_steps × 3 learners × 4 methods × 2 corpora × 5 k-shots × 5 seeds
    python -m scripts.m3_sanity_check

    # smoke mode (≈ 10 s) — small schedule for quick verification
    python -m scripts.m3_sanity_check --smoke

    # custom seed + output directory
    python -m scripts.m3_sanity_check --seed 7 --output-dir results/m3_seed7

    # select a subset of learners (e.g. skip B2 to save time)
    python -m scripts.m3_sanity_check --learners B1 P
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import torch

from geohead.experiments.sanity import (
    LEARNERS,
    SanityConfig,
    run_sanity_check,
)
from geohead.training.b1 import B1Config
from geohead.training.baseline import BaselineConfig
from geohead.training.geohead import GeoHeadConfig
from geohead.training.warmup import WarmupConfig


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run the M3 sanity check: B1 (source-only), B2 (DARE+ridge), "
            "P (GeoHead) × 4 test-time adaptation methods on the toy dataset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/m3_sanity_check"),
        help="Directory for artefacts (overwrites files of the same name).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Master seed; all sub-generators derive from it.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (e.g. 'cpu', 'cuda', 'cuda:0').",
    )
    p.add_argument(
        "--learners",
        nargs="+",
        default=list(LEARNERS),
        choices=list(LEARNERS),
        help="Subset of learners to run.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny schedule for a quick end-to-end smoke test (~10 s).",
    )

    # Schedule overrides (in case the user wants to tune without editing code).
    p.add_argument("--warmup-epochs", type=int, default=None)
    p.add_argument("--b1-outer-steps", type=int, default=None)
    p.add_argument("--baseline-outer-steps", type=int, default=None)
    p.add_argument("--geohead-outer-steps", type=int, default=None)
    p.add_argument("--n-seeds", type=int, default=None)
    return p


def _apply_overrides(
    config: SanityConfig, args: argparse.Namespace
) -> SanityConfig:
    """Return a new config with CLI overrides applied."""
    warmup = config.warmup
    b1 = config.b1
    baseline = config.baseline
    geohead = config.geohead
    eval_ = config.eval

    if args.warmup_epochs is not None:
        warmup = dataclasses.replace(warmup, epochs=int(args.warmup_epochs))
    if args.b1_outer_steps is not None:
        b1 = dataclasses.replace(b1, outer_steps=int(args.b1_outer_steps))
    if args.baseline_outer_steps is not None:
        baseline = dataclasses.replace(
            baseline, outer_steps=int(args.baseline_outer_steps)
        )
    if args.geohead_outer_steps is not None:
        geohead = dataclasses.replace(
            geohead, outer_steps=int(args.geohead_outer_steps)
        )
    if args.n_seeds is not None:
        eval_ = dataclasses.replace(eval_, n_seeds=int(args.n_seeds))

    return dataclasses.replace(
        config,
        warmup=warmup,
        b1=b1,
        baseline=baseline,
        geohead=geohead,
        eval=eval_,
        learners=tuple(args.learners),
        device=args.device,
        master_seed=int(args.seed),
    )


def _smoke_config(base: SanityConfig) -> SanityConfig:
    """Tiny schedule for the ``--smoke`` flag (≈10 s on CPU)."""
    warmup = WarmupConfig(epochs=2, batch_size=128, lr=1e-3)
    b1 = B1Config(
        outer_steps=30,
        support_size=16,
        query_size=32,
        batch_source_size=32,
        batch_target_size=32,
        log_every=10,
    )
    baseline = BaselineConfig(
        outer_steps=30,
        support_size=16,
        query_size=32,
        batch_source_size=32,
        batch_target_size=32,
        log_every=10,
    )
    geohead = GeoHeadConfig(
        outer_steps=30,
        inner_steps=1,
        support_size=16,
        query_size=32,
        batch_source_size=32,
        batch_target_size=32,
        log_every=10,
    )
    eval_ = dataclasses.replace(
        base.eval, k_shots=(1, 4), n_seeds=2
    )
    return dataclasses.replace(
        base,
        n_train_per_corpus=500,
        n_test_support=50,
        n_test_query=200,
        warmup=warmup,
        b1=b1,
        baseline=baseline,
        geohead=geohead,
        eval=eval_,
    )


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config = SanityConfig()
    if args.smoke:
        config = _smoke_config(config)
    config = _apply_overrides(config, args)

    print("[m3] config:")
    print(f"  output_dir       = {args.output_dir}")
    print(f"  master_seed      = {config.master_seed}")
    print(f"  device           = {config.device}")
    print(f"  learners         = {list(config.learners)}")
    print(f"  warmup.epochs    = {config.warmup.epochs}")
    print(f"  b1.steps         = {config.b1.outer_steps}")
    print(f"  baseline.steps   = {config.baseline.outer_steps}")
    print(f"  geohead.steps    = {config.geohead.outer_steps}")
    print(f"  eval.k_shots     = {list(config.eval.k_shots)}")
    print(f"  eval.n_seeds     = {config.eval.n_seeds}")

    torch.manual_seed(config.master_seed)
    t0 = time.perf_counter()
    result = run_sanity_check(config=config, output_dir=args.output_dir)
    dt = time.perf_counter() - t0
    print(f"[m3] done in {dt:.1f} s")
    print(f"[m3] artefacts written to: {result.output_dir}")
    print(f"[m3] records: {len(result.records)} rows")
    print(f"[m3] aggregated: {len(result.aggregated)} cells")
    print(f"[m3] summary: {result.output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
