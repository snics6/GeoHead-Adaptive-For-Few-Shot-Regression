"""M4 main-experiment driver (CLI wrapper around :func:`run_main_experiment`).

Usage
-----
::

    # Defaults (see docs/design.md §12 M4):
    #   n_train_seeds=3, eval.n_seeds=20, outer_steps=5000,
    #   k_shots=(1,4,8,16,24), learners=(B1,B2,P), preconditioned inner rule.
    python -m scripts.m4_main

    # Smoke test (~30 s on CPU): 2 train seeds × 2 eval seeds × 100 steps.
    python -m scripts.m4_main --smoke

    # Run on GPU with custom output directory.
    python -m scripts.m4_main --device cuda --output-dir results/m4_main_gpu

    # Scale up the training variance (5 train seeds).
    python -m scripts.m4_main --n-train-seeds 5
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import torch

from geohead.experiments.main import (
    M4Config,
    run_main_experiment,
)
from geohead.experiments.sanity import LEARNERS
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
            "Run the M4 main experiment: full-variance sample efficiency "
            "curves with 95% CI over n_train_seeds × eval.n_seeds samples "
            "per cell."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/m4_main"),
        help="Directory for artefacts (overwrites files of the same name).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Master seed; train_seed_i uses master_seed + i*1_000_000.",
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
        help="Tiny schedule for a quick end-to-end smoke test (~30 s).",
    )

    # Schedule overrides.
    p.add_argument("--n-train-seeds", type=int, default=None)
    p.add_argument("--n-seeds", type=int, default=None,
                   help="eval.n_seeds per train-seed run.")
    p.add_argument("--warmup-epochs", type=int, default=None)
    p.add_argument("--b1-outer-steps", type=int, default=None)
    p.add_argument("--baseline-outer-steps", type=int, default=None)
    p.add_argument("--geohead-outer-steps", type=int, default=None)
    return p


def _apply_overrides(
    config: M4Config, args: argparse.Namespace
) -> M4Config:
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

    kwargs: dict = dict(
        warmup=warmup,
        b1=b1,
        baseline=baseline,
        geohead=geohead,
        eval=eval_,
        learners=tuple(args.learners),
        device=args.device,
        master_seed=int(args.seed),
    )
    if args.n_train_seeds is not None:
        kwargs["n_train_seeds"] = int(args.n_train_seeds)

    return dataclasses.replace(config, **kwargs)


def _smoke_config(base: M4Config) -> M4Config:
    """Tiny schedule for ``--smoke`` (~30 s CPU)."""
    warmup = WarmupConfig(epochs=2, batch_size=128, lr=1e-3)
    b1 = B1Config(
        outer_steps=50,
        support_size=16,
        query_size=32,
        batch_source_size=32,
        batch_target_size=32,
        log_every=25,
    )
    baseline = BaselineConfig(
        outer_steps=50,
        support_size=16,
        query_size=32,
        batch_source_size=32,
        batch_target_size=32,
        log_every=25,
    )
    geohead = GeoHeadConfig(
        outer_steps=50,
        inner_steps=1,
        support_size=16,
        query_size=32,
        batch_source_size=32,
        batch_target_size=32,
        log_every=25,
    )
    eval_ = dataclasses.replace(base.eval, k_shots=(1, 4), n_seeds=2)
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
        n_train_seeds=2,
    )


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config = M4Config()
    if args.smoke:
        config = _smoke_config(config)
    config = _apply_overrides(config, args)

    n_samples = config.n_train_seeds * config.eval.n_seeds
    print("[m4] config:")
    print(f"  output_dir        = {args.output_dir}")
    print(f"  master_seed       = {config.master_seed}")
    print(f"  device            = {config.device}")
    print(f"  learners          = {list(config.learners)}")
    print(f"  n_train_seeds     = {config.n_train_seeds}")
    print(f"  eval.n_seeds      = {config.eval.n_seeds}")
    print(f"  samples per cell  = {n_samples}  (= n_train_seeds × n_seeds)")
    print(f"  warmup.epochs     = {config.warmup.epochs}")
    print(f"  b1.steps          = {config.b1.outer_steps}")
    print(f"  baseline.steps    = {config.baseline.outer_steps}")
    print(f"  geohead.steps     = {config.geohead.outer_steps}")
    print(f"  eval.k_shots      = {list(config.eval.k_shots)}")

    torch.manual_seed(config.master_seed)
    t0 = time.perf_counter()
    result = run_main_experiment(config=config, output_dir=args.output_dir)
    dt = time.perf_counter() - t0
    print(f"[m4] done in {dt:.1f} s")
    print(f"[m4] artefacts written to: {result.output_dir}")
    print(f"[m4] records: {len(result.records)} rows "
          f"({n_samples} per (learner, corpus, k, method) cell expected)")
    print(f"[m4] aggregated: {len(result.aggregated)} cells")
    print(f"[m4] summary: {result.output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
