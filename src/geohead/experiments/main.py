"""M4 main experiment: paper-ready sample-efficiency curves with CI.

Implements ``docs/design.md`` §12 (M4).

Extends the M3 sanity check along two axes:

1. **Training variance** — instead of one training run, we execute the
   full (toy data → warm-up → per-learner train → eval) pipeline for
   ``n_train_seeds`` independent master seeds.  Every seed draws a fresh
   toy dataset, fresh encoder initialisation, fresh training randomness,
   and a fresh block of ``n_seeds`` evaluation sub-samples.
2. **Longer schedules** — baseline and GeoHead are trained for
   ``outer_steps = 5000`` (v8 M3 used 1500), and evaluation uses
   ``n_seeds = 20`` (v8 used 5), so every ``(learner, corpus, k, method)``
   cell is backed by ``n_train_seeds × n_seeds`` samples (default
   ``3 × 20 = 60``).

The aggregated CI therefore absorbs four sources of variance:

* dataset realisation,
* model initialisation,
* training stochasticity (optimizer batches, episode sampling),
* test-time support sub-sampling.

This is the standard full-variance protocol for domain-generalisation
papers (cf. Gulrajani & Lopez-Paz, 2021).

Outputs (written under ``output_dir``)::

    config.json                              # M4Config (JSON-serialised)
    run_0/history/{warmup,b1,baseline,geohead}.json
    run_1/history/...
    ...
    records.jsonl                            # all rows, tagged with "train_seed"
    aggregated.csv                           # mean ± 95 % CI per cell
    plots/
      sample_efficiency_{corpus}_{learner}.png
      sample_efficiency_{corpus}_by_{method}.png
      head_correction_{corpus}_{learner}.png
    summary.md
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from geohead.data.toy import ToyConfig
from geohead.evaluation.runner import EvalConfig, EvalRecord
from geohead.experiments.sanity import (
    LEARNERS,
    SanityConfig,
    _aggregate_by_learner,
    _plot_all,
    _run_pipeline_once,
    _write_csv,
    _write_json,
    _write_jsonl,
    _default_toy,
    _default_warmup,
)
from geohead.training.b1 import B1Config, B1History
from geohead.training.baseline import BaselineConfig, BaselineHistory
from geohead.training.geohead import GeoHeadConfig, GeoHeadHistory
from geohead.training.warmup import WarmupConfig, WarmupHistory

__all__ = [
    "M4Config",
    "M4Result",
    "run_main_experiment",
]


# ---------------------------------------------------------------------------
# Default factories (M4-specific: longer schedules, more eval seeds)
# ---------------------------------------------------------------------------


#: Seed spacing between consecutive training runs.  Large enough that
#: the sub-seeds (``+1, +2, +3, +4, +1000..+1019``) derived inside
#: :func:`_run_pipeline_once` never collide across train seeds.
_TRAIN_SEED_STRIDE = 1_000_000


def _default_b1_m4() -> B1Config:
    # v5 (unified-episode) B1: 5 000 steps of pooled supervised MSE on the
    # shared (S, B_i, Q, B_j) episode.  Episode batch sizes match B2 / P
    # so the three learners consume the shared generator identically.
    return B1Config(
        outer_steps=5000,
        lr=1e-3,
        support_size=32,
        query_size=64,
        batch_source_size=64,
        batch_target_size=64,
        log_every=200,
    )


def _default_baseline_m4() -> BaselineConfig:
    # v5 (unified-episode) hyperparameters + 5 000 outer_steps.  Episode
    # batch sizes match :func:`_default_geohead_m4` so all three learners
    # share the same episode RNG consumption pattern (§8.0).
    return BaselineConfig(
        outer_steps=5000,
        lr=1e-3,
        support_size=32,
        query_size=64,
        batch_source_size=64,
        batch_target_size=64,
        alpha_cos=0.01,
        gamma_scale=1e-4,
        log_every=200,
    )


def _default_geohead_m4() -> GeoHeadConfig:
    # v8 hyperparameters + 5 000 outer_steps.  All M3 choices carry over:
    # preconditioned inner rule, ε=1e-4, inner_steps=5, support_size=32
    # (matches encoder_p=32 for train-at-rank-edge).
    return GeoHeadConfig(
        outer_steps=5000,
        outer_lr=1e-3,
        inner_steps=5,
        inner_lr=0.1,
        lambda_h=0.1,
        head_reg_eps=1e-4,
        preconditioned_inner=True,
        lambda_D=1.0,
        alpha_cos=0.01,
        gamma_scale=1e-4,
        support_size=32,
        query_size=64,
        batch_source_size=64,
        batch_target_size=64,
        log_every=200,
    )


def _default_eval_m4() -> EvalConfig:
    # v8 k_shots, but 20 evaluation seeds per cell (M3 used 5).
    return EvalConfig(
        k_shots=(1, 4, 8, 16, 24),
        n_seeds=20,
        methods=("none", "ridge", "geo", "inner"),
        inner_steps=5,
        head_reg_eps=1e-4,
        inner_preconditioned=True,
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class M4Config:
    """Top-level config for :func:`run_main_experiment`.

    Structurally a super-set of :class:`SanityConfig`: the toy /
    architecture / warm-up / eval sub-configs are identical in type so
    we can fan out to :func:`_run_pipeline_once` by building a
    :class:`SanityConfig` on the fly.
    """

    # Toy data (same v8 defaults as :class:`SanityConfig`).
    toy: ToyConfig = field(default_factory=_default_toy)
    n_train_per_corpus: int = 2000
    n_test_support: int = 200
    n_test_query: int = 1000

    # Encoder architecture.
    encoder_hidden: tuple[int, ...] = (128, 128)
    encoder_p: int = 32

    # Training schedules (M4 defaults — longer than M3).
    warmup: WarmupConfig = field(default_factory=_default_warmup)
    b1: B1Config = field(default_factory=_default_b1_m4)
    baseline: BaselineConfig = field(default_factory=_default_baseline_m4)
    geohead: GeoHeadConfig = field(default_factory=_default_geohead_m4)

    # Evaluation (M4 default: 20 seeds per cell).
    eval: EvalConfig = field(default_factory=_default_eval_m4)

    # Which learners to run (subset of LEARNERS).
    learners: tuple[str, ...] = LEARNERS

    # Number of fully independent training runs.  Each run has its own
    # toy dataset realisation, encoder init, and training stochasticity;
    # the evaluation sub-sample seeds are also disjoint across runs.
    n_train_seeds: int = 3

    # Misc
    device: str = "cpu"
    master_seed: int = 0

    def __post_init__(self) -> None:
        for name in self.learners:
            if name not in LEARNERS:
                raise ValueError(
                    f"unknown learner {name!r}; must be subset of {LEARNERS}"
                )
        if len(set(self.learners)) != len(self.learners):
            raise ValueError(f"duplicate learner in {self.learners!r}")
        if self.encoder_p <= 0:
            raise ValueError(f"encoder_p must be > 0, got {self.encoder_p}")
        if not self.encoder_hidden:
            raise ValueError("encoder_hidden must have at least one layer")
        if self.n_train_per_corpus <= 0:
            raise ValueError(
                f"n_train_per_corpus must be > 0, got {self.n_train_per_corpus}"
            )
        if self.n_test_support <= 0:
            raise ValueError(f"n_test_support must be > 0, got {self.n_test_support}")
        if self.n_test_query <= 0:
            raise ValueError(f"n_test_query must be > 0, got {self.n_test_query}")
        if not isinstance(self.n_train_seeds, int) or self.n_train_seeds <= 0:
            raise ValueError(
                f"n_train_seeds must be a positive int, got {self.n_train_seeds!r}"
            )

    def to_sanity_config(self, master_seed: int | None = None) -> SanityConfig:
        """Materialise a :class:`SanityConfig` snapshot for one train seed.

        Every field is deep-copied so subsequent mutation of the
        returned ``SanityConfig`` does not leak into this ``M4Config``.
        """
        ms = int(self.master_seed if master_seed is None else master_seed)
        return SanityConfig(
            toy=copy.deepcopy(self.toy),
            n_train_per_corpus=self.n_train_per_corpus,
            n_test_support=self.n_test_support,
            n_test_query=self.n_test_query,
            encoder_hidden=tuple(self.encoder_hidden),
            encoder_p=self.encoder_p,
            warmup=copy.deepcopy(self.warmup),
            b1=copy.deepcopy(self.b1),
            baseline=copy.deepcopy(self.baseline),
            geohead=copy.deepcopy(self.geohead),
            eval=copy.deepcopy(self.eval),
            learners=tuple(self.learners),
            device=self.device,
            master_seed=ms,
        )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class M4Result:
    """Everything produced by :func:`run_main_experiment`."""

    config: M4Config
    output_dir: Path
    # All rows across every train_seed × eval_seed.  Every record carries
    # ``"learner"`` AND ``"train_seed"`` fields so downstream analyses
    # can slice by either axis.
    records: list[EvalRecord]
    # Aggregated over the full 60-sample pool per cell.
    aggregated: list[EvalRecord]
    warmup_histories: list[WarmupHistory]
    per_learner_histories: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def run_main_experiment(
    config: M4Config | None = None,
    output_dir: str | Path = "results/m4_main",
) -> M4Result:
    """Run the M4 full-variance experiment.

    Parameters
    ----------
    config:
        M4 hyperparameters.  ``M4Config()`` matches the v8-derived
        defaults (§12 of ``docs/design.md``).
    output_dir:
        Directory to write artefacts into.  Created if absent; existing
        files with the same names are overwritten.

    Returns
    -------
    M4Result:
        Contains the full long-format records, the aggregated table
        with CI, training histories per run, and the resolved output
        path.
    """
    if config is None:
        config = M4Config()

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "plots").mkdir(parents=True, exist_ok=True)

    device = torch.device(config.device)

    _write_json(out_root / "config.json", config)

    all_records: list[EvalRecord] = []
    warmup_histories: list[WarmupHistory] = []
    per_learner_histories: list[dict[str, Any]] = []
    test_names: list[str] = []

    for i in range(config.n_train_seeds):
        master_seed_i = int(config.master_seed) + i * _TRAIN_SEED_STRIDE
        run_dir = out_root / f"run_{i}"
        # Each run_i snapshots the SanityConfig it was driven with; this
        # is handy when debugging a single run in isolation.
        sanity_cfg_i = config.to_sanity_config(master_seed=master_seed_i)
        _write_json(run_dir / "config.json", sanity_cfg_i)

        recs_i, warm_i, hist_i, names_i = _run_pipeline_once(
            sanity_cfg_i,
            master_seed=master_seed_i,
            device=device,
            history_dir=run_dir / "history",
        )
        for r in recs_i:
            r["train_seed"] = i

        all_records.extend(recs_i)
        warmup_histories.append(warm_i)
        per_learner_histories.append(hist_i)
        if not test_names:
            test_names = names_i

    # Aggregation: the existing ``aggregate`` helper groups by
    # (corpus, k_shot, method) — train_seed is silently collapsed,
    # which is exactly what we want for a full-variance CI.
    _write_jsonl(out_root / "records.jsonl", all_records)
    aggregated = _aggregate_by_learner(all_records)
    _write_csv(out_root / "aggregated.csv", aggregated)

    # Unified axis ranges so every plot on a given corpus shares the same
    # y-scale — side-by-side PNGs become directly comparable by eye.
    mse_ylim, delta_xlim = _compute_unified_axes(aggregated)
    _plot_all(
        all_records,
        out_root / "plots",
        ds_test_names=test_names,
        mse_ylim=mse_ylim,
        mse_yscale="log",
        delta_xlim=delta_xlim,
    )

    _write_summary_m4(
        out_root / "summary.md",
        config=config,
        aggregated=aggregated,
        warmup_histories=warmup_histories,
        per_learner_histories=per_learner_histories,
    )
    _write_comparison_tables(
        out_root / "comparison.md",
        config=config,
        aggregated=aggregated,
    )

    return M4Result(
        config=config,
        output_dir=out_root,
        records=all_records,
        aggregated=aggregated,
        warmup_histories=warmup_histories,
        per_learner_histories=per_learner_histories,
    )


# ---------------------------------------------------------------------------
# Unified axis-range computation for cross-plot comparability
# ---------------------------------------------------------------------------


def _compute_unified_axes(
    aggregated: Sequence[EvalRecord],
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    """Derive per-corpus ``(mse_ylim, delta_xlim)`` so every plot on the
    same corpus shares identical axes — B2's instability outliers no
    longer inflate P's plot and vice-versa.

    Strategy
    --------
    For the MSE y-axis we use a **log scale** keyed to the *median* CI
    band of the cells, widened by a small multiplicative margin.  This
    keeps B2's widest CI (~0.33 half-width on T2 k=4) visible while
    preserving the ~0.06 dynamic range of P's curves.  Log-scale is the
    right choice for MSE: multiplicative differences between methods
    (P/ridge beats B1/ridge by 1.4×) become linear distances on the plot.

    For the ``delta_geo`` x-axis we match the 99-th percentile across
    all (learner, method, k_shot) cells, with a small pad.  B2's
    ``delta_l2`` blow-up (scalar values up to ~18) does **not** feed
    into this because the aggregator stores ``delta_geo_*`` which is
    scale-invariant.
    """
    corpora = sorted({r["corpus"] for r in aggregated})

    mse_ylim: dict[str, tuple[float, float]] = {}
    delta_xlim: dict[str, tuple[float, float]] = {}

    for corpus in corpora:
        rows = [r for r in aggregated if r["corpus"] == corpus]
        if not rows:
            continue

        # ---- MSE range (log-y friendly) ------------------------------
        # min of (mean - ci95)   →  lower bound, clamped to 1e-6 so log works.
        # max of (mean + ci95)   →  upper bound, with 20 % head-room.
        lo_mse = min(max(r["mse_mean"] - r["mse_ci95"], 1e-6) for r in rows)
        hi_mse = max(r["mse_mean"] + r["mse_ci95"] for r in rows)
        # Multiplicative margin (log-axis friendly): 0.85× below, 1.25× above.
        mse_ylim[corpus] = (lo_mse * 0.85, hi_mse * 1.25)

        # ---- Δ_geo x-range (linear, 0-based) -------------------------
        # 0 is always hit by the ``none`` method (β̂ = β₀).
        hi_delta = max(r["delta_geo_mean"] + r["delta_geo_ci95"] for r in rows)
        # Small right margin so points at the max value aren't glued to
        # the frame.
        delta_xlim[corpus] = (0.0, hi_delta * 1.10 + 1e-4)

    return mse_ylim, delta_xlim


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------


def _fmt(x: float, digits: int = 4) -> str:
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return str(x)
    return f"{x:.{digits}g}"


def _write_summary_m4(
    path: Path,
    *,
    config: M4Config,
    aggregated: Sequence[EvalRecord],
    warmup_histories: Sequence[WarmupHistory],
    per_learner_histories: Sequence[Mapping[str, Any]],
) -> None:
    """Write a human-readable ``summary.md`` with CI across all train seeds."""
    n_train = config.n_train_seeds
    n_eval = config.eval.n_seeds

    lines: list[str] = []
    lines.append("# M4 main experiment — summary\n")
    lines.append(f"- `master_seed` = **{config.master_seed}**")
    lines.append(
        f"- `n_train_seeds` = **{n_train}**，`eval.n_seeds` = **{n_eval}** "
        f"→ **{n_train * n_eval}** samples per (learner, corpus, k, method) cell"
    )
    lines.append(f"- learners = **{list(config.learners)}**")
    lines.append(
        f"- k_shots = **{list(config.eval.k_shots)}**"
    )
    lines.append(
        f"- warm-up epochs = **{config.warmup.epochs}**，"
        f"baseline outer_steps = **{config.baseline.outer_steps}**，"
        f"geohead outer_steps = **{config.geohead.outer_steps}**"
    )
    lines.append(
        f"- toy shifts: `head_shift_train`=**{config.toy.head_shift_train}**，"
        f"`head_shift_extrap`=**{config.toy.head_shift_extrap}**，"
        f"`mu_norm`=**{config.toy.mu_norm}**，"
        f"`mu_norm_extrap`=**{config.toy.mu_norm_extrap}**"
    )
    lines.append(
        f"- test-time `inner_steps` = **{config.eval.inner_steps}** "
        f"(training-time = {config.geohead.inner_steps})，"
        f"preconditioned={config.eval.inner_preconditioned}"
    )

    # Per-run final training losses (compact summary).
    for i, (wh, hist) in enumerate(
        zip(warmup_histories, per_learner_histories)
    ):
        if wh.train_loss:
            lines.append(
                f"- run_{i}: warm-up MSE {_fmt(wh.train_loss[0])} "
                f"→ {_fmt(wh.train_loss[-1])}"
            )
        for learner, h in hist.items():
            if isinstance(h, BaselineHistory) and h.total_loss:
                lines.append(
                    f"  - B2 final total_loss = {_fmt(h.total_loss[-1])}"
                )
            elif isinstance(h, GeoHeadHistory) and h.total_loss:
                lines.append(
                    f"  - P  final total_loss = {_fmt(h.total_loss[-1])} "
                    f"(||β'-β₀||={_fmt(h.inner_delta_norm[-1])})"
                )
    lines.append("")

    corpora = sorted({r["corpus"] for r in aggregated})
    k_shots = sorted({r["k_shot"] for r in aggregated})
    learners = sorted({r["learner"] for r in aggregated})
    methods = sorted({r["method"] for r in aggregated})

    for corpus in corpora:
        lines.append(f"## Query MSE on `{corpus}` (mean ± 95 % CI)\n")
        header = ["learner / method"] + [f"k={k}" for k in k_shots]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for learner in learners:
            for method in methods:
                row = [f"{learner} / `{method}`"]
                for k in k_shots:
                    match = [
                        r for r in aggregated
                        if r["learner"] == learner
                        and r["corpus"] == corpus
                        and r["method"] == method
                        and r["k_shot"] == k
                    ]
                    if not match:
                        row.append("—")
                    else:
                        m = match[0]
                        row.append(
                            f"{_fmt(m['mse_mean'])} ± {_fmt(m['mse_ci95'])}"
                        )
                lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append("## Head correction on each corpus (mean over all samples)\n")
    lines.append("| learner / method | corpus | mean ‖β̂−β₀‖₂ | mean Δ_geo |")
    lines.append("|---|---|---|---|")
    for learner in learners:
        for corpus in corpora:
            for method in methods:
                rows = [
                    r for r in aggregated
                    if r["learner"] == learner
                    and r["corpus"] == corpus
                    and r["method"] == method
                ]
                if not rows:
                    continue
                rows.sort(key=lambda r: r["k_shot"])
                m = rows[-1]
                lines.append(
                    f"| {learner} / `{method}` | {corpus} | "
                    f"{_fmt(m['delta_l2_mean'])} | {_fmt(m['delta_geo_mean'])} |"
                )
    lines.append("")

    lines.append("## Plots\n")
    lines.append("- `plots/sample_efficiency_{corpus}_{learner}.png`: per-learner view，4 methods.")
    lines.append("- `plots/sample_efficiency_{corpus}_by_{method}.png`: per-method view，3 learners.")
    lines.append("- `plots/head_correction_{corpus}_{learner}.png`: geometry-weighted Δ vs MSE scatter.")
    lines.append("")
    lines.append(
        "All plots use the aggregated records (train_seed × eval_seed combined), "
        f"so error bars absorb all {n_train * n_eval} samples per cell."
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Comparison tables (one markdown file per experiment)
# ---------------------------------------------------------------------------


def _write_comparison_tables(
    path: Path,
    *,
    config: M4Config,
    aggregated: Sequence[EvalRecord],
) -> None:
    """Write paper-oriented comparison tables into ``comparison.md``.

    Sections
    --------
    1. **Headline**: the single best (learner, method) cell for every
       (corpus, k), with absolute MSE and relative improvement over the
       simplest baseline ``B1/none``.
    2. **Relative improvement**: per (corpus, k, method) rows showing
       ``P vs B1`` and ``P vs B2`` in percent.
    3. **Method ranking**: within each learner, every method gets an
       average rank across all (corpus, k).  Lower is better.
    4. **Full tables**: the standard ``summary.md`` tables but with the
       per-(corpus, k) winner marked in bold.
    5. **Head-correction sanity**: mean ``‖β̂-β₀‖₂`` and ``Δ_geo`` per
       (learner, method, corpus) at the largest k, highlighting any
       learner whose L2 head norm blows up (a failure mode of
       non-geometry-aware encoders).
    """
    by_cell: dict[tuple[str, str, int, str], EvalRecord] = {
        (r["learner"], r["corpus"], int(r["k_shot"]), r["method"]): r
        for r in aggregated
    }
    corpora = sorted({r["corpus"] for r in aggregated})
    k_shots = sorted({int(r["k_shot"]) for r in aggregated})
    learners = sorted({r["learner"] for r in aggregated})
    methods = sorted({r["method"] for r in aggregated})

    n_samples = config.n_train_seeds * config.eval.n_seeds

    lines: list[str] = []
    lines.append("# M4 comparison tables\n")
    lines.append(
        f"- samples per cell = **{n_samples}** "
        f"(n_train_seeds={config.n_train_seeds} × eval.n_seeds={config.eval.n_seeds})"
    )
    lines.append(f"- corpora = {corpora}，k_shots = {k_shots}")
    lines.append(f"- learners = {learners}，methods = {methods}\n")

    # ---- 1. Headline: best cell per (corpus, k) ------------------------
    lines.append("## 1. Headline: best (learner, method) per (corpus, k)\n")
    lines.append(
        "Winner per cell，with absolute MSE and **relative improvement "
        "over `B1/none`** (the simplest baseline)．\n"
    )
    lines.append(
        "| corpus | k | winner | MSE | ±CI95 | vs B1/none |"
    )
    lines.append("|---|---|---|---|---|---|")
    for corpus in corpora:
        for k in k_shots:
            candidates = [
                ((ll, m), by_cell[(ll, corpus, k, m)])
                for ll in learners
                for m in methods
                if (ll, corpus, k, m) in by_cell
            ]
            if not candidates:
                continue
            (best_ll, best_m), best_row = min(
                candidates, key=lambda x: x[1]["mse_mean"]
            )
            ref = by_cell.get(("B1", corpus, k, "none"))
            if ref is not None and ref["mse_mean"] > 0:
                rel = (
                    (best_row["mse_mean"] - ref["mse_mean"])
                    / ref["mse_mean"]
                    * 100.0
                )
                rel_str = f"{rel:+.1f}%"
            else:
                rel_str = "—"
            lines.append(
                f"| {corpus} | {k} | {best_ll} / `{best_m}` | "
                f"{_fmt(best_row['mse_mean'])} | "
                f"±{_fmt(best_row['mse_ci95'])} | {rel_str} |"
            )
    lines.append("")

    # ---- 2. Relative improvement: P vs {B1, B2} ------------------------
    lines.append("## 2. Relative improvement: P vs baselines\n")
    lines.append(
        "Negative = P is better．Same method compared fairly (e.g. P/ridge vs B1/ridge)．\n"
    )
    for corpus in corpora:
        lines.append(f"### {corpus}\n")
        lines.append(
            "| method | k | P MSE | B1 MSE | P vs B1 | B2 MSE | P vs B2 |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for method in methods:
            for k in k_shots:
                p = by_cell.get(("P", corpus, k, method))
                b1 = by_cell.get(("B1", corpus, k, method))
                b2 = by_cell.get(("B2", corpus, k, method))
                if p is None or b1 is None or b2 is None:
                    continue
                rel_b1 = (
                    (p["mse_mean"] - b1["mse_mean"]) / b1["mse_mean"] * 100.0
                    if b1["mse_mean"] > 0 else float("nan")
                )
                rel_b2 = (
                    (p["mse_mean"] - b2["mse_mean"]) / b2["mse_mean"] * 100.0
                    if b2["mse_mean"] > 0 else float("nan")
                )
                lines.append(
                    f"| `{method}` | {k} | "
                    f"{_fmt(p['mse_mean'])} | "
                    f"{_fmt(b1['mse_mean'])} | {rel_b1:+.1f}% | "
                    f"{_fmt(b2['mse_mean'])} | {rel_b2:+.1f}% |"
                )
        lines.append("")

    # ---- 3. Method ranking per learner --------------------------------
    lines.append("## 3. Method ranking within each learner\n")
    lines.append(
        "Lower avg rank = better across all (corpus, k)．"
        "Ranking is Borda-style (1 = best per cell)．\n"
    )
    rank_sum: dict[tuple[str, str], float] = {}
    rank_n: dict[tuple[str, str], int] = {}
    for ll in learners:
        for corpus in corpora:
            for k in k_shots:
                cells = [
                    (m, by_cell[(ll, corpus, k, m)])
                    for m in methods
                    if (ll, corpus, k, m) in by_cell
                ]
                if not cells:
                    continue
                cells.sort(key=lambda x: x[1]["mse_mean"])
                for rank, (m, _) in enumerate(cells, start=1):
                    key = (ll, m)
                    rank_sum[key] = rank_sum.get(key, 0.0) + rank
                    rank_n[key] = rank_n.get(key, 0) + 1
    lines.append("| learner | method | avg rank | n cells |")
    lines.append("|---|---|---|---|")
    per_learner: dict[str, list[tuple[str, float, int]]] = {}
    for (ll, m), total in rank_sum.items():
        per_learner.setdefault(ll, []).append(
            (m, total / rank_n[(ll, m)], rank_n[(ll, m)])
        )
    for ll in sorted(per_learner):
        rows_sorted = sorted(per_learner[ll], key=lambda t: t[1])
        for m, avg, n in rows_sorted:
            lines.append(f"| {ll} | `{m}` | {avg:.2f} | {n} |")
    lines.append("")

    # ---- 4. Full tables, bold-best -------------------------------------
    lines.append("## 4. Full MSE tables (bold = best in column)\n")
    for corpus in corpora:
        lines.append(f"### {corpus}\n")
        header = ["learner / method"] + [f"k={k}" for k in k_shots]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        # winner per column
        best_cell_per_k: dict[int, tuple[str, str]] = {}
        for k in k_shots:
            cands = [
                ((ll, m), by_cell[(ll, corpus, k, m)])
                for ll in learners
                for m in methods
                if (ll, corpus, k, m) in by_cell
            ]
            if cands:
                best_cell_per_k[k] = min(
                    cands, key=lambda x: x[1]["mse_mean"]
                )[0]
        for ll in learners:
            for m in methods:
                row = [f"{ll} / `{m}`"]
                for k in k_shots:
                    cell = by_cell.get((ll, corpus, k, m))
                    if cell is None:
                        row.append("—")
                        continue
                    val = (
                        f"{_fmt(cell['mse_mean'])} ± "
                        f"{_fmt(cell['mse_ci95'])}"
                    )
                    if best_cell_per_k.get(k) == (ll, m):
                        val = f"**{val}**"
                    row.append(val)
                lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # ---- 5. Head-correction sanity ------------------------------------
    lines.append("## 5. Head-correction sanity (at largest k)\n")
    lines.append(
        "Large `‖β̂−β₀‖₂` with moderate `Δ_geo` signals an encoder that "
        "inflates feature norm in unimportant directions — a failure "
        "mode of non-geometry-aware training (see B2)．\n"
    )
    lines.append(
        "| learner | corpus | method | ‖β̂−β₀‖₂ | Δ_geo | ratio ‖·‖₂/Δ_geo |"
    )
    lines.append("|---|---|---|---|---|---|")
    k_max = max(k_shots)
    for ll in learners:
        for corpus in corpora:
            for m in methods:
                cell = by_cell.get((ll, corpus, k_max, m))
                if cell is None:
                    continue
                l2 = cell["delta_l2_mean"]
                dg = cell["delta_geo_mean"]
                ratio = l2 / dg if dg > 1e-9 else float("inf")
                lines.append(
                    f"| {ll} | {corpus} | `{m}` | "
                    f"{_fmt(l2)} | {_fmt(dg)} | "
                    f"{'—' if ratio == float('inf') else f'{ratio:.2f}'} |"
                )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Re-export for test convenience.
__all__.append("_default_baseline_m4")
__all__.append("_default_geohead_m4")
__all__.append("_default_eval_m4")
__all__.append("_compute_unified_axes")
__all__.append("_write_comparison_tables")
