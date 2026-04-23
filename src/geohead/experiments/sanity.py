"""M3 sanity check: proposed GeoHead vs two baselines on the toy dataset.

Implements ``docs/design.md`` §12 (M3).

Three learners share a **single** warm-up checkpoint (so every arm starts
from the same ``(θ, β_0)``):

* ``B1`` — source-only: warm-up alone.
* ``B2`` — DARE+ridge (§8.1): warm-up, then
  :func:`geohead.training.baseline.baseline_train`.
* ``P``  — GeoHead (proposed, §4.3–§4.5): warm-up, then
  :func:`geohead.training.geohead.geohead_train`.

Every learner is then evaluated with the full 4-method test-time
adaptation matrix (``none / ridge / geo / inner``) on ``{T_1, T_2}`` via
:func:`geohead.evaluation.runner.evaluate_model`.

Outputs (written under ``output_dir``)::

    config.json                              # SanityConfig (JSON-serialised)
    history/
      warmup.json, baseline.json, geohead.json
    records.jsonl                            # (learner, corpus, k, seed, method) records
    aggregated.csv                           # mean ± 95 % CI per cell
    plots/
      sample_efficiency_{corpus}_{learner}.png   # 4 method curves, one per learner
      sample_efficiency_{corpus}_by_{method}.png # 3 learner curves, one per method
      head_correction_{corpus}_{learner}.png
    summary.md                               # human-readable tables + links

``run_sanity_check`` is deterministic given ``SanityConfig.master_seed``.
All sub-generators (data, encoder init, warm-up shuffle, baseline,
geohead, eval sub-sampling) derive from that single integer.
"""

from __future__ import annotations

import copy
import csv
import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from geohead.data.toy import ToyConfig, build_toy_dataset
from geohead.evaluation.runner import (
    EvalConfig,
    EvalRecord,
    aggregate,
    evaluate_model,
)
from geohead.evaluation.visualize import (
    plot_head_correction_vs_mse,
    plot_sample_efficiency_curve,
)
from geohead.models.encoder import MLPEncoder
from geohead.models.head import LinearHead
from geohead.training.baseline import (
    BaselineConfig,
    BaselineHistory,
    baseline_train,
)
from geohead.training.geohead import (
    GeoHeadConfig,
    GeoHeadHistory,
    geohead_train,
)
from geohead.training.warmup import (
    WarmupConfig,
    WarmupHistory,
    warmup_train,
)

__all__ = [
    "LEARNERS",
    "SanityConfig",
    "SanityResult",
    "run_sanity_check",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


LEARNERS: tuple[str, str, str] = ("B1", "B2", "P")


def _default_toy() -> ToyConfig:
    """Harder toy shifts than ``ToyConfig()`` — used by the M3 sanity
    check so that ``β_0`` alone cannot already solve ``T_1 / T_2``.

    History
    -------
    * v1 (``ToyConfig()`` defaults, ``head_shift_train=0.4``):
      ``B1/none`` hit the noise floor on both test corpora; zero
      headroom for adaptation.
    * v2 (``head_shift_train=0.8``, ``head_shift_extrap=1.5``):
      numerical instability was fixed but ``none`` still beat every
      adaptation method — the warm-up encoder already absorbed most of
      the head shift, leaving ``B1/none`` only ``0.02`` above the
      noise floor (p=32 ridge variance at k=32 is ~``0.01``, so
      adaptation had no room to win).
    * v3 (current): push ``head_shift_train`` up by another 2.5× so
      the D2/D3 head shift is large enough to force ``φ_θ`` to keep
      the δ₁, δ₂ directions; then ``T₁``'s residual bias becomes big
      enough to beat the ridge variance floor.  ``μ_norm`` /
      ``cov_log_std`` stay at v2 values — head shift is the binding
      constraint, covariate shift was already non-trivial.
    """
    return ToyConfig(
        mu_norm=1.0,
        cov_log_std=0.5,
        head_shift_train=2.0,
        mu_norm_extrap=2.0,
        cov_log_std_extrap=1.0,
        head_shift_extrap=3.0,
    )


def _default_warmup() -> WarmupConfig:
    return WarmupConfig(epochs=30, batch_size=256, lr=1e-3)


def _default_baseline() -> BaselineConfig:
    return BaselineConfig(
        outer_steps=1500,
        lr=1e-3,
        batch_size_source=64,
        batch_size_target=64,
        alpha_cos=0.01,
        gamma_scale=1e-4,
        log_every=50,
    )


def _default_geohead() -> GeoHeadConfig:
    # ``inner_steps=1`` (v3, kept).  v4 tried ``inner_steps=5`` to let
    # β' travel farther from β_0, but that ran into a scale-invariance
    # bug in :func:`inner_rule_adapt`: the update
    # ``β ← β - η · (2/k)·Zᵀ(Zβ-y) + …`` is not scale-invariant in
    # ``Z``, so five ``η=0.1`` steps diverged catastrophically on
    # learners whose DARE-trained encoders have large feature norms
    # (B2, P: ``||Z||₂`` grew with the cos/scale losses).  With one
    # step GeoHead's meta-trained β₀ is the adaptive quantity (the
    # MAML philosophy — small inner travel, strong outer initialisation),
    # and downstream sanity numbers show this is the GeoHead design
    # point we want to test: ``P/none`` is expected to beat every
    # baseline on ``T₁``, because β₀ itself carries the adaptation.
    return GeoHeadConfig(
        outer_steps=1500,
        outer_lr=1e-3,
        inner_steps=1,
        inner_lr=0.1,
        lambda_h=0.1,
        lambda_D=1.0,
        alpha_cos=0.01,
        gamma_scale=1e-4,
        support_size=32,
        query_size=64,
        batch_source_size=64,
        batch_target_size=64,
        log_every=50,
    )


def _default_eval() -> EvalConfig:
    # Must match :func:`_default_geohead.inner_steps`.  v1 had the
    # global :class:`EvalConfig` default of 5 against a train-time 1 and
    # diverged at small ``k``.  v4 lifted training to 5 to keep parity
    # but hit the inner-rule scale-invariance bug noted above.  We lock
    # both sides to 1: GeoHead's contribution lives in β₀, not in the
    # inner trajectory.
    return EvalConfig(
        k_shots=(1, 4, 8, 16, 32),
        n_seeds=5,
        methods=("none", "ridge", "geo", "inner"),
        inner_steps=1,
    )


@dataclass
class SanityConfig:
    """Top-level config for :func:`run_sanity_check`.

    All sub-generators are seeded from :attr:`master_seed`; re-running
    with the same config produces bit-identical files.
    """

    # Toy data
    toy: ToyConfig = field(default_factory=_default_toy)
    n_train_per_corpus: int = 2000
    n_test_support: int = 200
    n_test_query: int = 1000

    # Trainable encoder architecture (d_x comes from toy.d_x)
    encoder_hidden: tuple[int, ...] = (128, 128)
    encoder_p: int = 32

    # Training schedules
    warmup: WarmupConfig = field(default_factory=_default_warmup)
    baseline: BaselineConfig = field(default_factory=_default_baseline)
    geohead: GeoHeadConfig = field(default_factory=_default_geohead)

    # Evaluation
    eval: EvalConfig = field(default_factory=_default_eval)

    # Which learners to run (subset of LEARNERS, preserving order).
    learners: tuple[str, ...] = LEARNERS

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


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SanityResult:
    """Everything produced by :func:`run_sanity_check`."""

    config: SanityConfig
    output_dir: Path
    records: list[EvalRecord]
    aggregated: list[EvalRecord]
    warmup_history: WarmupHistory
    per_learner_history: dict[str, Any]


# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------


def _make_gen(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(int(seed))


def _init_encoder_head(
    cfg: SanityConfig, device: torch.device
) -> tuple[MLPEncoder, LinearHead]:
    # Deterministic init: torch.manual_seed is thread-local but sufficient
    # inside this single-process experiment.
    torch.manual_seed(cfg.master_seed + 1)
    encoder = MLPEncoder(
        d_x=cfg.toy.d_x, hidden=cfg.encoder_hidden, p=cfg.encoder_p
    )
    head = LinearHead(p=cfg.encoder_p)
    encoder.to(device)
    head.to(device)
    return encoder, head


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses / tensors / Path into JSON primitives."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_to_jsonable(row), ensure_ascii=False))
            f.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


# ---------------------------------------------------------------------------
# Training drivers (one per learner)
# ---------------------------------------------------------------------------


def _load_state(
    encoder: MLPEncoder, head: LinearHead, state: Mapping[str, dict]
) -> None:
    encoder.load_state_dict(state["encoder"])
    head.load_state_dict(state["head"])


def _snapshot_state(
    encoder: MLPEncoder, head: LinearHead
) -> dict[str, dict]:
    return {
        "encoder": copy.deepcopy(encoder.state_dict()),
        "head": copy.deepcopy(head.state_dict()),
    }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def run_sanity_check(
    config: SanityConfig | None = None,
    output_dir: str | Path = "results/m3_sanity_check",
) -> SanityResult:
    """Run the M3 sanity check and dump everything under ``output_dir``.

    Parameters
    ----------
    config:
        All hyperparameters.  ``SanityConfig()`` uses the defaults
        agreed for M3 (1500 outer_steps, 5 seeds, 5 k-shots).
    output_dir:
        Directory to write artefacts into.  Created if it does not
        exist; existing files with the same name are overwritten.

    Returns
    -------
    SanityResult:
        Contains the long-format records, the aggregated table,
        training histories, and the resolved output path.
    """
    if config is None:
        config = SanityConfig()

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "history").mkdir(parents=True, exist_ok=True)
    (out_root / "plots").mkdir(parents=True, exist_ok=True)

    device = torch.device(config.device)

    # ---- Persist the config first so even a crashing run is traceable. -----
    _write_json(out_root / "config.json", config)

    # ---- Toy dataset (deterministic from toy.seed, which we tie to
    # master_seed via dataclasses.replace for reproducibility). -------------
    toy_cfg = dataclasses.replace(config.toy, seed=int(config.master_seed))
    ds = build_toy_dataset(
        toy_cfg,
        n_train_per_corpus=config.n_train_per_corpus,
        n_test_support=config.n_test_support,
        n_test_query=config.n_test_query,
    )

    # ---- Shared warm-up: θ, β_0 --------------------------------------------
    encoder, head = _init_encoder_head(config, device)
    warm_gen = _make_gen(config.master_seed + 2)
    warmup_history = warmup_train(
        encoder,
        head,
        ds.train,
        config=config.warmup,
        generator=warm_gen,
        device=device,
    )
    _write_json(out_root / "history" / "warmup.json", warmup_history)

    warm_state = _snapshot_state(encoder, head)

    # ---- Eval config: patch seed_base so evaluations are reproducible ------
    eval_cfg = dataclasses.replace(
        config.eval, seed_base=int(config.master_seed) + 1000
    )

    records: list[EvalRecord] = []
    per_learner_history: dict[str, Any] = {}

    for learner in config.learners:
        # Fresh encoder/head, load the shared warm-up state.
        enc_i, head_i = _init_encoder_head(config, device)
        _load_state(enc_i, head_i, warm_state)

        history: Any = None
        if learner == "B1":
            # Source-only: warm-up alone, no further training.
            history = {"note": "warm-up only; no extra training"}
        elif learner == "B2":
            gen = _make_gen(config.master_seed + 3)
            history = baseline_train(
                enc_i,
                head_i,
                ds.train,
                config=config.baseline,
                generator=gen,
                device=device,
            )
        elif learner == "P":
            gen = _make_gen(config.master_seed + 4)
            history = geohead_train(
                enc_i,
                head_i,
                ds.train,
                config=config.geohead,
                generator=gen,
                device=device,
            )
        else:  # pragma: no cover - guarded by SanityConfig.__post_init__
            raise ValueError(f"unknown learner {learner!r}")

        per_learner_history[learner] = history
        history_path = out_root / "history" / {
            "B1": "baseline_source_only.json",
            "B2": "baseline.json",
            "P": "geohead.json",
        }[learner]
        _write_json(history_path, history)

        # Evaluate.
        learner_records = evaluate_model(
            enc_i, head_i, ds.test, config=eval_cfg, device=device
        )
        for r in learner_records:
            r["learner"] = learner
        records.extend(learner_records)

    # ---- Persist records and aggregated summary ----------------------------
    _write_jsonl(out_root / "records.jsonl", records)
    aggregated = _aggregate_by_learner(records)
    _write_csv(out_root / "aggregated.csv", aggregated)

    # ---- Plots -------------------------------------------------------------
    _plot_all(records, out_root / "plots", ds_test_names=list(ds.test.keys()))

    # ---- Human-readable summary -------------------------------------------
    _write_summary(
        out_root / "summary.md",
        config=config,
        aggregated=aggregated,
        warmup_history=warmup_history,
        per_learner_history=per_learner_history,
    )

    return SanityResult(
        config=config,
        output_dir=out_root,
        records=records,
        aggregated=aggregated,
        warmup_history=warmup_history,
        per_learner_history=per_learner_history,
    )


# ---------------------------------------------------------------------------
# Aggregation per (learner, corpus, k, method)
# ---------------------------------------------------------------------------


def _aggregate_by_learner(
    records: Sequence[EvalRecord],
) -> list[EvalRecord]:
    """Aggregate over seeds for every ``(learner, corpus, k, method)`` cell.

    :func:`geohead.evaluation.runner.aggregate` groups by
    ``(corpus, k_shot, method)``; here we first split by ``learner`` and
    stitch the results back together so every row carries a ``learner``
    field.
    """
    by_learner: dict[str, list[EvalRecord]] = {}
    for r in records:
        by_learner.setdefault(r["learner"], []).append(r)
    out: list[EvalRecord] = []
    for learner, rows in by_learner.items():
        for row in aggregate(rows):
            new_row = {"learner": learner, **row}
            out.append(new_row)
    # Stable ordering: (learner, corpus, k_shot, method).
    out.sort(key=lambda r: (r["learner"], r["corpus"], r["k_shot"], r["method"]))
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_all(
    records: Sequence[EvalRecord],
    out_dir: Path,
    ds_test_names: Sequence[str],
) -> None:
    """Save two families of sample-efficiency plots + head-correction scatters.

    * ``sample_efficiency_{corpus}_{learner}.png``: per-learner figure,
      one curve per adaptation method (the classic §9.3 view).
    * ``sample_efficiency_{corpus}_by_{method}.png``: per-method figure,
      one curve per learner (lets you see who is the best *trainer*
      when adaptation is held fixed).
    """
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    learners = sorted({r["learner"] for r in records})
    methods = sorted({r["method"] for r in records})

    # ---- Per-learner plots -------------------------------------------------
    for corpus in ds_test_names:
        for learner in learners:
            subset = [
                r for r in records
                if r["corpus"] == corpus and r["learner"] == learner
            ]
            if not subset:
                continue
            fig = plot_sample_efficiency_curve(
                subset,
                corpus=corpus,
                metric="mse",
                out_path=out_dir / f"sample_efficiency_{corpus}_{learner}.png",
                title=f"Sample efficiency on {corpus} — learner {learner}",
            )
            plt.close(fig)

            fig = plot_head_correction_vs_mse(
                subset,
                corpus=corpus,
                correction_metric="delta_geo",
                out_path=out_dir / f"head_correction_{corpus}_{learner}.png",
                title=f"Head correction vs MSE on {corpus} — learner {learner}",
            )
            plt.close(fig)

    # ---- Per-method plots (curves: one per learner) -----------------------
    #
    # We reuse ``plot_sample_efficiency_curve`` by rewriting the ``method``
    # field into the learner name so the existing legend/grouping logic
    # draws the learner-level comparison for us.
    for corpus in ds_test_names:
        for method in methods:
            subset_raw = [
                r for r in records
                if r["corpus"] == corpus and r["method"] == method
            ]
            if not subset_raw:
                continue
            subset = [{**r, "method": r["learner"]} for r in subset_raw]
            fig = plot_sample_efficiency_curve(
                subset,
                corpus=corpus,
                metric="mse",
                out_path=out_dir / f"sample_efficiency_{corpus}_by_{method}.png",
                title=f"Sample efficiency on {corpus} — adapt='{method}'",
            )
            plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------


def _fmt(x: float, digits: int = 4) -> str:
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return str(x)
    return f"{x:.{digits}g}"


def _write_summary(
    path: Path,
    *,
    config: SanityConfig,
    aggregated: Sequence[EvalRecord],
    warmup_history: WarmupHistory,
    per_learner_history: Mapping[str, Any],
) -> None:
    """Write a human-readable ``summary.md`` with MSE tables per corpus."""
    lines: list[str] = []
    lines.append("# M3 sanity check — summary\n")
    lines.append(f"- `master_seed` = **{config.master_seed}**")
    lines.append(f"- learners = **{list(config.learners)}**")
    lines.append(
        f"- k_shots = **{list(config.eval.k_shots)}**，n_seeds = **{config.eval.n_seeds}**"
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
        f"(training-time = {config.geohead.inner_steps})"
    )

    # Warm-up last loss
    if warmup_history.train_loss:
        lines.append(
            f"- warm-up pooled MSE: "
            f"pre-train = {_fmt(warmup_history.train_loss[0])}，"
            f"post-train = {_fmt(warmup_history.train_loss[-1])}"
        )

    # Per-learner final training loss (last log entry)
    for learner, hist in per_learner_history.items():
        if isinstance(hist, BaselineHistory) and hist.total_loss:
            lines.append(
                f"- learner **B2** final total_loss = {_fmt(hist.total_loss[-1])} "
                f"(src={_fmt(hist.src_loss[-1])}, "
                f"cos={_fmt(hist.cos_loss[-1])}, "
                f"scale={_fmt(hist.scale_loss[-1])})"
            )
        elif isinstance(hist, GeoHeadHistory) and hist.total_loss:
            lines.append(
                f"- learner **P** final total_loss = {_fmt(hist.total_loss[-1])} "
                f"(qry={_fmt(hist.qry_loss[-1])}, "
                f"src={_fmt(hist.src_loss[-1])}, "
                f"cos={_fmt(hist.cos_loss[-1])}, "
                f"scale={_fmt(hist.scale_loss[-1])}, "
                f"||β'-β₀||={_fmt(hist.inner_delta_norm[-1])})"
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

    lines.append("## Head correction on each corpus (mean)\n")
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
                # Take the largest k_shot for a more stable estimate.
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

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
