"""Matplotlib plots for the evaluation matrix (``docs/design.md`` §9.3).

Two plots are provided:

* :func:`plot_sample_efficiency_curve` — ``k_shot`` vs mean target query
  metric (MSE by default), one curve per method, shaded 95 % CI.  One
  figure per test corpus.
* :func:`plot_head_correction_vs_mse` — scatter of
  ``(head correction magnitude, target query MSE)`` with points coloured
  by method, intended to show the theoretical correlation described in
  §9.3 (head-correction budget vs achievable error).

Both helpers accept a ``records: list[EvalRecord]`` list as produced by
:func:`geohead.evaluation.runner.evaluate_model`, optionally save the
figure to ``out_path``, and return the :class:`matplotlib.figure.Figure`
instance for further customisation.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from geohead.evaluation.runner import EvalRecord, aggregate

__all__ = [
    "plot_sample_efficiency_curve",
    "plot_head_correction_vs_mse",
]


def _filter_corpus(records: Sequence[EvalRecord], corpus: str) -> list[EvalRecord]:
    rows = [r for r in records if r["corpus"] == corpus]
    if not rows:
        raise ValueError(
            f"no records for corpus {corpus!r}; available: "
            f"{sorted({r['corpus'] for r in records})}"
        )
    return rows


def _method_colour_map(methods: Sequence[str]) -> dict[str, Any]:
    """Stable method→colour mapping (tab10 palette)."""
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    return {m: cmap(i % 10) for i, m in enumerate(methods)}


def plot_sample_efficiency_curve(
    records: Sequence[EvalRecord],
    corpus: str,
    metric: str = "mse",
    out_path: str | Path | None = None,
    title: str | None = None,
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xscale: str = "log",
    yscale: str = "linear",
):
    """Plot ``k_shot`` vs mean ± 95 % CI for ``metric`` on ``corpus``.

    One curve per method (``records`` may contain an arbitrary subset of
    ``{"none", "ridge", "geo", "inner"}``).  X axis is log-scaled because
    the default ``k_shots`` span 1–20.

    Parameters
    ----------
    records:
        Long-format evaluation records (see ``evaluation.runner``).
    corpus:
        Test corpus name to plot (e.g. ``"T1"``).
    metric:
        Metric name, one of ``{"mse", "mae", "delta_l2", "delta_geo"}``.
    out_path:
        If given, the figure is saved to this path via
        :meth:`Figure.savefig` (PNG / PDF inferred from extension).
    title:
        Optional axes title override.  Defaults to
        ``f"Sample efficiency on {corpus} ({metric})"``.
    xlim, ylim:
        Optional axis-range overrides ``(low, high)``.  When ``None``
        matplotlib autoscales.  Use these to unify ranges across a
        family of comparable plots.
    xscale, yscale:
        Matplotlib axis scales (``"linear"`` / ``"log"`` / ``"symlog"``).
        Default: log x-axis (support sizes span 1–24), linear y.  Pass
        ``yscale="log"`` for a log-log view that compresses outliers
        and emphasises multiplicative differences between methods.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    rows = _filter_corpus(records, corpus)
    agg = aggregate(rows, metrics=(metric,))
    methods: list[str] = []
    for r in agg:
        if r["method"] not in methods:
            methods.append(r["method"])
    palette = _method_colour_map(methods)

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for method in methods:
        method_rows = sorted(
            (r for r in agg if r["method"] == method),
            key=lambda r: r["k_shot"],
        )
        ks = [r["k_shot"] for r in method_rows]
        means = [r[f"{metric}_mean"] for r in method_rows]
        ci = [r[f"{metric}_ci95"] for r in method_rows]
        lo = [m - c for m, c in zip(means, ci)]
        hi = [m + c for m, c in zip(means, ci)]
        colour = palette[method]
        ax.plot(ks, means, marker="o", color=colour, label=method)
        ax.fill_between(ks, lo, hi, color=colour, alpha=0.2)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("support size k")
    ax.set_ylabel(f"query {metric}")
    ax.set_title(title or f"Sample efficiency on {corpus} ({metric})")
    ax.legend(title="method", loc="best", frameon=False)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig


def plot_head_correction_vs_mse(
    records: Sequence[EvalRecord],
    corpus: str,
    correction_metric: str = "delta_geo",
    out_path: str | Path | None = None,
    title: str | None = None,
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xscale: str = "linear",
    yscale: str = "linear",
):
    """Scatter ``(head correction magnitude, query MSE)`` for one corpus.

    Each point is one ``(k_shot, seed, method)`` observation.  Points are
    coloured by method; when ``correction_metric="delta_geo"`` the
    resulting plot is the §9.3 "head correction vs query MSE" diagnostic.

    Parameters
    ----------
    records:
        Long-format evaluation records.
    corpus:
        Test corpus name to plot (e.g. ``"T1"``).
    correction_metric:
        One of ``{"delta_l2", "delta_geo"}``.  Default ``"delta_geo"``
        matches the §4.3 inner-loss geometry.
    out_path, title:
        See :func:`plot_sample_efficiency_curve`.
    xlim, ylim:
        Optional axis ranges for cross-plot comparison.
    xscale, yscale:
        Matplotlib scales.  Use ``xscale="symlog"`` when
        ``correction_metric="delta_l2"`` — the ``none`` method sits at
        ``x = 0`` and must remain plottable on a log-ish axis.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if correction_metric not in {"delta_l2", "delta_geo"}:
        raise ValueError(
            f"correction_metric must be 'delta_l2' or 'delta_geo'; "
            f"got {correction_metric!r}"
        )

    rows = _filter_corpus(records, corpus)
    methods: list[str] = []
    for r in rows:
        if r["method"] not in methods:
            methods.append(r["method"])
    palette = _method_colour_map(methods)

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for method in methods:
        xs = [r[correction_metric] for r in rows if r["method"] == method]
        ys = [r["mse"] for r in rows if r["method"] == method]
        ax.scatter(xs, ys, color=palette[method], alpha=0.6, s=20, label=method)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(
        "||β̂ - β₀||₂" if correction_metric == "delta_l2"
        else "(β̂ - β₀)ᵀ (Σ̂ + εI) (β̂ - β₀)"
    )
    ax.set_ylabel("query MSE")
    ax.set_title(
        title or f"Head correction vs query MSE on {corpus}"
    )
    ax.legend(title="method", loc="best", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig
