"""Tests for ``src/geohead/evaluation/visualize.py``.

The plotting helpers are deliberately thin wrappers around matplotlib;
the tests here verify contract (figure returned, axes labelled, file
written when ``out_path`` is given) rather than pixel-exact rendering.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest
import torch

matplotlib.use("Agg")  # headless backend for CI / non-GUI environments

import matplotlib.pyplot as plt  # noqa: E402

from geohead.evaluation.runner import EvalConfig, evaluate_model  # noqa: E402
from geohead.evaluation.visualize import (  # noqa: E402
    plot_head_correction_vs_mse,
    plot_sample_efficiency_curve,
)
from geohead.models.encoder import MLPEncoder  # noqa: E402
from geohead.models.head import LinearHead  # noqa: E402


def _small_records():
    g = torch.Generator().manual_seed(0)
    true_w = torch.randn(6, generator=g)
    corpora: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    for k, name in enumerate(("T1", "T2")):
        gs = torch.Generator().manual_seed(10 * (k + 1))
        gq = torch.Generator().manual_seed(10 * (k + 1) + 1)
        x_s = torch.randn(40, 6, generator=gs)
        y_s = x_s @ true_w + 0.05 * torch.randn(40, generator=gs)
        x_q = torch.randn(100, 6, generator=gq)
        y_q = x_q @ true_w + 0.05 * torch.randn(100, generator=gq)
        corpora[name] = {"support": (x_s, y_s), "query": (x_q, y_q)}

    torch.manual_seed(0)
    encoder = MLPEncoder(d_x=6, hidden=(16,), p=8)
    head = LinearHead(p=8)

    cfg = EvalConfig(k_shots=(1, 3, 5), n_seeds=3)
    return evaluate_model(encoder, head, corpora, config=cfg), cfg


# ---------------------------------------------------------------------------
# plot_sample_efficiency_curve
# ---------------------------------------------------------------------------


def test_sample_efficiency_curve_returns_figure_with_methods() -> None:
    records, cfg = _small_records()
    fig = plot_sample_efficiency_curve(records, corpus="T1", metric="mse")
    try:
        ax = fig.axes[0]
        labels = [line.get_label() for line in ax.get_lines() if not line.get_label().startswith("_")]
        assert set(labels) == set(cfg.methods)
        assert ax.get_xscale() == "log"
        assert "mse" in ax.get_ylabel()
    finally:
        plt.close(fig)


def test_sample_efficiency_curve_saves_file(tmp_path: Path) -> None:
    records, _ = _small_records()
    out = tmp_path / "sub" / "eff.png"
    fig = plot_sample_efficiency_curve(records, corpus="T1", out_path=out)
    try:
        assert out.exists() and out.stat().st_size > 0
    finally:
        plt.close(fig)


def test_sample_efficiency_curve_rejects_unknown_corpus() -> None:
    records, _ = _small_records()
    with pytest.raises(ValueError):
        plot_sample_efficiency_curve(records, corpus="T42")


def test_sample_efficiency_curve_supports_other_metrics() -> None:
    records, _ = _small_records()
    for metric in ("mae", "delta_l2", "delta_geo"):
        fig = plot_sample_efficiency_curve(records, corpus="T2", metric=metric)
        try:
            assert metric in fig.axes[0].get_ylabel()
        finally:
            plt.close(fig)


# ---------------------------------------------------------------------------
# plot_head_correction_vs_mse
# ---------------------------------------------------------------------------


def test_head_correction_vs_mse_returns_figure() -> None:
    records, cfg = _small_records()
    fig = plot_head_correction_vs_mse(records, corpus="T1")
    try:
        ax = fig.axes[0]
        # One scatter collection per method.
        assert len(ax.collections) == len(cfg.methods)
        assert "MSE" in ax.get_ylabel()
    finally:
        plt.close(fig)


def test_head_correction_vs_mse_saves_file(tmp_path: Path) -> None:
    records, _ = _small_records()
    out = tmp_path / "corr.png"
    fig = plot_head_correction_vs_mse(records, corpus="T1", out_path=out)
    try:
        assert out.exists() and out.stat().st_size > 0
    finally:
        plt.close(fig)


def test_head_correction_vs_mse_rejects_bad_metric() -> None:
    records, _ = _small_records()
    with pytest.raises(ValueError):
        plot_head_correction_vs_mse(records, corpus="T1", correction_metric="bogus")


def test_head_correction_vs_mse_rejects_unknown_corpus() -> None:
    records, _ = _small_records()
    with pytest.raises(ValueError):
        plot_head_correction_vs_mse(records, corpus="T9")
