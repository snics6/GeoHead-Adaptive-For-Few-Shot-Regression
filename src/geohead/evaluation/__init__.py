"""Evaluation suite for GeoHead (``docs/design.md`` §9)."""

from geohead.evaluation.metrics import (
    evaluate_head,
    head_correction_geo,
    head_correction_l2,
    query_mae,
    query_mse,
)
from geohead.evaluation.runner import (
    EvalConfig,
    EvalRecord,
    aggregate,
    evaluate_model,
    to_pandas,
)
from geohead.evaluation.visualize import (
    plot_head_correction_vs_mse,
    plot_sample_efficiency_curve,
)

__all__ = [
    "EvalConfig",
    "EvalRecord",
    "aggregate",
    "evaluate_head",
    "evaluate_model",
    "head_correction_geo",
    "head_correction_l2",
    "plot_head_correction_vs_mse",
    "plot_sample_efficiency_curve",
    "query_mae",
    "query_mse",
    "to_pandas",
]
