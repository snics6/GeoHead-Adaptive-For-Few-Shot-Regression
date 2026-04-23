"""Training routines for GeoHead: warm-up, baseline, and meta-learning."""

from geohead.training.baseline import (
    BaselineConfig,
    BaselineHistory,
    baseline_train,
)
from geohead.training.warmup import (
    WarmupConfig,
    WarmupHistory,
    pooled_dataset,
    warmup_train,
)

__all__ = [
    "BaselineConfig",
    "BaselineHistory",
    "WarmupConfig",
    "WarmupHistory",
    "baseline_train",
    "pooled_dataset",
    "warmup_train",
]
