"""Training routines for GeoHead: warm-up, baseline, and meta-learning."""

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
    pooled_dataset,
    warmup_train,
)

__all__ = [
    "BaselineConfig",
    "BaselineHistory",
    "GeoHeadConfig",
    "GeoHeadHistory",
    "WarmupConfig",
    "WarmupHistory",
    "baseline_train",
    "geohead_train",
    "pooled_dataset",
    "warmup_train",
]
