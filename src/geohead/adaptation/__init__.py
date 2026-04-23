"""Head adaptation rules (test-time and shared with bilevel inner loop)."""

from geohead.adaptation.test_time import (
    geo_adapt,
    inner_rule_adapt,
    ridge_adapt,
)

__all__ = [
    "geo_adapt",
    "inner_rule_adapt",
    "ridge_adapt",
]
