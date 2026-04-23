"""Model modules for GeoHead: encoder and linear head."""

from geohead.models.encoder import MLPEncoder
from geohead.models.head import LinearHead

__all__ = ["LinearHead", "MLPEncoder"]
