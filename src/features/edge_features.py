"""
Edge-level graph features for the CSOP prediction pipeline.

Future edge features should inherit from :class:`Feature`, expose
``FeatureLevel.EDGE`` through their ``level`` property, and declare their
computation scope with ``FeatureScope.GLOBAL`` or ``FeatureScope.LOCAL``.
Their ``compute`` method should receive a ``FeatureContext`` and select
``full_graph`` for global features or ``subgraph`` for local features.

This module intentionally contains no concrete edge feature yet.
"""

from .base import Feature, FeatureContext, FeatureLevel, FeatureScope


__all__ = [
	"Feature",
	"FeatureContext",
	"FeatureLevel",
	"FeatureScope",
]
