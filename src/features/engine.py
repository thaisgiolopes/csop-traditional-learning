from collections.abc import Collection, Iterable
from typing import Any

from .base import Feature, FeatureContext, FeatureScope


class FeatureEngine:
    """
    Executes configured graph features.

    The engine separates global feature computation from local feature
    computation. Global results are cached per complete graph so they can
    be reused for candidate subgraphs from the same graph instance.

    This class does not generate subgraphs, perform pooling, calculate
    objective values, build datasets, or perform machine learning.
    """

    def __init__(self, features: Collection[Feature]):
        """
        Initialize the feature engine.

        Args:
            features: The feature objects to execute.

        Raises:
            TypeError: If features is not a collection of feature-like
                objects.
            ValueError: If two features have the same name.
        """
        if not isinstance(features, Collection):
            raise TypeError("features must be a collection of Feature objects.")

        self._features = tuple(features)

        if not all(self._is_feature_like(feature) for feature in self._features):
            raise TypeError(
                "All items in features must implement the Feature interface."
            )

        feature_names = [feature.name for feature in self._features]

        if len(feature_names) != len(set(feature_names)):
            raise ValueError("Feature names must be unique.")

        self._global_cache: dict[Any, dict[str, Any]] = {}

    @property
    def features(self) -> tuple[Feature, ...]:
        """
        Return the configured features in their original order.

        Returns:
            tuple[Feature, ...]: The configured feature objects.
        """
        return self._features

    def compute_global(
        self,
        context: FeatureContext,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Compute and cache all configured global features.

        Global features are evaluated using the complete graph from the
        context. Cached results are reused when another context references
        the same complete graph.

        Args:
            context: The complete graph and candidate subgraph available
                to the features.
            force: Whether to recompute global features even if cached.

        Returns:
            dict[str, Any]: Global feature results keyed by feature name.

        Raises:
            TypeError: If context is not a FeatureContext instance.
        """
        self._validate_context(context)

        complete_graph = context.full_graph

        if not force and complete_graph in self._global_cache:
            return self._global_cache[complete_graph].copy()

        results = {
            feature.name: feature.compute(context)
            for feature in self._features
            if feature.scope is FeatureScope.GLOBAL
        }

        self._global_cache[complete_graph] = results
        return results.copy()

    def compute_local(self, context: FeatureContext) -> dict[str, Any]:
        """
        Compute all configured local features for a candidate subgraph.

        Local features are evaluated using the candidate subgraph from the
        context. They are intentionally not cached because each candidate
        subgraph may be different.

        Args:
            context: The complete graph and candidate subgraph available
                to the features.

        Returns:
            dict[str, Any]: Local feature results keyed by feature name.

        Raises:
            TypeError: If context is not a FeatureContext instance.
        """
        self._validate_context(context)

        return {
            feature.name: feature.compute(context)
            for feature in self._features
            if feature.scope is FeatureScope.LOCAL
        }

    @staticmethod
    def _validate_context(context: FeatureContext) -> None:
        """
        Validate the context supplied to an engine operation.

        Args:
            context: The context to validate.

        Raises:
            TypeError: If context is not a FeatureContext instance.
        """
        if not isinstance(context, FeatureContext):
            raise TypeError("context must be a FeatureContext instance.")

    @staticmethod
    def _is_feature_like(feature: object) -> bool:
        """Return whether an object exposes the feature engine interface."""
        return all(
            hasattr(feature, attribute)
            for attribute in ("name", "scope", "level", "compute")
        )