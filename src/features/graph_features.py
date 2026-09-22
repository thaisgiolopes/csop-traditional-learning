"""Graph-level features for the CSOP prediction pipeline."""

from .base import Feature, FeatureContext, FeatureLevel, FeatureScope


class NumVerticesFeature(Feature):
	"""
	Computes the number of vertices in the complete graph.

	This is a GLOBAL + GRAPH feature: it describes the complete graph
	instance. A future LOCAL + GRAPH feature would instead describe the
	candidate subgraph available in the same FeatureContext.
	"""

	@property
	def name(self) -> str:
		"""Return the identifier of the feature."""
		return "num_vertices"

	@property
	def scope(self) -> FeatureScope:
		"""Return the global scope of the feature."""
		return FeatureScope.GLOBAL

	@property
	def level(self) -> FeatureLevel:
		"""Return the graph level of the feature."""
		return FeatureLevel.GRAPH

	def compute(self, context: FeatureContext) -> int:
		"""
		Return the number of vertices in the complete graph.

		Args:
			context: The complete graph and candidate subgraph available
				to the feature.

		Returns:
			int: The number of vertices in ``context.full_graph``.

		Raises:
			TypeError: If context is not a FeatureContext instance.
		"""
		if not isinstance(context, FeatureContext):
			raise TypeError("context must be a FeatureContext instance.")

		return context.full_graph.num_vertices


class NumEdgesFeature(Feature):
	"""
	Computes the number of edges in the complete graph.

	This is a GLOBAL + GRAPH feature: it describes the complete graph
	instance. A future LOCAL + GRAPH feature would instead describe the
	candidate subgraph available in the same FeatureContext.
	"""

	@property
	def name(self) -> str:
		"""Return the identifier of the feature."""
		return "num_edges"

	@property
	def scope(self) -> FeatureScope:
		"""Return the global scope of the feature."""
		return FeatureScope.GLOBAL

	@property
	def level(self) -> FeatureLevel:
		"""Return the graph level of the feature."""
		return FeatureLevel.GRAPH

	def compute(self, context: FeatureContext) -> int:
		"""
		Return the number of edges in the complete graph.

		Args:
			context: The complete graph and candidate subgraph available
				to the feature.

		Returns:
			int: The number of edges in ``context.full_graph``.

		Raises:
			TypeError: If context is not a FeatureContext instance.
		"""
		if not isinstance(context, FeatureContext):
			raise TypeError("context must be a FeatureContext instance.")

		return context.full_graph.num_edges
