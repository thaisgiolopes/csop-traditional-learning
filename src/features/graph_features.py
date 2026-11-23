"""Graph-level features for the CSOP prediction pipeline."""

from .base import Feature, FeatureContext, FeatureLevel, FeatureScope


class SubgraphSizeFeature(Feature):
	"""Return the number of vertices in the candidate subgraph."""

	@property
	def name(self) -> str:
		return "subgraph_size"

	@property
	def scope(self) -> FeatureScope:
		return FeatureScope.LOCAL

	@property
	def level(self) -> FeatureLevel:
		return FeatureLevel.GRAPH

	def compute(self, context: FeatureContext) -> int:
		if not isinstance(context, FeatureContext):
			raise TypeError("context must be a FeatureContext instance.")
		return context.subgraph.num_vertices


class NumVerticesFeature(Feature):
	"""Return the number of vertices in the candidate subgraph."""

	@property
	def name(self) -> str:
		"""Return the identifier of the feature."""
		return "num_vertices"

	@property
	def scope(self) -> FeatureScope:
		"""Return the candidate-local scope of the feature."""
		return FeatureScope.LOCAL

	@property
	def level(self) -> FeatureLevel:
		"""Return the graph level of the feature."""
		return FeatureLevel.GRAPH

	def compute(self, context: FeatureContext) -> int:
		"""
		Return the number of vertices in the candidate subgraph.

		Args:
			context: The complete graph and candidate subgraph available
				to the feature.

		Returns:
				int: The number of vertices in ``context.subgraph``.

		Raises:
			TypeError: If context is not a FeatureContext instance.
		"""
		if not isinstance(context, FeatureContext):
			raise TypeError("context must be a FeatureContext instance.")

		return context.subgraph.num_vertices


class NumEdgesFeature(Feature):
	"""Return the number of edges in the candidate subgraph."""

	@property
	def name(self) -> str:
		"""Return the identifier of the feature."""
		return "num_edges"

	@property
	def scope(self) -> FeatureScope:
		"""Return the candidate-local scope of the feature."""
		return FeatureScope.LOCAL

	@property
	def level(self) -> FeatureLevel:
		"""Return the graph level of the feature."""
		return FeatureLevel.GRAPH

	def compute(self, context: FeatureContext) -> int:
		"""
		Return the number of edges in the candidate subgraph.

		Args:
			context: The complete graph and candidate subgraph available
				to the feature.

		Returns:
				int: The number of edges in ``context.subgraph``.

		Raises:
			TypeError: If context is not a FeatureContext instance.
		"""
		if not isinstance(context, FeatureContext):
			raise TypeError("context must be a FeatureContext instance.")

		return context.subgraph.num_edges
