from ..graph import Graph
from .base import Feature, FeatureContext, FeatureLevel, FeatureScope


class DegreeFeature(Feature):
	"""
	Computes the degree of every vertex at a configurable graph scope.

	The same feature can be computed globally from the complete graph or
	locally from the candidate subgraph. The feature remains node-level in
	both cases; only the graph used for the calculation changes.
	"""

	def __init__(self, scope: FeatureScope = FeatureScope.LOCAL):
		"""
		Initialize a degree feature.

		Args:
			scope: Whether degrees should be computed from the complete
				graph or the candidate subgraph.

		Raises:
			TypeError: If scope is not a FeatureScope value.
			ValueError: If scope is not GLOBAL or LOCAL.
		"""
		if not isinstance(scope, FeatureScope):
			raise TypeError("scope must be a FeatureScope value.")

		if scope not in (FeatureScope.GLOBAL, FeatureScope.LOCAL):
			raise ValueError("scope must be GLOBAL or LOCAL.")

		self._scope = scope

	@property
	def name(self) -> str:
		"""Return the identifier of the feature."""
		return "degree"

	@property
	def scope(self) -> FeatureScope:
		"""Return the graph scope used by this feature."""
		return self._scope

	@property
	def level(self) -> FeatureLevel:
		"""Return the level associated with the feature values."""
		return FeatureLevel.NODE

	def compute(self, context: FeatureContext) -> dict:
		"""
		Compute the degree of every vertex in the selected graph.

		Args:
			context: The complete graph and candidate subgraph available
				to the feature.

		Returns:
			dict: A mapping from vertex identifiers to their degree values.

		Raises:
			TypeError: If context is not a FeatureContext instance.
		"""
		if not isinstance(context, FeatureContext):
			raise TypeError("context must be a FeatureContext instance.")

		graph: Graph = (
			context.full_graph
			if self.scope is FeatureScope.GLOBAL
			else context.subgraph
		)

		return dict(graph.networkx_graph.degree())
