from typing import Any

import networkx as nx

from ..graph import Graph
from .base import Feature, FeatureContext, FeatureLevel, FeatureScope


class _ScopedNetworkXNodeFeature(Feature):
	"""Base for unweighted features on undirected, loop-free Graph values.

	The scope suffix in each feature name distinguishes calculations on the
	complete graph from calculations on the candidate graph.
	"""

	_feature_name = ""

	def __init__(self, scope: FeatureScope = FeatureScope.LOCAL):
		if not isinstance(scope, FeatureScope):
			raise TypeError("scope must be a FeatureScope value.")
		if scope not in (FeatureScope.GLOBAL, FeatureScope.LOCAL):
			raise ValueError("scope must be GLOBAL or LOCAL.")
		self._scope = scope

	@property
	def name(self) -> str:
		"""Return a scope-qualified, stable feature identifier."""
		return f"{self._feature_name}_{self.scope.name.lower()}"

	@property
	def scope(self) -> FeatureScope:
		"""Return the graph scope used by this feature."""
		return self._scope

	@property
	def level(self) -> FeatureLevel:
		"""Return the node level associated with the feature values."""
		return FeatureLevel.NODE

	def _networkx_graph(self, context: FeatureContext) -> nx.Graph:
		if not isinstance(context, FeatureContext):
			raise TypeError("context must be a FeatureContext instance.")

		graph = (
			context.full_graph
			if self.scope is FeatureScope.GLOBAL
			else context.subgraph
		)
		if not isinstance(graph, Graph):
			raise TypeError("selected graph must be a Graph instance.")
		return graph.networkx_graph

	def compute(self, context: FeatureContext) -> dict[Any, int | float]:
		"""Compute node values on the graph selected by the feature scope."""
		return self._compute(self._networkx_graph(context))

	def _compute(self, graph: nx.Graph) -> dict[Any, int | float]:
		raise NotImplementedError


class PageRankFeature(_ScopedNetworkXNodeFeature):
	"""Return unweighted PageRank scores, which sum to one per selected graph.

	NetworkX's damping factor defaults to 0.85; scores describe the graph
	identified by this feature's scope.
	"""

	_feature_name = "pagerank"

	def __init__(
		self,
		scope: FeatureScope = FeatureScope.LOCAL,
		*,
		alpha: float = 0.85,
		max_iter: int = 100,
		tol: float = 1.0e-6,
	):
		super().__init__(scope)
		self._parameters = {"alpha": alpha, "max_iter": max_iter, "tol": tol}

	def _compute(self, graph: nx.Graph) -> dict[Any, float]:
		return dict(nx.pagerank(graph, weight=None, **self._parameters))


class BetweennessCentralityFeature(_ScopedNetworkXNodeFeature):
	"""Count shortest paths passing through each node.

	The default returns normalized, unweighted betweenness and excludes path
	endpoints. NetworkX's exact all-pairs calculation is used when ``k`` is
	None; when sampling is enabled, ``seed`` controls reproducibility.
	"""

	_feature_name = "betweenness_centrality"

	def __init__(
		self,
		scope: FeatureScope = FeatureScope.LOCAL,
		*,
		normalized: bool = True,
		endpoints: bool = False,
		k: int | None = None,
		seed: int | None = None,
	):
		super().__init__(scope)
		self._parameters = {
			"normalized": normalized,
			"endpoints": endpoints,
			"k": k,
			"seed": seed,
		}

	def _compute(self, graph: nx.Graph) -> dict[Any, float]:
		return dict(
			nx.betweenness_centrality(
				graph,
				weight=None,
				**self._parameters,
			)
		)


class ClosenessCentralityFeature(_ScopedNetworkXNodeFeature):
	"""Compute unweighted closeness from shortest-path distances.

	With NetworkX's default ``wf_improved=True``, the reachable-component
	closeness is multiplied by the fraction of other graph nodes reachable
	from the node. This makes disconnected-graph values graph-scope-specific.
	"""

	_feature_name = "closeness_centrality"

	def __init__(
		self,
		scope: FeatureScope = FeatureScope.LOCAL,
		*,
		wf_improved: bool = True,
	):
		super().__init__(scope)
		self._wf_improved = wf_improved

	def _compute(self, graph: nx.Graph) -> dict[Any, float]:
		return dict(
			nx.closeness_centrality(
				graph,
				wf_improved=self._wf_improved,
				distance=None,
			)
		)


class EigenvectorCentralityFeature(_ScopedNetworkXNodeFeature):
	"""Return the leading adjacency-eigenvector centrality by power iteration.

	A node's score is proportional to the scores of its neighbors. For
	disconnected graphs, NetworkX's power iteration determines the returned
	vector; convergence failures are allowed to surface to the caller.
	"""

	_feature_name = "eigenvector_centrality"

	def __init__(
		self,
		scope: FeatureScope = FeatureScope.LOCAL,
		*,
		max_iter: int = 100,
		tol: float = 1.0e-6,
	):
		super().__init__(scope)
		self._max_iter = max_iter
		self._tol = tol

	def _compute(self, graph: nx.Graph) -> dict[Any, float]:
		if graph.number_of_nodes() == 0:
			return {}
		return dict(
			nx.eigenvector_centrality(
				graph,
				max_iter=self._max_iter,
				tol=self._tol,
				weight=None,
			)
		)


class KCoreNumberFeature(_ScopedNetworkXNodeFeature):
	"""Return each node's largest ``k`` for which it belongs to a ``k``-core."""

	_feature_name = "k_core_number"

	def _compute(self, graph: nx.Graph) -> dict[Any, int]:
		return dict(nx.core_number(graph))


class TriangleCountFeature(_ScopedNetworkXNodeFeature):
	"""Count triangles incident to each node.

	Each undirected 3-cycle contributes one to each of its three vertices;
	this mapping is not the graph's total triangle count.
	"""

	_feature_name = "triangle_count"

	def _compute(self, graph: nx.Graph) -> dict[Any, int]:
		return dict(nx.triangles(graph))


class LocalClusteringCoefficientFeature(_ScopedNetworkXNodeFeature):
	"""Return ``2T / (d * (d - 1))`` for each node.

	Here ``T`` is its incident triangle count and ``d`` its degree; NetworkX
	returns zero when fewer than two neighbors exist.
	"""

	_feature_name = "local_clustering_coefficient"

	def _compute(self, graph: nx.Graph) -> dict[Any, float]:
		return dict(nx.clustering(graph, weight=None))


class SquareClusteringFeature(_ScopedNetworkXNodeFeature):
	"""Return the fraction of possible 4-cycles through each node that exist.

	This uses NetworkX's unweighted square-clustering definition; values are
	zero when no square is possible for a node.
	"""

	_feature_name = "square_clustering"

	def _compute(self, graph: nx.Graph) -> dict[Any, float]:
		return dict(nx.square_clustering(graph))


class AverageNeighborDegreeFeature(_ScopedNetworkXNodeFeature):
	"""Return the arithmetic mean of each node's neighbors' degrees.

	Degrees are measured in the same selected graph, and a node with no
	neighbors receives zero.
	"""

	_feature_name = "average_neighbor_degree"

	def _compute(self, graph: nx.Graph) -> dict[Any, float]:
		return dict(nx.average_neighbor_degree(graph, weight=None))


class NormalizedDegreeFeature(_ScopedNetworkXNodeFeature):
	"""Return ``degree(v) / (n - 1)`` on the selected graph.

	The result is defined as zero for every vertex when ``n <= 1``. For
	``n > 1`` this is the simple undirected degree divided by the maximum
	possible degree, with no self-loops.
	"""

	_feature_name = "normalized_degree"

	def _compute(self, graph: nx.Graph) -> dict[Any, float]:
		node_count = graph.number_of_nodes()
		if node_count <= 1:
			return {node: 0.0 for node in graph}
		denominator = node_count - 1
		return {
			node: degree / denominator
			for node, degree in graph.degree()
		}


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
