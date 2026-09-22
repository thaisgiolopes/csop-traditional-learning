from collections.abc import Collection
from typing import Optional

import networkx as nx

from ..graph import Graph
from .base import SubgraphGenerator


class NetworkXSubgraphGenerator(SubgraphGenerator):
    """
    Generates candidate connected subgraphs using NetworkX random paths.

    This is the first experimental subgraph-generation strategy for the
    CSOP objective-value prediction pipeline. Other strategies can be
    added later without changing DatasetBuilder.
    """

    def __init__(
        self,
        num_subgraphs: int,
        path_length: int = 5,
        seed: Optional[int] = None,
    ):
        """
        Initialize the random-path subgraph generator.

        Args:
            num_subgraphs: Number of random paths to generate.
            path_length: Maximum path length accepted by NetworkX.
                A generated path contains up to ``path_length + 1`` vertices.
            seed: Optional random seed.

        Raises:
            TypeError: If a parameter has an invalid type.
            ValueError: If sample_size is not positive or path_length
                is negative.
        """
        if not isinstance(num_subgraphs, int):
            raise TypeError("num_subgraphs must be an integer.")

        if not isinstance(path_length, int):
            raise TypeError("path_length must be an integer.")

        if not isinstance(seed, (int, type(None))):
            raise TypeError("seed must be an integer or None.")

        if num_subgraphs <= 0:
            raise ValueError("num_subgraphs must be positive.")

        if path_length < 0:
            raise ValueError("path_length must be non-negative.")

        self._sample_size = num_subgraphs
        self._path_length = path_length
        self._seed = seed

    @property
    def name(self) -> str:
        """Return the identifier of the generation strategy."""
        return "networkx_random_paths"

    def generate(self, graph: Graph) -> Collection[Graph]:
        """
        Generate connected candidate subgraphs from a complete graph.

        NetworkX's ``generate_random_paths`` returns paths as lists of
        vertices. Each path is converted into a vertex set, and NetworkX
        then creates the subgraph induced by those vertices.

        Args:
            graph: The complete graph representing the problem instance.

        Returns:
            Collection[Graph]: Unique candidate subgraphs.

        Raises:
            ValueError: If graph is not a Graph instance.
        """
        if not isinstance(graph, Graph):
            raise ValueError("graph must be a Graph instance.")

        networkx_graph = graph.networkx_graph

        if networkx_graph.number_of_nodes() == 0:
            return []

        paths = nx.generate_random_paths(
            networkx_graph,
            sample_size=self._sample_size,
            path_length=self._path_length,
            seed=self._seed,
        )

        candidates = []
        seen_vertex_sets = set()

        for path in paths:
            vertex_set = frozenset(path)

            if not vertex_set or vertex_set in seen_vertex_sets:
                continue

            candidate = networkx_graph.subgraph(vertex_set)

            if not nx.is_connected(candidate):
                continue

            candidate_graph = Graph(
                vertices=candidate.nodes,
                edges=candidate.edges,
            )

            seen_vertex_sets.add(vertex_set)
            candidates.append(candidate_graph)

        return candidates