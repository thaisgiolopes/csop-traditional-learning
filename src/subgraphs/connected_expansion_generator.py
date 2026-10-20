"""Connected induced subgraphs grown by randomized frontier expansion."""

import hashlib
import random
from collections.abc import Collection

from ..graph import Graph
from .base import SubgraphGenerator


class ConnectedExpansionSubgraphGenerator(SubgraphGenerator):
    """Generate fixed-size connected induced candidates by vertex expansion."""

    def __init__(
        self,
        num_subgraphs: int,
        subgraph_size: int,
        seed: int | None = None,
        max_attempts: int | None = None,
    ) -> None:
        if isinstance(num_subgraphs, bool) or not isinstance(num_subgraphs, int):
            raise TypeError("num_subgraphs must be an integer.")
        if num_subgraphs <= 0:
            raise ValueError("num_subgraphs must be positive.")
        if isinstance(subgraph_size, bool) or not isinstance(subgraph_size, int):
            raise TypeError("subgraph_size must be an integer.")
        if subgraph_size <= 0:
            raise ValueError("subgraph_size must be positive.")
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int)
        ):
            raise TypeError("seed must be an integer or None.")
        if max_attempts is None:
            max_attempts = min(max(100, num_subgraphs * 50), 100_000)
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
            raise TypeError("max_attempts must be an integer.")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive.")

        self._num_subgraphs = num_subgraphs
        self._subgraph_size = subgraph_size
        self._seed = seed
        self._max_attempts = max_attempts

    @property
    def name(self) -> str:
        return "networkx_connected_expansion"

    def _seed_for_index(self, index: int) -> int:
        seed_material = f"{self._seed!r}:{index}".encode("utf-8")
        return int.from_bytes(
            hashlib.sha256(seed_material).digest()[:8],
            byteorder="big",
        )

    def generate_at_index(self, graph: Graph, index: int) -> Graph | None:
        """Attempt one fixed-size connected sample at a deterministic index."""
        if not isinstance(graph, Graph):
            raise ValueError("graph must be a Graph instance.")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("index must be a non-negative integer.")

        networkx_graph = graph.networkx_graph
        if networkx_graph.number_of_nodes() < self._subgraph_size:
            return None

        generator = random.Random(self._seed_for_index(index))
        initial_vertex = generator.choice(list(networkx_graph.nodes))
        selected = {initial_vertex}
        frontier: list[object] = []
        frontier_positions: dict[object, int] = {}

        def add_frontier_neighbors(vertex: object) -> None:
            for neighbor in networkx_graph.neighbors(vertex):
                if neighbor in selected or neighbor in frontier_positions:
                    continue
                frontier_positions[neighbor] = len(frontier)
                frontier.append(neighbor)

        add_frontier_neighbors(initial_vertex)

        while len(selected) < self._subgraph_size and frontier:
            chosen_index = generator.randrange(len(frontier))
            chosen_vertex = frontier[chosen_index]
            last_vertex = frontier.pop()
            del frontier_positions[chosen_vertex]
            if chosen_index < len(frontier):
                frontier[chosen_index] = last_vertex
                frontier_positions[last_vertex] = chosen_index

            selected.add(chosen_vertex)
            add_frontier_neighbors(chosen_vertex)

        if len(selected) != self._subgraph_size:
            return None

        induced_subgraph = networkx_graph.subgraph(selected)
        return Graph(
            vertices=induced_subgraph.nodes,
            edges=induced_subgraph.edges,
        )

    def generate(self, graph: Graph) -> Collection[Graph]:
        """Generate the requested number of unique candidates within a limit."""
        candidates: list[Graph] = []
        seen_vertex_sets: set[frozenset[object]] = set()
        for index in range(self._max_attempts):
            candidate = self.generate_at_index(graph, index)
            if candidate is None:
                continue
            vertex_set = frozenset(candidate.vertices)
            if vertex_set in seen_vertex_sets:
                continue
            seen_vertex_sets.add(vertex_set)
            candidates.append(candidate)
            if len(candidates) == self._num_subgraphs:
                return candidates

        raise RuntimeError(
            f"Could generate only {len(candidates)} unique connected "
            f"subgraphs out of {self._num_subgraphs} requested after "
            f"{self._max_attempts} deterministic attempts."
        )