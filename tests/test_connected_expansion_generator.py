import networkx as nx
import pytest

from src.graph import Graph
from src.subgraphs.base import IndexedSubgraphGenerator
from src.subgraphs.connected_expansion_generator import (
    ConnectedExpansionSubgraphGenerator,
)
from src.subgraphs.networkx_generator import NetworkXSubgraphGenerator


def make_graph() -> Graph:
    return Graph(
        vertices=range(8),
        edges=[
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ],
    )


def test_generated_subgraph_has_exact_size_and_is_connected():
    graph = make_graph()
    generator = ConnectedExpansionSubgraphGenerator(
        num_subgraphs=1,
        subgraph_size=5,
        seed=42,
    )

    candidate = generator.generate_at_index(graph, 0)

    assert candidate is not None
    assert candidate.num_vertices == 5
    assert nx.is_connected(candidate.networkx_graph)


def test_generated_subgraph_contains_induced_edges_without_mutating_source():
    graph = make_graph()
    original_nodes = set(graph.networkx_graph.nodes)
    original_edges = set(graph.networkx_graph.edges)
    candidate = ConnectedExpansionSubgraphGenerator(
        num_subgraphs=1,
        subgraph_size=5,
        seed=42,
    ).generate_at_index(graph, 0)

    assert candidate is not None
    expected_edges = set(
        graph.networkx_graph.subgraph(candidate.vertices).edges
    )
    assert set(candidate.networkx_graph.edges) == expected_edges
    assert set(graph.networkx_graph.nodes) == original_nodes
    assert set(graph.networkx_graph.edges) == original_edges


@pytest.mark.parametrize("subgraph_size", [0, -1])
def test_invalid_target_sizes_are_rejected(subgraph_size):
    with pytest.raises(ValueError, match="subgraph_size must be positive"):
        ConnectedExpansionSubgraphGenerator(
            num_subgraphs=1,
            subgraph_size=subgraph_size,
        )


def test_empty_and_too_small_graphs_fail_without_returning_partial_candidate():
    generator = ConnectedExpansionSubgraphGenerator(
        num_subgraphs=1,
        subgraph_size=2,
        seed=1,
    )

    assert generator.generate_at_index(Graph([], []), 0) is None
    assert generator.generate_at_index(Graph([0], []), 0) is None


def test_disconnected_graph_can_generate_from_large_enough_component():
    graph = Graph(
        vertices=range(6),
        edges=[(0, 1), (1, 2), (2, 3), (4, 5)],
    )
    generator = ConnectedExpansionSubgraphGenerator(
        num_subgraphs=1,
        subgraph_size=4,
        seed=7,
        max_attempts=20,
    )

    candidate = next(
        (
            generator.generate_at_index(graph, index)
            for index in range(20)
            if generator.generate_at_index(graph, index) is not None
        ),
        None,
    )

    assert candidate is not None
    assert candidate.num_vertices == 4
    assert nx.is_connected(candidate.networkx_graph)


def test_components_smaller_than_target_and_exhausted_frontier_fail():
    graph = Graph(
        vertices=range(5),
        edges=[(0, 1), (1, 2), (3, 4)],
    )
    generator = ConnectedExpansionSubgraphGenerator(
        num_subgraphs=1,
        subgraph_size=4,
        seed=7,
    )

    assert all(
        generator.generate_at_index(graph, index) is None
        for index in range(10)
    )


def test_indexed_generation_is_reproducible_and_independent_of_batch_size():
    graph = make_graph()
    first = ConnectedExpansionSubgraphGenerator(1, 4, seed=23)
    later = ConnectedExpansionSubgraphGenerator(12, 4, seed=23)

    assert first.generate_at_index(graph, 8).vertices == later.generate_at_index(
        graph, 8
    ).vertices
    assert first._seed_for_index(8) == later._seed_for_index(8)
    assert first._seed_for_index(8) != first._seed_for_index(9)


def test_both_generators_implement_runner_indexed_capability():
    assert isinstance(
        ConnectedExpansionSubgraphGenerator(1, 3),
        IndexedSubgraphGenerator,
    )
    assert isinstance(
        NetworkXSubgraphGenerator(1),
        IndexedSubgraphGenerator,
    )


def test_generator_uses_current_graph_nodes_after_graph_mutation():
    graph = make_graph()
    generator = ConnectedExpansionSubgraphGenerator(
        num_subgraphs=1,
        subgraph_size=4,
        seed=42,
    )
    original_candidate = generator.generate_at_index(graph, 0)
    assert original_candidate is not None

    networkx_graph = graph.networkx_graph
    networkx_graph.clear()
    networkx_graph.add_edges_from([(20, 21), (21, 22), (22, 23)])

    candidate = generator.generate_at_index(graph, 0)

    assert candidate is not None
    assert candidate.num_vertices == 4
    assert candidate.vertices <= {20, 21, 22, 23}