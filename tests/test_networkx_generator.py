import networkx as nx
import pytest

from src.graph.graph import Graph
from src.subgraphs.networkx_generator import NetworkXSubgraphGenerator


def create_test_graph():
    """
    Create a small connected graph suitable for testing random
    subgraph generation.
    """
    return Graph(
        vertices=[0, 1, 2, 3, 4, 5],
        edges=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (0, 5),
            (1, 4),
            (2, 5),
        ],
    )


def test_generator_returns_requested_number_of_subgraphs():
    """
    Test that the generator produces the requested number of candidate
    subgraphs.
    """
    graph = create_test_graph()

    generator = NetworkXSubgraphGenerator(
        num_subgraphs=5,
        seed=42,
    )

    subgraphs = generator.generate(graph)

    assert len(subgraphs) == 5


def test_generator_returns_graph_objects():
    """
    Test that generated candidates use the project's Graph representation.
    """
    graph = create_test_graph()

    generator = NetworkXSubgraphGenerator(
        num_subgraphs=5,
        seed=42,
    )

    subgraphs = generator.generate(graph)

    assert all(isinstance(subgraph, Graph) for subgraph in subgraphs)


def test_generated_vertices_belong_to_original_graph():
    """
    Test that generated subgraphs contain only vertices from the original
    graph.
    """
    graph = create_test_graph()

    generator = NetworkXSubgraphGenerator(
        num_subgraphs=10,
        seed=42,
    )

    subgraphs = generator.generate(graph)

    for subgraph in subgraphs:
        assert subgraph.vertices.issubset(graph.vertices)


def test_generated_subgraphs_have_valid_edges():
    """
    Test that every edge of every generated subgraph connects vertices
    belonging to that subgraph.
    """
    graph = create_test_graph()

    generator = NetworkXSubgraphGenerator(
        num_subgraphs=10,
        seed=42,
    )

    subgraphs = generator.generate(graph)

    for subgraph in subgraphs:
        for u, v in subgraph.edges:
            assert u in subgraph.vertices
            assert v in subgraph.vertices


def test_generated_subgraphs_are_connected():
    """
    Test that generated candidate subgraphs are connected.
    """
    graph = create_test_graph()

    generator = NetworkXSubgraphGenerator(
        num_subgraphs=10,
        seed=42,
    )

    subgraphs = generator.generate(graph)

    for subgraph in subgraphs:
        networkx_graph = nx.Graph()

        networkx_graph.add_nodes_from(subgraph.vertices)
        networkx_graph.add_edges_from(subgraph.edges)

        assert nx.is_connected(networkx_graph)


def test_generator_rejects_invalid_graph():
    """
    Test that the generator rejects objects that are not Graph instances.
    """
    generator = NetworkXSubgraphGenerator(
        num_subgraphs=5,
        seed=42,
    )

    with pytest.raises(ValueError):
        generator.generate(None)


def test_generator_rejects_non_positive_number_of_subgraphs():
    """
    Test that the generator rejects an invalid number of requested
    candidate subgraphs.
    """
    with pytest.raises(ValueError):
        NetworkXSubgraphGenerator(
            num_subgraphs=0,
            seed=42,
        )


def test_generator_is_reproducible_with_same_seed():
    """
    Test that two generators configured with the same random seed produce
    equivalent candidate subgraphs.
    """
    graph = create_test_graph()

    generator_a = NetworkXSubgraphGenerator(
        num_subgraphs=5,
        seed=42,
    )

    generator_b = NetworkXSubgraphGenerator(
        num_subgraphs=5,
        seed=42,
    )

    subgraphs_a = generator_a.generate(graph)
    subgraphs_b = generator_b.generate(graph)

    assert len(subgraphs_a) == len(subgraphs_b)

    for subgraph_a, subgraph_b in zip(subgraphs_a, subgraphs_b):
        assert subgraph_a.vertices == subgraph_b.vertices
        assert subgraph_a.edges == subgraph_b.edges