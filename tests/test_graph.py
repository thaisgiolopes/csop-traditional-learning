import pytest
from src.graph import Graph


def test_graph_creation():
    """Test the creation of a graph with vertices and edges."""
    vertices = {0, 1, 2, 3}
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]

    graph = Graph(vertices, edges)

    assert graph.num_vertices == 4
    assert graph.num_edges == 4


def test_graph_removes_duplicate_edges():
    """Test that duplicate undirected edges are stored only once."""
    vertices = {0, 1}
    edges = [(0, 1), (1, 0), (0, 1)]

    graph = Graph(vertices, edges)

    assert graph.num_edges == 1


def test_has_vertex():
    """Test vertex membership queries."""
    graph = Graph({0, 1, 2}, [(0, 1)])

    assert graph.has_vertex(0)
    assert graph.has_vertex(1)
    assert graph.has_vertex(2)
    assert not graph.has_vertex(3)


def test_has_edge():
    """Test undirected edge membership queries."""
    graph = Graph({0, 1, 2}, [(0, 1), (1, 2)])

    assert graph.has_edge(0, 1)
    assert graph.has_edge(1, 0)
    assert graph.has_edge(1, 2)
    assert graph.has_edge(2, 1)
    assert not graph.has_edge(0, 2)


def test_has_edge_with_invalid_vertex():
    """Test that querying an invalid vertex raises an error."""
    graph = Graph({0, 1}, [(0, 1)])

    with pytest.raises(ValueError):
        graph.has_edge(0, 2)


def test_neighbors():
    """Test retrieving the neighbors of a vertex."""
    graph = Graph(
        {0, 1, 2, 3},
        [(0, 1), (0, 2), (1, 3), (2, 3)],
    )

    assert graph.neighbors(0) == {1, 2}
    assert graph.neighbors(1) == {0, 3}
    assert graph.neighbors(2) == {0, 3}
    assert graph.neighbors(3) == {1, 2}


def test_neighbors_with_invalid_vertex():
    """Test that requesting neighbors of an invalid vertex raises an error."""
    graph = Graph({0, 1}, [(0, 1)])

    with pytest.raises(ValueError):
        graph.neighbors(2)


def test_graph_rejects_self_loop():
    """Test that self-loops are rejected."""
    with pytest.raises(ValueError):
        Graph({0, 1}, [(0, 0)])


def test_graph_rejects_unknown_vertex_in_edge():
    """Test that edges cannot reference unknown vertices."""
    with pytest.raises(ValueError):
        Graph({0, 1}, [(0, 2)])