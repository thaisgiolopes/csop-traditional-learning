import pytest

from src.graph import Graph
from src.features.base import Feature
from src.features.global_features import (
    NumEdgesFeature,
    NumVerticesFeature,
)
from src.features.local_features import DegreeFeature


def create_test_graph():
    """
    Create the graph used by the feature tests.

    The graph contains four vertices and four undirected edges:

        0 --- 1
        |     |
        |     |
        2 --- 3

    Returns:
        Graph: A graph containing four vertices and four edges.
    """
    vertices = {0, 1, 2, 3}
    edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
    ]

    return Graph(vertices, edges)


def test_feature_is_abstract():
    """Test that the Feature class cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Feature()


def test_num_vertices_feature_name():
    """Test the name of the number-of-vertices feature."""
    feature = NumVerticesFeature()

    assert feature.name == "num_vertices"


def test_num_vertices_feature_compute():
    """Test that the number-of-vertices feature returns the correct value."""
    graph = create_test_graph()
    feature = NumVerticesFeature()

    result = feature.compute(graph)

    assert result == 4


def test_num_edges_feature_name():
    """Test the name of the number-of-edges feature."""
    feature = NumEdgesFeature()

    assert feature.name == "num_edges"


def test_num_edges_feature_compute():
    """Test that the number-of-edges feature returns the correct value."""
    graph = create_test_graph()
    feature = NumEdgesFeature()

    result = feature.compute(graph)

    assert result == 4


def test_degree_feature_name():
    """Test the name of the degree feature."""
    feature = DegreeFeature()

    assert feature.name == "degree"


def test_degree_feature_compute():
    """Test that the degree feature returns the correct degree for every vertex."""
    graph = create_test_graph()
    feature = DegreeFeature()

    result = feature.compute(graph)

    expected = {
        0: 2,
        1: 2,
        2: 2,
        3: 2,
    }

    assert result == expected


def test_degree_feature_preserves_vertex_identifiers():
    """Test that degree values are associated with the correct vertex identifiers."""
    graph = Graph(
        vertices={10, 20, 30},
        edges=[
            (10, 20),
            (20, 30),
        ],
    )

    feature = DegreeFeature()

    result = feature.compute(graph)

    expected = {
        10: 1,
        20: 2,
        30: 1,
    }

    assert result == expected


def test_features_reject_invalid_graph():
    """Test that feature computation rejects objects that are not Graph instances."""
    invalid_graph = "not a graph"

    features = [
        NumVerticesFeature(),
        NumEdgesFeature(),
        DegreeFeature(),
    ]

    for feature in features:
        with pytest.raises(TypeError):
            feature.compute(invalid_graph)


def test_num_vertices_feature_with_empty_graph():
    """Test the number-of-vertices feature on an empty graph."""
    graph = Graph(set(), [])
    feature = NumVerticesFeature()

    assert feature.compute(graph) == 0


def test_num_edges_feature_with_empty_graph():
    """Test the number-of-edges feature on an empty graph."""
    graph = Graph(set(), [])
    feature = NumEdgesFeature()

    assert feature.compute(graph) == 0


def test_degree_feature_with_empty_graph():
    """Test the degree feature on an empty graph."""
    graph = Graph(set(), [])
    feature = DegreeFeature()

    assert feature.compute(graph) == {}