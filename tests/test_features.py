import pytest

from src.graph import Graph
from src.features.base import (
    Feature,
    FeatureContext,
    FeatureLevel,
    FeatureScope,
)
from src.features.graph_features import (
    NumEdgesFeature,
    NumVerticesFeature,
)
from src.features.node_features import DegreeFeature


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


def create_test_context():
    """Create a context with distinct complete and candidate graphs."""
    full_graph = create_test_graph()
    subgraph = Graph(
        vertices={0, 1, 2},
        edges=[(0, 1), (0, 2)],
    )

    return FeatureContext(full_graph, subgraph)


def test_feature_is_abstract():
    """Test that the Feature class cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Feature()


def test_num_vertices_feature_name():
    """Test the name of the number-of-vertices feature."""
    feature = NumVerticesFeature()

    assert feature.name == "num_vertices"
    assert feature.scope is FeatureScope.GLOBAL
    assert feature.level is FeatureLevel.GRAPH


def test_num_vertices_feature_compute():
    """Test that the number-of-vertices feature returns the correct value."""
    context = create_test_context()
    feature = NumVerticesFeature()

    result = feature.compute(context)

    assert result == 4


def test_num_edges_feature_name():
    """Test the name of the number-of-edges feature."""
    feature = NumEdgesFeature()

    assert feature.name == "num_edges"
    assert feature.scope is FeatureScope.GLOBAL
    assert feature.level is FeatureLevel.GRAPH


def test_num_edges_feature_compute():
    """Test that the number-of-edges feature returns the correct value."""
    context = create_test_context()
    feature = NumEdgesFeature()

    result = feature.compute(context)

    assert result == 4


def test_degree_feature_name():
    """Test the name of the degree feature."""
    feature = DegreeFeature()

    assert feature.name == "degree"
    assert feature.scope is FeatureScope.LOCAL
    assert feature.level is FeatureLevel.NODE


def test_degree_feature_compute():
    """Test that the degree feature returns the correct degree for every vertex."""
    context = create_test_context()
    feature = DegreeFeature()

    result = feature.compute(context)

    expected = {
        0: 2,
        1: 1,
        2: 1,
    }

    assert result == expected


def test_degree_feature_global_scope_uses_full_graph():
    """Test that global degree uses the complete graph."""
    context = create_test_context()
    feature = DegreeFeature(scope=FeatureScope.GLOBAL)

    result = feature.compute(context)

    assert result == {0: 2, 1: 2, 2: 2, 3: 2}


def test_degree_feature_preserves_vertex_identifiers():
    """Test that degree values are associated with the correct vertex identifiers."""
    context = FeatureContext(
        full_graph=Graph(
            vertices={10, 20, 30},
            edges=[
                (10, 20),
                (20, 30),
            ],
        ),
        subgraph=Graph(
            vertices={10, 20, 30},
            edges=[
                (10, 20),
                (20, 30),
            ],
        ),
    )

    feature = DegreeFeature()

    result = feature.compute(context)

    expected = {
        10: 1,
        20: 2,
        30: 1,
    }

    assert result == expected


def test_features_reject_invalid_context():
    """Test that feature computation rejects invalid contexts."""
    invalid_context = "not a feature context"

    features = [
        NumVerticesFeature(),
        NumEdgesFeature(),
        DegreeFeature(),
    ]

    for feature in features:
        with pytest.raises(TypeError):
            feature.compute(invalid_context)


def test_num_vertices_feature_with_empty_graph():
    """Test the number-of-vertices feature on an empty graph."""
    context = FeatureContext(Graph(set(), []), Graph(set(), []))
    feature = NumVerticesFeature()

    assert feature.compute(context) == 0


def test_num_edges_feature_with_empty_graph():
    """Test the number-of-edges feature on an empty graph."""
    context = FeatureContext(Graph(set(), []), Graph(set(), []))
    feature = NumEdgesFeature()

    assert feature.compute(context) == 0


def test_degree_feature_with_empty_graph():
    """Test the degree feature on an empty graph."""
    context = FeatureContext(Graph(set(), []), Graph(set(), []))
    feature = DegreeFeature()

    assert feature.compute(context) == {}