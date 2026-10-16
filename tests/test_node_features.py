import networkx as nx
import pytest

from src.application.csop_experiment_runner import _build_feature_set
from src.features.base import FeatureContext, FeatureLevel, FeatureScope
from src.features.engine import FeatureEngine
from src.features.node_features import (
    AverageNeighborDegreeFeature,
    BetweennessCentralityFeature,
    ClosenessCentralityFeature,
    EigenvectorCentralityFeature,
    KCoreNumberFeature,
    LocalClusteringCoefficientFeature,
    NormalizedDegreeFeature,
    PageRankFeature,
    SquareClusteringFeature,
    TriangleCountFeature,
)
from src.graph import Graph


FEATURE_TYPES = (
    PageRankFeature,
    BetweennessCentralityFeature,
    ClosenessCentralityFeature,
    EigenvectorCentralityFeature,
    KCoreNumberFeature,
    TriangleCountFeature,
    LocalClusteringCoefficientFeature,
    SquareClusteringFeature,
    AverageNeighborDegreeFeature,
    NormalizedDegreeFeature,
)


def make_graph(vertices, edges):
    return Graph(vertices=vertices, edges=edges)


def expected_values(feature_type, graph):
    if feature_type is PageRankFeature:
        return nx.pagerank(graph, weight=None)
    if feature_type is BetweennessCentralityFeature:
        return nx.betweenness_centrality(graph, weight=None)
    if feature_type is ClosenessCentralityFeature:
        return nx.closeness_centrality(graph, distance=None)
    if feature_type is EigenvectorCentralityFeature:
        return nx.eigenvector_centrality(graph, weight=None)
    if feature_type is KCoreNumberFeature:
        return nx.core_number(graph)
    if feature_type is TriangleCountFeature:
        return nx.triangles(graph)
    if feature_type is LocalClusteringCoefficientFeature:
        return nx.clustering(graph, weight=None)
    if feature_type is SquareClusteringFeature:
        return nx.square_clustering(graph)
    if feature_type is AverageNeighborDegreeFeature:
        return nx.average_neighbor_degree(graph, weight=None)
    if feature_type is NormalizedDegreeFeature:
        node_count = graph.number_of_nodes()
        return {
            node: degree / (node_count - 1) if node_count > 1 else 0.0
            for node, degree in graph.degree()
        }
    raise AssertionError(f"Unexpected feature type: {feature_type!r}")


def test_features_are_node_level_and_have_scope_qualified_names():
    for feature_type in FEATURE_TYPES:
        local_feature = feature_type()
        global_feature = feature_type(scope=FeatureScope.GLOBAL)

        assert local_feature.level is FeatureLevel.NODE
        assert local_feature.scope is FeatureScope.LOCAL
        assert local_feature.name.endswith("_local")
        assert global_feature.scope is FeatureScope.GLOBAL
        assert global_feature.name.endswith("_global")
        assert local_feature.name != global_feature.name


@pytest.mark.parametrize("feature_type", FEATURE_TYPES)
def test_features_preserve_networkx_node_mappings(feature_type):
    full_graph = make_graph(
        ["a", "b", "c", "d", "isolated"],
        [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")],
    )
    candidate = make_graph(["a", "b", "c"], [("a", "b"), ("b", "c")])
    context = FeatureContext(full_graph, candidate)

    feature = feature_type()
    result = feature.compute(context)
    expected = expected_values(feature_type, candidate.networkx_graph)

    assert set(result) == candidate.vertices
    assert result == pytest.approx(expected)


def test_triangle_count_and_clustering_have_explicit_per_node_values():
    graph = make_graph(
        [10, 20, 30, 40],
        [(10, 20), (20, 30), (30, 10)],
    )
    context = FeatureContext(graph, graph)
    expected = {10: 1, 20: 1, 30: 1, 40: 0}

    assert TriangleCountFeature().compute(context) == expected
    assert LocalClusteringCoefficientFeature().compute(context) == expected


def test_square_clustering_detects_cycle_and_zeroes_isolate():
    graph = make_graph(
        ["a", "b", "c", "d", "isolated"],
        [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")],
    )

    result = SquareClusteringFeature().compute(FeatureContext(graph, graph))

    assert result == pytest.approx(
        {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0, "isolated": 0.0}
    )


def test_normalized_degree_uses_n_minus_one_and_zero_for_small_graphs():
    graph = make_graph(["a", "b", "c", "isolated"], [("a", "b"), ("b", "c")])
    feature = NormalizedDegreeFeature()

    assert feature.compute(FeatureContext(graph, graph)) == pytest.approx(
        {"a": 1 / 3, "b": 2 / 3, "c": 1 / 3, "isolated": 0.0}
    )

    singleton = make_graph(["only"], [])
    assert feature.compute(FeatureContext(singleton, singleton)) == {"only": 0.0}


@pytest.mark.parametrize("feature_type", FEATURE_TYPES)
def test_feature_uses_selected_graph_scope(feature_type):
    full_graph = make_graph(
        [0, 1, 2, 3],
        [(0, 1), (1, 2), (2, 0), (2, 3)],
    )
    induced_candidate = make_graph([0, 1, 2], [(0, 1), (1, 2), (2, 0)])
    context = FeatureContext(full_graph, induced_candidate)

    local = feature_type(scope=FeatureScope.LOCAL).compute(context)
    global_values = feature_type(scope=FeatureScope.GLOBAL).compute(context)

    assert set(local) == induced_candidate.vertices
    assert set(global_values) == full_graph.vertices
    assert local == pytest.approx(
        expected_values(feature_type, induced_candidate.networkx_graph)
    )
    assert global_values == pytest.approx(
        expected_values(feature_type, full_graph.networkx_graph)
    )


@pytest.mark.parametrize("feature_type", FEATURE_TYPES)
def test_features_handle_empty_and_isolated_graphs(feature_type):
    feature = feature_type()
    empty = make_graph([], [])
    isolates = make_graph(["a", "b"], [])

    assert feature.compute(FeatureContext(empty, empty)) == {}
    result = feature.compute(FeatureContext(isolates, isolates))
    assert set(result) == {"a", "b"}
    assert all(value >= 0 for value in result.values())


def test_disconnected_feature_values_match_networkx_conventions():
    graph = make_graph(
        [0, 1, 2, 3, 4, 5],
        [(0, 1), (1, 2), (2, 0), (3, 4)],
    )
    context = FeatureContext(graph, graph)

    closeness = ClosenessCentralityFeature().compute(context)
    pagerank = PageRankFeature().compute(context)
    cores = KCoreNumberFeature().compute(context)
    eigenvector = EigenvectorCentralityFeature().compute(context)

    assert closeness == pytest.approx(
        nx.closeness_centrality(graph.networkx_graph, distance=None)
    )
    assert pagerank == pytest.approx(nx.pagerank(graph.networkx_graph, weight=None))
    assert eigenvector == pytest.approx(
        nx.eigenvector_centrality(graph.networkx_graph, weight=None)
    )
    assert cores == {0: 2, 1: 2, 2: 2, 3: 1, 4: 1, 5: 0}


def test_global_feature_mapping_is_reused_by_feature_engine(monkeypatch):
    graph = make_graph([0, 1, 2], [(0, 1), (1, 2)])
    candidate_a = make_graph([0, 1], [(0, 1)])
    candidate_b = make_graph([1, 2], [(1, 2)])
    calls = []
    pagerank = nx.pagerank

    def counting_pagerank(*args, **kwargs):
        calls.append(args[0])
        return pagerank(*args, **kwargs)

    monkeypatch.setattr(nx, "pagerank", counting_pagerank)
    feature = PageRankFeature(scope=FeatureScope.GLOBAL)
    engine = FeatureEngine([feature])

    first = engine.compute_global(FeatureContext(graph, candidate_a))
    second = engine.compute_global(FeatureContext(graph, candidate_b))

    assert len(calls) == 1
    assert first == second
    assert list(first) == ["pagerank_global"]
    assert set(first["pagerank_global"]) == graph.vertices


def test_feature_engine_preserves_configured_order_within_each_scope():
    graph = make_graph([0, 1, 2], [(0, 1), (1, 2)])
    features = [
        NormalizedDegreeFeature(),
        PageRankFeature(),
        NormalizedDegreeFeature(scope=FeatureScope.GLOBAL),
        PageRankFeature(scope=FeatureScope.GLOBAL),
    ]
    engine = FeatureEngine(features)
    context = FeatureContext(graph, graph)

    assert list(engine.compute_local(context)) == [
        "normalized_degree_local",
        "pagerank_local",
    ]
    assert list(engine.compute_global(context)) == [
        "normalized_degree_global",
        "pagerank_global",
    ]


@pytest.mark.parametrize(
    ("feature_set_id", "scope"),
    [
        ("graph-size-and-node-structure-local-v1", FeatureScope.LOCAL),
        ("graph-size-and-node-structure-global-v1", FeatureScope.GLOBAL),
    ],
)
def test_runner_feature_sets_preserve_scope_and_pooling_order(
    feature_set_id,
    scope,
):
    engine, pooling_strategies = _build_feature_set(feature_set_id)

    assert [feature.name for feature in engine.features[:2]] == [
        "num_vertices",
        "num_edges",
    ]
    node_features = engine.features[2:]
    assert len(node_features) == len(FEATURE_TYPES)
    assert all(feature.scope is scope for feature in node_features)
    assert list(pooling_strategies) == [feature.name for feature in node_features]
    assert all(
        tuple(pooling.name for pooling in pooling_strategies[feature.name])
        == ("mean", "max", "min", "std")
        for feature in node_features
    )


def test_existing_runner_feature_set_remains_unchanged():
    engine, pooling_strategies = _build_feature_set(
        "graph-size-and-local-degree-v1"
    )

    assert [feature.name for feature in engine.features] == [
        "num_vertices",
        "num_edges",
        "degree",
    ]
    assert list(pooling_strategies) == ["degree"]


def test_runner_feature_set_can_enable_selected_features_only():
    engine, pooling_strategies = _build_feature_set(
        "graph-size-and-node-structure-local-v1",
        enabled_features=["pagerank", "triangle_count"],
    )

    assert [feature.name for feature in engine.features] == [
        "num_vertices",
        "num_edges",
        "pagerank_local",
        "triangle_count_local",
    ]
    assert list(pooling_strategies) == [
        "pagerank_local",
        "triangle_count_local",
    ]


def test_runner_feature_set_can_disable_all_node_features():
    engine, pooling_strategies = _build_feature_set(
        "graph-size-and-node-structure-global-v1",
        enabled_features=[],
    )

    assert [feature.name for feature in engine.features] == [
        "num_vertices",
        "num_edges",
    ]
    assert pooling_strategies == {}


def test_runner_rejects_unknown_enabled_feature():
    with pytest.raises(ValueError, match="Unknown enabled features"):
        _build_feature_set(
            "graph-size-and-node-structure-local-v1",
            enabled_features=["not_a_feature"],
        )


@pytest.mark.parametrize("feature_type", FEATURE_TYPES)
def test_features_reject_invalid_context_and_selected_graph(feature_type):
    feature = feature_type()
    with pytest.raises(TypeError, match="FeatureContext"):
        feature.compute(None)

    valid_graph = make_graph([0], [])
    invalid_context = FeatureContext(valid_graph, object())
    with pytest.raises(TypeError, match="Graph instance"):
        feature.compute(invalid_context)


@pytest.mark.parametrize("feature_type", FEATURE_TYPES)
def test_features_reject_invalid_scope(feature_type):
    with pytest.raises(TypeError, match="FeatureScope"):
        feature_type(scope="local")