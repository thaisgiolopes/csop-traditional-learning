import pytest

from src.features.base import FeatureContext, FeatureLevel, FeatureScope
from src.features.engine import FeatureEngine


class FakeGraph:
    """
    Minimal graph representation used for testing the FeatureEngine.
    """

    pass


class FakeFeature:
    """
    Minimal feature implementation used to test FeatureEngine behavior.
    """

    def __init__(self, name, scope, level, value):
        self._name = name
        self.scope = scope
        self.level = level
        self.value = value
        self.calls = []

    @property
    def name(self):
        return self._name

    def compute(self, context):
        self.calls.append(context)
        return self.value


def test_compute_global_executes_global_features():
    """
    Test that compute_global executes features configured for the global
    scope.
    """
    graph = FakeGraph()

    feature = FakeFeature(
        name="test_global",
        scope=FeatureScope.GLOBAL,
        level=FeatureLevel.GRAPH,
        value=42,
    )

    engine = FeatureEngine([feature])

    context = FeatureContext(
        full_graph=graph,
        subgraph=None,
    )

    results = engine.compute_global(context)

    assert results == {
        "test_global": 42,
    }

    assert len(feature.calls) == 1
    assert feature.calls[0] is context


def test_compute_local_executes_local_features():
    """
    Test that compute_local executes features configured for the local
    scope.
    """
    full_graph = FakeGraph()
    subgraph = FakeGraph()

    feature = FakeFeature(
        name="test_local",
        scope=FeatureScope.LOCAL,
        level=FeatureLevel.NODE,
        value={0: 2, 1: 3},
    )

    engine = FeatureEngine([feature])

    context = FeatureContext(
        full_graph=full_graph,
        subgraph=subgraph,
    )

    results = engine.compute_local(context)

    assert results == {
        "test_local": {0: 2, 1: 3},
    }

    assert len(feature.calls) == 1
    assert feature.calls[0] is context


def test_compute_global_does_not_execute_local_features():
    """
    Test that compute_global ignores features configured for the local
    scope.
    """
    feature = FakeFeature(
        name="local_feature",
        scope=FeatureScope.LOCAL,
        level=FeatureLevel.NODE,
        value={0: 1},
    )

    engine = FeatureEngine([feature])

    context = FeatureContext(
        full_graph=FakeGraph(),
        subgraph=FakeGraph(),
    )

    results = engine.compute_global(context)

    assert results == {}
    assert feature.calls == []


def test_compute_local_does_not_execute_global_features():
    """
    Test that compute_local ignores features configured for the global
    scope.
    """
    feature = FakeFeature(
        name="global_feature",
        scope=FeatureScope.GLOBAL,
        level=FeatureLevel.GRAPH,
        value=10,
    )

    engine = FeatureEngine([feature])

    context = FeatureContext(
        full_graph=FakeGraph(),
        subgraph=FakeGraph(),
    )

    results = engine.compute_local(context)

    assert results == {}
    assert feature.calls == []


def test_compute_global_executes_multiple_global_features():
    """
    Test that multiple global features are computed and returned using
    their feature names.
    """
    features = [
        FakeFeature(
            name="feature_a",
            scope=FeatureScope.GLOBAL,
            level=FeatureLevel.GRAPH,
            value=10,
        ),
        FakeFeature(
            name="feature_b",
            scope=FeatureScope.GLOBAL,
            level=FeatureLevel.NODE,
            value={0: 2, 1: 4},
        ),
    ]

    engine = FeatureEngine(features)

    context = FeatureContext(
        full_graph=FakeGraph(),
        subgraph=None,
    )

    results = engine.compute_global(context)

    assert results == {
        "feature_a": 10,
        "feature_b": {0: 2, 1: 4},
    }


def test_compute_local_executes_multiple_local_features():
    """
    Test that multiple local features are computed and returned using
    their feature names.
    """
    features = [
        FakeFeature(
            name="degree",
            scope=FeatureScope.LOCAL,
            level=FeatureLevel.NODE,
            value={0: 2, 1: 4},
        ),
        FakeFeature(
            name="edge_count",
            scope=FeatureScope.LOCAL,
            level=FeatureLevel.GRAPH,
            value=3,
        ),
    ]

    engine = FeatureEngine(features)

    context = FeatureContext(
        full_graph=FakeGraph(),
        subgraph=FakeGraph(),
    )

    results = engine.compute_local(context)

    assert results == {
        "degree": {0: 2, 1: 4},
        "edge_count": 3,
    }


def test_engine_rejects_duplicate_feature_names():
    """
    Test that FeatureEngine rejects multiple features with the same name.

    Feature names are used as identifiers in the dataset and therefore
    must be unique.
    """
    features = [
        FakeFeature(
            name="degree",
            scope=FeatureScope.LOCAL,
            level=FeatureLevel.NODE,
            value={0: 1},
        ),
        FakeFeature(
            name="degree",
            scope=FeatureScope.GLOBAL,
            level=FeatureLevel.GRAPH,
            value=10,
        ),
    ]

    with pytest.raises(ValueError):
        FeatureEngine(features)