from types import SimpleNamespace

from src.dataset.dataset import Dataset
from src.dataset.builder import DatasetBuilder
from src.features.base import FeatureLevel
from src.features.pooling import (
    MaxPooling,
    MeanPooling,
    MinPooling,
    StdPooling,
)


class FakeGraph:
    """
    Minimal graph representation used only for testing DatasetBuilder.
    """

    def __init__(self, name, vertices=None):
        self.name = name
        self.vertices = vertices


class FakeGraphLoader:
    """
    Fake graph loader used to isolate DatasetBuilder from file I/O.
    """

    def __init__(self, graph):
        self.graph = graph
        self.instance_path = SimpleNamespace(name="test_graph")

    def load(self):
        return self.graph


class FakeSubgraphGenerator:
    """
    Fake subgraph generator that returns predefined candidate subgraphs.
    """

    def __init__(self, subgraphs):
        self.subgraphs = subgraphs
        self.received_graph = None

    def generate(self, graph):
        self.received_graph = graph
        return self.subgraphs


class FakeFeature:
    """
    Minimal feature representation used by the fake feature engine.
    """

    def __init__(self, name, level):
        self.name = name
        self.level = level


class FakeFeatureEngine:
    """
    Fake feature engine used to test DatasetBuilder orchestration.
    """

    def __init__(self, features, global_results, local_results):
        self.features = features
        self.global_results = global_results
        self.local_results = local_results

        self.global_calls = []
        self.local_calls = []

    def compute_global(self, context):
        self.global_calls.append(context)
        return self.global_results

    def compute_local(self, context):
        self.local_calls.append(context)
        return self.local_results[context.subgraph]


class FakeObjective:
    """
    Fake objective function that returns predefined values for subgraphs.
    """

    def __init__(self, values):
        self.values = values
        self.calls = []

    def compute(self, subgraph):
        self.calls.append(subgraph)
        return self.values[subgraph]


def create_builder(
    graph,
    subgraphs,
    features,
    global_results,
    local_results,
    objective_values,
    pooling_strategies=None,
):
    """
    Create a DatasetBuilder using fake dependencies.
    """
    graph_loader = FakeGraphLoader(graph)
    subgraph_generator = FakeSubgraphGenerator(subgraphs)
    feature_engine = FakeFeatureEngine(
        features=features,
        global_results=global_results,
        local_results=local_results,
    )
    objective = FakeObjective(objective_values)

    builder = DatasetBuilder(
        graph_loader=graph_loader,
        subgraph_generator=subgraph_generator,
        feature_engine=feature_engine,
        pooling_strategies=pooling_strategies or {},
        objective_function=objective,
    )

    return (
        builder,
        graph_loader,
        subgraph_generator,
        feature_engine,
        objective,
    )


def test_builder_creates_one_sample_per_subgraph():
    """
    Test that DatasetBuilder creates exactly one Sample for each candidate
    subgraph.
    """
    graph = FakeGraph("G")
    subgraphs = [
        FakeGraph("S1"),
        FakeGraph("S2"),
        FakeGraph("S3"),
    ]

    features = [
        FakeFeature("num_vertices", FeatureLevel.GRAPH),
    ]

    global_results = {
        "num_vertices": 10,
    }

    local_results = {
        subgraphs[0]: {},
        subgraphs[1]: {},
        subgraphs[2]: {},
    }

    objective_values = {
        subgraphs[0]: 1.0,
        subgraphs[1]: 2.0,
        subgraphs[2]: 3.0,
    }

    builder, _, _, _, _ = create_builder(
        graph=graph,
        subgraphs=subgraphs,
        features=features,
        global_results=global_results,
        local_results=local_results,
        objective_values=objective_values,
    )

    samples = builder.build(graph_id="graph_001")

    assert len(samples) == 3


def test_builder_uses_supplied_subgraphs_without_regenerating():
    graph = FakeGraph("G")
    subgraphs = [FakeGraph("S1"), FakeGraph("S2")]
    builder, _, generator, _, _ = create_builder(
        graph=graph,
        subgraphs=subgraphs,
        features=[FakeFeature("num_vertices", FeatureLevel.GRAPH)],
        global_results={"num_vertices": 10},
        local_results={subgraph: {} for subgraph in subgraphs},
        objective_values={subgraphs[0]: 1.0, subgraphs[1]: 2.0},
    )
    generator.generate = lambda unused_graph: (_ for _ in ()).throw(
        AssertionError("preselected subgraphs must not be regenerated")
    )

    samples = builder.build(
        graph_id="graph_001",
        candidate_subgraphs=subgraphs,
        sample_ids=["SMP_001", "SMP_002"],
    )

    assert [sample.sample_id for sample in samples] == [
        "SMP_001",
        "SMP_002",
    ]
    assert [sample.subgraph_id for sample in samples] == [0, 1]


def test_builder_preserves_graph_and_subgraph_ids():
    """
    Test that graph_id and generated subgraph_id values are correctly
    stored in each Sample.
    """
    graph = FakeGraph("G")
    subgraphs = [
        FakeGraph("S1"),
        FakeGraph("S2"),
    ]

    features = [
        FakeFeature("num_vertices", FeatureLevel.GRAPH),
    ]

    global_results = {
        "num_vertices": 10,
    }

    local_results = {
        subgraphs[0]: {},
        subgraphs[1]: {},
    }

    objective_values = {
        subgraphs[0]: 1.0,
        subgraphs[1]: 2.0,
    }

    builder, _, _, _, _ = create_builder(
        graph=graph,
        subgraphs=subgraphs,
        features=features,
        global_results=global_results,
        local_results=local_results,
        objective_values=objective_values,
    )

    samples = builder.build(graph_id="graph_001")

    assert samples[0].graph_id == "graph_001"
    assert samples[1].graph_id == "graph_001"

    assert samples[0].subgraph_id == 0
    assert samples[1].subgraph_id == 1


def test_builder_uses_loader_directory_name_when_graph_id_is_not_provided():
    """
    Test that the graph instance directory name is used as the default
    graph identifier.
    """
    graph = FakeGraph("G")
    subgraphs = [FakeGraph("S1")]

    features = [
        FakeFeature("num_vertices", FeatureLevel.GRAPH),
    ]

    global_results = {
        "num_vertices": 10,
    }

    local_results = {
        subgraphs[0]: {},
    }

    objective_values = {
        subgraphs[0]: 1.0,
    }

    builder, _, _, _, _ = create_builder(
        graph=graph,
        subgraphs=subgraphs,
        features=features,
        global_results=global_results,
        local_results=local_results,
        objective_values=objective_values,
    )

    samples = builder.build()

    assert samples[0].graph_id == "test_graph"


def test_global_features_are_computed_once():
    """
    Test that global features are computed once for the complete graph,
    rather than once for each candidate subgraph.
    """
    graph = FakeGraph("G")
    subgraphs = [
        FakeGraph("S1"),
        FakeGraph("S2"),
        FakeGraph("S3"),
    ]

    features = [
        FakeFeature("num_vertices", FeatureLevel.GRAPH),
    ]

    global_results = {
        "num_vertices": 10,
    }

    local_results = {
        subgraphs[0]: {},
        subgraphs[1]: {},
        subgraphs[2]: {},
    }

    objective_values = {
        subgraphs[0]: 1.0,
        subgraphs[1]: 2.0,
        subgraphs[2]: 3.0,
    }

    builder, _, _, feature_engine, _ = create_builder(
        graph=graph,
        subgraphs=subgraphs,
        features=features,
        global_results=global_results,
        local_results=local_results,
        objective_values=objective_values,
    )

    builder.build()

    assert len(feature_engine.global_calls) == 1
    assert feature_engine.global_calls[0].full_graph is graph


def test_local_features_are_computed_for_each_subgraph():
    """
    Test that local features are computed once for each candidate
    subgraph.
    """
    graph = FakeGraph("G")
    subgraphs = [
        FakeGraph("S1"),
        FakeGraph("S2"),
        FakeGraph("S3"),
    ]

    features = [
        FakeFeature("degree", FeatureLevel.NODE),
    ]

    global_results = {}

    local_results = {
        subgraphs[0]: {"degree": {0: 2, 1: 4}},
        subgraphs[1]: {"degree": {0: 3, 1: 5}},
        subgraphs[2]: {"degree": {0: 1, 1: 2}},
    }

    objective_values = {
        subgraphs[0]: 1.0,
        subgraphs[1]: 2.0,
        subgraphs[2]: 3.0,
    }

    builder, _, _, feature_engine, _ = create_builder(
        graph=graph,
        subgraphs=subgraphs,
        features=features,
        global_results=global_results,
        local_results=local_results,
        objective_values=objective_values,
    )

    builder.build()

    assert len(feature_engine.local_calls) == 3

    for call, subgraph in zip(feature_engine.local_calls, subgraphs):
        assert call.full_graph is graph
        assert call.subgraph is subgraph


def test_graph_level_features_are_kept_without_pooling():
    """
    Test that graph-level features are included directly in the sample
    feature vector.
    """
    graph = FakeGraph("G")
    subgraphs = [FakeGraph("S1")]

    features = [
        FakeFeature("num_vertices", FeatureLevel.GRAPH),
    ]

    global_results = {
        "num_vertices": 10,
    }

    local_results = {
        subgraphs[0]: {},
    }

    objective_values = {
        subgraphs[0]: 1.0,
    }

    builder, _, _, _, _ = create_builder(
        graph=graph,
        subgraphs=subgraphs,
        features=features,
        global_results=global_results,
        local_results=local_results,
        objective_values=objective_values,
    )

    samples = builder.build()

    assert samples[0].features == {
        "num_vertices": 10,
    }


def test_node_level_features_are_pooled():
    """
    Test that node-level features are transformed into graph-level
    features using the configured pooling strategy.
    """
    graph = FakeGraph("G")
    subgraphs = [FakeGraph("S1")]

    features = [
        FakeFeature("degree", FeatureLevel.NODE),
    ]

    global_results = {}

    local_results = {
        subgraphs[0]: {
            "degree": {
                0: 2,
                1: 4,
                2: 6,
            }
        }
    }

    objective_values = {
        subgraphs[0]: 1.0,
    }

    pooling_strategies = {
        "degree": (
            MeanPooling(),
        )
    }

    builder, _, _, _, _ = create_builder(
        graph=graph,
        subgraphs=subgraphs,
        features=features,
        global_results=global_results,
        local_results=local_results,
        objective_values=objective_values,
        pooling_strategies=pooling_strategies,
    )

    samples = builder.build()

    assert samples[0].features["degree_mean"] == 4.0


def test_multiple_pooling_strategies_create_multiple_features():
    """
    Test that multiple pooling strategies produce separate feature values.
    """
    graph = FakeGraph("G")
    subgraphs = [FakeGraph("S1")]

    features = [
        FakeFeature("degree", FeatureLevel.NODE),
    ]

    global_results = {}

    local_results = {
        subgraphs[0]: {
            "degree": {
                0: 2,
                1: 4,
                2: 6,
            }
        }
    }

    objective_values = {
        subgraphs[0]: 1.0,
    }

    pooling_strategies = {
        "degree": (
            MeanPooling(),
        )
    }

    builder, _, _, _, _ = create_builder(
        graph=graph,
        subgraphs=subgraphs,
        features=features,
        global_results=global_results,
        local_results=local_results,
        objective_values=objective_values,
        pooling_strategies=pooling_strategies,
    )

    samples = builder.build()

    assert "degree_mean" in samples[0].features
    assert samples[0].features["degree_mean"] == 4.0


def test_builder_and_dataset_preserve_feature_and_pooling_order():
    graph = FakeGraph("G")
    subgraph = FakeGraph("S1")
    features = [
        FakeFeature("local_first", FeatureLevel.NODE),
        FakeFeature("graph_value", FeatureLevel.GRAPH),
        FakeFeature("local_second", FeatureLevel.NODE),
    ]
    builder, _, _, _, _ = create_builder(
        graph=graph,
        subgraphs=[subgraph],
        features=features,
        global_results={"graph_value": 7},
        local_results={
            subgraph: {
                "local_first": {0: 1, 1: 3},
                "local_second": {0: 2, 1: 4},
            }
        },
        objective_values={subgraph: 1.0},
        pooling_strategies={
            "local_first": (MaxPooling(), MeanPooling()),
            "local_second": (MeanPooling(),),
        },
    )

    dataset = Dataset(builder.build())

    assert dataset.feature_names == [
        "graph_value",
        "local_first_max",
        "local_first_mean",
        "local_second_mean",
    ]


def test_global_node_values_are_filtered_per_candidate_and_reused():
    graph = FakeGraph("G", vertices={0, 1, 2})
    subgraphs = [
        FakeGraph("S1", vertices={0, 1}),
        FakeGraph("S2", vertices={1, 2}),
    ]
    original_values = {0: 0.1, 1: 0.9, 2: 0.5}
    global_results = {"score_global": original_values}
    builder, _, _, feature_engine, _ = create_builder(
        graph=graph,
        subgraphs=subgraphs,
        features=[FakeFeature("score_global", FeatureLevel.NODE)],
        global_results=global_results,
        local_results={subgraph: {} for subgraph in subgraphs},
        objective_values={subgraphs[0]: 1.0, subgraphs[1]: 2.0},
        pooling_strategies={
            "score_global": (
                MeanPooling(),
                MaxPooling(),
                MinPooling(),
                StdPooling(),
            ),
        },
    )

    samples = builder.build()
    dataset = Dataset(samples)

    assert len(feature_engine.global_calls) == 1
    assert samples[0].features == {
        "score_global_mean": 0.5,
        "score_global_max": 0.9,
        "score_global_min": 0.1,
        "score_global_std": 0.4,
    }
    assert samples[1].features == {
        "score_global_mean": 0.7,
        "score_global_max": 0.9,
        "score_global_min": 0.5,
        "score_global_std": 0.2,
    }
    assert original_values == {0: 0.1, 1: 0.9, 2: 0.5}
    assert dataset.num_features == 4
    assert dataset.feature_names == [
        "score_global_mean",
        "score_global_max",
        "score_global_min",
        "score_global_std",
    ]


def test_global_node_pooling_supports_singleton_candidate():
    graph = FakeGraph("G", vertices={0, 1})
    subgraph = FakeGraph("single", vertices={1})
    builder, _, _, _, _ = create_builder(
        graph=graph,
        subgraphs=[subgraph],
        features=[FakeFeature("score_global", FeatureLevel.NODE)],
        global_results={"score_global": {0: 0.2, 1: 0.8}},
        local_results={subgraph: {}},
        objective_values={subgraph: 1.0},
        pooling_strategies={
            "score_global": (
                MeanPooling(),
                MaxPooling(),
                MinPooling(),
                StdPooling(),
            ),
        },
    )

    sample = builder.build()[0]

    assert sample.features == {
        "score_global_mean": 0.8,
        "score_global_max": 0.8,
        "score_global_min": 0.8,
        "score_global_std": 0.0,
    }


def test_objective_is_computed_for_each_subgraph():
    """
    Test that the objective function is evaluated independently for every
    candidate subgraph.
    """
    graph = FakeGraph("G")
    subgraphs = [
        FakeGraph("S1"),
        FakeGraph("S2"),
    ]

    features = [
        FakeFeature("num_vertices", FeatureLevel.GRAPH),
    ]

    global_results = {
        "num_vertices": 10,
    }

    local_results = {
        subgraphs[0]: {},
        subgraphs[1]: {},
    }

    objective_values = {
        subgraphs[0]: 1.5,
        subgraphs[1]: 2.75,
    }

    builder, _, _, _, objective = create_builder(
        graph=graph,
        subgraphs=subgraphs,
        features=features,
        global_results=global_results,
        local_results=local_results,
        objective_values=objective_values,
    )

    samples = builder.build()

    assert objective.calls == subgraphs

    assert samples[0].target == 1.5
    assert samples[1].target == 2.75


def test_builder_combines_global_and_local_features():
    """
    Test that global and local feature values are combined into the final
    feature vector of each sample.
    """
    graph = FakeGraph("G")
    subgraphs = [FakeGraph("S1")]

    features = [
        FakeFeature("num_vertices", FeatureLevel.GRAPH),
        FakeFeature("degree", FeatureLevel.NODE),
    ]

    global_results = {
        "num_vertices": 10,
    }

    local_results = {
        subgraphs[0]: {
            "degree": {
                0: 2,
                1: 4,
            }
        }
    }

    objective_values = {
        subgraphs[0]: 1.5,
    }

    pooling_strategies = {
        "degree": (
            MeanPooling(),
        )
    }

    builder, _, _, _, _ = create_builder(
        graph=graph,
        subgraphs=subgraphs,
        features=features,
        global_results=global_results,
        local_results=local_results,
        objective_values=objective_values,
        pooling_strategies=pooling_strategies,
    )

    samples = builder.build()

    assert samples[0].features == {
        "num_vertices": 10,
        "degree_mean": 3.0,
    }

    assert samples[0].target == 1.5