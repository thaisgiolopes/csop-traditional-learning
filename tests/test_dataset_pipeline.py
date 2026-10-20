from types import SimpleNamespace

import networkx as nx
import pytest

from src.dataset.builder import DatasetBuilder
from src.dataset.loader import DatasetLoader
from src.features.base import FeatureScope
from src.features.engine import FeatureEngine
from src.features.node_features import DegreeFeature, PageRankFeature
from src.features.pooling import MaxPooling, MeanPooling
from src.graph import Graph
from src.pipeline.dataset_pipeline import DatasetPipeline


class FakeGraphLoader:
    def __init__(self, graph):
        self.graph = graph
        self.instance_path = SimpleNamespace(name="pipeline_graph")

    def load(self):
        return self.graph


class FakeSubgraphGenerator:
    def generate(self, graph):
        raise AssertionError("provided candidates should be used directly")


class FakeObjective:
    def __init__(self, targets):
        self.targets = targets

    def compute(self, subgraph):
        return self.targets[subgraph]


def create_pipeline(output_dir, *, include_local_degree=True):
    full_graph = Graph(
        vertices={0, 1, 2, 3},
        edges=[(0, 1), (1, 2), (2, 0), (2, 3)],
    )
    triangle = Graph(
        vertices={0, 1, 2},
        edges=[(0, 1), (1, 2), (2, 0)],
    )
    edge = Graph(vertices={2, 3}, edges=[(2, 3)])
    features = [PageRankFeature(scope=FeatureScope.GLOBAL)]
    pooling = {
        "pagerank_global": (MeanPooling(), MaxPooling()),
    }
    if include_local_degree:
        features.append(DegreeFeature(scope=FeatureScope.LOCAL))
        pooling["degree"] = (MeanPooling(), MaxPooling())

    builder = DatasetBuilder(
        graph_loader=FakeGraphLoader(full_graph),
        subgraph_generator=FakeSubgraphGenerator(),
        feature_engine=FeatureEngine(features),
        pooling_strategies=pooling,
        objective_function=FakeObjective({triangle: 3.0, edge: 1.0}),
    )
    pipeline = DatasetPipeline(builder, output_dir)
    return pipeline, full_graph, [triangle, edge]


def test_pipeline_persists_aligned_candidate_features_targets_and_metadata(
    tmp_path,
    monkeypatch,
):
    pipeline, _, candidates = create_pipeline(tmp_path)
    pagerank = nx.pagerank
    calls = []

    def counted_pagerank(*args, **kwargs):
        calls.append(args[0])
        return pagerank(*args, **kwargs)

    monkeypatch.setattr(nx, "pagerank", counted_pagerank)
    experiment_metadata = {
        "feature_set_id": "global-pagerank-local-degree-v1",
        "feature_names": [
            "pagerank_global_mean",
            "pagerank_global_max",
            "degree_mean",
            "degree_max",
        ],
        "objective_name": "test_objective",
    }

    built = pipeline.run(
        graph_id="graph-A",
        output_path=tmp_path / "processed" / "graph-A.csv",
        candidate_subgraphs=candidates,
        sample_ids=["sample-triangle", "sample-edge"],
        experiment_metadata=experiment_metadata,
    )

    dataset_path = tmp_path / "processed" / "graph-A.csv"
    expected_configuration = {
        "feature_pipeline": pipeline._dataset_builder.feature_configuration,
        "experiment": experiment_metadata,
    }
    loaded = DatasetLoader(
        dataset_path,
        expected_configuration=expected_configuration,
    ).load()

    assert built.num_samples == 2
    assert built.num_features == 4
    assert built.feature_names == [
        "pagerank_global_mean",
        "pagerank_global_max",
        "degree_mean",
        "degree_max",
    ]
    assert list(loaded.X.columns) == built.feature_names
    assert loaded.X.shape == (2, 4)
    assert loaded.y.shape == (2,)
    assert loaded.graph_ids == ["graph-A", "graph-A"]
    assert loaded.subgraph_ids == [0, 1]
    assert loaded.samples[0].sample_id == "sample-triangle"
    assert loaded.samples[1].sample_id == "sample-edge"
    assert loaded.y.tolist() == [3.0, 1.0]
    global_pagerank = pagerank(calls[0], weight=None)
    assert loaded.X["pagerank_global_mean"].tolist() == pytest.approx(
        [
            sum(global_pagerank[node] for node in candidates[0].vertices) / 3,
            sum(global_pagerank[node] for node in candidates[1].vertices) / 2,
        ]
    )
    assert loaded.X["degree_mean"].tolist() == [2.0, 1.0]
    assert loaded.X["degree_max"].tolist() == [2, 1]
    assert loaded.metadata == expected_configuration
    assert len(calls) == 1
    assert calls[0] is not candidates[0].networkx_graph

    with pytest.raises(ValueError, match="does not match"):
        DatasetLoader(
            dataset_path,
            expected_configuration={
                **expected_configuration,
                "experiment": {
                    **experiment_metadata,
                    "feature_set_id": "different-feature-set",
                },
            },
        ).load()

    dataset_path.write_text(
        dataset_path.read_text(encoding="utf-8").replace("3.0", "4.0"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match its configuration"):
        DatasetLoader(dataset_path).load()


def test_pipeline_configuration_changes_when_features_are_disabled(tmp_path):
    pipeline_with_degree, _, candidates = create_pipeline(tmp_path / "with")
    pipeline_without_degree, _, candidates_without_degree = create_pipeline(
        tmp_path / "without",
        include_local_degree=False,
    )
    feature_set_metadata = {
        "feature_set_id": "configurable-set-v1",
    }

    with_degree = pipeline_with_degree.run(
        graph_id="graph-A",
        candidate_subgraphs=candidates,
        experiment_metadata=feature_set_metadata,
    )
    without_degree = pipeline_without_degree.run(
        graph_id="graph-A",
        candidate_subgraphs=candidates_without_degree,
        experiment_metadata=feature_set_metadata,
    )

    assert with_degree.feature_names == [
        "pagerank_global_mean",
        "pagerank_global_max",
        "degree_mean",
        "degree_max",
    ]
    assert without_degree.feature_names == [
        "pagerank_global_mean",
        "pagerank_global_max",
    ]
    assert with_degree.metadata != without_degree.metadata


def test_pipeline_without_new_features_keeps_legacy_behavior(tmp_path):
    full_graph = Graph(vertices={0, 1, 2}, edges=[(0, 1), (1, 2)])
    candidate = Graph(vertices={0, 1}, edges=[(0, 1)])
    builder = DatasetBuilder(
        graph_loader=FakeGraphLoader(full_graph),
        subgraph_generator=FakeSubgraphGenerator(),
        feature_engine=FeatureEngine([DegreeFeature()]),
        pooling_strategies={"degree": (MeanPooling(),)},
        objective_function=FakeObjective({candidate: 1.0}),
    )

    dataset = DatasetPipeline(builder, tmp_path).run(
        candidate_subgraphs=[candidate]
    )

    assert dataset.feature_names == ["degree_mean"]
    assert dataset.X["degree_mean"].tolist() == [1.0]
    assert dataset.y.tolist() == [1.0]
    assert dataset.metadata["experiment"] == {}


def test_pipeline_records_feature_and_persistence_stages_separately(tmp_path):
    pipeline, _, candidates = create_pipeline(tmp_path)
    recorded_stages = []

    class StageContext:
        def __init__(self, stage_name):
            self.stage_name = stage_name

        def __enter__(self):
            recorded_stages.append((self.stage_name, "started"))

        def __exit__(self, exception_type, exception, traceback):
            recorded_stages.append((self.stage_name, "finished"))
            return False

    dataset = pipeline.run(
        graph_id="graph-A",
        candidate_subgraphs=candidates,
        stage_context_factory=StageContext,
    )

    assert dataset.num_samples == len(candidates)
    assert recorded_stages == [
        ("feature_extraction_and_targets", "started"),
        ("feature_extraction_and_targets", "finished"),
        ("dataset_construction_and_persistence", "started"),
        ("dataset_construction_and_persistence", "finished"),
    ]