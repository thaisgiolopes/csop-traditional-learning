import json
from contextlib import nullcontext

import networkx as nx
import pytest

from src.experiments.artifact_store import ArtifactStore
from src.experiments.feature_benchmark import FeatureExtractionBenchmark
from src.experiments.resource_monitor import ProcessMemoryMonitor
from src.features.base import FeatureContext, FeatureScope
from src.features.graph_features import NumVerticesFeature
from src.features.node_features import DegreeFeature, PageRankFeature
from src.graph import Graph


class FakeExperiment:
    def __init__(self, root):
        self.artifact_store = ArtifactStore(root)
        self.registered = []
        self.stages = []

    def track_stage(self, name):
        self.stages.append(name)
        return nullcontext()

    def register_artifact(self, category, filename, artifact_type):
        self.registered.append((category, filename, artifact_type))


def graph_and_candidates():
    full_graph = Graph(
        vertices={0, 1, 2, 3, 4},
        edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 3)],
    )
    candidates = [
        Graph(vertices={0, 1, 2}, edges=[(0, 1), (1, 2)]),
        Graph(vertices={2, 3, 4}, edges=[(2, 3), (3, 4)]),
    ]
    return full_graph, candidates


def test_benchmark_records_computation_scope_and_reuse_without_mutation():
    graph, candidates = graph_and_candidates()
    feature = PageRankFeature(scope=FeatureScope.GLOBAL)
    original_values = feature.compute(FeatureContext(graph, graph))
    benchmark = FeatureExtractionBenchmark(
        "benchmark-graph",
        graph,
        candidates,
        [feature, DegreeFeature(scope=FeatureScope.LOCAL)],
        candidate_ids=["left", "right"],
        repeats=2,
        random_seed=13,
    )

    report = benchmark.run()

    assert len(report.records) == 2 * (1 + len(candidates) * 2 * 2)
    assert report.metadata["graph_id"] == "benchmark-graph"
    assert report.metadata["graph_num_vertices"] == 5
    assert report.metadata["graph_num_edges"] == 6
    assert report.metadata["random_seed"] == 13
    assert report.metadata["repeats"] == 2
    assert len(report.metadata["candidate_sizes"]) == 2
    assert all(
        "graph_fingerprint" in candidate
        for candidate in report.metadata["candidate_sizes"]
    )

    required_fields = {
        "graph_id",
        "graph_num_vertices",
        "graph_num_edges",
        "subgraph_num_vertices",
        "subgraph_num_edges",
        "feature_name",
        "computation_scope",
        "operation",
        "elapsed_seconds",
        "result_reused",
        "result_status",
        "random_seed",
    }
    assert all(required_fields <= record.keys() for record in report.records)
    assert all(record["elapsed_seconds"] >= 0 for record in report.records)

    pagerank_records = [
        record for record in report.records
        if record["feature_name"] == feature.name
    ]
    assert sum(record["result_status"] == "newly_computed" for record in pagerank_records) == 1 + 4
    assert sum(record["result_status"] == "reused" for record in pagerank_records) == 4
    assert all(
        record["result_reused"]
        for record in pagerank_records
        if record["result_status"] == "reused"
    )
    assert all(
        record["operation"] == "mapping_filter_reuse"
        for record in pagerank_records
        if record["result_reused"]
    )
    assert {record["computation_scope"] for record in pagerank_records} == {
        "original_graph",
        "candidate_subgraph",
    }
    assert feature.compute(FeatureContext(graph, graph)) == pytest.approx(
        original_values
    )

    comparison = report.comparisons[0]
    assert comparison["feature_name"] == feature.name
    assert comparison["candidate_measurement_count"] == 4
    assert comparison["reuse_measurement_count"] == 4
    assert "not mathematically equivalent" in comparison["semantic_note"]


def test_benchmark_optionally_records_process_memory():
    graph, candidates = graph_and_candidates()
    reading = iter([100, 120, 130, 150, 200, 230, 240, 270])
    monitor = ProcessMemoryMonitor(memory_reader=lambda: next(reading))
    benchmark = FeatureExtractionBenchmark(
        "memory-test",
        graph,
        candidates[:1],
        [DegreeFeature()],
        memory_monitor=monitor,
    )

    report = benchmark.run()

    assert report.metadata["memory_measurement_enabled"] is True
    assert all(record["memory_before_bytes"] is not None for record in report.records)
    assert all(record["memory_after_bytes"] is not None for record in report.records)
    assert all(record["memory_unit"] == "bytes" for record in report.records)
    assert all(
        record["measurement_method"] == "linux_procfs_resident_set_size"
        for record in report.records
    )


def test_benchmark_values_match_networkx_and_feature_mapping_is_not_mutated():
    graph, candidates = graph_and_candidates()
    feature = PageRankFeature()
    full_before = nx.pagerank(graph.networkx_graph, weight=None)
    candidate_before = nx.pagerank(candidates[0].networkx_graph, weight=None)
    report = FeatureExtractionBenchmark(
        "value-check",
        graph,
        candidates[:1],
        [feature],
    ).run()

    full_after = nx.pagerank(graph.networkx_graph, weight=None)
    candidate_after = nx.pagerank(candidates[0].networkx_graph, weight=None)

    assert full_after == pytest.approx(full_before)
    assert candidate_after == pytest.approx(candidate_before)
    reused = next(record for record in report.records if record["result_reused"])
    computed = next(
        record for record in report.records
        if record["computation_scope"] == "candidate_subgraph"
    )
    assert reused["result_status"] == "reused"
    assert computed["result_status"] == "newly_computed"


def test_benchmark_report_can_be_saved_and_tracked(tmp_path):
    graph, candidates = graph_and_candidates()
    benchmark = FeatureExtractionBenchmark(
        "artifact-test",
        graph,
        candidates[:1],
        [DegreeFeature()],
    )
    experiment = FakeExperiment(tmp_path / "EXP_TEST")

    report = benchmark.run_and_track(experiment)

    report_path = (
        experiment.artifact_store.experiment_root
        / "evaluation"
        / "feature_benchmark.json"
    )
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved == report.to_dict()
    assert experiment.stages == ["feature_extraction_benchmark"]
    assert experiment.registered == [
        ("evaluation", "feature_benchmark.json", "application/json")
    ]


def test_benchmark_rejects_non_node_features_and_bad_repeats():
    graph, candidates = graph_and_candidates()
    with pytest.raises(ValueError, match="node-level"):
        FeatureExtractionBenchmark(
            "invalid-feature",
            graph,
            candidates,
            [NumVerticesFeature()],
            repeats=1,
        )

    with pytest.raises(ValueError, match="positive"):
        FeatureExtractionBenchmark(
            "invalid-repeats",
            graph,
            candidates,
            [DegreeFeature()],
            repeats=0,
        )
