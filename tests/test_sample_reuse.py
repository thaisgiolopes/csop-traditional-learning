"""Tests for deterministic sample reuse across experiment runs."""

from contextlib import contextmanager

import pytest

from src.application.csop_experiment_runner import (
    _build_generator,
    _resolve_candidate_samples,
    _sample_compatibility_key,
)
from src.experiments.config import SubgraphGenerationConfig
from src.experiments.database import ExperimentDatabase
from src.graph import Graph
from src.subgraphs.identity import (
    canonical_graph_structure,
    graph_fingerprint,
    graph_from_canonical_structure,
)
from src.subgraphs.networkx_generator import NetworkXSubgraphGenerator
from src.subgraphs.connected_expansion_generator import (
    ConnectedExpansionSubgraphGenerator,
)


@pytest.fixture
def database(tmp_path):
    registry = ExperimentDatabase(tmp_path / "experiments.sqlite3")
    yield registry
    registry.close()


def make_graph() -> Graph:
    return Graph(
        vertices=range(7),
        edges=[
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0),
            (0, 3), (1, 4), (2, 5), (3, 6),
        ],
    )


def create_experiment(database, experiment_id: str) -> None:
    database.create_experiment(
        experiment_id=experiment_id,
        name=experiment_id,
    )


def request_samples(database, experiment_id, graph, count, seed=42):
    create_experiment(database, experiment_id)
    config = SubgraphGenerationConfig(
        method="networkx_random_paths",
        sample_count=count,
        path_length=2,
        seed=seed,
    )
    generator = NetworkXSubgraphGenerator(
        num_subgraphs=count,
        path_length=config.path_length,
        seed=config.seed,
    )
    records, origins = _resolve_candidate_samples(
        database,
        experiment_id=experiment_id,
        graph=graph,
        graph_id="graph-shared",
        generation_config=config,
        generator=generator,
    )
    return records, origins


def test_first_experiment_generates_requested_samples(database):
    records, origins = request_samples(
        database,
        "EXP_001",
        make_graph(),
        5,
    )

    request = database.get_experiment_sample_request("EXP_001")
    assert len(records) == 5
    assert origins == ["generated"] * 5
    assert request.requested_count == 5
    assert request.reused_count == 0
    assert request.generated_count == 5
    assert request.total_available == 5


def test_compatible_samples_are_fully_reused(database):
    first, _ = request_samples(database, "EXP_001", make_graph(), 5)
    second, origins = request_samples(database, "EXP_002", make_graph(), 5)

    assert [record.sample_id for record in second] == [
        record.sample_id for record in first
    ]
    assert origins == ["reused"] * 5
    request = database.get_experiment_sample_request("EXP_002")
    assert request.reused_count == 5
    assert request.generated_count == 0


def test_partial_reuse_generates_only_missing_samples(database):
    first, _ = request_samples(database, "EXP_001", make_graph(), 4)
    second, origins = request_samples(database, "EXP_002", make_graph(), 7)

    assert len(second) == 7
    assert [record.sample_id for record in second[:4]] == [
        record.sample_id for record in first
    ]
    assert origins.count("reused") == 4
    assert origins.count("generated") == 3
    request = database.get_experiment_sample_request("EXP_002")
    assert request.requested_count == 7
    assert request.reused_count == 4
    assert request.generated_count == 3
    assert request.total_available == 7


def test_request_smaller_than_available_reuses_only_requested_prefix(database):
    available, _ = request_samples(database, "EXP_001", make_graph(), 6)
    selected, origins = request_samples(database, "EXP_002", make_graph(), 3)

    assert [record.sample_id for record in selected] == [
        record.sample_id for record in available[:3]
    ]
    assert origins == ["reused"] * 3
    request = database.get_experiment_sample_request("EXP_002")
    assert request.requested_count == 3
    assert request.reused_count == 3
    assert request.generated_count == 0
    assert request.total_available == 6


def test_sample_identity_is_content_based_and_experiment_independent(database):
    graph = make_graph()
    first, _ = request_samples(database, "EXP_001", graph, 3)
    second, _ = request_samples(database, "EXP_002", graph, 3)

    assert [record.sample_id for record in first] == [
        record.sample_id for record in second
    ]
    assert all(record.sample_id.startswith("SMP_") for record in first)


def test_duplicate_sample_and_experiment_associations_are_rejected(database):
    records, _ = request_samples(database, "EXP_001", make_graph(), 3)

    with pytest.raises(ValueError, match="duplicate samples"):
        database.associate_samples_with_experiment(
            experiment_id="EXP_001",
            compatibility_key=records[0].compatibility_key,
            requested_count=3,
            samples=[
                (records[0].sample_id, "generated"),
                (records[0].sample_id, "reused"),
                (records[2].sample_id, "reused"),
            ],
            total_available=3,
        )

    assert len(database.get_experiment_samples("EXP_001")) == 3


def test_duplicate_structure_does_not_duplicate_global_sample(database):
    graph = make_graph()
    records, _ = request_samples(database, "EXP_001", graph, 3)
    config = SubgraphGenerationConfig(
        method="networkx_random_paths",
        sample_count=3,
        path_length=2,
        seed=42,
    )
    compatibility_key, metadata = _sample_compatibility_key(
        graph_fingerprint(graph),
        config,
    )
    repeated_structure = graph_from_canonical_structure(
        {"vertices": records[0].vertices, "edges": records[0].edges}
    )

    inserted = database.register_generated_sample(
        sample_id=records[0].sample_id,
        graph_id="graph-shared",
        graph_fingerprint=graph_fingerprint(graph),
        structure=canonical_graph_structure(repeated_structure),
        compatibility_key=compatibility_key,
        sequence_index=999,
        generation_metadata={**metadata, "sequence_index": 999},
    )

    assert inserted is False
    assert len(database.get_compatible_samples(compatibility_key)) == 3


def test_generation_seed_and_graph_fingerprint_are_compatibility_inputs(database):
    graph = make_graph()
    _, _ = request_samples(database, "EXP_001", graph, 3, seed=42)
    with_different_seed, origins_seed = request_samples(
        database,
        "EXP_002",
        graph,
        3,
        seed=43,
    )
    changed_graph = Graph(
        vertices=range(7),
        edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
    )
    changed_records, origins_graph = request_samples(
        database,
        "EXP_003",
        changed_graph,
        3,
        seed=42,
    )

    assert origins_seed == ["generated"] * 3
    assert origins_graph == ["generated"] * 3
    assert len({item.compatibility_key for item in with_different_seed}) == 1
    assert len({item.compatibility_key for item in changed_records}) == 1
    assert graph_fingerprint(graph) != graph_fingerprint(changed_graph)


def test_incremental_generation_matches_single_request_sequence(tmp_path):
    graph = make_graph()
    incremental_db = ExperimentDatabase(tmp_path / "incremental.sqlite3")
    single_db = ExperimentDatabase(tmp_path / "single.sqlite3")
    try:
        first, _ = request_samples(incremental_db, "EXP_001", graph, 4)
        incremental, _ = request_samples(
            incremental_db,
            "EXP_002",
            graph,
            8,
        )
        single, _ = request_samples(single_db, "EXP_003", graph, 8)

        assert [item.sample_id for item in incremental] == [
            item.sample_id for item in single
        ]
        assert len(first) == 4
    finally:
        incremental_db.close()
        single_db.close()


def test_runner_builds_both_generators_from_configuration():
    random_paths = _build_generator(
        SubgraphGenerationConfig(
            method="networkx_random_paths",
            sample_count=3,
            path_length=2,
            seed=42,
        )
    )
    connected_expansion = _build_generator(
        SubgraphGenerationConfig(
            method="networkx_connected_expansion",
            sample_count=3,
            seed=42,
            parameters={"subgraph_size": 4, "max_attempts": 30},
        )
    )

    assert isinstance(random_paths, NetworkXSubgraphGenerator)
    assert isinstance(connected_expansion, ConnectedExpansionSubgraphGenerator)
    candidate = connected_expansion.generate_at_index(make_graph(), 0)
    assert candidate is not None
    assert candidate.num_vertices == 4


def test_connected_expansion_samples_are_persisted_and_reused(database):
    graph = make_graph()
    first_config = SubgraphGenerationConfig(
        method="networkx_connected_expansion",
        sample_count=3,
        seed=42,
        parameters={"subgraph_size": 4, "max_attempts": 30},
    )
    second_config = SubgraphGenerationConfig(
        method="networkx_connected_expansion",
        sample_count=3,
        seed=42,
        parameters={"subgraph_size": 4, "max_attempts": 1},
    )
    create_experiment(database, "EXP_CONNECTED_1")
    first, first_origins = _resolve_candidate_samples(
        database,
        experiment_id="EXP_CONNECTED_1",
        graph=graph,
        graph_id="graph-shared",
        generation_config=first_config,
        generator=_build_generator(first_config),
    )
    create_experiment(database, "EXP_CONNECTED_2")
    second, second_origins = _resolve_candidate_samples(
        database,
        experiment_id="EXP_CONNECTED_2",
        graph=graph,
        graph_id="graph-shared",
        generation_config=second_config,
        generator=_build_generator(second_config),
    )

    assert first_origins == ["generated"] * 3
    assert second_origins == ["reused"] * 3
    assert [record.sample_id for record in first] == [
        record.sample_id for record in second
    ]
    assert all(len(record.vertices) == 4 for record in first)


def test_failed_connected_expansion_attempts_are_not_persisted(database):
    graph = Graph(vertices=range(4), edges=[(0, 1), (2, 3)])
    config = SubgraphGenerationConfig(
        method="networkx_connected_expansion",
        sample_count=1,
        seed=42,
        parameters={"subgraph_size": 3, "max_attempts": 4},
    )
    create_experiment(database, "EXP_CONNECTED_FAIL")

    with pytest.raises(RuntimeError, match="after 4 deterministic attempts"):
        _resolve_candidate_samples(
            database,
            experiment_id="EXP_CONNECTED_FAIL",
            graph=graph,
            graph_id="graph-shared",
            generation_config=config,
            generator=_build_generator(config),
        )

    assert database.get_compatible_samples(
        _sample_compatibility_key(graph_fingerprint(graph), config)[0]
    ) == []


def test_duplicate_connected_candidates_are_not_persisted(database):
    graph = make_graph()
    candidate = Graph(vertices=[0, 1, 2, 3], edges=[(0, 1), (1, 2), (2, 3)])

    class RepeatingGenerator:
        def generate_at_index(self, graph, index):
            return candidate

    config = SubgraphGenerationConfig(
        method="networkx_connected_expansion",
        sample_count=2,
        seed=42,
        parameters={"subgraph_size": 4, "max_attempts": 5},
    )
    create_experiment(database, "EXP_CONNECTED_DUPLICATE")

    with pytest.raises(RuntimeError, match="Could generate only 1 unique"):
        _resolve_candidate_samples(
            database,
            experiment_id="EXP_CONNECTED_DUPLICATE",
            graph=graph,
            graph_id="graph-shared",
            generation_config=config,
            generator=RepeatingGenerator(),
        )

    compatibility_key, _ = _sample_compatibility_key(
        graph_fingerprint(graph),
        config,
    )
    records = database.get_compatible_samples(compatibility_key)
    assert len(records) == 1
    assert records[0].sequence_index == 0


def test_generation_method_and_target_size_are_compatibility_inputs():
    graph_digest = graph_fingerprint(make_graph())
    random_config = SubgraphGenerationConfig(
        method="networkx_random_paths",
        sample_count=3,
        path_length=2,
        seed=42,
    )
    connected_config = SubgraphGenerationConfig(
        method="networkx_connected_expansion",
        sample_count=3,
        seed=42,
        parameters={"subgraph_size": 4},
    )
    other_size_config = SubgraphGenerationConfig(
        method="networkx_connected_expansion",
        sample_count=3,
        seed=42,
        parameters={"subgraph_size": 5},
    )

    random_key = _sample_compatibility_key(graph_digest, random_config)[0]
    connected_key = _sample_compatibility_key(graph_digest, connected_config)[0]
    other_size_key = _sample_compatibility_key(graph_digest, other_size_config)[0]

    assert len({random_key, connected_key, other_size_key}) == 3


def test_connected_expansion_attempt_limit_does_not_change_compatibility():
    graph_digest = graph_fingerprint(make_graph())
    lower_limit = SubgraphGenerationConfig(
        method="networkx_connected_expansion",
        sample_count=3,
        seed=42,
        parameters={"subgraph_size": 4, "max_attempts": 10},
    )
    higher_limit = SubgraphGenerationConfig(
        method="networkx_connected_expansion",
        sample_count=3,
        seed=42,
        parameters={"max_attempts": 500, "subgraph_size": 4},
    )

    assert _sample_compatibility_key(graph_digest, lower_limit)[0] == (
        _sample_compatibility_key(graph_digest, higher_limit)[0]
    )


def test_generation_and_reuse_are_tracked_as_separate_stages(database):
    graph = make_graph()
    config = SubgraphGenerationConfig(
        method="networkx_connected_expansion",
        sample_count=2,
        seed=42,
        parameters={"subgraph_size": 3, "max_attempts": 100},
    )

    class RecordingExperiment:
        def __init__(self, experiment_id):
            self.experiment_id = experiment_id
            self.stage_names = []

        @contextmanager
        def track_stage(self, stage_name):
            self.stage_names.append(stage_name)
            yield

    first_experiment = RecordingExperiment("EXP_TRACK_1")
    create_experiment(database, first_experiment.experiment_id)
    first_records, first_origins = _resolve_candidate_samples(
        database,
        experiment_id=first_experiment.experiment_id,
        graph=graph,
        graph_id="graph-shared",
        generation_config=config,
        generator=_build_generator(config),
        experiment=first_experiment,
    )

    second_experiment = RecordingExperiment("EXP_TRACK_2")
    create_experiment(database, second_experiment.experiment_id)
    second_records, second_origins = _resolve_candidate_samples(
        database,
        experiment_id=second_experiment.experiment_id,
        graph=graph,
        graph_id="graph-shared",
        generation_config=config,
        generator=_build_generator(config),
        experiment=second_experiment,
    )

    expected_stages = [
        "candidate_retrieval",
        "candidate_generation",
        "candidate_selection_and_association",
    ]
    assert first_experiment.stage_names == expected_stages
    assert second_experiment.stage_names == expected_stages
    assert first_origins == ["generated", "generated"]
    assert second_origins == ["reused", "reused"]
    assert [record.sample_id for record in first_records] == [
        record.sample_id for record in second_records
    ]


def test_random_paths_compatibility_still_includes_all_parameters():
    graph_digest = graph_fingerprint(make_graph())
    lower_limit = SubgraphGenerationConfig(
        method="networkx_random_paths",
        sample_count=3,
        path_length=2,
        seed=42,
        parameters={"max_attempts": 10},
    )
    higher_limit = SubgraphGenerationConfig(
        method="networkx_random_paths",
        sample_count=3,
        path_length=2,
        seed=42,
        parameters={"max_attempts": 500},
    )

    assert _sample_compatibility_key(graph_digest, lower_limit)[0] != (
        _sample_compatibility_key(graph_digest, higher_limit)[0]
    )


def test_graph_fingerprint_ignores_input_order():
    forward = Graph(vertices=[0, 1, 2], edges=[(0, 1), (1, 2)])
    reversed_order = Graph(vertices=[2, 1, 0], edges=[(2, 1), (1, 0)])

    assert graph_fingerprint(forward) == graph_fingerprint(reversed_order)