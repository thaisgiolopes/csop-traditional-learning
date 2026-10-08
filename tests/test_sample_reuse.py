"""Tests for deterministic sample reuse across experiment runs."""

import pytest

from src.application.csop_experiment_runner import (
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


def test_graph_fingerprint_ignores_input_order():
    forward = Graph(vertices=[0, 1, 2], edges=[(0, 1), (1, 2)])
    reversed_order = Graph(vertices=[2, 1, 0], edges=[(2, 1), (1, 0)])

    assert graph_fingerprint(forward) == graph_fingerprint(reversed_order)