import sqlite3

import pytest

from src.experiments.database import ExperimentDatabase


@pytest.fixture
def database(tmp_path):
    db = ExperimentDatabase(tmp_path / "registry.sqlite3")
    yield db
    db.close()


def create_experiment(database, experiment_id="EXP_001", name="baseline"):
    return database.create_experiment(
        experiment_id=experiment_id,
        name=name,
        description="test experiment",
        configuration_path="config/config.json",
        environment_path="environment/environment.json",
    )


def test_database_initializes_schema(database):
    tables = {
        row["name"]
        for row in database._connection.execute(
            "SELECT name FROM sqlite_master WHERE type = ?",
            ("table",),
        )
    }

    assert {
        "experiments",
        "stages",
        "artifacts",
        "predictions",
        "evaluations",
    } <= tables
    assert database._connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_create_and_retrieve_experiment(database):
    created = create_experiment(database)

    retrieved = database.get_experiment("EXP_001")

    assert retrieved == created
    assert retrieved.status == "created"
    assert retrieved.configuration_path == "config/config.json"


def test_update_experiment_status(database):
    create_experiment(database)

    updated = database.update_experiment_status(
        "EXP_001",
        "running",
        started_at="2026-09-30T12:00:00Z",
    )

    assert updated.status == "running"
    assert updated.started_at == "2026-09-30T12:00:00Z"


def test_register_stage(database):
    create_experiment(database)

    stage = database.register_stage(
        experiment_id="EXP_001",
        stage_name="training",
        status="completed",
        started_at="2026-09-30T12:00:00Z",
        finished_at="2026-09-30T12:00:02Z",
        duration_seconds=2.0,
        memory_before_bytes=1000,
        memory_after_bytes=1200,
        memory_delta_bytes=200,
    )

    assert stage.stage_name == "training"
    assert stage.duration_seconds == 2.0
    assert database.get_experiment_stages("EXP_001") == [stage]


def test_register_artifact(database):
    create_experiment(database)

    artifact = database.register_artifact(
        artifact_id="ART_001",
        experiment_id="EXP_001",
        category="config",
        name="experiment.json",
        path="EXP_001/config/experiment.json",
        artifact_type="application/json",
        size_bytes=128,
    )

    assert artifact.artifact_id == "ART_001"
    assert artifact.size_bytes == 128
    assert database.get_experiment_artifacts("EXP_001") == [artifact]


def test_register_prediction(database):
    create_experiment(database)

    prediction = database.register_prediction(
        experiment_id="EXP_001",
        sample_id="sample-1",
        predicted_value=0.75,
        actual_value=0.8,
        prediction_error=-0.05,
    )

    assert prediction.predicted_value == 0.75
    assert database.get_experiment_predictions("EXP_001") == [prediction]


def test_register_evaluation(database):
    create_experiment(database)

    evaluation = database.register_evaluation(
        "EXP_001",
        "mae",
        0.125,
    )

    assert evaluation.metric_name == "mae"
    assert database.get_experiment_evaluations("EXP_001") == [evaluation]


def test_records_enforce_experiment_foreign_keys(database):
    with pytest.raises(sqlite3.IntegrityError):
        database.register_stage(
            "missing",
            "training",
            "completed",
        )

    with pytest.raises(sqlite3.IntegrityError):
        database.register_artifact(
            "ART_MISSING",
            "missing",
            "models",
            "model.bin",
            "missing/models/model.bin",
            "application/octet-stream",
        )

    with pytest.raises(sqlite3.IntegrityError):
        database.register_prediction(
            "missing",
            "sample-1",
            0.5,
        )

    with pytest.raises(sqlite3.IntegrityError):
        database.register_evaluation("missing", "mae", 0.1)


def test_list_experiments(database):
    create_experiment(database, "EXP_001", "first")
    create_experiment(database, "EXP_002", "second")

    experiments = database.list_experiments()

    assert {item.experiment_id for item in experiments} == {
        "EXP_001",
        "EXP_002",
    }


def test_queries_return_only_records_for_requested_experiment(database):
    create_experiment(database, "EXP_001")
    create_experiment(database, "EXP_002")

    database.register_stage("EXP_001", "training", "completed")
    database.register_stage("EXP_002", "evaluation", "completed")
    database.register_artifact(
        "ART_001", "EXP_001", "logs", "run.log", "EXP_001/logs/run.log", "text"
    )
    database.register_artifact(
        "ART_002", "EXP_002", "logs", "run.log", "EXP_002/logs/run.log", "text"
    )
    database.register_prediction("EXP_001", "sample-1", 0.5)
    database.register_prediction("EXP_002", "sample-2", 0.7)
    database.register_evaluation("EXP_001", "mae", 0.1)
    database.register_evaluation("EXP_002", "rmse", 0.2)

    assert len(database.get_experiment_stages("EXP_001")) == 1
    assert len(database.get_experiment_artifacts("EXP_001")) == 1
    assert database.get_experiment_predictions("EXP_001")[0].sample_id == "sample-1"
    assert database.get_experiment_evaluations("EXP_001")[0].metric_name == "mae"


def test_duplicate_experiment_id_is_rejected(database):
    create_experiment(database)

    with pytest.raises(sqlite3.IntegrityError):
        create_experiment(database)


def test_strings_with_quotes_are_inserted_as_values(database):
    quoted_name = "O'Brien's experiment"
    database.create_experiment(
        "EXP_QUOTES",
        name=quoted_name,
        description="value containing ' quotes",
    )

    assert database.get_experiment("EXP_QUOTES").name == quoted_name


def test_database_persists_after_close_and_reopen(tmp_path):
    database_path = tmp_path / "persistent.sqlite3"
    first_connection = ExperimentDatabase(database_path)
    create_experiment(first_connection, "EXP_PERSIST")
    first_connection.close()

    reopened = ExperimentDatabase(database_path)
    try:
        assert reopened.get_experiment("EXP_PERSIST").name == "baseline"
    finally:
        reopened.close()


def test_duplicate_artifact_id_is_rejected(database):
    create_experiment(database)
    args = (
        "ART_DUP",
        "EXP_001",
        "metadata",
        "record.json",
        "EXP_001/metadata/record.json",
        "application/json",
    )

    database.register_artifact(*args)

    with pytest.raises(sqlite3.IntegrityError):
        database.register_artifact(*args)


def test_values_containing_quotes_are_parameterized(database):
    create_experiment(database)
    sample_id = "sample' OR 1=1 --"

    record = database.register_prediction(
        "EXP_001",
        sample_id,
        0.4,
    )

    assert record.sample_id == sample_id
    assert database.get_experiment_predictions("EXP_001") == [record]