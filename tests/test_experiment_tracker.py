import logging

import pytest

import src.experiments.timer as timer_module
from src.experiments.artifact_store import ArtifactStore
from src.experiments.database import ExperimentDatabase
from src.experiments.logger import create_experiment_logger
from src.experiments.resource_monitor import ProcessMemoryMonitor
from src.experiments.timer import Timer
from src.experiments.tracker import ExperimentTracker


@pytest.fixture
def database(tmp_path):
    repository = ExperimentDatabase(tmp_path / "registry.sqlite3")
    repository.create_experiment("EXP_001", name="tracker-test")
    yield repository
    repository.close()


def make_tracker(database, tmp_path, *, readings=None, logger=None):
    values = iter(readings or [1000, 1200])
    monitor = ProcessMemoryMonitor(
        memory_reader=lambda: next(values),
        measurement_method="test_reader",
    )
    store = ArtifactStore(tmp_path / "EXP_001")

    return ExperimentTracker(
        "EXP_001",
        database,
        artifact_store=store,
        logger=logger,
        memory_monitor=monitor,
    )


def test_successful_stage_records_duration_and_memory(
    database,
    tmp_path,
    monkeypatch,
):
    clock_values = iter([10.0, 12.5])
    monkeypatch.setattr(
        timer_module.time,
        "perf_counter",
        lambda: next(clock_values),
    )
    tracker = make_tracker(database, tmp_path, readings=[2000, 2500])

    with tracker.track_stage("feature_extraction"):
        pass

    [stage] = database.get_experiment_stages("EXP_001")
    assert stage.status == "completed"
    assert stage.duration_seconds == 2.5
    assert stage.memory_before_bytes == 2000
    assert stage.memory_after_bytes == 2500
    assert stage.memory_delta_bytes == 500
    assert stage.started_at.endswith("Z")
    assert stage.finished_at.endswith("Z")


def test_failed_stage_is_registered_and_original_exception_propagates(
    database,
    tmp_path,
    monkeypatch,
):
    clock_values = iter([1.0, 1.5])
    monkeypatch.setattr(
        timer_module.time,
        "perf_counter",
        lambda: next(clock_values),
    )
    tracker = make_tracker(database, tmp_path, readings=[3000, 3200])

    with pytest.raises(ValueError, match="stage error"):
        with tracker.track_stage("training"):
            raise ValueError("stage error")

    [stage] = database.get_experiment_stages("EXP_001")
    assert stage.status == "failed"
    assert stage.duration_seconds == 0.5
    assert stage.memory_delta_bytes == 200


def test_multiple_stages_are_registered_once_each(database, tmp_path):
    tracker = make_tracker(
        database,
        tmp_path,
        readings=[100, 110, 120, 130],
    )

    with tracker.track_stage("graph_loading"):
        pass
    with tracker.track_stage("prediction"):
        pass

    stages = database.get_experiment_stages("EXP_001")
    assert [stage.stage_name for stage in stages] == [
        "graph_loading",
        "prediction",
    ]


def test_stage_logging_does_not_create_duplicate_stage_records(
    database,
    tmp_path,
):
    logger = create_experiment_logger(
        "EXP_001",
        tmp_path / "logs" / "experiment.log",
        level=logging.INFO,
    )
    tracker = make_tracker(database, tmp_path, logger=logger)

    with tracker.track_stage("training"):
        pass

    stages = database.get_experiment_stages("EXP_001")
    assert len(stages) == 1
    assert stages[0].stage_name == "training"


def test_artifact_registration_records_existing_artifact(
    database,
    tmp_path,
):
    store = ArtifactStore(tmp_path / "EXP_001")
    artifact_path = store.save_text("logs", "run.txt", "started")
    tracker = ExperimentTracker(
        "EXP_001",
        database,
        artifact_store=store,
        memory_monitor=ProcessMemoryMonitor(
            memory_reader=lambda: 1,
            measurement_method="test_reader",
        ),
    )

    record = tracker.register_artifact(
        "logs",
        "run.txt",
        "text/plain",
        artifact_id="ART_001",
    )

    assert record.path == str(artifact_path)
    assert record.size_bytes == len("started")
    assert database.get_experiment_artifacts("EXP_001") == [record]


def test_tracker_does_not_copy_artifacts(database, tmp_path):
    store = ArtifactStore(tmp_path / "EXP_001")
    original = store.save_text("datasets", "data.txt", "data")
    tracker = ExperimentTracker(
        "EXP_001",
        database,
        artifact_store=store,
    )

    tracker.register_artifact("datasets", "data.txt", "text/plain")

    assert original.read_text(encoding="utf-8") == "data"
    assert len(list(store.experiment_root.rglob("data.txt"))) == 1


def test_prediction_registration_supports_multiple_samples(
    database,
    tmp_path,
):
    tracker = make_tracker(database, tmp_path)

    first = tracker.register_prediction(
        "sample-1",
        predicted_value=0.4,
        actual_value=0.5,
        prediction_error=-0.1,
    )
    second = tracker.register_prediction(
        "sample-2",
        predicted_value=0.8,
    )

    predictions = database.get_experiment_predictions("EXP_001")
    assert predictions == [first, second]
    assert predictions[0].predicted_value == 0.4
    assert predictions[0].actual_value == 0.5
    assert predictions[0].prediction_error == -0.1
    assert predictions[1].actual_value is None


def test_evaluation_registration(database, tmp_path):
    tracker = make_tracker(database, tmp_path)

    record = tracker.register_evaluation("mae", 0.125)

    assert record.metric_name == "mae"
    assert record.metric_value == 0.125
    assert database.get_experiment_evaluations("EXP_001") == [record]


def test_tracker_reuses_timer_and_memory_monitor(
    database,
    tmp_path,
    monkeypatch,
):
    clock_values = iter([5.0, 6.0])
    monkeypatch.setattr(
        timer_module.time,
        "perf_counter",
        lambda: next(clock_values),
    )
    monitor = ProcessMemoryMonitor(
        memory_reader=iter([50, 75]).__next__,
        measurement_method="test_reader",
    )
    tracker = ExperimentTracker(
        "EXP_001",
        database,
        memory_monitor=monitor,
    )

    with tracker.track_stage("dataset_build"):
        pass

    [stage] = database.get_experiment_stages("EXP_001")
    assert stage.duration_seconds == 1.0
    assert stage.memory_delta_bytes == 25
    assert isinstance(Timer(), Timer)