import json
import logging

import pytest

from src.experiments.artifact_store import ArtifactStore
from src.experiments.config import (
    DatasetSplitConfig,
    ExperimentConfig,
    FeatureConfig,
    GraphConfig,
    ModelConfig,
    SubgraphGenerationConfig,
)
from src.experiments.database import ExperimentDatabase
from src.experiments.environment import EnvironmentMetadata
from src.experiments.manager import ExperimentManager
from src.experiments.resource_monitor import ProcessMemoryMonitor
from src.experiments.timer import Timer


def make_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="manager-test",
        description="Experiment manager test",
        random_seed=42,
        graph=GraphConfig(graph_id="graph-test"),
        subgraph_generation=SubgraphGenerationConfig(
            method="random-paths",
            sample_count=10,
            path_length=3,
        ),
        features=FeatureConfig(
            feature_set_id="structural-v1",
            feature_names=["degree_mean"],
            objective_name="tds",
        ),
        dataset_split=DatasetSplitConfig(
            strategy="random",
            train_ratio=0.8,
            test_ratio=0.2,
        ),
        model=ModelConfig(model_type="test-model"),
    )


def make_environment() -> EnvironmentMetadata:
    return EnvironmentMetadata(
        python_version="3.test",
        operating_system="TestOS",
        os_release="test-release",
        machine_architecture="test-arch",
        processor=None,
        hostname="test-host",
        repository_git_commit=None,
        repository_git_branch=None,
        repository_dirty=None,
        dependency_versions={},
    )


@pytest.fixture
def database(tmp_path):
    repository = ExperimentDatabase(tmp_path / "registry.sqlite3")
    yield repository
    repository.close()


def create_manager(database, tmp_path, **kwargs):
    return ExperimentManager.create(
        config=make_config(),
        experiments_root=tmp_path / "experiments",
        database=database,
        environment=make_environment(),
        **kwargs,
    )


def test_create_experiment_and_register_initial_metadata(database, tmp_path):
    manager = create_manager(database, tmp_path)

    record = database.get_experiment(manager.experiment_id)

    assert manager.status == ExperimentManager.CREATED
    assert record.status == ExperimentManager.CREATED
    assert manager.experiment_root.is_dir()
    assert (manager.experiment_root / "config" / "experiment.json").is_file()
    assert (manager.experiment_root / "environment" / "environment.json").is_file()
    assert (manager.experiment_root / "logs" / "experiment.log").is_file()
    assert record.configuration_path.endswith("config/experiment.json")
    assert record.environment_path.endswith("environment/environment.json")

    config_data = json.loads(
        (manager.experiment_root / "config" / "experiment.json").read_text(
            encoding="utf-8"
        )
    )
    environment_data = json.loads(
        (manager.experiment_root / "environment" / "environment.json").read_text(
            encoding="utf-8"
        )
    )
    assert config_data["name"] == "manager-test"
    assert environment_data["hostname"] == "test-host"

    artifacts = database.get_experiment_artifacts(manager.experiment_id)
    assert {artifact.category for artifact in artifacts} == {
        "config",
        "environment",
    }


def test_context_manager_marks_successful_experiment_completed(
    database,
    tmp_path,
):
    manager = create_manager(database, tmp_path)

    with manager as experiment:
        assert experiment.status == ExperimentManager.RUNNING

    record = database.get_experiment(manager.experiment_id)
    assert manager.status == ExperimentManager.COMPLETED
    assert record.status == ExperimentManager.COMPLETED
    assert record.started_at is not None
    assert record.finished_at is not None


def test_context_manager_marks_failed_experiment_and_reraises(
    database,
    tmp_path,
):
    manager = create_manager(database, tmp_path)

    with pytest.raises(ValueError, match="original failure"):
        with manager:
            raise ValueError("original failure")

    record = database.get_experiment(manager.experiment_id)
    assert manager.status == ExperimentManager.FAILED
    assert record.status == ExperimentManager.FAILED
    assert record.finished_at is not None


def test_tracked_stage_is_registered_through_manager(database, tmp_path):
    readings = iter([1000, 1250])
    monitor = ProcessMemoryMonitor(
        memory_reader=lambda: next(readings),
        measurement_method="test_reader",
    )
    manager = create_manager(database, tmp_path, memory_monitor=monitor)

    with manager as experiment:
        with experiment.track_stage("candidate_generation"):
            pass

    [stage] = database.get_experiment_stages(manager.experiment_id)
    assert stage.stage_name == "candidate_generation"
    assert stage.status == "completed"
    assert stage.duration_seconds is not None
    assert stage.memory_before_bytes == 1000
    assert stage.memory_after_bytes == 1250
    assert stage.memory_delta_bytes == 250


def test_artifact_registration_is_exposed_by_manager(database, tmp_path):
    manager = create_manager(database, tmp_path)

    with manager:
        manager.artifact_store.save_text("logs", "stage.txt", "complete")
        artifact = manager.register_artifact(
            "logs",
            "stage.txt",
            "text/plain",
        )

    assert artifact.experiment_id == manager.experiment_id
    assert artifact.name == "stage.txt"


def test_prediction_and_evaluation_registration_are_exposed(
    database,
    tmp_path,
):
    manager = create_manager(database, tmp_path)

    with manager:
        prediction = manager.register_prediction(
            "sample-1",
            predicted_value=0.7,
            actual_value=0.8,
            prediction_error=-0.1,
        )
        evaluation = manager.register_evaluation("mae", 0.1)

    assert prediction.predicted_value == 0.7
    assert evaluation.metric_value == 0.1
    assert database.get_experiment_predictions(manager.experiment_id) == [
        prediction
    ]
    assert database.get_experiment_evaluations(manager.experiment_id) == [
        evaluation
    ]


def test_repeated_creation_generates_distinct_experiment_ids(
    database,
    tmp_path,
):
    first = create_manager(database, tmp_path)
    second = create_manager(database, tmp_path)

    assert first.experiment_id != second.experiment_id
    assert first.experiment_root != second.experiment_root
    assert len(database.list_experiments()) == 2


def test_managers_have_independent_loggers_and_directories(
    database,
    tmp_path,
):
    first = create_manager(database, tmp_path)
    second = create_manager(database, tmp_path)

    assert first.logger is not second.logger
    assert first.logger.name != second.logger.name
    assert first.artifact_store is not second.artifact_store
    assert first.experiment_root != second.experiment_root


def test_tracker_reuses_injected_timer_factory(database, tmp_path):
    created_timers = []

    def timer_factory(stage_name):
        timer = Timer(stage_name)
        created_timers.append(timer)
        return timer

    manager = create_manager(
        database,
        tmp_path,
        timer_factory=timer_factory,
        memory_monitor=ProcessMemoryMonitor(
            memory_reader=lambda: 1,
            measurement_method="test_reader",
        ),
    )

    with manager:
        with manager.track_stage("test-stage"):
            pass

    assert len(created_timers) == 1
    assert created_timers[0].running is False