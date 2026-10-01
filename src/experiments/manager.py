from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any
import secrets
from .artifact_store import ArtifactStore
from .config import ExperimentConfig
from .database import ExperimentDatabase
from .environment import EnvironmentMetadata
from .logger import create_experiment_logger
from .resource_monitor import ProcessMemoryMonitor
from .timer import Timer
from .tracker import ExperimentTracker
from .report import ExperimentReport


class ExperimentManager:
    """Orchestrate the lifecycle and components of one experiment."""

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

    def __init__(
        self,
        experiment_id: str,
        config: ExperimentConfig,
        experiment_root: Path,
        database: ExperimentDatabase,
        artifact_store: ArtifactStore,
        logger: logging.Logger,
        tracker: ExperimentTracker,
        report_path: str | Path | None = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.config = config
        self.experiment_root = experiment_root
        self.database = database
        self.artifact_store = artifact_store
        self.logger = logger
        self.tracker = tracker
        self._status = self.CREATED
        self._entered = False
        self._finished = False
        self.report_path = (
            Path(report_path)
            if report_path is not None
            else experiment_root.parent.parent / "reports" / "experiments.xlsx"
        )

    @classmethod
    def create(
        cls,
        config: ExperimentConfig,
        experiments_root: str | Path,
        database: ExperimentDatabase,
        *,
        experiment_id: str | None = None,
        environment: EnvironmentMetadata | None = None,
        log_level: int | str = logging.INFO,
        console_logging: bool = False,
        memory_monitor: ProcessMemoryMonitor | None = None,
        timer_factory=Timer,
        report_path: str | Path | None = None,
    ) -> "ExperimentManager":
        """Create the experiment record and its initial metadata artifacts."""
        if not isinstance(config, ExperimentConfig):
            raise TypeError("config must be an ExperimentConfig.")
        if not isinstance(database, ExperimentDatabase):
            raise TypeError("database must be an ExperimentDatabase.")

        if experiment_id is None:
            date_prefix = datetime.now(timezone.utc).strftime("%Y%m%d")
            resolved_id = f"EXP_{date_prefix}_{secrets.token_hex(6)}"
        else:
            resolved_id = experiment_id
        if not isinstance(resolved_id, str) or not resolved_id.strip():
            raise ValueError("experiment_id must be a non-empty string.")

        root = Path(experiments_root).resolve() / resolved_id
        store = ArtifactStore(root)

        config_path = store.save_json("config", "experiment.json", config.to_dict())

        environment_metadata = environment or EnvironmentMetadata.collect()
        environment_path = store.save_json(
            "environment",
            "environment.json",
            environment_metadata.to_dict(),
        )

        logger = create_experiment_logger(
            resolved_id,
            root / "logs" / "experiment.log",
            level=log_level,
            console=console_logging,
        )

        database.create_experiment(
            experiment_id=resolved_id,
            name=config.name,
            description=config.description,
            status=cls.CREATED,
            configuration_path=str(config_path),
            environment_path=str(environment_path),
        )

        tracker = ExperimentTracker(
            experiment_id=resolved_id,
            database=database,
            artifact_store=store,
            logger=logger,
            memory_monitor=memory_monitor,
            timer_factory=timer_factory,
        )

        manager = cls(
            experiment_id=resolved_id,
            config=config,
            experiment_root=root,
            database=database,
            artifact_store=store,
            logger=logger,
            tracker=tracker,
            report_path=report_path,
        )

        tracker.register_artifact(
            "config",
            "experiment.json",
            "application/json",
            artifact_id=f"{resolved_id}:config",
        )
        tracker.register_artifact(
            "environment",
            "environment.json",
            "application/json",
            artifact_id=f"{resolved_id}:environment",
        )

        logger.info("Experiment created")
        return manager

    @property
    def status(self) -> str:
        """Return the manager's current lifecycle status."""
        return self._status

    def __enter__(self) -> "ExperimentManager":
        if self._entered:
            raise RuntimeError("Experiment manager has already been entered.")
        if self._finished:
            raise RuntimeError("Experiment has already been finalized.")

        self.database.update_experiment_status(
            self.experiment_id,
            self.RUNNING,
            started_at=self._utc_now(),
        )
        self._status = self.RUNNING
        self._entered = True
        self.logger.info("Experiment started")
        return self

    def __exit__(self, exception_type, exception, traceback) -> bool:
        if not self._entered:
            raise RuntimeError("Experiment manager was not entered.")

        final_status = self.FAILED if exception is not None else self.COMPLETED
        finished_at = self._utc_now()

        try:
            self.database.update_experiment_status(
                self.experiment_id,
                final_status,
                finished_at=finished_at,
            )
        except BaseException as finalize_error:
            if exception is None:
                raise
            exception.add_note(
                "The original experiment exception was preserved, but "
                f"finalizing its database status failed: {finalize_error!r}"
            )
        else:
            self._status = final_status
            self._finished = True

        try:
            ExperimentReport(self.database).update_xlsx(
                self.report_path,
                self.experiment_id,
            )
        except Exception:
            self.logger.exception(
                "Could not update experiment XLSX report at %s",
                self.report_path,
            )

        if exception is None:
            self.logger.info("Experiment completed")
        else:
            self.logger.exception(
                "Experiment failed",
                exc_info=(exception_type, exception, traceback),
            )

        return False

    def track_stage(self, stage_name: str):
        """Return the tracker's context manager for one experiment stage."""
        if self._status != self.RUNNING:
            raise RuntimeError("Stages can only be tracked while experiment is running.")
        return self.tracker.track_stage(stage_name)

    def register_artifact(
        self,
        category: str,
        filename: str | Path,
        artifact_type: str,
        artifact_id: str | None = None,
    ):
        """Register metadata for a file already stored by ArtifactStore."""
        return self.tracker.register_artifact(
            category=category,
            filename=filename,
            artifact_type=artifact_type,
            artifact_id=artifact_id,
        )

    def register_prediction(
        self,
        sample_id: str,
        predicted_value: float,
        actual_value: float | None = None,
        prediction_error: float | None = None,
    ):
        """Register one scalar prediction through the tracker."""
        return self.tracker.register_prediction(
            sample_id=sample_id,
            predicted_value=predicted_value,
            actual_value=actual_value,
            prediction_error=prediction_error,
        )

    def register_evaluation(self, metric_name: str, metric_value: float):
        """Register one scalar evaluation metric through the tracker."""
        return self.tracker.register_evaluation(metric_name, metric_value)

    @staticmethod
    def _utc_now() -> str:
        return (
            datetime.now(timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )