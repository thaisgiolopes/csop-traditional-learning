from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Iterator
from uuid import uuid4

from .artifact_store import ArtifactStore
from .database import (
    ArtifactRecord,
    EvaluationRecord,
    ExperimentDatabase,
    MetricRecord,
    PredictionRecord,
    StageRecord,
)
import logging
from .resource_monitor import MemoryMeasurement, ProcessMemoryMonitor
from .timer import Timer


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


@dataclass
class StageTrackingHandle:
    """Handle populated with its persisted record when a stage exits."""

    record: StageRecord | None = None

    @property
    def stage_id(self) -> int | None:
        """Return the database stage ID once the context has exited."""
        return self.record.stage_id if self.record is not None else None


class ExperimentTracker:
    """Record experiment measurements and results in the metadata database."""

    def __init__(
        self,
        experiment_id: str,
        database: ExperimentDatabase,
        artifact_store: ArtifactStore | None = None,
        logger=None,
        memory_monitor: ProcessMemoryMonitor | None = None,
        timer_factory=Timer,
    ) -> None:
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            raise ValueError("experiment_id must be a non-empty string.")

        self.experiment_id = experiment_id
        self._database = database
        self._artifact_store = artifact_store
        self._logger = logger
        self._memory_monitor = memory_monitor or ProcessMemoryMonitor()
        self._timer_factory = timer_factory

    @contextmanager
    def track_stage(self, stage_name: str) -> Iterator[StageTrackingHandle]:
        """Measure and register one successful or failed experiment stage."""
        if not isinstance(stage_name, str) or not stage_name.strip():
            raise ValueError("stage_name must be a non-empty string.")

        started_at = _utc_now()
        timer = self._timer_factory(stage_name)
        measurement: MemoryMeasurement | None = None
        status = "failed"
        stage_exception: BaseException | None = None
        stage_traceback: TracebackType | None = None
        handle = StageTrackingHandle()

        try:
            with timer:
                with self._memory_monitor.measure(stage_name) as measurement:
                    self._log("info", "Stage started: %s", stage_name)
                    yield handle
            status = "completed"
        except BaseException as exc:
            stage_exception = exc
            stage_traceback = exc.__traceback__
        finally:
            finished_at = _utc_now()

            try:
                duration_seconds = timer.elapsed_seconds
            except RuntimeError:
                duration_seconds = None

            stage_record = dict(
                experiment_id=self.experiment_id,
                stage_name=stage_name,
                status=status,
                started_at=started_at,
                finished_at=finished_at,
                duration_seconds=duration_seconds,
                memory_before_bytes=(
                    measurement.memory_before_bytes
                    if measurement is not None
                    else None
                ),
                memory_after_bytes=(
                    measurement.memory_after_bytes
                    if measurement is not None
                    else None
                ),
                memory_delta_bytes=(
                    measurement.memory_delta_bytes
                    if measurement is not None
                    else None
                ),
            )

            try:
                handle.record = self._database.register_stage(**stage_record)
            except BaseException as registration_error:
                if stage_exception is None:
                    raise
                stage_exception.add_note(
                    "The stage exception was preserved, but registering "
                    f"the failed stage also failed: {registration_error!r}"
                )

            if status == "completed":
                self._log("info", "Stage completed: %s", stage_name)
            else:
                self._log("error", "Stage failed: %s", stage_name)

        if stage_exception is not None:
            raise stage_exception.with_traceback(stage_traceback)

    def register_artifact(
        self,
        category: str,
        filename: str | Path,
        artifact_type: str,
        artifact_id: str | None = None,
    ) -> ArtifactRecord:
        """Register metadata for an artifact already stored by ArtifactStore."""
        if self._artifact_store is None:
            raise RuntimeError("ArtifactStore is required to register artifacts.")

        if not self._artifact_store.exists(category, filename):
            raise FileNotFoundError(
                f"Artifact does not exist in the store: "
                f"{category}/{filename}"
            )

        artifact_path = (
            self._artifact_store.experiment_root / category / Path(filename)
        ).resolve()
        identifier = artifact_id or f"ART_{uuid4().hex}"

        record = self._database.register_artifact(
            artifact_id=identifier,
            experiment_id=self.experiment_id,
            category=category,
            name=Path(filename).name,
            path=artifact_path,
            artifact_type=artifact_type,
            size_bytes=artifact_path.stat().st_size,
        )
        self._log("info", "Artifact registered: %s/%s", category, filename)
        return record

    def register_prediction(
        self,
        sample_id: str,
        predicted_value: float,
        actual_value: float | None = None,
        prediction_error: float | None = None,
    ) -> PredictionRecord:
        """Register scalar prediction values for one sample."""
        return self._database.register_prediction(
            experiment_id=self.experiment_id,
            sample_id=sample_id,
            predicted_value=predicted_value,
            actual_value=actual_value,
            prediction_error=prediction_error,
        )

    def register_evaluation(
        self,
        metric_name: str,
        metric_value: float,
    ) -> EvaluationRecord:
        """Register one scalar evaluation metric."""
        return self._database.register_evaluation(
            experiment_id=self.experiment_id,
            metric_name=metric_name,
            metric_value=metric_value,
        )

    def register_metric(
        self,
        metric_name: str,
        metric_value: float,
        *,
        stage_id: int | None = None,
        split: str | None = None,
    ) -> MetricRecord:
        """Register one metric with optional stage and data-split metadata."""
        return self._database.register_metric(
            experiment_id=self.experiment_id,
            stage_id=stage_id,
            metric_name=metric_name,
            split=split,
            metric_value=metric_value,
        )

    def _log(self, level: str, message: str, *args) -> None:
        if self._logger is None:
            return

        try:
            getattr(self._logger, level)(message, *args)
        except Exception:
            # Database registration is the primary responsibility.
            pass