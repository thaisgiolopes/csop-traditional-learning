from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import sqlite3


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _require_optional_text(value: str | None, field_name: str) -> None:
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or None.")


def _require_finite(value: float | None, field_name: str) -> None:
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field_name} must be numeric or None.")
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite.")


def _require_non_negative_int(value: int | None, field_name: str) -> None:
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field_name} must be an integer or None.")
        if value < 0:
            raise ValueError(f"{field_name} cannot be negative.")


@dataclass(frozen=True)
class ExperimentRecord:
    """Metadata for one registered experiment."""

    experiment_id: str
    name: str | None
    description: str | None
    status: str
    created_at: str
    started_at: str | None
    finished_at: str | None
    configuration_path: str | None
    environment_path: str | None


@dataclass(frozen=True)
class StageRecord:
    """Timing and resource metadata for one experiment stage."""

    stage_id: int
    experiment_id: str
    stage_name: str
    status: str
    started_at: str | None
    finished_at: str | None
    duration_seconds: float | None
    memory_before_bytes: int | None
    memory_after_bytes: int | None
    memory_delta_bytes: int | None


@dataclass(frozen=True)
class ArtifactRecord:
    """File metadata and reference for one experiment artifact."""

    artifact_id: str
    experiment_id: str
    category: str
    name: str
    path: str
    artifact_type: str
    created_at: str
    size_bytes: int | None


@dataclass(frozen=True)
class PredictionRecord:
    """One prediction record with optional actual value and error."""

    prediction_id: int
    experiment_id: str
    sample_id: str
    predicted_value: float
    actual_value: float | None
    prediction_error: float | None


@dataclass(frozen=True)
class EvaluationRecord:
    """One named evaluation metric for an experiment."""

    evaluation_id: int
    experiment_id: str
    metric_name: str
    metric_value: float


class ExperimentDatabase:
    """SQLite metadata registry for experiment reproducibility."""

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path)
        if str(self._database_path).strip() == "":
            raise ValueError("database_path must not be empty.")

        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = self._connect()
        self.create_schema()

    def _connect(self) -> sqlite3.Connection:
        """Open a connection with row access and foreign keys enabled."""
        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _ensure_open(self) -> None:
        if self._connection is None:
            raise RuntimeError("Database connection is closed.")

    def create_schema(self) -> None:
        """Create registry tables and constraints if they do not exist."""
        self._ensure_open()
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY
                    CHECK (length(trim(experiment_id)) > 0),
                name TEXT,
                description TEXT,
                status TEXT NOT NULL
                    CHECK (length(trim(status)) > 0),
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                configuration_path TEXT,
                environment_path TEXT
            );

            CREATE TABLE IF NOT EXISTS stages (
                stage_id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                stage_name TEXT NOT NULL
                    CHECK (length(trim(stage_name)) > 0),
                status TEXT NOT NULL
                    CHECK (length(trim(status)) > 0),
                started_at TEXT,
                finished_at TEXT,
                duration_seconds REAL
                    CHECK (duration_seconds IS NULL OR duration_seconds >= 0),
                memory_before_bytes INTEGER
                    CHECK (memory_before_bytes IS NULL OR memory_before_bytes >= 0),
                memory_after_bytes INTEGER
                    CHECK (memory_after_bytes IS NULL OR memory_after_bytes >= 0),
                memory_delta_bytes INTEGER,
                FOREIGN KEY (experiment_id)
                    REFERENCES experiments(experiment_id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY
                    CHECK (length(trim(artifact_id)) > 0),
                experiment_id TEXT NOT NULL,
                category TEXT NOT NULL
                    CHECK (length(trim(category)) > 0),
                name TEXT NOT NULL
                    CHECK (length(trim(name)) > 0),
                path TEXT NOT NULL
                    CHECK (length(trim(path)) > 0),
                artifact_type TEXT NOT NULL
                    CHECK (length(trim(artifact_type)) > 0),
                created_at TEXT NOT NULL,
                size_bytes INTEGER
                    CHECK (size_bytes IS NULL OR size_bytes >= 0),
                UNIQUE (experiment_id, category, name),
                FOREIGN KEY (experiment_id)
                    REFERENCES experiments(experiment_id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                sample_id TEXT NOT NULL
                    CHECK (length(trim(sample_id)) > 0),
                predicted_value REAL NOT NULL,
                actual_value REAL,
                prediction_error REAL,
                FOREIGN KEY (experiment_id)
                    REFERENCES experiments(experiment_id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS evaluations (
                evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                metric_name TEXT NOT NULL
                    CHECK (length(trim(metric_name)) > 0),
                metric_value REAL NOT NULL,
                UNIQUE (experiment_id, metric_name),
                FOREIGN KEY (experiment_id)
                    REFERENCES experiments(experiment_id)
                    ON DELETE CASCADE
            );
            """
        )
        self._connection.commit()

    def close(self) -> None:
        """Close the SQLite connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> "ExperimentDatabase":
        self._ensure_open()
        return self

    def __exit__(self, exception_type, exception, traceback) -> bool:
        self.close()
        return False

    @staticmethod
    def _record(record_type, row):
        return record_type(**dict(row))

    def create_experiment(
        self,
        experiment_id: str,
        name: str | None = None,
        description: str | None = None,
        status: str = "created",
        created_at: str | None = None,
        configuration_path: str | None = None,
        environment_path: str | None = None,
    ) -> ExperimentRecord:
        """Insert an experiment and return its stored record."""
        _require_text(experiment_id, "experiment_id")
        _require_text(status, "status")
        _require_optional_text(name, "name")
        _require_optional_text(description, "description")
        _require_optional_text(configuration_path, "configuration_path")
        _require_optional_text(environment_path, "environment_path")

        timestamp = created_at or _utc_now()
        _require_text(timestamp, "created_at")

        self._ensure_open()
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO experiments (
                    experiment_id, name, description, status, created_at,
                    configuration_path, environment_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    name,
                    description,
                    status,
                    timestamp,
                    configuration_path,
                    environment_path,
                ),
            )

        return self.get_experiment(experiment_id)

    def update_experiment_status(
        self,
        experiment_id: str,
        status: str,
        started_at: str | None = None,
        finished_at: str | None = None,
    ) -> ExperimentRecord:
        """Update an experiment status and optional lifecycle timestamps."""
        _require_text(experiment_id, "experiment_id")
        _require_text(status, "status")
        _require_optional_text(started_at, "started_at")
        _require_optional_text(finished_at, "finished_at")

        self._ensure_open()
        with self._connection:
            cursor = self._connection.execute(
                """
                UPDATE experiments
                SET status = ?,
                    started_at = COALESCE(?, started_at),
                    finished_at = COALESCE(?, finished_at)
                WHERE experiment_id = ?
                """,
                (status, started_at, finished_at, experiment_id),
            )

        if cursor.rowcount == 0:
            raise KeyError(f"Unknown experiment_id: {experiment_id}")

        return self.get_experiment(experiment_id)

    def register_stage(
        self,
        experiment_id: str,
        stage_name: str,
        status: str,
        started_at: str | None = None,
        finished_at: str | None = None,
        duration_seconds: float | None = None,
        memory_before_bytes: int | None = None,
        memory_after_bytes: int | None = None,
        memory_delta_bytes: int | None = None,
    ) -> StageRecord:
        """Register stage metadata without measuring time or memory."""
        _require_text(experiment_id, "experiment_id")
        _require_text(stage_name, "stage_name")
        _require_text(status, "status")
        _require_optional_text(started_at, "started_at")
        _require_optional_text(finished_at, "finished_at")
        _require_finite(duration_seconds, "duration_seconds")
        _require_non_negative_int(memory_before_bytes, "memory_before_bytes")
        _require_non_negative_int(memory_after_bytes, "memory_after_bytes")
        _require_finite(memory_delta_bytes, "memory_delta_bytes")

        self._ensure_open()
        with self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO stages (
                    experiment_id, stage_name, status, started_at, finished_at,
                    duration_seconds, memory_before_bytes,
                    memory_after_bytes, memory_delta_bytes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    stage_name,
                    status,
                    started_at,
                    finished_at,
                    duration_seconds,
                    memory_before_bytes,
                    memory_after_bytes,
                    memory_delta_bytes,
                ),
            )
            row = self._connection.execute(
                "SELECT * FROM stages WHERE stage_id = ?",
                (cursor.lastrowid,),
            ).fetchone()

        return self._record(StageRecord, row)

    def register_artifact(
        self,
        artifact_id: str,
        experiment_id: str,
        category: str,
        name: str,
        path: str | Path,
        artifact_type: str,
        size_bytes: int | None = None,
        created_at: str | None = None,
    ) -> ArtifactRecord:
        """Register an artifact file reference and its metadata."""
        for value, field_name in (
            (artifact_id, "artifact_id"),
            (experiment_id, "experiment_id"),
            (category, "category"),
            (name, "name"),
            (str(path), "path"),
            (artifact_type, "artifact_type"),
        ):
            _require_text(value, field_name)

        _require_non_negative_int(size_bytes, "size_bytes")
        timestamp = created_at or _utc_now()
        _require_text(timestamp, "created_at")

        self._ensure_open()
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO artifacts (
                    artifact_id, experiment_id, category, name, path,
                    artifact_type, created_at, size_bytes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    experiment_id,
                    category,
                    name,
                    str(path),
                    artifact_type,
                    timestamp,
                    size_bytes,
                ),
            )
            row = self._connection.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()

        return self._record(ArtifactRecord, row)

    def register_prediction(
        self,
        experiment_id: str,
        sample_id: str,
        predicted_value: float,
        actual_value: float | None = None,
        prediction_error: float | None = None,
    ) -> PredictionRecord:
        """Register scalar prediction metadata, not prediction arrays."""
        _require_text(experiment_id, "experiment_id")
        _require_text(sample_id, "sample_id")
        _require_finite(predicted_value, "predicted_value")
        _require_finite(actual_value, "actual_value")
        _require_finite(prediction_error, "prediction_error")

        self._ensure_open()
        with self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO predictions (
                    experiment_id, sample_id, predicted_value,
                    actual_value, prediction_error
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    sample_id,
                    predicted_value,
                    actual_value,
                    prediction_error,
                ),
            )
            row = self._connection.execute(
                "SELECT * FROM predictions WHERE prediction_id = ?",
                (cursor.lastrowid,),
            ).fetchone()

        return self._record(PredictionRecord, row)

    def register_evaluation(
        self,
        experiment_id: str,
        metric_name: str,
        metric_value: float,
    ) -> EvaluationRecord:
        """Register one scalar evaluation metric."""
        _require_text(experiment_id, "experiment_id")
        _require_text(metric_name, "metric_name")
        _require_finite(metric_value, "metric_value")

        self._ensure_open()
        with self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO evaluations (
                    experiment_id, metric_name, metric_value
                ) VALUES (?, ?, ?)
                """,
                (experiment_id, metric_name, metric_value),
            )
            row = self._connection.execute(
                "SELECT * FROM evaluations WHERE evaluation_id = ?",
                (cursor.lastrowid,),
            ).fetchone()

        return self._record(EvaluationRecord, row)

    def get_experiment(self, experiment_id: str) -> ExperimentRecord:
        """Return one experiment or raise KeyError if it is absent."""
        self._ensure_open()
        row = self._connection.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()

        if row is None:
            raise KeyError(f"Unknown experiment_id: {experiment_id}")

        return self._record(ExperimentRecord, row)

    def list_experiments(self) -> list[ExperimentRecord]:
        """Return experiments in creation order."""
        self._ensure_open()
        rows = self._connection.execute(
            "SELECT * FROM experiments ORDER BY created_at, experiment_id"
        ).fetchall()
        return [self._record(ExperimentRecord, row) for row in rows]

    def get_experiment_stages(self, experiment_id: str) -> list[StageRecord]:
        """Return stages belonging to one experiment."""
        self._ensure_open()
        rows = self._connection.execute(
            """
            SELECT * FROM stages
            WHERE experiment_id = ?
            ORDER BY stage_id
            """,
            (experiment_id,),
        ).fetchall()
        return [self._record(StageRecord, row) for row in rows]

    def get_experiment_artifacts(
        self,
        experiment_id: str,
    ) -> list[ArtifactRecord]:
        """Return artifact references belonging to one experiment."""
        self._ensure_open()
        rows = self._connection.execute(
            """
            SELECT * FROM artifacts
            WHERE experiment_id = ?
            ORDER BY created_at, artifact_id
            """,
            (experiment_id,),
        ).fetchall()
        return [self._record(ArtifactRecord, row) for row in rows]

    def get_experiment_predictions(
        self,
        experiment_id: str,
    ) -> list[PredictionRecord]:
        """Return scalar predictions belonging to one experiment."""
        self._ensure_open()
        rows = self._connection.execute(
            """
            SELECT * FROM predictions
            WHERE experiment_id = ?
            ORDER BY prediction_id
            """,
            (experiment_id,),
        ).fetchall()
        return [self._record(PredictionRecord, row) for row in rows]

    def get_experiment_evaluations(
        self,
        experiment_id: str,
    ) -> list[EvaluationRecord]:
        """Return metrics belonging to one experiment."""
        self._ensure_open()
        rows = self._connection.execute(
            """
            SELECT * FROM evaluations
            WHERE experiment_id = ?
            ORDER BY evaluation_id
            """,
            (experiment_id,),
        ).fetchall()
        return [self._record(EvaluationRecord, row) for row in rows]