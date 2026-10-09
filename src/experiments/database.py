from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import sqlite3
import json
from typing import Any, Sequence
from uuid import uuid4


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


@dataclass(frozen=True)
class MetricRecord:
    """One scalar metric associated with an experiment and optional stage."""

    metric_id: str
    experiment_id: str
    stage_id: int | None
    metric_name: str
    split: str | None
    metric_value: float
    recorded_at: str


@dataclass(frozen=True)
class SampleRecord:
    """One globally stored candidate-subgraph structure."""

    sample_id: str
    graph_id: str
    graph_fingerprint: str
    vertices: list[Any]
    edges: list[list[Any]]
    compatibility_key: str
    sequence_index: int
    generation_metadata: dict[str, Any]


@dataclass(frozen=True)
class ExperimentSampleRecord:
    """Association between an experiment and one shared sample."""

    experiment_id: str
    sample_id: str
    position: int
    origin: str


@dataclass(frozen=True)
class SampleRequestRecord:
    """Sample request and reuse/generation counts for an experiment."""

    experiment_id: str
    compatibility_key: str
    requested_count: int
    reused_count: int
    generated_count: int
    total_available: int


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

            CREATE TABLE IF NOT EXISTS metrics (
                metric_id TEXT PRIMARY KEY
                    CHECK (length(trim(metric_id)) > 0),
                experiment_id TEXT NOT NULL,
                stage_id INTEGER,
                metric_name TEXT NOT NULL
                    CHECK (length(trim(metric_name)) > 0),
                split TEXT,
                metric_value REAL NOT NULL,
                recorded_at TEXT NOT NULL,
                FOREIGN KEY (experiment_id)
                    REFERENCES experiments(experiment_id)
                    ON DELETE CASCADE,
                FOREIGN KEY (stage_id)
                    REFERENCES stages(stage_id)
                    ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS samples (
                sample_id TEXT PRIMARY KEY
                    CHECK (length(trim(sample_id)) > 0),
                graph_id TEXT NOT NULL
                    CHECK (length(trim(graph_id)) > 0),
                graph_fingerprint TEXT NOT NULL
                    CHECK (length(trim(graph_fingerprint)) > 0),
                vertices_json TEXT NOT NULL,
                edges_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sample_generation_streams (
                compatibility_key TEXT PRIMARY KEY
                    CHECK (length(trim(compatibility_key)) > 0),
                next_index INTEGER NOT NULL DEFAULT 0
                    CHECK (next_index >= 0)
            );

            CREATE TABLE IF NOT EXISTS sample_generations (
                compatibility_key TEXT NOT NULL,
                sequence_index INTEGER NOT NULL CHECK (sequence_index >= 0),
                sample_id TEXT NOT NULL,
                generation_metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (compatibility_key, sequence_index),
                UNIQUE (compatibility_key, sample_id),
                FOREIGN KEY (compatibility_key)
                    REFERENCES sample_generation_streams(compatibility_key),
                FOREIGN KEY (sample_id)
                    REFERENCES samples(sample_id)
            );

            CREATE TABLE IF NOT EXISTS experiment_samples (
                experiment_id TEXT NOT NULL,
                sample_id TEXT NOT NULL,
                position INTEGER NOT NULL CHECK (position >= 0),
                origin TEXT NOT NULL CHECK (origin IN ('reused', 'generated')),
                PRIMARY KEY (experiment_id, sample_id),
                UNIQUE (experiment_id, position),
                FOREIGN KEY (experiment_id)
                    REFERENCES experiments(experiment_id)
                    ON DELETE CASCADE,
                FOREIGN KEY (sample_id)
                    REFERENCES samples(sample_id)
            );

            CREATE TABLE IF NOT EXISTS experiment_sample_requests (
                experiment_id TEXT PRIMARY KEY,
                compatibility_key TEXT NOT NULL,
                requested_count INTEGER NOT NULL CHECK (requested_count > 0),
                reused_count INTEGER NOT NULL CHECK (reused_count >= 0),
                generated_count INTEGER NOT NULL CHECK (generated_count >= 0),
                total_available INTEGER NOT NULL CHECK (total_available >= 0),
                FOREIGN KEY (experiment_id)
                    REFERENCES experiments(experiment_id)
                    ON DELETE CASCADE
            );
            """
        )
        self._migrate_evaluations_to_metrics()
        self._connection.commit()

    def _migrate_evaluations_to_metrics(self) -> None:
        """Copy legacy evaluation rows to metrics, safely on every open."""
        columns = {
            row["name"]
            for row in self._connection.execute(
                "PRAGMA table_info(evaluations)"
            )
        }
        if not {"evaluation_id", "experiment_id", "metric_name", "metric_value"} <= columns:
            return

        rows = self._connection.execute(
            """SELECT evaluation_id, experiment_id, metric_name, metric_value
               FROM evaluations ORDER BY evaluation_id"""
        ).fetchall()
        timestamp = _utc_now()
        for row in rows:
            metric_name = row["metric_name"]
            split: str | None = None
            base_name = metric_name
            for known_split in ("train", "validation", "test"):
                prefix = f"{known_split}_"
                if metric_name.startswith(prefix):
                    split = known_split
                    base_name = metric_name[len(prefix):]
                    break
            self._connection.execute(
                """INSERT OR IGNORE INTO metrics
                   (metric_id, experiment_id, stage_id, metric_name, split,
                    metric_value, recorded_at)
                   VALUES (?, ?, NULL, ?, ?, ?, ?)""",
                (
                    f"LEGACY_{row['evaluation_id']}",
                    row["experiment_id"],
                    base_name,
                    split,
                    row["metric_value"],
                    timestamp,
                ),
            )

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
        """Register a legacy experiment-level evaluation metric.

        New callers should use ``register_metric`` to record split and stage
        associations. This adapter keeps existing consumers operational.
        """
        _require_text(experiment_id, "experiment_id")
        _require_text(metric_name, "metric_name")
        _require_finite(metric_value, "metric_value")

        split: str | None = None
        base_name = metric_name
        for known_split in ("train", "validation", "test"):
            prefix = f"{known_split}_"
            if metric_name.startswith(prefix):
                split = known_split
                base_name = metric_name[len(prefix):]
                break
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
            evaluation_id = int(cursor.lastrowid)
            self._connection.execute(
                """INSERT OR IGNORE INTO metrics
                   (metric_id, experiment_id, stage_id, metric_name, split,
                    metric_value, recorded_at)
                   VALUES (?, ?, NULL, ?, ?, ?, ?)""",
                (
                    f"LEGACY_{evaluation_id}",
                    experiment_id,
                    base_name,
                    split,
                    metric_value,
                    _utc_now(),
                ),
            )
            row = self._connection.execute(
                "SELECT * FROM evaluations WHERE evaluation_id = ?",
                (cursor.lastrowid,),
            ).fetchone()

        return self._record(EvaluationRecord, row)

    def register_metric(
        self,
        *,
        experiment_id: str,
        metric_name: str,
        metric_value: float,
        stage_id: int | None = None,
        split: str | None = None,
        recorded_at: str | None = None,
    ) -> MetricRecord:
        """Register a metric without imposing uniqueness by metric name."""
        _require_text(experiment_id, "experiment_id")
        _require_text(metric_name, "metric_name")
        _require_optional_text(split, "split")
        _require_finite(metric_value, "metric_value")
        if stage_id is not None:
            _require_non_negative_int(stage_id, "stage_id")
            self._ensure_open()
            stage = self._connection.execute(
                "SELECT experiment_id FROM stages WHERE stage_id = ?",
                (stage_id,),
            ).fetchone()
            if stage is None or stage["experiment_id"] != experiment_id:
                raise ValueError(
                    "stage_id must identify a stage belonging to experiment_id."
                )

        timestamp = recorded_at or _utc_now()
        _require_text(timestamp, "recorded_at")
        metric_id = f"MET_{uuid4().hex}"
        self._ensure_open()
        with self._connection:
            self._connection.execute(
                """INSERT INTO metrics
                   (metric_id, experiment_id, stage_id, metric_name, split,
                    metric_value, recorded_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    metric_id,
                    experiment_id,
                    stage_id,
                    metric_name,
                    split,
                    metric_value,
                    timestamp,
                ),
            )
            row = self._connection.execute(
                "SELECT * FROM metrics WHERE metric_id = ?",
                (metric_id,),
            ).fetchone()
        return self._record(MetricRecord, row)

    def get_experiment_metrics(
        self,
        experiment_id: str | None = None,
    ) -> list[MetricRecord]:
        """Return metrics ordered by creation ID, optionally filtered."""
        self._ensure_open()
        if experiment_id is None:
            rows = self._connection.execute(
                "SELECT * FROM metrics ORDER BY rowid"
            ).fetchall()
        else:
            rows = self._connection.execute(
                """SELECT * FROM metrics WHERE experiment_id = ?
                   ORDER BY rowid""",
                (experiment_id,),
            ).fetchall()
        return [self._record(MetricRecord, row) for row in rows]

    def list_stages(self) -> list[StageRecord]:
        """Return all stages in stable database order."""
        self._ensure_open()
        rows = self._connection.execute(
            "SELECT * FROM stages ORDER BY stage_id"
        ).fetchall()
        return [self._record(StageRecord, row) for row in rows]

    def reserve_sample_indices(
        self,
        compatibility_key: str,
        count: int,
    ) -> range:
        """Atomically reserve unique deterministic indices in one stream."""
        _require_text(compatibility_key, "compatibility_key")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive integer.")

        self._ensure_open()
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            self._connection.execute(
                """INSERT OR IGNORE INTO sample_generation_streams
                   (compatibility_key, next_index) VALUES (?, 0)""",
                (compatibility_key,),
            )
            row = self._connection.execute(
                """SELECT next_index FROM sample_generation_streams
                   WHERE compatibility_key = ?""",
                (compatibility_key,),
            ).fetchone()
            start = int(row["next_index"])
            self._connection.execute(
                """UPDATE sample_generation_streams SET next_index = ?
                   WHERE compatibility_key = ?""",
                (start + count, compatibility_key),
            )
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise

        return range(start, start + count)

    def register_generated_sample(
        self,
        *,
        sample_id: str,
        graph_id: str,
        graph_fingerprint: str,
        structure: dict[str, Any],
        compatibility_key: str,
        sequence_index: int,
        generation_metadata: dict[str, Any],
    ) -> bool:
        """Persist one structural sample and its generating sequence entry.

        Returns True only when this compatibility stream gained a new unique
        sample. The cache stores subgraph structure, not features or targets.
        """
        for value, name in (
            (sample_id, "sample_id"),
            (graph_id, "graph_id"),
            (graph_fingerprint, "graph_fingerprint"),
            (compatibility_key, "compatibility_key"),
        ):
            _require_text(value, name)
        if isinstance(sequence_index, bool) or not isinstance(
            sequence_index, int
        ) or sequence_index < 0:
            raise ValueError("sequence_index must be a non-negative integer.")
        if set(structure) != {"vertices", "edges"}:
            raise ValueError("structure must contain vertices and edges.")

        vertices_json = json.dumps(
            structure["vertices"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        edges_json = json.dumps(
            structure["edges"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        metadata_json = json.dumps(
            generation_metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

        self._ensure_open()
        with self._connection:
            self._connection.execute(
                """INSERT OR IGNORE INTO samples
                   (sample_id, graph_id, graph_fingerprint, vertices_json,
                    edges_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    sample_id,
                    graph_id,
                    graph_fingerprint,
                    vertices_json,
                    edges_json,
                    _utc_now(),
                ),
            )
            stored = self._connection.execute(
                """SELECT graph_fingerprint, vertices_json, edges_json
                   FROM samples WHERE sample_id = ?""",
                (sample_id,),
            ).fetchone()
            if (
                stored["graph_fingerprint"] != graph_fingerprint
                or stored["vertices_json"] != vertices_json
                or stored["edges_json"] != edges_json
            ):
                raise ValueError("Sample identity collision detected.")

            cursor = self._connection.execute(
                """INSERT OR IGNORE INTO sample_generations
                   (compatibility_key, sequence_index, sample_id,
                    generation_metadata_json, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    compatibility_key,
                    sequence_index,
                    sample_id,
                    metadata_json,
                    _utc_now(),
                ),
            )

        return cursor.rowcount == 1

    def get_compatible_samples(
        self,
        compatibility_key: str,
        limit: int | None = None,
    ) -> list[SampleRecord]:
        """Return unique samples in deterministic generation order."""
        _require_text(compatibility_key, "compatibility_key")
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
        ):
            raise ValueError("limit must be a non-negative integer or None.")

        self._ensure_open()
        query = """SELECT s.sample_id, s.graph_id, s.graph_fingerprint,
                          s.vertices_json, s.edges_json,
                          g.compatibility_key, g.sequence_index,
                          g.generation_metadata_json
                   FROM sample_generations AS g
                   JOIN samples AS s ON s.sample_id = g.sample_id
                   WHERE g.compatibility_key = ?
                   ORDER BY g.sequence_index, s.sample_id"""
        parameters: tuple[Any, ...] = (compatibility_key,)
        if limit is not None:
            query += " LIMIT ?"
            parameters += (limit,)
        rows = self._connection.execute(query, parameters).fetchall()

        return [
            SampleRecord(
                sample_id=row["sample_id"],
                graph_id=row["graph_id"],
                graph_fingerprint=row["graph_fingerprint"],
                vertices=json.loads(row["vertices_json"]),
                edges=json.loads(row["edges_json"]),
                compatibility_key=row["compatibility_key"],
                sequence_index=row["sequence_index"],
                generation_metadata=json.loads(
                    row["generation_metadata_json"]
                ),
            )
            for row in rows
        ]

    def associate_samples_with_experiment(
        self,
        *,
        experiment_id: str,
        compatibility_key: str,
        requested_count: int,
        samples: Sequence[tuple[str, str]],
        total_available: int,
    ) -> SampleRequestRecord:
        """Link selected shared samples and persist request accounting."""
        _require_text(experiment_id, "experiment_id")
        _require_text(compatibility_key, "compatibility_key")
        if requested_count <= 0 or len(samples) != requested_count:
            raise ValueError(
                "The association count must equal the positive requested count."
            )
        if total_available < requested_count:
            raise ValueError("total_available is below requested_count.")
        if len({sample_id for sample_id, _ in samples}) != len(samples):
            raise ValueError("An experiment cannot associate duplicate samples.")
        if any(origin not in {"reused", "generated"} for _, origin in samples):
            raise ValueError("Sample origin must be reused or generated.")

        reused_count = sum(origin == "reused" for _, origin in samples)
        generated_count = requested_count - reused_count
        self._ensure_open()
        with self._connection:
            self._connection.executemany(
                """INSERT INTO experiment_samples
                   (experiment_id, sample_id, position, origin)
                   VALUES (?, ?, ?, ?)""",
                [
                    (experiment_id, sample_id, position, origin)
                    for position, (sample_id, origin) in enumerate(samples)
                ],
            )
            self._connection.execute(
                """INSERT INTO experiment_sample_requests
                   (experiment_id, compatibility_key, requested_count,
                    reused_count, generated_count, total_available)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    experiment_id,
                    compatibility_key,
                    requested_count,
                    reused_count,
                    generated_count,
                    total_available,
                ),
            )

        return SampleRequestRecord(
            experiment_id=experiment_id,
            compatibility_key=compatibility_key,
            requested_count=requested_count,
            reused_count=reused_count,
            generated_count=generated_count,
            total_available=total_available,
        )

    def get_experiment_samples(
        self,
        experiment_id: str,
    ) -> list[ExperimentSampleRecord]:
        """Return samples associated with an experiment in requested order."""
        self._ensure_open()
        rows = self._connection.execute(
            """SELECT experiment_id, sample_id, position, origin
               FROM experiment_samples WHERE experiment_id = ?
               ORDER BY position""",
            (experiment_id,),
        ).fetchall()
        return [self._record(ExperimentSampleRecord, row) for row in rows]

    def get_experiment_sample_request(
        self,
        experiment_id: str,
    ) -> SampleRequestRecord:
        """Return requested/reused/generated sample counts for an experiment."""
        self._ensure_open()
        row = self._connection.execute(
            """SELECT experiment_id, compatibility_key, requested_count,
                      reused_count, generated_count, total_available
               FROM experiment_sample_requests WHERE experiment_id = ?""",
            (experiment_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"No sample request for experiment {experiment_id}.")
        return self._record(SampleRequestRecord, row)

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