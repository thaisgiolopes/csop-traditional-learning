import csv
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterable

from .database import (
    ExperimentDatabase,
    ExperimentRecord,
)


SUMMARY_COLUMNS = (
    "experiment_id",
    "name",
    "description",
    "status",
    "created_at",
    "started_at",
    "finished_at",
    "total_duration_seconds",
    "graph_reference_json",
    "subgraph_generation_method",
    "sample_count",
    "path_length",
    "feature_set_id",
    "feature_names_json",
    "number_of_features",
    "dataset_split_strategy",
    "train_ratio",
    "test_ratio",
    "model_type",
    "model_hyperparameters_json",
    "recorded_stage_count",
    "prediction_count",
    "evaluation_metrics_json",
)


def _json_cell(value: Any) -> str:
    """Serialize complex values deterministically for a CSV cell."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _total_duration_seconds(
    started_at: str | None,
    finished_at: str | None,
) -> float | None:
    """Calculate duration from recorded ISO-8601 timestamps."""
    if not started_at or not finished_at:
        return None

    try:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        finish = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
    except ValueError:
        return None

    return (finish - start).total_seconds()


class ExperimentReport:
    """Generate tabular reports from records in the experiment database."""

    def __init__(self, database: ExperimentDatabase) -> None:
        self._database = database

    def write_csv(
        self,
        path: str | Path,
        experiment_ids: str | Iterable[str] | None = None,
    ) -> Path:
        """Write one summary row per selected experiment to UTF-8 CSV."""
        rows = self._summary_rows(experiment_ids)
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=SUMMARY_COLUMNS,
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(rows)

        return output_path

    def get_experiment_details(self, experiment_id: str) -> dict[str, Any]:
        """Return a summary and database records for one experiment."""
        experiment = self._database.get_experiment(experiment_id)
        summary = self._summary_row(experiment)

        return {
            "summary": summary,
            "stages": [
                asdict(record)
                for record in self._database.get_experiment_stages(
                    experiment_id
                )
            ],
            "artifacts": [
                asdict(record)
                for record in self._database.get_experiment_artifacts(
                    experiment_id
                )
            ],
            "predictions": [
                asdict(record)
                for record in self._database.get_experiment_predictions(
                    experiment_id
                )
            ],
            "evaluations": [
                asdict(record)
                for record in self._database.get_experiment_evaluations(
                    experiment_id
                )
            ],
        }

    def _summary_rows(
        self,
        experiment_ids: str | Iterable[str] | None,
    ) -> list[dict[str, Any]]:
        if experiment_ids is None:
            experiments = self._database.list_experiments()
        else:
            requested_ids = (
                [experiment_ids]
                if isinstance(experiment_ids, str)
                else list(dict.fromkeys(experiment_ids))
            )
            experiments = [
                self._database.get_experiment(experiment_id)
                for experiment_id in requested_ids
            ]

        return [self._summary_row(item) for item in experiments]

    def _summary_row(
        self,
        experiment: ExperimentRecord,
    ) -> dict[str, Any]:
        config = self._load_configuration(experiment.configuration_path)
        graph = config.get("graph") or {}
        generation = config.get("subgraph_generation") or {}
        features = config.get("features") or {}
        split = config.get("dataset_split") or {}
        model = config.get("model") or {}

        stages = self._database.get_experiment_stages(
            experiment.experiment_id
        )
        predictions = self._database.get_experiment_predictions(
            experiment.experiment_id
        )
        evaluations = self._database.get_experiment_evaluations(
            experiment.experiment_id
        )

        graph_reference = {
            key: graph[key]
            for key in ("graph_id", "path", "source")
            if graph.get(key) is not None
        }
        feature_names = features.get("feature_names") or []
        evaluation_metrics = {
            record.metric_name: record.metric_value
            for record in sorted(
                evaluations,
                key=lambda item: item.metric_name,
            )
        }

        return {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name or "",
            "description": experiment.description or "",
            "status": experiment.status,
            "created_at": experiment.created_at,
            "started_at": experiment.started_at or "",
            "finished_at": experiment.finished_at or "",
            "total_duration_seconds": _total_duration_seconds(
                experiment.started_at,
                experiment.finished_at,
            ),
            "graph_reference_json": _json_cell(graph_reference),
            "subgraph_generation_method": generation.get("method", ""),
            "sample_count": generation.get("sample_count", ""),
            "path_length": (
                generation.get("path_length")
                if generation.get("path_length") is not None
                else ""
            ),
            "feature_set_id": features.get("feature_set_id", ""),
            "feature_names_json": _json_cell(feature_names),
            "number_of_features": len(feature_names),
            "dataset_split_strategy": split.get("strategy", ""),
            "train_ratio": split.get("train_ratio", ""),
            "test_ratio": split.get("test_ratio", ""),
            "model_type": model.get("model_type", ""),
            "model_hyperparameters_json": _json_cell(
                model.get("hyperparameters") or {}
            ),
            "recorded_stage_count": len(stages),
            "prediction_count": len(predictions),
            "evaluation_metrics_json": _json_cell(evaluation_metrics),
        }

    @staticmethod
    def _load_configuration(path: str | None) -> dict[str, Any]:
        """Load configuration from the exact artifact path stored in SQLite."""
        if not path:
            return {}

        try:
            with Path(path).open("r", encoding="utf-8") as json_file:
                configuration = json.load(json_file)
        except (OSError, json.JSONDecodeError):
            return {}

        return configuration if isinstance(configuration, dict) else {}