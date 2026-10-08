from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any


def _to_json_compatible(value: Any) -> Any:
    """Convert supported values into JSON-compatible values."""
    if isinstance(value, Enum):
        return _to_json_compatible(value.value)

    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value) and not isinstance(value, type):
        return _to_json_compatible(asdict(value))

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Configuration values cannot contain NaN or infinity.")
        return value

    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]

    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Configuration dictionary keys must be strings.")

        return {
            key: _to_json_compatible(value[key])
            for key in sorted(value)
        }

    raise TypeError(
        f"Value of type {type(value).__name__} is not JSON-compatible."
    )


def _validate_parameters(value: dict[str, Any], field_name: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a dictionary.")

    _to_json_compatible(value)


def _validate_non_empty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


@dataclass
class GraphConfig:
    """Identifies the graph input used by an experiment."""

    graph_id: str | None = None
    path: str | Path | None = None
    source: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not any(
            value is not None and value != ""
            for value in (self.graph_id, self.path, self.source)
        ):
            raise ValueError(
                "At least one of graph_id, path, or source must be provided."
            )

        if self.graph_id is not None:
            _validate_non_empty_string(self.graph_id, "graph_id")

        if self.path is not None:
            if isinstance(self.path, Path):
                self.path = str(self.path)
            _validate_non_empty_string(self.path, "path")

        if self.source is not None:
            _validate_non_empty_string(self.source, "source")

        _validate_parameters(self.parameters, "graph parameters")


@dataclass
class SubgraphGenerationConfig:
    """Describes the method and parameters used to generate candidates."""

    method: str
    sample_count: int
    path_length: int | None = None
    seed: int | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.method, "method")

        if isinstance(self.sample_count, bool) or not isinstance(
            self.sample_count, int
        ):
            raise TypeError("sample_count must be an integer.")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive.")

        if self.path_length is not None:
            if isinstance(self.path_length, bool) or not isinstance(
                self.path_length, int
            ):
                raise TypeError("path_length must be an integer or None.")
            if self.path_length < 0:
                raise ValueError("path_length cannot be negative.")

        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("seed must be an integer or None.")

        _validate_parameters(self.parameters, "subgraph generation parameters")


@dataclass
class FeatureConfig:
    """Records the exact feature names and objective used in an experiment."""

    feature_set_id: str
    feature_names: list[str]
    objective_name: str

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.feature_set_id, "feature_set_id")
        _validate_non_empty_string(self.objective_name, "objective_name")

        if not isinstance(self.feature_names, list):
            raise TypeError("feature_names must be a list.")

        if not all(
            isinstance(name, str) and name.strip()
            for name in self.feature_names
        ):
            raise ValueError(
                "feature_names must contain only non-empty strings."
            )

        if len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError("feature_names must not contain duplicates.")


@dataclass
class DatasetSplitConfig:
    """Configures the final test and development train/validation splits.

    validation_size is a fraction of the development data, after removing
    the final test set.
    """

    strategy: str
    test_ratio: float
    validation_size: float
    random_state: int | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.strategy, "strategy")

        for field_name in ("test_ratio", "validation_size"):
            ratio = getattr(self, field_name)
            if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
                raise TypeError(f"{field_name} must be numeric.")
            if not math.isfinite(ratio) or not 0 < ratio < 1:
                raise ValueError(
                    f"{field_name} must be greater than 0 and less than 1."
                )

        if self.random_state is not None and (
            isinstance(self.random_state, bool)
            or not isinstance(self.random_state, int)
        ):
            raise TypeError("random_state must be an integer or None.")

        _validate_parameters(self.parameters, "dataset split parameters")


@dataclass
class ModelConfig:
    """Identifies the model implementation and its hyperparameters."""

    model_type: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.model_type, "model_type")
        _validate_parameters(self.hyperparameters, "hyperparameters")


@dataclass
class ExperimentConfig:
    """Stores the complete, reproducible configuration for one experiment."""

    graph: GraphConfig
    subgraph_generation: SubgraphGenerationConfig
    features: FeatureConfig
    dataset_split: DatasetSplitConfig
    model: ModelConfig
    name: str | None = None
    description: str | None = None
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.name is not None:
            _validate_non_empty_string(self.name, "name")

        if self.description is not None and not isinstance(
            self.description, str
        ):
            raise TypeError("description must be a string or None.")

        if isinstance(self.random_seed, bool) or not isinstance(
            self.random_seed, int
        ):
            raise TypeError("random_seed must be an integer.")

        expected_types = {
            "graph": GraphConfig,
            "subgraph_generation": SubgraphGenerationConfig,
            "features": FeatureConfig,
            "dataset_split": DatasetSplitConfig,
            "model": ModelConfig,
        }

        for field_name, expected_type in expected_types.items():
            if not isinstance(getattr(self, field_name), expected_type):
                raise TypeError(
                    f"{field_name} must be an instance of "
                    f"{expected_type.__name__}."
                )

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as deterministic JSON-compatible data."""
        return _to_json_compatible(asdict(self))

    def save_json(self, path: str | Path) -> None:
        """Save the configuration as readable UTF-8 JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as json_file:
            json.dump(
                self.to_dict(),
                json_file,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            json_file.write("\n")

    @classmethod
    def load_json(cls, path: str | Path) -> "ExperimentConfig":
        """Load and validate an experiment configuration from JSON."""
        input_path = Path(path)

        with input_path.open("r", encoding="utf-8") as json_file:
            data = json.load(json_file)

        if not isinstance(data, dict):
            raise ValueError("Experiment configuration JSON must contain an object.")

        try:
            data["graph"] = GraphConfig(**data["graph"])
            data["subgraph_generation"] = SubgraphGenerationConfig(
                **data["subgraph_generation"]
            )
            data["features"] = FeatureConfig(**data["features"])
            data["dataset_split"] = DatasetSplitConfig(**data["dataset_split"])
            data["model"] = ModelConfig(**data["model"])
            return cls(**data)
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Invalid experiment configuration in {input_path}."
            ) from exc