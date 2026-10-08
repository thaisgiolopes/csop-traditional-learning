import json

from pathlib import Path

import pytest

from src.experiments.config import (
    DatasetSplitConfig,
    ExperimentConfig,
    FeatureConfig,
    GraphConfig,
    ModelConfig,
    SubgraphGenerationConfig,
)


def make_config(graph_path: Path | None = None) -> ExperimentConfig:
    """Create a valid experiment configuration for tests."""
    return ExperimentConfig(
        name="tds-baseline",
        description="Baseline TDS regression experiment",
        random_seed=42,
        graph=GraphConfig(
            graph_id="graph_001",
            path=graph_path,
            parameters={
                "directed": False,
                "format": "adjlist",
            },
        ),
        subgraph_generation=SubgraphGenerationConfig(
            method="networkx_random_paths",
            sample_count=20,
            path_length=3,
            seed=7,
            parameters={"allow_duplicates": False},
        ),
        features=FeatureConfig(
            feature_set_id="structural-v1",
            feature_names=["num_vertices", "degree_mean"],
            objective_name="tds",
        ),
        dataset_split=DatasetSplitConfig(
            strategy="random",
            test_ratio=0.2,
            validation_size=0.25,
            random_state=11,
            parameters={"shuffle": True},
        ),
        model=ModelConfig(
            model_type="lightgbm",
            hyperparameters={
                "n_estimators": 50,
                "learning_rate": 0.05,
            },
        ),
    )


def test_valid_experiment_configuration_creation():
    config = make_config()

    assert config.name == "tds-baseline"
    assert config.random_seed == 42
    assert config.graph.graph_id == "graph_001"
    assert config.subgraph_generation.sample_count == 20
    assert config.features.feature_set_id == "structural-v1"
    assert config.features.feature_names == [
        "num_vertices",
        "degree_mean",
    ]
    assert config.dataset_split.validation_size == 0.25
    assert config.dataset_split.random_state == 11
    assert config.model.model_type == "lightgbm"


def test_graph_requires_identifier_path_or_source():
    with pytest.raises(ValueError, match="graph_id, path, or source"):
        GraphConfig()


@pytest.mark.parametrize("sample_count", [0, -1])
def test_sample_count_must_be_positive(sample_count):
    with pytest.raises(ValueError, match="sample_count must be positive"):
        SubgraphGenerationConfig(
            method="random",
            sample_count=sample_count,
        )


@pytest.mark.parametrize("path_length", [-1, 1.5, True])
def test_path_length_must_be_non_negative_integer_or_none(path_length):
    with pytest.raises((TypeError, ValueError)):
        SubgraphGenerationConfig(
            method="random",
            sample_count=5,
            path_length=path_length,
        )


@pytest.mark.parametrize(
    "feature_names",
    [
        ["degree", ""],
        ["degree", "   "],
        ["degree", 3],
        "degree",
    ],
)
def test_feature_names_must_be_explicit_non_empty_strings(feature_names):
    with pytest.raises((TypeError, ValueError)):
        FeatureConfig(
            feature_set_id="structural-v1",
            feature_names=feature_names,
            objective_name="tds",
        )


def test_feature_names_must_be_unique():
    with pytest.raises(ValueError, match="duplicates"):
        FeatureConfig(
            feature_set_id="structural-v1",
            feature_names=["degree", "degree"],
            objective_name="tds",
        )


@pytest.mark.parametrize(
    ("test_ratio", "validation_size"),
    [
        (0, 0.2),
        (1, 0.2),
        (0.2, 0),
        (0.2, 1),
        (-0.1, 0.2),
        (float("nan"), 0.2),
        (True, 0.2),
    ],
)
def test_split_sizes_must_be_numeric_and_between_zero_and_one(
    test_ratio,
    validation_size,
):
    with pytest.raises((TypeError, ValueError)):
        DatasetSplitConfig(
            strategy="random",
            test_ratio=test_ratio,
            validation_size=validation_size,
        )


def test_split_random_state_must_be_an_integer_or_none():
    with pytest.raises(TypeError, match="random_state"):
        DatasetSplitConfig(
            strategy="random",
            test_ratio=0.2,
            validation_size=0.25,
            random_state=1.5,
        )


def test_to_dict_returns_json_compatible_nested_values(tmp_path):
    config = make_config(graph_path=tmp_path / "graph")

    config_dict = config.to_dict()

    assert config_dict["graph"]["path"] == str(tmp_path / "graph")
    assert config_dict["features"]["feature_set_id"] == "structural-v1"
    assert config_dict["features"]["feature_names"] == [
        "num_vertices",
        "degree_mean",
    ]
    assert config_dict["dataset_split"]["parameters"] == {"shuffle": True}
    assert config_dict["model"]["hyperparameters"]["n_estimators"] == 50

    # The returned structure can be encoded by the standard JSON library.
    json.dumps(config_dict, allow_nan=False)


def test_save_and_load_json_round_trip_preserves_nested_configuration(tmp_path):
    config = make_config(graph_path=tmp_path / "graph")
    config_path = tmp_path / "nested" / "experiment.json"

    config.save_json(config_path)
    loaded = ExperimentConfig.load_json(config_path)

    assert loaded.to_dict() == config.to_dict()
    assert loaded.graph.path == str(tmp_path / "graph")
    assert loaded.graph.parameters == {
        "directed": False,
        "format": "adjlist",
    }
    assert loaded.subgraph_generation.parameters == {
        "allow_duplicates": False,
    }
    assert loaded.features.feature_names == [
        "num_vertices",
        "degree_mean",
    ]
    assert loaded.dataset_split.parameters == {"shuffle": True}
    assert loaded.model.hyperparameters == {
        "n_estimators": 50,
        "learning_rate": 0.05,
    }


def test_to_dict_rejects_unsupported_values():
    config = make_config()
    config.model.hyperparameters["unsupported"] = object()

    with pytest.raises(TypeError, match="not JSON-compatible"):
        config.to_dict()