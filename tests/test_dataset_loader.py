import pandas as pd
import pytest

from src.dataset.dataset import Dataset
from src.dataset.loader import DatasetLoader
from src.dataset.sample import Sample


def create_dataset_csv(path):
    """Create a small valid processed dataset CSV."""
    dataframe = pd.DataFrame(
        {
            "graph_id": ["graph_001", "graph_001", "graph_002"],
            "subgraph_id": [0, 1, 0],
            "num_vertices": [4, 3, 5],
            "degree_mean": [2.0, 1.5, 2.4],
            "target": [0.25, 0.5, 0.8],
        }
    )
    dataframe.to_csv(path, index=False)


def test_load_valid_csv_reconstructs_dataset(tmp_path):
    """Test CSV loading into a Dataset with preserved tabular values."""
    dataset_path = tmp_path / "dataset.csv"
    create_dataset_csv(dataset_path)

    dataset = DatasetLoader(dataset_path).load()

    expected_X = pd.DataFrame(
        {
            "num_vertices": [4, 3, 5],
            "degree_mean": [2.0, 1.5, 2.4],
        }
    )
    expected_y = pd.Series(
        [0.25, 0.5, 0.8],
        name="target",
    )

    pd.testing.assert_frame_equal(dataset.X, expected_X)
    pd.testing.assert_series_equal(dataset.y, expected_y)


def test_loader_reconstructs_samples_and_metadata(tmp_path):
    """Test that Samples retain identifiers, features, and targets."""
    dataset_path = tmp_path / "dataset.csv"
    create_dataset_csv(dataset_path)

    dataset = DatasetLoader(dataset_path).load()

    expected_samples = [
        Sample(
            graph_id="graph_001",
            subgraph_id=0,
            features={
                "num_vertices": 4,
                "degree_mean": 2.0,
            },
            target=0.25,
        ),
        Sample(
            graph_id="graph_001",
            subgraph_id=1,
            features={
                "num_vertices": 3,
                "degree_mean": 1.5,
            },
            target=0.5,
        ),
        Sample(
            graph_id="graph_002",
            subgraph_id=0,
            features={
                "num_vertices": 5,
                "degree_mean": 2.4,
            },
            target=0.8,
        ),
    ]

    assert dataset.samples == expected_samples
    assert dataset.graph_ids == [
        "graph_001",
        "graph_001",
        "graph_002",
    ]
    assert dataset.subgraph_ids == [0, 1, 0]


def test_loader_preserves_features_and_targets_for_training(tmp_path):
    """Test that loaded X and y contain all model-training information."""
    dataset_path = tmp_path / "dataset.csv"
    create_dataset_csv(dataset_path)

    dataset = DatasetLoader(dataset_path).load()

    assert dataset.feature_names == ["num_vertices", "degree_mean"]
    assert dataset.X["num_vertices"].tolist() == [4, 3, 5]
    assert dataset.X["degree_mean"].tolist() == [2.0, 1.5, 2.4]
    assert dataset.y.tolist() == [0.25, 0.5, 0.8]
    assert dataset.num_samples == 3
    assert dataset.num_features == 2


def test_loader_rejects_missing_file(tmp_path):
    """Test that a missing processed dataset file raises an error."""
    dataset_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="not found"):
        DatasetLoader(dataset_path)


def test_loader_rejects_missing_required_column(tmp_path):
    """Test that a CSV without required columns is rejected."""
    dataset_path = tmp_path / "invalid.csv"

    pd.DataFrame(
        {
            "graph_id": ["graph_001"],
            "num_vertices": [4],
            "target": [0.25],
        }
    ).to_csv(dataset_path, index=False)

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        DatasetLoader(dataset_path).load()


def test_loader_rejects_invalid_feature_content(tmp_path):
    """Test that non-numerical feature values are rejected."""
    dataset_path = tmp_path / "invalid_values.csv"

    pd.DataFrame(
        {
            "graph_id": ["graph_001"],
            "subgraph_id": [0],
            "num_vertices": ["not-a-number"],
            "target": [0.25],
        }
    ).to_csv(dataset_path, index=False)

    with pytest.raises(
        ValueError,
        match="invalid sample data",
    ):
        DatasetLoader(dataset_path).load()