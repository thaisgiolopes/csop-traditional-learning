import pandas as pd
import pytest

from src.dataset.dataset import Dataset
from src.dataset.sample import Sample


@pytest.fixture
def samples():
    """Create ordered samples from two candidate subgraphs of one graph."""
    return [
        Sample(
            graph_id="graph_001",
            subgraph_id="subgraph_001",
            features={"num_vertices": 3, "degree_mean": 1.5},
            target=0.5,
        ),
        Sample(
            graph_id="graph_001",
            subgraph_id="subgraph_002",
            features={"num_vertices": 5, "degree_mean": 2.25},
            target=0.75,
        ),
        Sample(
            graph_id="graph_002",
            subgraph_id="subgraph_001",
            features={"num_vertices": 4, "degree_mean": 1.75},
            target=1.0,
        ),
    ]


def test_dataset_converts_samples_to_feature_matrix_and_target_vector(
    samples,
):
    """Test that samples are converted to the expected tabular values."""
    dataset = Dataset(samples)

    expected_X = pd.DataFrame(
        {
            "num_vertices": [3, 5, 4],
            "degree_mean": [1.5, 2.25, 1.75],
        }
    )
    expected_y = pd.Series([0.5, 0.75, 1.0], name="target")

    pd.testing.assert_frame_equal(dataset.X, expected_X)
    pd.testing.assert_series_equal(dataset.y, expected_y)


def test_dataset_preserves_feature_columns_and_sample_order(samples):
    """Test that feature names and row order match the input samples."""
    dataset = Dataset(samples)

    assert list(dataset.X.columns) == ["num_vertices", "degree_mean"]
    assert dataset.X["num_vertices"].tolist() == [3, 5, 4]
    assert dataset.X["degree_mean"].tolist() == [1.5, 2.25, 1.75]
    assert dataset.samples == samples


def test_dataset_preserves_graph_and_subgraph_metadata(samples):
    """Test that each tabular row retains its source identifiers."""
    dataset = Dataset(samples)

    assert dataset.graph_ids == ["graph_001", "graph_001", "graph_002"]
    assert dataset.subgraph_ids == [
        "subgraph_001",
        "subgraph_002",
        "subgraph_001",
    ]


def test_dataset_reports_dimensions_and_feature_names(samples):
    """Test the dataset dimensions and ordered feature-name property."""
    dataset = Dataset(samples)

    assert dataset.num_samples == 3
    assert dataset.num_features == 2
    assert dataset.feature_names == ["num_vertices", "degree_mean"]


def test_dataset_handles_multiple_samples_from_same_graph(samples):
    """Test that samples sharing a graph remain separate dataset rows."""
    dataset = Dataset(samples[:2])

    assert dataset.num_samples == 2
    assert dataset.graph_ids == ["graph_001", "graph_001"]
    assert dataset.subgraph_ids == ["subgraph_001", "subgraph_002"]
    assert dataset.y.tolist() == [0.5, 0.75]


def test_dataset_rejects_inconsistent_feature_sets(samples):
    """Test that missing or unexpected feature columns are rejected."""
    inconsistent_samples = samples[:2]
    inconsistent_samples[1] = Sample(
        graph_id="graph_001",
        subgraph_id="subgraph_002",
        features={"num_vertices": 5, "edge_count": 4},
        target=0.75,
    )

    with pytest.raises(ValueError, match="inconsistent feature schema"):
        Dataset(inconsistent_samples)


def test_dataset_rejects_empty_sample_collection():
    """Test that a dataset cannot be created without samples."""
    with pytest.raises(ValueError, match="at least one sample"):
        Dataset([])


def test_dataset_rejects_non_numerical_feature_values():
    """Test that feature values must be suitable for numerical models."""
    invalid_samples = [
        Sample(
            graph_id="graph_001",
            subgraph_id="subgraph_001",
            features={"num_vertices": "three"},
            target=0.5,
        )
    ]

    with pytest.raises(ValueError, match="feature"):
        Dataset(invalid_samples)


def test_dataset_rejects_non_finite_feature_values():
    """Test that non-finite feature values are rejected."""
    invalid_samples = [
        Sample(
            graph_id="graph_001",
            subgraph_id="subgraph_001",
            features={"num_vertices": float("nan")},
            target=0.5,
        )
    ]

    with pytest.raises(ValueError, match="finite numerical"):
        Dataset(invalid_samples)
