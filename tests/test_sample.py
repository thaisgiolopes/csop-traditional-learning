from src.dataset.sample import Sample


def test_sample_creation():
    """
    Test that a Sample stores its graph identifier, subgraph identifier,
    features, and target correctly.
    """
    features = {
        "num_vertices": 10,
        "degree_mean": 2.5,
    }

    sample = Sample(
        graph_id="graph_001",
        subgraph_id=3,
        features=features,
        target=1.25,
    )

    assert sample.graph_id == "graph_001"
    assert sample.subgraph_id == 3
    assert sample.features == features
    assert sample.target == 1.25


def test_sample_is_a_dataclass():
    """
    Test that Sample behaves as a dataclass.
    """
    sample = Sample(
        graph_id="graph_001",
        subgraph_id=0,
        features={"feature": 10},
        target=2.0,
    )

    assert sample.__dataclass_fields__ is not None


def test_sample_accepts_empty_features():
    """
    Test that a Sample can be created with an empty feature mapping.

    Validation of feature completeness is not the responsibility of Sample.
    """
    sample = Sample(
        graph_id="graph_001",
        subgraph_id=0,
        features={},
        target=0.0,
    )

    assert sample.features == {}


def test_sample_preserves_feature_values():
    """
    Test that Sample preserves different types of feature values without
    modifying them.
    """
    features = {
        "num_vertices": 10,
        "degree_mean": 2.5,
        "custom_feature": 7,
    }

    sample = Sample(
        graph_id="graph_001",
        subgraph_id=1,
        features=features,
        target=3.75,
    )

    assert sample.features["num_vertices"] == 10
    assert sample.features["degree_mean"] == 2.5
    assert sample.features["custom_feature"] == 7