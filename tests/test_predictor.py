from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.dataset.dataset import Dataset
from src.dataset.sample import Sample
from src.model.base import Model
from src.model.lightgbm_model import LightGBMModel
from src.model.predictor import PredictionResult, Predictor


@pytest.fixture
def dataset():
    """Create a small synthetic dataset with ordered samples."""
    return Dataset(
        [
            Sample(
                graph_id="graph_001",
                subgraph_id="subgraph_001",
                features={"feature_a": 0.0, "feature_b": 1.0},
                target=0.5,
            ),
            Sample(
                graph_id="graph_001",
                subgraph_id="subgraph_002",
                features={"feature_a": 1.0, "feature_b": 1.5},
                target=1.5,
            ),
            Sample(
                graph_id="graph_002",
                subgraph_id="subgraph_001",
                features={"feature_a": 2.0, "feature_b": 2.0},
                target=2.5,
            ),
        ]
    )


class FakeModel(Model):
    """Lightweight model double for testing Predictor behavior."""

    def __init__(self, predictions):
        self.predictions = np.asarray(predictions)
        self.received_X = None

    def fit(self, X, y):
        """Return this fake model without training."""
        return self

    def predict(self, X):
        """Return predefined predictions and record the input matrix."""
        self.received_X = X
        return self.predictions


def test_predictor_generates_one_result_per_sample(dataset):
    """Test that Predictor creates one result for every dataset sample."""
    model = FakeModel([10.0, 20.0, 30.0])

    results = Predictor(model).predict(dataset)

    assert len(results) == dataset.num_samples
    assert all(isinstance(result, PredictionResult) for result in results)


def test_predictor_preserves_order_and_metadata(dataset):
    """Test that prediction results preserve sample order and identifiers."""
    model = FakeModel([10.0, 20.0, 30.0])

    results = Predictor(model).predict(dataset)

    assert [(result.graph_id, result.subgraph_id) for result in results] == [
        ("graph_001", "subgraph_001"),
        ("graph_001", "subgraph_002"),
        ("graph_002", "subgraph_001"),
    ]
    assert [result.predicted_target for result in results] == [
        10.0,
        20.0,
        30.0,
    ]
    assert [result.true_target for result in results] == [0.5, 1.5, 2.5]


def test_predictor_calls_generic_model_with_dataset_X(dataset):
    """Test that Predictor passes the Dataset feature matrix to Model."""
    model = Mock(spec=Model)
    model.predict.return_value = np.array([4.0, 5.0, 6.0])

    Predictor(model).predict(dataset)

    model.predict.assert_called_once_with(dataset.X)
    assert model.predict.call_args.args[0] is dataset.X


def test_predictor_has_no_direct_lightgbm_dependency():
    """Test that Predictor depends on Model rather than LightGBM."""
    import src.model.predictor as predictor_module

    assert "lightgbm" not in predictor_module.__dict__


def test_predictor_integrates_with_trained_lightgbm_model(dataset):
    """Test Predictor integration with a trained LightGBMModel."""
    model = LightGBMModel(
        n_estimators=5,
        learning_rate=0.1,
        num_leaves=5,
        verbosity=-1,
        random_state=42,
    )
    model.fit(dataset.X, pd.Series(dataset.y))

    results = Predictor(model).predict(dataset)

    assert len(results) == dataset.num_samples
    assert all(
        isinstance(result.predicted_target, float)
        for result in results
    )
    assert np.isfinite(
        [result.predicted_target for result in results]
    ).all()