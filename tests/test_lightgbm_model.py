from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor

from src.model.base import Model
from src.model.lightgbm_model import LightGBMModel


@pytest.fixture
def feature_data():
    """Create a small synthetic regression feature matrix."""
    return pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, 2.0, 3.0],
            "feature_b": [1.0, 1.5, 2.0, 2.5],
        }
    )


@pytest.fixture
def target_data():
    """Create a small synthetic regression target vector."""
    return pd.Series([0.5, 1.5, 2.5, 3.5], name="target")


@pytest.fixture
def model():
    """Create a small LightGBM model for fast tests."""
    return LightGBMModel(
        n_estimators=5,
        learning_rate=0.1,
        num_leaves=5,
        verbosity=-1,
        random_state=42,
    )


def test_lightgbm_model_can_be_instantiated(model):
    """Test that LightGBMModel can be created."""
    assert isinstance(model, LightGBMModel)


def test_lightgbm_model_implements_generic_model_interface(model):
    """Test that LightGBMModel implements the project's Model interface."""
    assert isinstance(model, Model)


def test_lightgbm_model_uses_lightgbm_regressor(model):
    """Test that the implementation uses the official LightGBM estimator."""
    assert isinstance(model._estimator, LGBMRegressor)


def test_fit_trains_model_and_returns_instance(
    model,
    feature_data,
    target_data,
):
    """Test that fit trains the model and follows estimator conventions."""
    result = model.fit(feature_data, target_data)

    assert result is model
    assert model._is_fitted is True
    assert hasattr(model._estimator, "booster_")


def test_fit_accepts_numpy_target(model, feature_data):
    """Test that fit accepts a NumPy-compatible target vector."""
    target = np.array([0.5, 1.5, 2.5, 3.5])

    result = model.fit(feature_data, target)

    assert result is model


def test_predict_returns_one_numeric_prediction_per_sample(
    model,
    feature_data,
    target_data,
):
    """Test that prediction returns numerical output with matching length."""
    model.fit(feature_data, target_data)

    predictions = model.predict(feature_data)

    assert len(predictions) == len(feature_data)
    assert np.issubdtype(np.asarray(predictions).dtype, np.number)
    assert np.isfinite(predictions).all()


def test_predict_before_fit_raises_clear_error(model, feature_data):
    """Test that prediction is rejected before the model is fitted."""
    with pytest.raises(RuntimeError, match="must be fitted"):
        model.predict(feature_data)


def test_fit_and_predict_delegate_to_lightgbm_estimator(
    model,
    feature_data,
    target_data,
):
    """Test that fitting and prediction delegate to LGBMRegressor."""
    with patch.object(
        model._estimator,
        "fit",
        wraps=model._estimator.fit,
    ) as fit_mock:
        model.fit(feature_data, target_data)

    assert fit_mock.called
    fit_mock.assert_called_once_with(feature_data, target_data)

    with patch.object(
        model._estimator,
        "predict",
        wraps=model._estimator.predict,
    ) as predict_mock:
        predictions = model.predict(feature_data)

    assert predict_mock.called
    predict_mock.assert_called_once_with(feature_data)
    assert len(predictions) == len(feature_data)