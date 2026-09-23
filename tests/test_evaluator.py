import numpy as np
import pytest
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.evaluation.evaluator import Evaluator
from src.evaluation.result import EvaluationResult


def test_evaluator_calculates_regression_metrics():
    """Test MAE, RMSE, and R² against sklearn reference metrics."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.5, 2.0, 2.5, 5.0])

    result = Evaluator().evaluate(y_true, y_pred)

    assert result.mae == pytest.approx(
        mean_absolute_error(y_true, y_pred)
    )
    assert result.rmse == pytest.approx(
        np.sqrt(mean_squared_error(y_true, y_pred))
    )
    assert result.r2 == pytest.approx(r2_score(y_true, y_pred))


def test_evaluator_returns_evaluation_result_with_sample_count():
    """Test the result type and evaluated sample count."""
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.5, 2.5]

    result = Evaluator().evaluate(y_true, y_pred)

    assert isinstance(result, EvaluationResult)
    assert result.num_samples == 3


def test_perfect_predictions_produce_zero_errors_and_expected_r2():
    """Test the metrics for perfect predictions."""
    y_true = np.array([0.5, 1.5, 2.5, 3.5])
    y_pred = y_true.copy()

    result = Evaluator().evaluate(y_true, y_pred)

    assert result.mae == pytest.approx(0.0)
    assert result.rmse == pytest.approx(0.0)
    assert result.r2 == pytest.approx(r2_score(y_true, y_pred))
    assert result.r2 == pytest.approx(1.0)


def test_evaluator_rejects_mismatched_lengths():
    """Test that incompatible target and prediction lengths are rejected."""
    with pytest.raises(
        ValueError,
        match="must have the same length",
    ):
        Evaluator().evaluate([1.0, 2.0, 3.0], [1.0, 2.0])


def test_evaluation_layer_does_not_import_lightgbm():
    """Test that the evaluator module has no LightGBM dependency."""
    import src.evaluation.evaluator as evaluator_module

    assert "lightgbm" not in evaluator_module.__dict__