from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    mean_absolute_error as sklearn_mean_absolute_error,
    mean_squared_error,
    r2_score as sklearn_r2_score,
)


def _validate_inputs(y_true: ArrayLike, y_pred: ArrayLike) -> None:
    """
    Validate regression target and prediction arrays.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.

    Raises:
        ValueError: If the inputs are not one-dimensional, are empty, or
            have different lengths.
    """
    true_values = np.asarray(y_true)
    predicted_values = np.asarray(y_pred)

    if true_values.ndim != 1 or predicted_values.ndim != 1:
        raise ValueError(
            "Regression targets and predictions must be one-dimensional."
        )

    if len(true_values) == 0:
        raise ValueError("Regression targets and predictions cannot be empty.")

    if len(true_values) != len(predicted_values):
        raise ValueError(
            "Regression targets and predictions must have the same length."
        )


def mean_absolute_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> float:
    """
    Calculate the mean absolute error.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.

    Returns:
        float: Mean absolute error.
    """
    _validate_inputs(y_true, y_pred)
    return float(sklearn_mean_absolute_error(y_true, y_pred))


def root_mean_squared_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> float:
    """
    Calculate the root mean squared error.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.

    Returns:
        float: Root mean squared error.
    """
    _validate_inputs(y_true, y_pred)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> float:
    """
    Calculate the coefficient of determination.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.

    Returns:
        float: R² score.
    """
    _validate_inputs(y_true, y_pred)
    return float(sklearn_r2_score(y_true, y_pred))