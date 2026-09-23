from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from .result import EvaluationResult


class Evaluator:
    """
    Orchestrates regression metric calculations for model predictions.

    The evaluator receives true and predicted target values, delegates metric
    calculations to ``evaluation.metrics``, and packages the results into an
    ``EvaluationResult``. It does not train models, calculate features,
    generate subgraphs, or calculate objective functions.
    """

    def evaluate(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
    ) -> EvaluationResult:
        """
        Evaluate predicted target values against their true values.

        Args:
            y_true: True objective values.
            y_pred: Predicted objective values.

        Returns:
            EvaluationResult: Calculated MAE, RMSE, R², and sample count.

        Raises:
            ValueError: If the inputs have incompatible lengths or are not
                one-dimensional arrays.
        """
        true_values = np.asarray(y_true)
        predicted_values = np.asarray(y_pred)

        if true_values.ndim != 1 or predicted_values.ndim != 1:
            raise ValueError(
                "True and predicted values must be one-dimensional."
            )

        if len(true_values) != len(predicted_values):
            raise ValueError(
                "True and predicted values must have the same length."
            )

        if len(true_values) == 0:
            raise ValueError(
                "True and predicted values cannot be empty."
            )

        return EvaluationResult(
            mae=mean_absolute_error(true_values, predicted_values),
            rmse=root_mean_squared_error(true_values, predicted_values),
            r2=r2_score(true_values, predicted_values),
            num_samples=len(true_values),
        )