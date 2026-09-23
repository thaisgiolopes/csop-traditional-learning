from abc import ABC, abstractmethod
from typing import Any


class Model(ABC):
    """
    Abstract interface for regression models used in the CSOP pipeline.

    Implementations receive prepared machine-learning data and are
    responsible only for fitting a regression model and generating
    predictions. This class does not perform preprocessing, feature
    calculation, dataset splitting, metric calculation, or evaluation.
    """

    @abstractmethod
    def fit(self, X: Any, y: Any) -> "Model":
        """
        Fit the model using a feature matrix and target vector.

        Args:
            X: Feature matrix prepared for machine learning.
            y: Numerical target vector containing objective values.

        Returns:
            Model: The fitted model instance.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """
        Predict objective values for a feature matrix.

        Args:
            X: Feature matrix prepared for machine learning.

        Returns:
            Any: Numerical predictions for the provided samples.
        """
        raise NotImplementedError