from typing import Any

from lightgbm import LGBMRegressor

from .base import Model


class LightGBMModel(Model):
    """
    LightGBM-based regression model for predicting CSOP objective values.

    This class adapts LightGBM's scikit-learn-compatible regressor to the
    project's generic Model interface. It expects already-prepared feature
    matrices and target vectors and does not perform preprocessing, feature
    extraction, splitting, or evaluation.
    """

    _DEFAULT_PARAMS = {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": 42,
        "verbosity": -1,
    }

    def __init__(self, **params: Any):
        """
        Initialize a LightGBM regression model.

        Args:
            **params: LightGBM ``LGBMRegressor`` parameters. Supplied values
                override the exploratory default parameters.
        """
        estimator_params = {
            **self._DEFAULT_PARAMS,
            **params,
        }

        self._estimator = LGBMRegressor(**estimator_params)
        self._is_fitted = False

    def fit(self, X: Any, y: Any) -> "LightGBMModel":
        """
        Fit the LightGBM regressor.

        Args:
            X: Prepared feature matrix.
            y: Numerical target vector containing objective values.

        Returns:
            LightGBMModel: This fitted model instance.
        """
        self._estimator.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: Any) -> Any:
        """
        Predict objective values for the provided feature matrix.

        Args:
            X: Prepared feature matrix.

        Returns:
            Any: Numerical predictions produced by LightGBM.

        Raises:
            RuntimeError: If called before ``fit``.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "LightGBMModel must be fitted before predictions can be made."
            )

        return self._estimator.predict(X)