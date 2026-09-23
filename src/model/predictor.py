from dataclasses import dataclass
from typing import Any

import numpy as np

from ..dataset.dataset import Dataset
from .base import Model


@dataclass(frozen=True)
class PredictionResult:
    """
    Associates a predicted target value with its source candidate subgraph.

    The original target is retained when available so that later evaluation
    components can compare predictions with true objective values.
    """

    graph_id: Any
    subgraph_id: Any
    predicted_target: float
    true_target: float | None = None


class Predictor:
    """
    Applies a trained generic model to a prepared project Dataset.

    This class only coordinates prediction and metadata association. It does
    not calculate features or objectives, train models, or evaluate results.
    """

    def __init__(self, model: Model):
        """
        Initialize a predictor with a trained model.

        Args:
            model: Trained implementation of the project's Model interface.
        """
        self._model = model

    def predict(self, dataset: Dataset) -> list[PredictionResult]:
        """
        Generate predictions for all samples in a dataset.

        Predictions are generated from ``dataset.X`` and returned in the same
        order as the original samples.

        Args:
            dataset: Prepared dataset containing features and sample metadata.

        Returns:
            list[PredictionResult]: Predictions associated with their source
            graph and subgraph identifiers.

        Raises:
            ValueError: If the model returns a different number of
                predictions than the dataset contains.
        """
        predictions = np.asarray(self._model.predict(dataset.X))

        if predictions.ndim != 1:
            predictions = predictions.reshape(-1)

        if len(predictions) != dataset.num_samples:
            raise ValueError(
                "The model returned a different number of predictions "
                "than the dataset contains."
            )

        return [
            PredictionResult(
                graph_id=sample.graph_id,
                subgraph_id=sample.subgraph_id,
                predicted_target=float(prediction),
                true_target=sample.target,
            )
            for sample, prediction in zip(
                dataset.samples,
                predictions,
                strict=True,
            )
        ]