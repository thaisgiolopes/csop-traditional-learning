from dataclasses import dataclass

from ..dataset.dataset import Dataset
from ..evaluation.evaluator import Evaluator
from ..evaluation.result import EvaluationResult
from ..model.base import Model
from ..model.predictor import PredictionResult, Predictor


@dataclass(frozen=True)
class PredictionEvaluationResult:
    """
    Groups prediction details with the corresponding evaluation metrics.

    The prediction results preserve graph and subgraph metadata, while the
    evaluation result contains the aggregate regression metrics.
    """

    predictions: list[PredictionResult]
    evaluation: EvaluationResult


class PredictionPipeline:
    """
    Orchestrates prediction and evaluation for a prepared Dataset.

    The trained model is supplied through the Predictor dependency. This
    pipeline does not train models, calculate features, generate subgraphs,
    calculate objectives, or implement evaluation metrics.
    """

    def __init__(
        self,
        model: Model,
        dataset: Dataset,
        predictor: Predictor,
        evaluator: Evaluator,
    ):
        """
        Initialize the prediction pipeline.

        Args:
            model: The trained generic model used by the predictor.
            dataset: Prepared dataset containing features and true targets.
            predictor: Component responsible for generating predictions.
            evaluator: Component responsible for calculating regression
                metrics.
        """
        self._model = model
        self._dataset = dataset
        self._predictor = predictor
        self._evaluator = evaluator

    def run(self) -> PredictionEvaluationResult:
        """
        Generate predictions and evaluate them against Dataset.y.

        Returns:
            PredictionEvaluationResult: Prediction records with preserved
            metadata and the calculated evaluation metrics.
        """
        prediction_results = self._predictor.predict(self._dataset)

        predicted_targets = [
            prediction.predicted_target
            for prediction in prediction_results
        ]

        evaluation = self._evaluator.evaluate(
            self._dataset.y,
            predicted_targets,
        )

        return PredictionEvaluationResult(
            predictions=prediction_results,
            evaluation=evaluation,
        )