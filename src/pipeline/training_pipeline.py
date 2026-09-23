from ..dataset.dataset import Dataset
from ..model.base import Model


class TrainingPipeline:
    """
    Orchestrates model training using an already prepared Dataset.

    This pipeline passes ``Dataset.X`` and ``Dataset.y`` to an injected
    generic Model. It does not load raw graphs, construct datasets,
    calculate features, split data, or evaluate predictions.
    """

    def __init__(self, dataset: Dataset, model: Model):
        """
        Initialize the training pipeline.

        Args:
            dataset: Prepared dataset containing the feature matrix and
                target vector.
            model: Generic regression model to train.
        """
        self._dataset = dataset
        self._model = model

    def run(self) -> Model:
        """
        Train the injected model using the prepared dataset.

        Returns:
            Model: The trained model instance.
        """
        return self._model.fit(
            self._dataset.X,
            self._dataset.y,
        )