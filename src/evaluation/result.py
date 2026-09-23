from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """
    Stores regression metrics calculated for model predictions.

    This class is a data container only. Metric calculations are performed
    by functions in ``evaluation.metrics`` before an instance is created.
    """

    mae: float
    rmse: float
    r2: float
    num_samples: int

    def __post_init__(self) -> None:
        """
        Validate the basic structure of the evaluation result.

        Raises:
            ValueError: If the number of evaluated samples is negative.
        """
        if self.num_samples < 0:
            raise ValueError(
                "The number of evaluated samples cannot be negative."
            )