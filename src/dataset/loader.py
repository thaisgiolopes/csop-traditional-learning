from pathlib import Path

import pandas as pd

from .dataset import Dataset
from .sample import Sample


class DatasetLoader:
    """
    Loads persisted tabular datasets into the project's Dataset format.

    This class reads processed CSV files and reconstructs Sample objects
    without loading graphs, recalculating features, generating subgraphs,
    or evaluating objective functions.
    """

    REQUIRED_COLUMNS = {
        "graph_id",
        "subgraph_id",
        "target",
    }

    def __init__(self, dataset_path: Path | str):
        """
        Initialize a dataset loader.

        Args:
            dataset_path: Path to a processed CSV dataset file.

        Raises:
            FileNotFoundError: If the dataset file does not exist.
            IsADirectoryError: If the path refers to a directory.
        """
        self._dataset_path = Path(dataset_path)

        if not self._dataset_path.exists():
            raise FileNotFoundError(
                f"Processed dataset file not found: {self._dataset_path}"
            )

        if not self._dataset_path.is_file():
            raise IsADirectoryError(
                f"Processed dataset path is not a file: {self._dataset_path}"
            )

    def load(self) -> Dataset:
        """
        Load the processed CSV file into a Dataset.

        Returns:
            Dataset: Reconstructed dataset containing Samples, features,
            targets, and preserved identifiers.

        Raises:
            ValueError: If the CSV cannot be parsed, required columns are
                missing, or the file contains no samples.
        """
        try:
            dataframe = pd.read_csv(self._dataset_path)
        except pd.errors.EmptyDataError as exc:
            raise ValueError(
                f"Processed dataset file is empty: {self._dataset_path}"
            ) from exc
        except pd.errors.ParserError as exc:
            raise ValueError(
                f"Could not parse processed dataset file: "
                f"{self._dataset_path}"
            ) from exc

        missing_columns = self.REQUIRED_COLUMNS - set(dataframe.columns)
        if missing_columns:
            raise ValueError(
                "Processed dataset is missing required columns: "
                f"{sorted(missing_columns)!r}."
            )

        feature_names = [
            column
            for column in dataframe.columns
            if column not in self.REQUIRED_COLUMNS
        ]

        samples = [
            Sample(
                graph_id=row["graph_id"],
                subgraph_id=row["subgraph_id"],
                features={
                    feature_name: row[feature_name]
                    for feature_name in feature_names
                },
                target=row["target"],
            )
            for _, row in dataframe.iterrows()
        ]

        try:
            return Dataset(samples)
        except ValueError as exc:
            raise ValueError(
                f"Processed dataset contains invalid sample data: "
                f"{self._dataset_path}"
            ) from exc