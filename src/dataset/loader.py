from collections.abc import Mapping
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

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
    OPTIONAL_COLUMNS = {"sample_id"}

    def __init__(
        self,
        dataset_path: Path | str,
        *,
        expected_configuration: Mapping[str, Any] | None = None,
    ):
        """
        Initialize a dataset loader.

        Args:
            dataset_path: Path to a processed CSV dataset file.

        Raises:
            FileNotFoundError: If the dataset file does not exist.
            IsADirectoryError: If the path refers to a directory.
        """
        self._dataset_path = Path(dataset_path)
        self._expected_configuration = (
            dict(expected_configuration)
            if expected_configuration is not None
            else None
        )

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
        configuration = self._load_manifest_configuration()

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
            if column not in self.REQUIRED_COLUMNS | self.OPTIONAL_COLUMNS
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
                sample_id=(
                    None
                    if "sample_id" not in dataframe.columns
                    or pd.isna(row["sample_id"])
                    else str(row["sample_id"])
                ),
            )
            for _, row in dataframe.iterrows()
        ]

        try:
            return Dataset(samples, metadata=configuration)
        except ValueError as exc:
            raise ValueError(
                f"Processed dataset contains invalid sample data: "
                f"{self._dataset_path}"
            ) from exc

    def _load_manifest_configuration(self) -> dict[str, Any] | None:
        """Read and validate a neighboring dataset configuration manifest."""
        manifest_path = self._dataset_path.with_suffix(".metadata.json")
        if not manifest_path.exists():
            if self._expected_configuration is not None:
                raise ValueError(
                    "Processed dataset has no configuration manifest: "
                    f"{self._dataset_path}"
                )
            return None

        try:
            with manifest_path.open(encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Could not read dataset manifest: {manifest_path}"
            ) from exc

        if not isinstance(manifest, dict) or manifest.get("format_version") != 1:
            raise ValueError(f"Unsupported dataset manifest: {manifest_path}")

        expected_digest = manifest.get("csv_sha256")
        actual_digest = sha256(self._dataset_path.read_bytes()).hexdigest()
        if expected_digest != actual_digest:
            raise ValueError(
                "Dataset CSV does not match its configuration manifest: "
                f"{self._dataset_path}"
            )

        configuration = manifest.get("configuration")
        if not isinstance(configuration, dict):
            raise ValueError(
                f"Dataset manifest has invalid configuration: {manifest_path}"
            )
        if (
            self._expected_configuration is not None
            and configuration != self._expected_configuration
        ):
            raise ValueError(
                "Processed dataset feature configuration does not match the "
                "expected configuration."
            )
        return configuration