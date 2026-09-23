from pathlib import Path
from typing import Any

import pandas as pd

from ..dataset.dataset import Dataset
from ..dataset.builder import DatasetBuilder


class DatasetPipeline:
    """
    Orchestrates dataset construction and persistence for one graph instance.

    Dataset construction is delegated to ``DatasetBuilder``. This class only
    converts the resulting samples into a ``Dataset`` and persists its
    tabular representation. It does not calculate features, generate
    subgraphs, apply pooling, calculate objectives, or run machine learning.
    """

    def __init__(
        self,
        dataset_builder: DatasetBuilder,
        processed_dir: Path | str,
    ):
        """
        Initialize the dataset pipeline.

        Args:
            dataset_builder: Builder configured with the graph loader,
                subgraph generator, feature engine, pooling strategies, and
                objective function.
            processed_dir: Directory where generated CSV datasets are stored.
        """
        self._dataset_builder = dataset_builder
        self._processed_dir = Path(processed_dir)

    def run(
        self,
        graph_id: Any = None,
        output_path: Path | str | None = None,
    ) -> Dataset:
        """
        Build and persist a dataset for one graph instance.

        The raw graph instance is read through the ``GraphLoader`` owned by
        the injected ``DatasetBuilder``. Raw files are never modified.

        Args:
            graph_id: Optional identifier passed to ``DatasetBuilder``.
                When omitted, the builder derives it from the graph loader.
            output_path: Optional CSV file path. When omitted, the output is
                written to ``processed_dir/<graph_id>.csv``.

        Returns:
            Dataset: The constructed dataset that was persisted.

        Raises:
            ValueError: If the builder produces no samples or samples with
                inconsistent graph identifiers.
        """
        samples = self._dataset_builder.build(graph_id=graph_id)
        dataset = Dataset(samples)

        if not dataset.graph_ids:
            raise ValueError("DatasetBuilder produced no graph identifiers.")

        graph_ids = set(dataset.graph_ids)
        if len(graph_ids) != 1:
            raise ValueError(
                "DatasetPipeline expects samples from one graph instance."
            )

        resolved_graph_id = next(iter(graph_ids))

        if output_path is None:
            output_file = self._processed_dir / f"{resolved_graph_id}.csv"
        else:
            output_file = Path(output_path)

        output_file.parent.mkdir(parents=True, exist_ok=True)

        tabular_dataset = dataset.X.copy()
        tabular_dataset.insert(0, "graph_id", dataset.graph_ids)
        tabular_dataset.insert(1, "subgraph_id", dataset.subgraph_ids)
        tabular_dataset["target"] = dataset.y.to_numpy()

        tabular_dataset.to_csv(output_file, index=False)

        return dataset