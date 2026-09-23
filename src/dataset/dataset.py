import math
from numbers import Real
from typing import Any

import pandas as pd

from .sample import Sample


class Dataset:
	"""
	Represents samples in a tabular format for machine learning models.

	This class adapts already-computed ``Sample`` objects into a feature
	matrix and target vector. It does not calculate features or objectives,
	generate subgraphs, apply pooling, or train a model.
	"""

	def __init__(self, samples: list[Sample]):
		"""
		Initialize a dataset from an ordered collection of samples.

		Args:
			samples: Samples with identical feature schemas and numerical
				feature and target values.

		Raises:
			ValueError: If the collection is empty, feature schemas differ,
				or a feature or target value is not finite and numerical.
		"""
		if not samples:
			raise ValueError("Dataset requires at least one sample.")

		self._samples = list(samples)
		self._feature_names = list(self._samples[0].features.keys())
		expected_features = set(self._feature_names)

		feature_rows = []
		targets = []

		for sample_index, sample in enumerate(self._samples):
			actual_features = set(sample.features.keys())
			if actual_features != expected_features:
				missing = expected_features - actual_features
				unexpected = actual_features - expected_features
				raise ValueError(
					"Sample at index "
					f"{sample_index} has an inconsistent feature schema. "
					f"Missing: {sorted(missing)!r}; "
					f"unexpected: {sorted(unexpected)!r}."
				)

			row = {}
			for feature_name in self._feature_names:
				value = sample.features[feature_name]
				self._validate_numeric(
					value,
					f"feature {feature_name!r} in sample {sample_index}",
				)
				row[feature_name] = value

			self._validate_numeric(
				sample.target,
				f"target in sample {sample_index}",
			)
			feature_rows.append(row)
			targets.append(sample.target)

		self._X = pd.DataFrame(feature_rows, columns=self._feature_names)
		self._y = pd.Series(targets, name="target")
		self._graph_ids = [sample.graph_id for sample in self._samples]
		self._subgraph_ids = [sample.subgraph_id for sample in self._samples]

	@staticmethod
	def _validate_numeric(value: Any, description: str) -> None:
		"""Validate that a value can be used as a finite ML number."""
		if isinstance(value, bool) or not isinstance(value, Real):
			raise ValueError(
				f"{description} must be a finite numerical value; "
				f"received {type(value).__name__}."
			)

		if not math.isfinite(value):
			raise ValueError(
				f"{description} must be a finite numerical value; "
				f"received {value!r}."
			)

	@property
	def X(self) -> pd.DataFrame:
		"""Return the feature matrix in sample order."""
		return self._X

	@property
	def y(self) -> pd.Series:
		"""Return the target vector in sample order."""
		return self._y

	@property
	def samples(self) -> list[Sample]:
		"""Return the original samples in their original order."""
		return list(self._samples)

	@property
	def graph_ids(self) -> list[Any]:
		"""Return graph identifiers in sample order."""
		return list(self._graph_ids)

	@property
	def subgraph_ids(self) -> list[Any]:
		"""Return subgraph identifiers in sample order."""
		return list(self._subgraph_ids)

	@property
	def num_samples(self) -> int:
		"""Return the number of samples in the dataset."""
		return len(self._samples)

	@property
	def num_features(self) -> int:
		"""Return the number of feature columns in the dataset."""
		return len(self._feature_names)

	@property
	def feature_names(self) -> list[str]:
		"""Return feature column names in their tabular order."""
		return list(self._feature_names)
