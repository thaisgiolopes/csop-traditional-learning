from abc import ABC, abstractmethod
import numpy as np

class Pooling(ABC):
    """
    Abstract base class for graph-level pooling strategies.

    Pooling aggregates vertex-level feature values into a single
    graph-level value.
    """

    @property
    @abstractmethod
    def name(self):
        """
        Return the pooling strategy name.

        Returns:
            str: The pooling strategy name.
        """
        raise NotImplementedError(
            "Subclasses must define a pooling name."
        )

    @abstractmethod
    def compute(self, values):
        """
        Aggregate vertex-level feature values.

        Args:
            values: A mapping from vertex identifiers to feature values.

        Returns:
            float: The aggregated value.

        Raises:
            TypeError: If values is not a dictionary.
            ValueError: If values is empty.
        """
        raise NotImplementedError(
            "Subclasses must implement pooling computation."
        )

    def _validate(self, values):
        """
        Validate the pooling input.

        Args:
            values: A mapping from vertex identifiers to feature values.

        Raises:
            TypeError: If values is not a dictionary.
            ValueError: If values is empty.
        """
        if not isinstance(values, dict):
            raise TypeError("values must be a dictionary.")

        if not values:
            raise ValueError("values cannot be empty.")


class MeanPooling(Pooling):
    """Computes the mean of vertex-level feature values."""

    @property
    def name(self):
        """
        Return the pooling strategy name.

        Returns:
            str: The pooling strategy name.
        """
        return "mean"

    def compute(self, values):
        """
        Compute the mean of the feature values.

        Args:
            values: A mapping from vertex identifiers to feature values.

        Returns:
            float: The mean value.
        """
        self._validate(values)
        return sum(values.values()) / len(values)


class MaxPooling(Pooling):
    """Computes the maximum of vertex-level feature values."""

    @property
    def name(self):
        """
        Return the pooling strategy name.

        Returns:
            str: The pooling strategy name.
        """
        return "max"

    def compute(self, values):
        """
        Compute the maximum feature value.

        Args:
            values: A mapping from vertex identifiers to feature values.

        Returns:
            float: The maximum value.
        """
        self._validate(values)
        return max(values.values())


class MinPooling(Pooling):
    """Computes the minimum of vertex-level feature values."""

    @property
    def name(self):
        """
        Return the pooling strategy name.

        Returns:
            str: The pooling strategy name.
        """
        return "min"

    def compute(self, values):
        """
        Compute the minimum feature value.

        Args:
            values: A mapping from vertex identifiers to feature values.

        Returns:
            float: The minimum value.
        """
        self._validate(values)
        return min(values.values())


class StdPooling(Pooling):
    """Computes the population standard deviation of vertex-level feature values."""

    @property
    def name(self):
        """
        Return the pooling strategy name.

        Returns:
            str: The pooling strategy name.
        """
        return "std"

    def compute(self, values):
        """
        Compute the population standard deviation of the feature values.

        Args:
            values: A mapping from vertex identifiers to feature values.

        Returns:
            float: The population standard deviation.
        """
        self._validate(values)
        return float(np.std(list(values.values()), ddof=0))