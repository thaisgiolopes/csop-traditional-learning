from abc import ABC, abstractmethod


class Pooling(ABC):
    """
    Abstract base class for graph-level pooling strategies.

    A pooling strategy transforms vertex-level feature values into a
    fixed-size representation describing the graph as a whole.

    Pooling operates on the output of local feature extractors and must
    remain independent from the Graph class, machine learning models,
    objective functions, and optimization algorithms.

    Different pooling strategies can be implemented independently and
    combined later to construct graph-level feature vectors.
    """

    @property
    @abstractmethod
    def name(self):
        """
        Return the name that identifies the pooling strategy.

        Returns:
            str: The pooling strategy name.
        """
        raise NotImplementedError(
            "Subclasses must define a pooling name."
        )

    @abstractmethod
    def compute(self, values):
        """
        Aggregate vertex-level feature values into a graph-level value.

        Args:
            values: A mapping from vertex identifiers to numerical
                feature values.

        Returns:
            float: The aggregated graph-level value.

        Raises:
            TypeError: If values is not a dictionary.
            ValueError: If the mapping is empty.
        """
        raise NotImplementedError(
            "Subclasses must implement pooling computation."
        )

    def _validate(self, values):
        """
        Validate the input values before pooling.

        Args:
            values: A mapping from vertex identifiers to numerical
                feature values.

        Raises:
            TypeError: If values is not a dictionary.
            ValueError: If the mapping is empty.
        """
        if not isinstance(values, dict):
            raise TypeError("values must be a dictionary.")

        if not values:
            raise ValueError("values cannot be empty.")


class MeanPooling(Pooling):
    """
    Computes the arithmetic mean of vertex-level feature values.
    """

    @property
    def name(self):
        """
        Return the identifier of the pooling strategy.

        Returns:
            str: The pooling strategy name.
        """
        return "mean"

    def compute(self, values):
        """
        Compute the arithmetic mean of the feature values.

        Args:
            values: A mapping from vertex identifiers to numerical
                feature values.

        Returns:
            float: The mean of the feature values.
        """
        self._validate(values)
        return sum(values.values()) / len(values)


class MaxPooling(Pooling):
    """
    Computes the maximum value among all vertex-level feature values.
    """

    @property
    def name(self):
        """
        Return the identifier of the pooling strategy.

        Returns:
            str: The pooling strategy name.
        """
        return "max"

    def compute(self, values):
        """
        Compute the maximum feature value.

        Args:
            values: A mapping from vertex identifiers to numerical
                feature values.

        Returns:
            float: The maximum feature value.
        """
        self._validate(values)
        return max(values.values())


class MinPooling(Pooling):
    """
    Computes the minimum value among all vertex-level feature values.
    """

    @property
    def name(self):
        """
        Return the identifier of the pooling strategy.

        Returns:
            str: The pooling strategy name.
        """
        return "min"

    def compute(self, values):
        """
        Compute the minimum feature value.

        Args:
            values: A mapping from vertex identifiers to numerical
                feature values.

        Returns:
            float: The minimum feature value.
        """
        self._validate(values)
        return min(values.values())


class StdPooling(Pooling):
    """
    Computes the population standard deviation of vertex-level
    feature values.
    """

    @property
    def name(self):
        """
        Return the identifier of the pooling strategy.

        Returns:
            str: The pooling strategy name.
        """
        return "std"

    def compute(self, values):
        """
        Compute the population standard deviation of the feature values.

        The population standard deviation must be used because the
        values represent all vertices of the given graph rather than
        a statistical sample.

        Args:
            values: A mapping from vertex identifiers to numerical
                feature values.

        Returns:
            float: The population standard deviation.
        """
        self._validate(values)
        values_list = list(values.values())
        mean_value = sum(values_list) / len(values_list)
        variance = sum((value - mean_value) ** 2 for value in values_list) / len(values_list)
        return variance ** 0.5