from abc import ABC, abstractmethod

from ..graph import Graph


class Feature(ABC):
    """
    Abstract base class for graph feature extractors.

    A feature extractor transforms information from a Graph object into
    a numerical representation that can be used by later stages of the
    CSOP objective-value prediction pipeline.

    Feature extractors are divided conceptually into two categories:

        - Global features: describe the graph as a whole.
        - Local features: describe individual vertices.

    This class defines the common interface that all feature extractors
    must implement.

    Feature extraction must remain independent from machine learning
    models, pooling strategies, objective functions, and optimization
    algorithms.
    """

    @property
    @abstractmethod
    def name(self):
        """
        Return the name that identifies the feature.

        The name should be stable and suitable for use as a feature
        identifier in datasets and later machine learning pipelines.

        Returns:
            str: The feature name.
        """
        raise NotImplementedError("Subclasses must define a feature name.")

    @abstractmethod
    def compute(self, graph: Graph):
        """
        Compute the feature for a given graph.

        The concrete behavior depends on the feature being implemented.

        Args:
            graph: The graph from which the feature should be extracted.

        Returns:
            The computed feature value or representation.

        Raises:
            TypeError: If the graph is invalid for the feature.
        """
        if not isinstance(graph, Graph):
            raise TypeError("A valid Graph instance is required.")

        raise NotImplementedError("Subclasses must implement feature computation.")
