from abc import ABC, abstractmethod
from enum import Enum, auto

from ..graph import Graph


class FeatureScope(Enum):
    """Identifies the graph scope used to compute a feature."""

    GLOBAL = auto()
    LOCAL = auto()


class FeatureLevel(Enum):
    """Identifies the entities associated with a feature's values."""

    NODE = auto()
    EDGE = auto()
    GRAPH = auto()


class FeatureContext:
    """
    Provides the graph instances available during feature computation.

    Attributes:
        full_graph: The complete graph instance.
        subgraph: The candidate subgraph instance.
    """

    def __init__(self, full_graph: Graph, subgraph: Graph):
        """
        Initialize a feature context.

        Args:
            full_graph: The complete graph instance.
            subgraph: The candidate subgraph instance.

        """
        self.full_graph = full_graph
        self.subgraph = subgraph


class Feature(ABC):
    """
    Abstract base class for graph feature extractors.

    A feature extractor transforms information from a FeatureContext into
    a representation that can be used by later stages of the CSOP
    objective-value prediction pipeline.

    Each feature has an independent scope and level. The scope identifies
    which graph is relevant to the computation, while the level identifies
    the entities associated with the result.

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

    @property
    @abstractmethod
    def scope(self) -> FeatureScope:
        """
        Return the scope used to compute the feature.

        Returns:
            FeatureScope: The feature's computation scope.
        """
        raise NotImplementedError("Subclasses must define a feature scope.")

    @property
    @abstractmethod
    def level(self) -> FeatureLevel:
        """
        Return the level associated with the feature values.

        Returns:
            FeatureLevel: The feature's output level.
        """
        raise NotImplementedError("Subclasses must define a feature level.")

    @abstractmethod
    def compute(self, context: FeatureContext):
        """
        Compute the feature for a feature context.

        The concrete behavior depends on the feature being implemented.

        Args:
            context: The complete graph and candidate subgraph available
                to the feature.

        Returns:
            The computed feature value or representation.
        """
        raise NotImplementedError("Subclasses must implement feature computation.")
