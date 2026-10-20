from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import Protocol, runtime_checkable

from ..graph import Graph


class SubgraphGenerator(ABC):
    """
    Abstract base class for candidate subgraph generators.

    A subgraph generator receives a complete graph and produces candidate
    subgraphs for later stages of the CSOP objective-value prediction
    pipeline.

    Subgraph generation is independent from feature extraction, pooling,
    objective-function calculation, dataset construction, and machine
    learning.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the identifier of the subgraph generation strategy.

        Returns:
            str: The generator name.
        """
        raise NotImplementedError(
            "Subclasses must define a generator name."
        )

    @abstractmethod
    def generate(self, graph: Graph) -> Collection[Graph]:
        """
        Generate candidate subgraphs from a complete graph.

        Args:
            graph: The complete graph representing the original problem
                instance.

        Returns:
            A collection of candidate subgraphs.

        Raises:
            TypeError: If graph is not a Graph instance.
        """
        if not isinstance(graph, Graph):
            raise TypeError("A valid Graph instance is required.")

        raise NotImplementedError(
            "Subclasses must implement subgraph generation."
        )


@runtime_checkable
class IndexedSubgraphGenerator(Protocol):
    """Capability for generators that reproduce one attempt by index."""

    @property
    def name(self) -> str:
        """Return the identifier of the generation strategy."""
        ...

    def generate(self, graph: Graph) -> Collection[Graph]:
        """Generate a collection of candidate subgraphs."""
        ...

    def generate_at_index(self, graph: Graph, index: int) -> Graph | None:
        """Generate one deterministic candidate attempt, or return None."""
        ...