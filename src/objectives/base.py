from abc import ABC, abstractmethod

from ..graph import Graph


class ObjectiveFunction(ABC):
    """
    Abstract base class for objective functions.

    An objective function assigns a numerical value to a candidate
    subgraph. Concrete implementations may represent Triangle Density
    Search or other objectives defined by the research problem.

    Objective functions are independent from subgraph generation,
    feature extraction, pooling, dataset construction, and machine
    learning.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the identifier of the objective function.

        Returns:
            str: The objective function name.
        """
        raise NotImplementedError(
            "Subclasses must define an objective function name."
        )

    @abstractmethod
    def compute(self, subgraph: Graph) -> float:
        """
        Compute the objective value of a candidate subgraph.

        Args:
            subgraph: The candidate subgraph to evaluate.

        Returns:
            float: The numerical objective value.

        Raises:
            TypeError: If subgraph is not a Graph instance.
        """
        if not isinstance(subgraph, Graph):
            raise TypeError("A valid Graph instance is required.")

        raise NotImplementedError(
            "Subclasses must implement objective computation."
        )