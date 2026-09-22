import networkx as nx

from ..graph import Graph
from .base import ObjectiveFunction


class TDSObjective(ObjectiveFunction):
    """
    Computes the Triangle Density Search objective.

    TDS is defined as the number of triangles in a candidate subgraph
    divided by its number of vertices. This value is the target that the
    machine learning model will eventually learn to predict for candidate
    subgraphs.

    NetworkX-specific triangle calculation remains isolated in this
    objective implementation.
    """

    @property
    def name(self) -> str:
        """Return the identifier of the objective function."""
        return "tds"

    def compute(self, subgraph: Graph) -> float:
        """
        Compute the Triangle Density Search value.

        Args:
            subgraph: The candidate subgraph to evaluate.

        Returns:
            float: The number of triangles divided by the number of
                vertices.

        Raises:
            ValueError: If subgraph is not a Graph instance.
            ValueError: If subgraph has no vertices.
        """
        if not isinstance(subgraph, Graph):
            raise ValueError("subgraph must be a Graph instance.")

        number_of_vertices = subgraph.num_vertices

        if number_of_vertices == 0:
            raise ValueError(
                "TDS is undefined for a graph with no vertices."
            )

        nx_graph = nx.Graph(subgraph.networkx_graph)
        triangle_counts = nx.triangles(nx_graph)
        number_of_triangles = sum(triangle_counts.values()) // 3

        return number_of_triangles / number_of_vertices