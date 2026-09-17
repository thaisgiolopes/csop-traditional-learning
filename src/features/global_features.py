from ..graph import Graph
from .base import Feature


class NumVerticesFeature(Feature):
    """
    Extracts the number of vertices of a graph.

    This is a global graph feature because it describes the graph as a
    whole rather than an individual vertex.
    """

    @property
    def name(self):
        """
        Return the identifier of the feature.

        Returns:
            str: The feature name.
        """
        return "num_vertices"

    def compute(self, graph: Graph):
        """
        Compute the number of vertices in the graph.

        Args:
            graph: The graph from which the feature should be extracted.

        Returns:
            int: The number of vertices in the graph.
        """
        if not isinstance(graph, Graph):
            raise TypeError("A valid Graph instance is required.")

        return graph.num_vertices


class NumEdgesFeature(Feature):
    """
    Extracts the number of edges of a graph.

    This is a global graph feature because it describes the graph as a
    whole rather than an individual vertex.
    """

    @property
    def name(self):
        """
        Return the identifier of the feature.

        Returns:
            str: The feature name.
        """
        return "num_edges"

    def compute(self, graph: Graph):
        """
        Compute the number of edges in the graph.

        Args:
            graph: The graph from which the feature should be extracted.

        Returns:
            int: The number of undirected edges in the graph.
        """
        if not isinstance(graph, Graph):
            raise TypeError("A valid Graph instance is required.")

        return graph.num_edges