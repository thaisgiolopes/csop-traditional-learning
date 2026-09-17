from ..graph import Graph
from .base import Feature


class DegreeFeature(Feature):
    """
    Extracts the degree of every vertex in a graph.

    The degree of a vertex is the number of vertices directly connected
    to it.

    Unlike a global feature, this feature produces one value for each
    vertex. The resulting vertex-level values can later be transformed
    into a graph-level representation through a pooling strategy.

    The feature extractor must not perform pooling itself.
    """

    @property
    def name(self):
        """
        Return the identifier of the feature.

        Returns:
            str: The feature name.
        """
        return "degree"

    def compute(self, graph: Graph):
        """
        Compute the degree of every vertex in the graph.

        The result must preserve the association between each vertex and
        its degree. The representation should be deterministic with
        respect to the graph's vertex identifiers.

        Args:
            graph: The graph from which the feature should be extracted.

        Returns:
            A mapping from vertex identifiers to their corresponding
            degree values.

        Raises:
            ValueError: If the graph is invalid for feature extraction.
        """
        if not isinstance(graph, Graph):
            raise TypeError("A valid Graph instance is required.")

        return {
            vertex: len(graph.neighbors(vertex))
            for vertex in sorted(graph.vertices)
        }