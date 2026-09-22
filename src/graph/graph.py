import networkx as nx

class Graph:
    """
    Represents an undirected graph.

    Stores the graph structure and provides access to its basic
    structural information. Feature extraction, objective functions,
    machine learning, and optimization are handled separately.

    Attributes:
        vertices: The set of vertex identifiers.
        edges: The set of undirected edges.
    """

    def __init__(self, vertices, edges):
        """
        Initialize an undirected graph.

        Args:
            vertices: The graph's vertex identifiers.
            edges: The graph's edges.

        Raises:
            ValueError: If an edge is invalid, references a vertex that
                is not in the graph, or is a self-loop.
        """
        self._graph = nx.Graph()
        self._graph.add_nodes_from(vertices)

        for edge in edges:
            if len(edge) != 2:
                raise ValueError("Invalid edge representation.")

            u, v = edge

            if u not in self._graph or v not in self._graph:
                raise ValueError(
                    "Edge references a vertex that does not belong "
                    "to the graph."
                )

            if u == v:
                raise ValueError("Edge is a self-loop.")

            self._graph.add_edge(u, v)

    @property
    def networkx_graph(self):
        return self._graph

    @property
    def vertices(self):
        return set(self._graph.nodes)

    @property
    def edges(self):
        return {
            frozenset(edge)
            for edge in self._graph.edges
        }
    
    @property
    def num_vertices(self):
        """
        Return the number of vertices in the graph.

        Returns:
            int: Number of vertices.
        """
        return self._graph.number_of_nodes()

    @property
    def num_edges(self):
        """
        Return the number of edges in the graph.

        Returns:
            int: Number of undirected edges.
        """
        return self._graph.number_of_edges()

    def has_vertex(self, vertex):
        """
        Check whether a vertex belongs to the graph.

        Args:
            vertex: The vertex identifier.

        Returns:
            bool: True if the vertex belongs to the graph, False otherwise.
        """
        return self._graph.has_node(vertex)

    def has_edge(self, u, v):
        """
        Check whether an edge connects two vertices.

        Args:
            u: The first vertex identifier.
            v: The second vertex identifier.

        Returns:
            bool: True if an edge connects u and v, False otherwise.

        Raises:
            ValueError: If either vertex does not belong to the graph.
        """
        if not self.has_vertex(u) or not self.has_vertex(v):
            raise ValueError(
                "One or both vertices do not belong to the graph."
            )
        return self._graph.has_edge(u, v)

    def neighbors(self, vertex):
        """
        Return the vertices directly connected to a vertex.

        Args:
            vertex: The vertex identifier.

        Returns:
            Set: The identifiers of the vertex's neighbors.

        Raises:
            ValueError: If the vertex does not belong to the graph.
        """
        if not self.has_vertex(vertex):
            raise ValueError("Vertex does not belong to the graph.")

        return set(self._graph.neighbors(vertex))