class Graph:
    """
    Represents an undirected graph used by the CSOP objective-value
    prediction pipeline.

    The class is responsible exclusively for representing the graph and
    providing access to its basic structural information. Graph feature
    calculations, objective-function calculations, machine learning
    operations, and optimization logic must be implemented in separate
    components.

    The graph is represented by a collection of vertices and a collection
    of undirected edges. Vertex identifiers are generic and should not be
    restricted to consecutive integers.

    The implementation should provide efficient access to basic structural
    operations such as checking whether a vertex or edge exists and
    retrieving the neighbors of a vertex.

    This class should remain independent from machine learning models,
    feature extraction components, objective functions, optimization
    algorithms, and future Java integration layers.

    Attributes:
        vertices: The set of vertex identifiers in the graph.
        edges: The set of undirected edges in the graph.
    """

    def __init__(self, vertices, edges):
        """
        Initialize an undirected graph.

        Args:
            vertices: A collection containing the identifiers of the
                graph's vertices.
            edges: A collection containing the graph's edges. Each edge
                must connect two vertices that belong to the graph.

        Raises:
            ValueError: If an edge references a vertex that does not
                belong to the graph.
            ValueError: If an edge is a self-loop.
            ValueError: If an edge has an invalid representation.
        """
        self.vertices = set(vertices)
        self.edges = set()
        self._adjacency = {vertex: set() for vertex in self.vertices}

        for edge in edges:
            if len(edge) != 2:
                raise ValueError("Invalid edge representation.")

            u, v = edge

            if u not in self.vertices or v not in self.vertices:
                raise ValueError(
                    "Edge references a vertex that does not belong "
                    "to the graph."
                )

            if u == v:
                raise ValueError("Edge is a self-loop.")

            edge_set = frozenset((u, v))

            if edge_set not in self.edges:
                self.edges.add(edge_set)
                self._adjacency[u].add(v)
                self._adjacency[v].add(u)

    @property
    def num_vertices(self):
        """
        Return the number of vertices in the graph.

        Returns:
            int: The number of vertices in the graph.
        """
        return len(self.vertices)

    @property
    def num_edges(self):
        """
        Return the number of undirected edges in the graph.

        Each connection between two vertices is counted only once.
        Therefore, (u, v) and (v, u) represent the same edge.

        Returns:
            int: The number of undirected edges in the graph.
        """
        return len(self.edges)

    def has_vertex(self, vertex):
        """
        Check whether a vertex belongs to the graph.

        Args:
            vertex: The identifier of the vertex to check.

        Returns:
            bool: True if the vertex belongs to the graph; False otherwise.
        """
        return vertex in self.vertices

    def has_edge(self, u, v):
        """
        Check whether an undirected edge exists between two vertices.

        Since the graph is undirected, (u, v) and (v, u) represent the
        same edge.

        Args:
            u: The identifier of the first vertex.
            v: The identifier of the second vertex.

        Returns:
            bool: True if an edge connects u and v; False otherwise.

        Raises:
            ValueError: If either u or v does not belong to the graph.
        """
        if u not in self.vertices or v not in self.vertices:
            raise ValueError(
                "One or both vertices do not belong to the graph."
            )

        return frozenset((u, v)) in self.edges

    def neighbors(self, vertex):
        """
        Return the vertices directly connected to the given vertex.

        A neighbor of a vertex is a vertex connected to it by an edge.

        Args:
            vertex: The identifier of the vertex whose neighbors should
                be retrieved.

        Returns:
            Set: A set containing the identifiers of all vertices
                directly connected to vertex.

        Raises:
            ValueError: If the vertex does not belong to the graph.
        """
        if vertex not in self.vertices:
            raise ValueError("Vertex does not belong to the graph.")

        return self._adjacency[vertex].copy()