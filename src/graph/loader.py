from pathlib import Path
from .graph import Graph

class GraphLoader:
    """
    Loads graph instances from the project's file-based representation.

    Each graph instance is stored in a directory containing two files:

        metadata
        adjlist

    The metadata file contains the number of vertices and the number of
    edges in its first line, separated by a whitespace character.

    The adjacency-list file contains one line for each vertex. The line
    number corresponds to the vertex identifier, starting from vertex 0.

    Each adjacency entry has the following format:

        vertex_id,edge_id

    Multiple adjacency entries in the same line are separated by spaces.

    Example:

        3,10 5,12 8,15

    means that the vertex corresponding to that line is connected to
    vertices 3, 5, and 8 through edges 10, 12, and 15, respectively.

    The loader is responsible only for translating the file representation
    into a Graph object. It must not perform feature extraction, objective
    function calculations, machine learning operations, or optimization
    logic.
    """

    METADATA_FILENAME = "metadata"
    ADJLIST_FILENAME = "adjlist"

    def __init__(self, instance_path):
        """
        Initialize a graph loader for a specific graph instance.

        Args:
            instance_path: Path to the directory containing the graph
                instance files.

        Raises:
            FileNotFoundError: If the instance directory does not exist.
            NotADirectoryError: If instance_path is not a directory.
        """
        self.instance_path = Path(instance_path)

        if not self.instance_path.exists():
            raise FileNotFoundError(f"Graph instance directory not found: {self.instance_path}")

        if not self.instance_path.is_dir():
            raise NotADirectoryError(f"Graph instance path is not a directory: {self.instance_path}")

    def load(self):
        """
        Load the graph instance and return a Graph object.

        The method reads the metadata and adjacency-list files, validates
        their consistency, and constructs the corresponding Graph.

        In an undirected graph, each edge is expected to appear in the
        adjacency lists of both endpoints using the same edge identifier.

        Returns:
            Graph: The graph represented by the instance files.

        Raises:
            FileNotFoundError: If the metadata or adjacency-list file
                does not exist.
            ValueError: If the instance contains invalid or inconsistent
                data.
        """
        num_vertices, num_edges = self._read_metadata()
        adjacency = self._read_adjacency_list(num_vertices)

        # Maps each undirected edge to its edge identifier.
        seen_edges = {}

        # Maps each edge identifier to its undirected edge.
        edge_id_to_edge = {}

        edges = []

        for vertex_id, neighbors in enumerate(adjacency):
            for neighbor_id, edge_id in neighbors:
                edge = frozenset((vertex_id, neighbor_id))

                # Check whether the same edge was already found.
                if edge in seen_edges:
                    if seen_edges[edge] != edge_id:
                        raise ValueError(
                            f"Inconsistent edge ID for edge "
                            f"({vertex_id}, {neighbor_id})."
                        )
                    continue

                # Check whether the edge ID is already associated
                # with a different edge.
                if edge_id in edge_id_to_edge:
                    if edge_id_to_edge[edge_id] != edge:
                        raise ValueError(
                            f"Edge ID {edge_id} is associated with "
                            f"multiple edges."
                        )

                seen_edges[edge] = edge_id
                edge_id_to_edge[edge_id] = edge
                edges.append((vertex_id, neighbor_id))

        if len(edges) != num_edges:
            raise ValueError(
                "Adjacency list edge count does not match metadata."
            )

        return Graph(list(range(num_vertices)), edges)

    def _read_metadata(self):
        """
        Read the graph metadata from the metadata file.

        The first line of the metadata file must contain two values
        separated by whitespace:

            number_of_vertices number_of_edges

        Returns:
            tuple[int, int]: A tuple containing the number of vertices
                and the number of edges.

        Raises:
            FileNotFoundError: If the metadata file does not exist.
            ValueError: If the metadata format is invalid.
        """
        metadata_path = self.instance_path / self.METADATA_FILENAME

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            first_line = metadata_file.readline().strip()

        if not first_line:
            raise ValueError("Metadata file is empty.")

        parts = first_line.split()
        if len(parts) != 2:
            raise ValueError("Metadata file must contain two integers.")

        try:
            num_vertices = int(parts[0])
            num_edges = int(parts[1])
        except ValueError as exc:
            raise ValueError("Metadata file contains non-integer values.") from exc

        if num_vertices < 0 or num_edges < 0:
            raise ValueError("Number of vertices and edges must be non-negative.")

        return num_vertices, num_edges

    def _read_adjacency_list(self, num_vertices):
        """
        Read the adjacency-list file.

        Each line corresponds to one vertex. The first line corresponds
        to vertex 0, the second line to vertex 1, and so on.

        Each adjacency entry must have the format:

            vertex_id,edge_id

        Multiple entries on the same line are separated by spaces.

        Args:
            num_vertices: Expected number of vertices according to the
                metadata file.

        Returns:
            list: A collection containing the parsed adjacency information.

        Raises:
            FileNotFoundError: If the adjacency-list file does not exist.
            ValueError: If the number of lines or an adjacency entry is
                invalid.
        """
        adjacency_path = self.instance_path / self.ADJLIST_FILENAME

        if not adjacency_path.exists():
            raise FileNotFoundError(f"Adjacency-list file not found: {adjacency_path}")

        with adjacency_path.open("r", encoding="utf-8") as adjacency_file:
            lines = adjacency_file.read().splitlines()

        if len(lines) != num_vertices:
            raise ValueError(
                "Adjacency-list line count does not match the number of vertices."
            )

        adjacency = []

        for vertex_id, line in enumerate(lines):
            if not line.strip():
                adjacency.append([])
                continue

            entries = []
            for token in line.split():
                neighbor_id, edge_id = self._parse_adjacency_entry(token)

                if neighbor_id < 0 or neighbor_id >= num_vertices:
                    raise ValueError(
                        f"Vertex identifier {neighbor_id} is outside the valid range."
                    )

                if vertex_id == neighbor_id:
                    raise ValueError("Adjacency list contains a self-loop.")

                entries.append((neighbor_id, edge_id))

            adjacency.append(entries)

        return adjacency

    def _parse_adjacency_entry(self, entry):
        """
        Parse a single adjacency-list entry.

        An entry must have the following format:

            vertex_id,edge_id

        There must be exactly one comma separating the vertex identifier
        from the edge identifier.

        Args:
            entry: A string representing one adjacency-list entry.

        Returns:
            tuple[int, int]: A tuple containing the adjacent vertex
                identifier and the edge identifier.

        Raises:
            ValueError: If the entry does not follow the expected format.
        """
        if not isinstance(entry, str):
            raise ValueError("Adjacency entry must be a string.")

        entry = entry.strip()
        if not entry:
            raise ValueError("Adjacency entry cannot be empty.")

        parts = entry.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid adjacency entry format: {entry}")

        vertex_id_str, edge_id_str = parts
        if not vertex_id_str or not edge_id_str:
            raise ValueError(f"Invalid adjacency entry format: {entry}")

        try:
            vertex_id = int(vertex_id_str)
            edge_id = int(edge_id_str)
        except ValueError as exc:
            raise ValueError(f"Invalid adjacency entry format: {entry}") from exc

        if vertex_id < 0 or edge_id < 0:
            raise ValueError(f"Adjacency entry identifiers must be non-negative: {entry}")

        return vertex_id, edge_id
        