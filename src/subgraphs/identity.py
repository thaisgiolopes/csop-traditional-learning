"""Stable identities for graph structures and candidate subgraphs."""

import hashlib
import json
import math
from typing import Any

from ..graph import Graph


def _canonical_node(value: Any) -> dict[str, Any]:
    """Encode supported graph node identifiers deterministically."""
    if value is None or isinstance(value, (str, bool, int)):
        normalized = value
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Graph node identifiers must be finite.")
        normalized = value
    elif isinstance(value, tuple):
        return {
            "type": "tuple",
            "value": [_canonical_node(item) for item in value],
        }
    else:
        raise TypeError(
            "Stable graph identities support only JSON scalar or tuple "
            f"node identifiers, not {type(value).__name__}."
        )

    return {
        "type": type(value).__name__,
        "value": normalized,
    }


def canonical_graph_structure(graph: Graph) -> dict[str, Any]:
    """Return the canonical node and edge representation of a graph."""
    node_tokens = {
        node: json.dumps(
            _canonical_node(node),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        for node in graph.vertices
    }
    vertices = [json.loads(token) for token in sorted(node_tokens.values())]
    edges = []

    for edge in graph.edges:
        left, right = sorted(node_tokens[node] for node in edge)
        edges.append([json.loads(left), json.loads(right)])

    edges.sort(
        key=lambda edge: json.dumps(
            edge,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    )
    return {"vertices": vertices, "edges": edges}


def _digest(value: Any) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def graph_fingerprint(graph: Graph) -> str:
    """Return a content-based fingerprint independent of graph_id."""
    return _digest(canonical_graph_structure(graph))


def sample_identity(graph_fingerprint_value: str, subgraph: Graph) -> str:
    """Return a deterministic identity for one graph/subgraph pair."""
    return "SMP_" + _digest(
        {
            "graph_fingerprint": graph_fingerprint_value,
            "subgraph": canonical_graph_structure(subgraph),
        }
    )


def graph_from_canonical_structure(structure: dict[str, Any]) -> Graph:
    """Rebuild a Graph from the canonical representation stored in SQLite."""
    def decode_node(encoded: dict[str, Any]) -> Any:
        node_type = encoded["type"]
        value = encoded["value"]
        if node_type == "tuple":
            return tuple(decode_node(item) for item in value)
        if node_type == "NoneType":
            return None
        if node_type == "bool":
            return bool(value)
        if node_type == "int":
            return int(value)
        if node_type == "float":
            return float(value)
        if node_type == "str":
            return str(value)
        raise ValueError(f"Unsupported canonical node type: {node_type!r}.")

    vertices = [decode_node(node) for node in structure["vertices"]]
    edges = [
        (decode_node(edge[0]), decode_node(edge[1]))
        for edge in structure["edges"]
    ]
    return Graph(vertices=vertices, edges=edges)