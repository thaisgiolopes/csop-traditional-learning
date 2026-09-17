from pathlib import Path

import pytest

from src.loader import GraphLoader


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "simple_graph"


def test_load_graph():
    """Test loading a valid graph instance."""
    loader = GraphLoader(FIXTURE_PATH)

    graph = loader.load()

    assert graph.num_vertices == 4
    assert graph.num_edges == 4

    assert graph.has_edge(0, 1)
    assert graph.has_edge(0, 2)
    assert graph.has_edge(1, 3)
    assert graph.has_edge(2, 3)


def test_load_graph_neighbors():
    """Test that the loaded graph contains the expected adjacency structure."""
    loader = GraphLoader(FIXTURE_PATH)

    graph = loader.load()

    assert graph.neighbors(0) == {1, 2}
    assert graph.neighbors(1) == {0, 3}
    assert graph.neighbors(2) == {0, 3}
    assert graph.neighbors(3) == {1, 2}


def test_invalid_metadata(tmp_path):
    """Test that invalid metadata is rejected."""
    instance_path = tmp_path / "invalid_graph"
    instance_path.mkdir()

    (instance_path / "metadata").write_text("invalid metadata\n")
    (instance_path / "adjlist").write_text("")

    loader = GraphLoader(instance_path)

    with pytest.raises(ValueError):
        loader.load()


def test_invalid_adjacency_list_line_count(tmp_path):
    """Test that the adjacency list must contain one line per vertex."""
    instance_path = tmp_path / "invalid_graph"
    instance_path.mkdir()

    (instance_path / "metadata").write_text("4 2\n")

    (instance_path / "adjlist").write_text(
        "1,10\n"
        "0,10\n"
        "1,20\n"
    )

    loader = GraphLoader(instance_path)

    with pytest.raises(ValueError):
        loader.load()


def test_inconsistent_edge_id(tmp_path):
    """Test that the same undirected edge must use the same edge ID."""
    instance_path = tmp_path / "invalid_graph"
    instance_path.mkdir()

    (instance_path / "metadata").write_text("2 1\n")

    (instance_path / "adjlist").write_text(
        "1,10\n"
        "0,20\n"
    )

    loader = GraphLoader(instance_path)

    with pytest.raises(ValueError):
        loader.load()


def test_edge_id_assigned_to_multiple_edges(tmp_path):
    """Test that an edge ID cannot identify multiple different edges."""
    instance_path = tmp_path / "invalid_graph"
    instance_path.mkdir()

    (instance_path / "metadata").write_text("4 2\n")

    (instance_path / "adjlist").write_text(
        "1,10\n"
        "0,10\n"
        "3,10\n"
        "2,10\n"
    )

    loader = GraphLoader(instance_path)

    with pytest.raises(ValueError):
        loader.load()