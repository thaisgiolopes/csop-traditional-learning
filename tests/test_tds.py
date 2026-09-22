import pytest

from src.graph.graph import Graph
from src.objectives.tds import TDSObjective


def test_tds_of_single_triangle():
    """
    Test that a graph containing exactly one triangle has TDS equal to
    one triangle divided by three vertices.
    """
    graph = Graph(
        vertices=[0, 1, 2],
        edges=[
            (0, 1),
            (1, 2),
            (0, 2),
        ],
    )

    objective = TDSObjective()

    result = objective.compute(graph)

    assert result == pytest.approx(1 / 3)


def test_tds_of_graph_without_triangles():
    """
    Test that a graph without triangles has TDS equal to zero.
    """
    graph = Graph(
        vertices=[0, 1, 2, 3],
        edges=[
            (0, 1),
            (1, 2),
            (2, 3),
        ],
    )

    objective = TDSObjective()

    result = objective.compute(graph)

    assert result == 0.0


def test_tds_of_two_triangles_sharing_a_vertex():
    """
    Test a graph containing two distinct triangles.

    The graph contains:

        triangle (0, 1, 2)
        triangle (0, 2, 3)

    Therefore, the TDS is 2 / 4.
    """
    graph = Graph(
        vertices=[0, 1, 2, 3],
        edges=[
            (0, 1),
            (1, 2),
            (2, 0),
            (0, 3),
            (3, 2),
        ],
    )

    objective = TDSObjective()

    result = objective.compute(graph)

    assert result == pytest.approx(2 / 4)


def test_tds_of_complete_graph_k4():
    """
    Test the TDS of the complete graph with four vertices.

    K4 contains four triangles, so:

        TDS = 4 / 4 = 1.
    """
    graph = Graph(
        vertices=[0, 1, 2, 3],
        edges=[
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
        ],
    )

    objective = TDSObjective()

    result = objective.compute(graph)

    assert result == pytest.approx(1.0)


def test_tds_of_graph_with_isolated_vertex():
    """
    Test that isolated vertices are included in the denominator.

    A triangle plus one isolated vertex has:

        1 triangle
        4 vertices

    Therefore:

        TDS = 1 / 4.
    """
    graph = Graph(
        vertices=[0, 1, 2, 3],
        edges=[
            (0, 1),
            (1, 2),
            (0, 2),
        ],
    )

    objective = TDSObjective()

    result = objective.compute(graph)

    assert result == pytest.approx(1 / 4)


def test_tds_rejects_invalid_input():
    """
    Test that the objective function rejects objects that are not Graph
    instances.
    """
    objective = TDSObjective()

    with pytest.raises(ValueError):
        objective.compute(None)


def test_tds_rejects_empty_graph():
    """
    Test that TDS cannot be computed for a graph with zero vertices.
    """
    graph = Graph(
        vertices=[],
        edges=[],
    )

    objective = TDSObjective()

    with pytest.raises(ValueError):
        objective.compute(graph)

def test_tds_can_differentiate_candidate_subgraphs():
    """
    Test that TDS assigns different objective values to candidate
    subgraphs with different triangle densities.
    """
    low_density = Graph(
        vertices=[0, 1, 2, 3],
        edges=[
            (0, 1),
            (1, 2),
            (2, 3),
        ],
    )

    high_density = Graph(
        vertices=[0, 1, 2],
        edges=[
            (0, 1),
            (1, 2),
            (0, 2),
        ],
    )

    objective = TDSObjective()

    low_value = objective.compute(low_density)
    high_value = objective.compute(high_density)

    assert low_value == 0.0
    assert high_value == pytest.approx(1 / 3)