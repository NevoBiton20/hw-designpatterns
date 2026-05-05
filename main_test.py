import pytest

from main import mst, kruskal, prim, MSTWeight, MSTEdges


@pytest.fixture
def adjacency_graph():
    return {
        "A": {"B": 1, "C": 4},
        "B": {"A": 1, "C": 2, "D": 5},
        "C": {"A": 4, "B": 2, "D": 3},
        "D": {"B": 5, "C": 3},
    }


@pytest.fixture
def matrix_graph():
    return [
        [0,    1, 4,    None],
        [1,    0, 2,    5],
        [4,    2, 0,    3],
        [None, 5, 3,    0],
    ]


@pytest.mark.parametrize(
    "algorithm, output_type, expected",
    [
        (kruskal, MSTWeight, 6),
        (prim, MSTWeight, 6),
        (kruskal, MSTEdges, [("A", "B", 1), ("B", "C", 2), ("C", "D", 3)]),
        (prim, MSTEdges, [("A", "B", 1), ("B", "C", 2), ("C", "D", 3)]),
    ],
)
def test_adjacency_graph_all_combinations(adjacency_graph, algorithm, output_type, expected):
    assert mst(adjacency_graph, algorithm=algorithm, output_type=output_type) == expected


@pytest.mark.parametrize(
    "algorithm, output_type, expected",
    [
        (kruskal, MSTWeight, 6),
        (prim, MSTWeight, 6),
        (kruskal, MSTEdges, [(0, 1, 1), (1, 2, 2), (2, 3, 3)]),
        (prim, MSTEdges, [(0, 1, 1), (1, 2, 2), (2, 3, 3)]),
    ],
)
def test_matrix_graph_all_combinations(matrix_graph, algorithm, output_type, expected):
    assert mst(matrix_graph, algorithm=algorithm, output_type=output_type) == expected
