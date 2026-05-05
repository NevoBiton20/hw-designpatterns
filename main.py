"""
Minimum Spanning Tree system using design patterns.

Supported input types:
1. Adjacency representation:
2. Weight matrix:

Supported output types:
1. MSTWeight - only the total weight of the MST.
2. MSTEdges  - the list of edges in the MST.

Supported algorithms:
1. kruskal
2. prim

Design patterns used:
1. Strategy:
2. Flyweight:
3. Iterator:
4. Decorator:
"""

from dataclasses import dataclass
from heapq import heappop, heappush
from math import inf
from typing import Any, Callable, Iterable


# Basic edge representation

@dataclass(frozen=True)
class Edge:
    u: Any
    v: Any
    weight: float

    def as_tuple(self):
        return (self.u, self.v, self.weight)


# Decorator pattern: registering input adapters

_INPUT_ADAPTERS = []


def input_adapter(cls):
    """
    Class decorator for registering a new graph input adapter.

    This lets mst() support new input types without changing mst() itself.
    """
    _INPUT_ADAPTERS.append(cls)
    return cls


def create_graph_adapter(data):
    """
    Detect the input type and return a matching adapter.
    """
    for adapter_cls in _INPUT_ADAPTERS:
        if adapter_cls.can_handle(data):
            return adapter_cls(data)

    raise TypeError("unsupported graph input type")


# Input adapters

class GraphAdapter:
    """
    Abstract interface for graph adapters.

    Concrete adapters must expose:
    - vertices
    - edges()
    - neighbors(vertex)
    """

    @property
    def vertices(self):
        raise NotImplementedError

    def edges(self):
        raise NotImplementedError

    def neighbors(self, vertex):
        raise NotImplementedError


@input_adapter
class AdjacencyGraphAdapter(GraphAdapter):
    """
    Adapter for adjacency dictionaries.
    """

    @staticmethod
    def can_handle(data):
        return isinstance(data, dict)

    def __init__(self, data):
        self._data = data
        vertices = set(data.keys())

        for u, neighbors in data.items():
            for v, _ in self._neighbor_items(neighbors):
                vertices.add(v)

        self._vertices = list(vertices)

    @property
    def vertices(self):
        return self._vertices

    def _neighbor_items(self, neighbors):
        if isinstance(neighbors, dict):
            return neighbors.items()

        return neighbors

    def edges(self):
        """
        Create each undirected edge once.

        If the same undirected edge appears twice, for example A-B and B-A,
        it is yielded only once.
        """
        best_edges = {}

        for u, neighbors in self._data.items():
            for v, weight in self._neighbor_items(neighbors):
                if u == v:
                    continue

                key = frozenset((u, v))
                edge = Edge(u, v, weight)

                if key not in best_edges or weight < best_edges[key].weight:
                    best_edges[key] = edge

        for edge in best_edges.values():
            yield edge

    def neighbors(self, vertex):
        if vertex not in self._data:
            return

        for neighbor, weight in self._neighbor_items(self._data[vertex]):
            if neighbor != vertex:
                yield Edge(vertex, neighbor, weight)


@input_adapter
class MatrixGraphAdapter(GraphAdapter):
    """
    Adapter for a weight matrix.
    """

    @staticmethod
    def can_handle(data):
        if isinstance(data, dict):
            return False

        try:
            rows = list(data)
        except TypeError:
            return False

        return all(hasattr(row, "__iter__") for row in rows)

    def __init__(self, data):
        self._matrix = [list(row) for row in data]
        self._vertices = list(range(len(self._matrix)))

        for row in self._matrix:
            if len(row) != len(self._matrix):
                raise ValueError("weight matrix must be square")

    @property
    def vertices(self):
        return self._vertices

    def _has_edge(self, i, j):
        if i == j:
            return False

        weight = self._matrix[i][j]
        return weight is not None and weight != inf

    def edges(self):
        n = len(self._matrix)

        for i in range(n):
            for j in range(i + 1, n):
                if self._has_edge(i, j):
                    yield Edge(i, j, self._matrix[i][j])

    def neighbors(self, vertex):
        for neighbor in range(len(self._matrix)):
            if self._has_edge(vertex, neighbor):
                yield Edge(vertex, neighbor, self._matrix[vertex][neighbor])


# Flyweight pattern: output builders

class MSTWeight:
    """
    Output type that returns only the total MST weight.
    """

    @staticmethod
    def create_builder():
        return _MSTWeightBuilder()


class MSTEdges:
    """
    Output type that returns the list of MST edges.
    """

    @staticmethod
    def create_builder():
        return _MSTEdgesBuilder()


class _MSTWeightBuilder:
    def __init__(self):
        self.total_weight = 0
        self.edge_count = 0

    def add_edge(self, edge):
        self.total_weight += edge.weight
        self.edge_count += 1

    def result(self):
        return self.total_weight


class _MSTEdgesBuilder:
    def __init__(self):
        self.edges = []
        self.edge_count = 0

    def add_edge(self, edge):
        self.edges.append(edge.as_tuple())
        self.edge_count += 1

    def result(self):
        return self.edges


# Strategy pattern: MST algorithms

def kruskal(graph: GraphAdapter, output_builder):
    """
    Kruskal's MST algorithm.

    The algorithm receives a graph adapter and an output builder.
    It does not know whether the caller wants only the total weight
    or the actual list of edges.
    """
    vertices = list(graph.vertices)
    target_edges = max(0, len(vertices) - 1)

    if target_edges == 0:
        return output_builder.result()

    parent = {v: v for v in vertices}
    rank = {v: 0 for v in vertices}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        root_x = find(x)
        root_y = find(y)

        if root_x == root_y:
            return False

        if rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        elif rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
        else:
            parent[root_y] = root_x
            rank[root_x] += 1

        return True

    for edge in sorted(graph.edges(), key=lambda e: (e.weight, str(e.u), str(e.v))):
        if union(edge.u, edge.v):
            output_builder.add_edge(edge)

            if output_builder.edge_count == target_edges:
                return output_builder.result()

    raise ValueError("graph is disconnected; MST does not exist")


def prim(graph: GraphAdapter, output_builder):
    """
    Prim's MST algorithm.

    The algorithm starts from one arbitrary vertex and repeatedly adds
    the cheapest edge connecting the current tree to a new vertex.
    """
    vertices = list(graph.vertices)
    target_edges = max(0, len(vertices) - 1)

    if target_edges == 0:
        return output_builder.result()

    start = min(vertices, key=str)
    visited = {start}
    heap = []
    counter = 0

    for edge in graph.neighbors(start):
        heappush(heap, (edge.weight, counter, edge))
        counter += 1

    while heap and output_builder.edge_count < target_edges:
        _, _, edge = heappop(heap)

        if edge.v in visited:
            continue

        visited.add(edge.v)
        output_builder.add_edge(edge)

        for next_edge in graph.neighbors(edge.v):
            if next_edge.v not in visited:
                heappush(heap, (next_edge.weight, counter, next_edge))
                counter += 1

    if output_builder.edge_count != target_edges:
        raise ValueError("graph is disconnected; MST does not exist")

    return output_builder.result()


# Main

def mst(graph_data, algorithm: Callable = kruskal, output_type=MSTEdges):
    """
    Compute a minimum spanning tree according to the chosen algorithm
    and output type.

    Parameters
    ----------
    graph_data:
        The graph input. Its type is detected automatically.

    algorithm:
        The MST algorithm strategy.
        For example: kruskal or prim.

    output_type:
        The requested output type.
        For example: MSTWeight or MSTEdges.

    Returns
    -------
    The result according to output_type.
    """
    graph = create_graph_adapter(graph_data)
    output_builder = output_type.create_builder()
    return algorithm(graph, output_builder)

if __name__ == '__main__':
    import doctest
    print(doctest.testmod())
