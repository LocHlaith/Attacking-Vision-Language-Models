from __future__ import annotations

from collections import deque
from collections.abc import Iterable

Vertex = tuple[int, int]
Arc = tuple[Vertex, Vertex]


def four_neighbours(vertex: Vertex, height: int, width: int) -> list[Vertex]:
    row, column = vertex
    result = []
    for next_row, next_column in (
        (row - 1, column),
        (row + 1, column),
        (row, column - 1),
        (row, column + 1),
    ):
        if 0 <= next_row < height and 0 <= next_column < width:
            result.append((next_row, next_column))
    return result


def connected_components(vertices: Iterable[Vertex], height: int, width: int) -> list[set[Vertex]]:
    remaining = set(vertices)
    components: list[set[Vertex]] = []
    while remaining:
        start = next(iter(remaining))
        component = {start}
        queue = deque([start])
        remaining.remove(start)
        while queue:
            vertex = queue.popleft()
            for neighbour in four_neighbours(vertex, height, width):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    component.add(neighbour)
                    queue.append(neighbour)
        components.append(component)
    return components


def is_connected(vertices: Iterable[Vertex], height: int, width: int) -> bool:
    selected = set(vertices)
    return not selected or len(connected_components(selected, height, width)) == 1


def ford_fulkerson(
    capacities: dict[Arc, float],
    source: Vertex,
    sink: Vertex,
) -> tuple[float, dict[Arc, float]]:
    """Hand-written augmenting-chain maximum-flow solver."""
    flow = {arc: 0.0 for arc in capacities}
    adjacency: dict[Vertex, set[Vertex]] = {}
    for start, end in capacities:
        adjacency.setdefault(start, set()).add(end)
        adjacency.setdefault(end, set()).add(start)
    total = 0.0
    while True:
        parent: dict[Vertex, Vertex | None] = {source: None}
        queue = deque([source])
        while queue and sink not in parent:
            start = queue.popleft()
            for end in adjacency.get(start, set()):
                forward = capacities.get((start, end), 0.0) - flow.get((start, end), 0.0)
                reverse = flow.get((end, start), 0.0)
                if end not in parent and forward + reverse > 1e-12:
                    parent[end] = start
                    queue.append(end)
        if sink not in parent:
            break
        residual = float("inf")
        end = sink
        while parent[end] is not None:
            start = parent[end]
            forward = capacities.get((start, end), 0.0) - flow.get((start, end), 0.0)
            reverse = flow.get((end, start), 0.0)
            residual = min(residual, forward + reverse)
            end = start
        end = sink
        while parent[end] is not None:
            start = parent[end]
            forward = capacities.get((start, end), 0.0) - flow.get((start, end), 0.0)
            used_forward = min(residual, max(forward, 0.0))
            if used_forward:
                flow[(start, end)] = flow.get((start, end), 0.0) + used_forward
            remainder = residual - used_forward
            if remainder:
                flow[(end, start)] = flow.get((end, start), 0.0) - remainder
            end = start
        total += residual
    return total, flow


def network_flow_connected(
    vertices: Iterable[Vertex],
    root: Vertex,
    height: int,
    width: int,
) -> bool:
    """Verify the paper's single-root connectivity condition by maximum flow."""
    selected = set(vertices)
    if not selected:
        return True
    if root not in selected:
        return False
    sink = (-1, -1)
    capacity = float(max(len(selected) - 1, 1))
    capacities: dict[Arc, float] = {}
    for vertex in selected:
        if vertex != root:
            capacities[(vertex, sink)] = 1.0
        for neighbour in four_neighbours(vertex, height, width):
            if neighbour in selected:
                capacities[(vertex, neighbour)] = capacity
    value, _ = ford_fulkerson(capacities, root, sink)
    return abs(value - (len(selected) - 1)) <= 1e-8
