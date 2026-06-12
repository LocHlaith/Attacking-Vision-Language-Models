from avlm_or.solvers.graph import (
    connected_components,
    ford_fulkerson,
    is_connected,
    network_flow_connected,
)


def test_components_and_connectivity() -> None:
    vertices = {(0, 0), (0, 1), (2, 2)}
    assert len(connected_components(vertices, 3, 3)) == 2
    assert not is_connected(vertices, 3, 3)
    assert is_connected({(0, 0), (0, 1)}, 3, 3)


def test_ford_fulkerson() -> None:
    source = (0, 0)
    middle_a = (0, 1)
    middle_b = (1, 0)
    sink = (1, 1)
    capacities = {
        (source, middle_a): 2.0,
        (source, middle_b): 1.0,
        (middle_a, sink): 2.0,
        (middle_b, sink): 1.0,
    }
    value, _ = ford_fulkerson(capacities, source, sink)
    assert value == 3.0


def test_network_flow_connectivity_certificate() -> None:
    assert network_flow_connected({(0, 0), (0, 1), (1, 1)}, (0, 0), 3, 3)
    assert not network_flow_connected({(0, 0), (2, 2)}, (0, 0), 3, 3)
