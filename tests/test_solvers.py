import torch

from avlm_or.solvers.manual import (
    dfp_change_scale,
    external_point_method,
    nonlinear_conjugate_gradient,
    project_l2_ball,
    projected_gradient,
    steepest_descent,
)
from avlm_or.types import SolverSettings


SETTINGS = SolverSettings(max_iterations=80, initial_step=0.25)


def quadratic(point: torch.Tensor) -> torch.Tensor:
    return (point[0] - 2.0).square() + 2.0 * (point[1] + 1.0).square()


def test_unconstrained_manual_solvers() -> None:
    initial = torch.tensor([8.0, 5.0])
    for solver in (steepest_descent, nonlinear_conjugate_gradient, dfp_change_scale):
        result = solver(quadratic, initial, SETTINGS)
        assert torch.allclose(result.point, torch.tensor([2.0, -1.0]), atol=2e-3)
    result = nonlinear_conjugate_gradient(
        quadratic, initial, SETTINGS, beta_method="fletcher_reeves"
    )
    assert torch.allclose(result.point, torch.tensor([2.0, -1.0]), atol=2e-3)


def test_projected_gradient_stays_in_ball() -> None:
    initial = torch.zeros(2)

    def projector(point: torch.Tensor) -> torch.Tensor:
        return project_l2_ball(point, 1.0)

    result = projected_gradient(lambda point: -point[0], initial, projector, SETTINGS)
    assert torch.linalg.vector_norm(result.point) <= 1.00001
    assert result.point[0] > 0.99


def test_external_point_method_reduces_violation() -> None:
    result = external_point_method(
        lambda point: point.square().sum(),
        lambda point: torch.clamp(1.0 - point[0], min=0.0),
        torch.zeros(1),
        SolverSettings(max_iterations=40, initial_step=0.1),
        initial_penalty=10.0,
        outer_iterations=5,
        violation_tolerance=2e-2,
    )
    assert result.point[0] >= 0.97
