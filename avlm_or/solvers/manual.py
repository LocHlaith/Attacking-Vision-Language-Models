from __future__ import annotations

from collections.abc import Callable

import torch

from ..types import IterationRecord, SolverResult, SolverSettings

Objective = Callable[[torch.Tensor], torch.Tensor]
Projector = Callable[[torch.Tensor], torch.Tensor]


def value_and_gradient(objective: Objective, point: torch.Tensor) -> tuple[float, torch.Tensor]:
    variable = point.detach().clone().requires_grad_(True)
    value = objective(variable)
    gradient = torch.autograd.grad(value, variable)[0]
    return float(value.detach().item()), gradient.detach()


def approximate_line_search(
    objective: Objective,
    point: torch.Tensor,
    direction: torch.Tensor,
    gradient: torch.Tensor,
    settings: SolverSettings,
    projector: Projector | None = None,
) -> tuple[float, torch.Tensor, float]:
    current_value = float(objective(point).detach().item())
    directional_derivative = float(torch.dot(gradient.reshape(-1), direction.reshape(-1)).item())
    step_size = settings.initial_step
    candidate = point
    candidate_value = current_value
    for _ in range(settings.line_search_trials):
        candidate = point + step_size * direction
        if projector is not None:
            candidate = projector(candidate)
        candidate_value = float(objective(candidate).detach().item())
        bound = current_value + settings.sufficient_decrease * step_size * directional_derivative
        if candidate_value <= bound:
            break
        step_size *= settings.line_search_reduction
    return step_size, candidate.detach(), candidate_value


def _finished(
    gradient_norm: float,
    step_norm: float,
    objective_change: float,
    settings: SolverSettings,
) -> bool:
    return (
        gradient_norm <= settings.gradient_tolerance
        or step_norm <= settings.point_tolerance
        or objective_change <= settings.objective_tolerance
    )


def steepest_descent(
    objective: Objective,
    initial_point: torch.Tensor,
    settings: SolverSettings | None = None,
    projector: Projector | None = None,
) -> SolverResult:
    settings = settings or SolverSettings()
    point = initial_point.detach().clone()
    history: list[IterationRecord] = []
    converged = False
    message = "maximum iterations reached"
    value, gradient = value_and_gradient(objective, point)
    for iteration in range(settings.max_iterations):
        direction = -gradient
        gradient_norm = float(torch.linalg.vector_norm(gradient).item())
        step_size, candidate, candidate_value = approximate_line_search(
            objective, point, direction, gradient, settings, projector
        )
        step_norm = float(torch.linalg.vector_norm(candidate - point).item())
        objective_change = abs(candidate_value - value)
        history.append(
            IterationRecord(iteration, candidate_value, gradient_norm, step_norm, step_size)
        )
        point = candidate
        value, gradient = value_and_gradient(objective, point)
        if _finished(gradient_norm, step_norm, objective_change, settings):
            converged = True
            message = "termination criterion satisfied"
            break
    return SolverResult(point, value, len(history), converged, message, history)


def nonlinear_conjugate_gradient(
    objective: Objective,
    initial_point: torch.Tensor,
    settings: SolverSettings | None = None,
    projector: Projector | None = None,
    beta_method: str = "polak_ribiere_plus",
) -> SolverResult:
    settings = settings or SolverSettings()
    point = initial_point.detach().clone()
    value, gradient = value_and_gradient(objective, point)
    direction = -gradient
    history: list[IterationRecord] = []
    converged = False
    message = "maximum iterations reached"
    for iteration in range(settings.max_iterations):
        step_size, candidate, candidate_value = approximate_line_search(
            objective, point, direction, gradient, settings, projector
        )
        next_value, next_gradient = value_and_gradient(objective, candidate)
        denominator = torch.dot(gradient.reshape(-1), gradient.reshape(-1)).clamp_min(1e-30)
        if beta_method == "fletcher_reeves":
            beta = torch.dot(next_gradient.reshape(-1), next_gradient.reshape(-1)) / denominator
        elif beta_method == "polak_ribiere_plus":
            difference = next_gradient - gradient
            beta = torch.dot(next_gradient.reshape(-1), difference.reshape(-1)) / denominator
            beta = torch.clamp(beta, min=0.0)
        else:
            raise ValueError(f"unknown conjugate-gradient parameter: {beta_method}")
        next_direction = -next_gradient + beta * direction
        if torch.dot(next_gradient.reshape(-1), next_direction.reshape(-1)) >= 0:
            next_direction = -next_gradient
        gradient_norm = float(torch.linalg.vector_norm(next_gradient).item())
        step_norm = float(torch.linalg.vector_norm(candidate - point).item())
        objective_change = abs(candidate_value - value)
        history.append(
            IterationRecord(iteration, candidate_value, gradient_norm, step_norm, step_size)
        )
        point, value, gradient, direction = (
            candidate,
            next_value,
            next_gradient,
            next_direction.detach(),
        )
        if _finished(gradient_norm, step_norm, objective_change, settings):
            converged = True
            message = "termination criterion satisfied"
            break
    return SolverResult(point, value, len(history), converged, message, history)


class _DfpInverse:
    """Exact DFP inverse-Hessian action stored as rank-two corrections."""

    def __init__(self) -> None:
        self.positive: list[torch.Tensor] = []
        self.negative: list[torch.Tensor] = []

    def apply(self, vector: torch.Tensor) -> torch.Tensor:
        result = vector.clone()
        flat = vector.reshape(-1)
        for positive, negative in zip(self.positive, self.negative):
            result = result + positive * torch.dot(positive.reshape(-1), flat)
            result = result - negative * torch.dot(negative.reshape(-1), flat)
        return result

    def update(self, displacement: torch.Tensor, gradient_change: torch.Tensor) -> bool:
        inverse_gradient_change = self.apply(gradient_change)
        curvature = torch.dot(displacement.reshape(-1), gradient_change.reshape(-1))
        inverse_curvature = torch.dot(
            gradient_change.reshape(-1), inverse_gradient_change.reshape(-1)
        )
        if curvature <= 1e-12 or inverse_curvature <= 1e-12:
            return False
        self.positive.append(displacement / torch.sqrt(curvature))
        self.negative.append(inverse_gradient_change / torch.sqrt(inverse_curvature))
        return True


def dfp_change_scale(
    objective: Objective,
    initial_point: torch.Tensor,
    settings: SolverSettings | None = None,
    projector: Projector | None = None,
) -> SolverResult:
    settings = settings or SolverSettings()
    point = initial_point.detach().clone()
    value, gradient = value_and_gradient(objective, point)
    inverse = _DfpInverse()
    history: list[IterationRecord] = []
    skipped_updates = 0
    converged = False
    message = "maximum iterations reached"
    for iteration in range(settings.max_iterations):
        direction = -inverse.apply(gradient)
        if torch.dot(gradient.reshape(-1), direction.reshape(-1)) >= 0:
            direction = -gradient
        step_size, candidate, candidate_value = approximate_line_search(
            objective, point, direction, gradient, settings, projector
        )
        next_value, next_gradient = value_and_gradient(objective, candidate)
        displacement = candidate - point
        if not inverse.update(displacement, next_gradient - gradient):
            skipped_updates += 1
        gradient_norm = float(torch.linalg.vector_norm(next_gradient).item())
        step_norm = float(torch.linalg.vector_norm(displacement).item())
        objective_change = abs(candidate_value - value)
        history.append(
            IterationRecord(iteration, candidate_value, gradient_norm, step_norm, step_size)
        )
        point, value, gradient = candidate, next_value, next_gradient
        if _finished(gradient_norm, step_norm, objective_change, settings):
            converged = True
            message = "termination criterion satisfied"
            break
    return SolverResult(
        point,
        value,
        len(history),
        converged,
        message,
        history,
        {"skipped_dfp_updates": skipped_updates},
    )


def conjugate_gradient_linear(
    operator: Callable[[torch.Tensor], torch.Tensor],
    right_hand_side: torch.Tensor,
    max_iterations: int = 20,
    tolerance: float = 1e-5,
) -> torch.Tensor:
    solution = torch.zeros_like(right_hand_side)
    residual = right_hand_side - operator(solution)
    direction = residual.clone()
    residual_square = torch.dot(residual.reshape(-1), residual.reshape(-1))
    for _ in range(max_iterations):
        product = operator(direction)
        denominator = torch.dot(direction.reshape(-1), product.reshape(-1))
        if denominator <= 1e-20:
            break
        step = residual_square / denominator
        solution = solution + step * direction
        residual = residual - step * product
        next_square = torch.dot(residual.reshape(-1), residual.reshape(-1))
        if torch.sqrt(next_square) <= tolerance:
            break
        direction = residual + (next_square / residual_square.clamp_min(1e-30)) * direction
        residual_square = next_square
    return solution.detach()


def levenberg_marquardt_newton(
    objective: Objective,
    initial_point: torch.Tensor,
    settings: SolverSettings | None = None,
    initial_mu: float = 1e-2,
    linear_iterations: int = 10,
) -> SolverResult:
    settings = settings or SolverSettings()
    point = initial_point.detach().clone()
    history: list[IterationRecord] = []
    mu = initial_mu
    converged = False
    message = "maximum iterations reached"
    for iteration in range(settings.max_iterations):
        variable = point.detach().clone().requires_grad_(True)
        value_tensor = objective(variable)
        gradient_graph = torch.autograd.grad(value_tensor, variable, create_graph=True)[0]
        gradient = gradient_graph.detach()

        def modified_hessian(vector: torch.Tensor) -> torch.Tensor:
            product = torch.autograd.grad(
                gradient_graph,
                variable,
                grad_outputs=vector,
                retain_graph=True,
            )[0]
            return product.detach() + mu * vector

        direction = conjugate_gradient_linear(
            modified_hessian, -gradient, max_iterations=linear_iterations
        )
        if torch.dot(gradient.reshape(-1), direction.reshape(-1)) >= 0:
            mu *= 10.0
            direction = -gradient
        old_value = float(value_tensor.detach().item())
        step_size, candidate, candidate_value = approximate_line_search(
            objective, point, direction, gradient, settings
        )
        if candidate_value < old_value:
            mu = max(mu / 2.0, 1e-8)
        else:
            mu *= 10.0
        gradient_norm = float(torch.linalg.vector_norm(gradient).item())
        step_norm = float(torch.linalg.vector_norm(candidate - point).item())
        objective_change = abs(candidate_value - old_value)
        history.append(
            IterationRecord(iteration, candidate_value, gradient_norm, step_norm, step_size)
        )
        point = candidate
        if _finished(gradient_norm, step_norm, objective_change, settings):
            converged = True
            message = "termination criterion satisfied"
            break
    final_value = float(objective(point).detach().item())
    return SolverResult(point, final_value, len(history), converged, message, history, {"mu": mu})


def project_l2_ball(point: torch.Tensor, radius: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(point)
    if norm <= radius:
        return point
    return point * (radius / norm.clamp_min(1e-30))


def projected_gradient(
    objective: Objective,
    initial_point: torch.Tensor,
    projector: Projector,
    settings: SolverSettings | None = None,
) -> SolverResult:
    settings = settings or SolverSettings()
    point = projector(initial_point.detach().clone())
    value, gradient = value_and_gradient(objective, point)
    history: list[IterationRecord] = []
    converged = False
    message = "maximum iterations reached"
    for iteration in range(settings.max_iterations):
        direction = -gradient
        step_size, candidate, candidate_value = approximate_line_search(
            objective, point, direction, gradient, settings, projector
        )
        mapping = (point - projector(point - step_size * gradient)) / max(step_size, 1e-30)
        mapping_norm = float(torch.linalg.vector_norm(mapping).item())
        step_norm = float(torch.linalg.vector_norm(candidate - point).item())
        objective_change = abs(candidate_value - value)
        history.append(
            IterationRecord(iteration, candidate_value, mapping_norm, step_norm, step_size)
        )
        point = candidate
        value, gradient = value_and_gradient(objective, point)
        if _finished(mapping_norm, step_norm, objective_change, settings):
            converged = True
            message = "projected-gradient mapping criterion satisfied"
            break
    return SolverResult(point, value, len(history), converged, message, history)


def external_point_method(
    base_objective: Objective,
    violation: Callable[[torch.Tensor], torch.Tensor],
    initial_point: torch.Tensor,
    settings: SolverSettings | None = None,
    initial_penalty: float = 1.0,
    penalty_growth: float = 10.0,
    outer_iterations: int = 5,
    violation_tolerance: float = 1e-3,
    inner_solver: Callable[..., SolverResult] = steepest_descent,
) -> SolverResult:
    settings = settings or SolverSettings()
    point = initial_point.detach().clone()
    penalty = initial_penalty
    all_history: list[IterationRecord] = []
    total_iterations = 0
    last_result: SolverResult | None = None
    for outer in range(outer_iterations):
        def penalized(candidate: torch.Tensor) -> torch.Tensor:
            return base_objective(candidate) + penalty * violation(candidate).square()

        last_result = inner_solver(penalized, point, settings)
        total_iterations += last_result.iterations
        point = last_result.point
        all_history.extend(last_result.history)
        current_violation = float(violation(point).detach().item())
        if current_violation <= violation_tolerance:
            return SolverResult(
                point,
                float(base_objective(point).detach().item()),
                total_iterations,
                True,
                "constraint violation tolerance satisfied",
                all_history,
                {"penalty": penalty, "outer_iterations": outer + 1},
            )
        penalty *= penalty_growth
    assert last_result is not None
    return SolverResult(
        point,
        float(base_objective(point).detach().item()),
        total_iterations,
        False,
        "maximum outer iterations reached",
        all_history,
        {"penalty": penalty, "outer_iterations": outer_iterations},
    )
