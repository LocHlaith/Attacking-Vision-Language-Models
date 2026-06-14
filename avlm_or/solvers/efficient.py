from __future__ import annotations

from collections.abc import Callable

import torch

from ..types import IterationRecord
from ..types import SolverResult, SolverSettings

Objective = Callable[[torch.Tensor], torch.Tensor]


def optimizer_steepest_descent(
    objective: Objective,
    initial_point: torch.Tensor,
    settings: SolverSettings | None = None,
) -> SolverResult:
    """Efficient backend using PyTorch's optimized tensor updates."""
    settings = settings or SolverSettings()
    point = initial_point.detach().clone().requires_grad_(True)
    optimizer = torch.optim.SGD([point], lr=settings.initial_step)
    previous = float("inf")
    previous_point = point.detach().clone()
    history: list[IterationRecord] = []
    converged = False
    for iteration in range(settings.max_iterations):
        optimizer.zero_grad(set_to_none=True)
        value = objective(point)
        value.backward()
        gradient_norm = float(torch.linalg.vector_norm(point.grad).item())
        optimizer.step()
        current = float(value.detach().item())
        current_point = point.detach().clone()
        step_norm = float(torch.linalg.vector_norm(current_point - previous_point).item())
        previous_point = current_point
        history.append(
            IterationRecord(
                iteration,
                current,
                gradient_norm,
                step_norm,
                settings.initial_step,
            )
        )
        if gradient_norm <= settings.gradient_tolerance or abs(previous - current) <= settings.objective_tolerance:
            converged = True
            break
        previous = current
    final = point.detach()
    return SolverResult(
        final,
        float(objective(final).detach().item()),
        iteration + 1,
        converged,
        "termination criterion satisfied" if converged else "maximum iterations reached",
        history,
    )


def library_quasi_newton(
    objective: Objective,
    initial_point: torch.Tensor,
    settings: SolverSettings | None = None,
) -> SolverResult:
    """Efficient quasi-Newton backend; the mathematical objective is unchanged."""
    settings = settings or SolverSettings()
    point = initial_point.detach().clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [point],
        lr=1.0,
        max_iter=max(settings.max_iterations * 4, 20),
        tolerance_grad=settings.gradient_tolerance,
        tolerance_change=settings.objective_tolerance,
        line_search_fn="strong_wolfe",
    )
    calls = 0
    previous_point = point.detach().clone()
    history: list[IterationRecord] = []

    def closure() -> torch.Tensor:
        nonlocal calls, previous_point
        calls += 1
        optimizer.zero_grad(set_to_none=True)
        value = objective(point)
        value.backward()
        current_point = point.detach().clone()
        step_norm = float(torch.linalg.vector_norm(current_point - previous_point).item())
        previous_point = current_point
        gradient_norm = (
            float(torch.linalg.vector_norm(point.grad).item())
            if point.grad is not None
            else 0.0
        )
        history.append(
            IterationRecord(
                calls - 1,
                float(value.detach().item()),
                gradient_norm,
                step_norm,
                1.0,
            )
        )
        return value

    optimizer.step(closure)
    final = point.detach()
    return SolverResult(
        final,
        float(objective(final).detach().item()),
        calls,
        True,
        "library quasi-Newton backend completed",
        history,
    )
