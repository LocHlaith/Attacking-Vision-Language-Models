from __future__ import annotations

from collections.abc import Callable

import torch

from ..types import SolverResult, SolverSettings

Objective = Callable[[torch.Tensor], torch.Tensor]


def compiled_steepest_descent(
    objective: Objective,
    initial_point: torch.Tensor,
    settings: SolverSettings | None = None,
) -> SolverResult:
    """Efficient experimental backend using PyTorch's compiled tensor updates."""
    settings = settings or SolverSettings()
    point = initial_point.detach().clone().requires_grad_(True)
    optimizer = torch.optim.SGD([point], lr=settings.initial_step)
    previous = float("inf")
    converged = False
    for iteration in range(settings.max_iterations):
        optimizer.zero_grad(set_to_none=True)
        value = objective(point)
        value.backward()
        gradient_norm = float(torch.linalg.vector_norm(point.grad).item())
        optimizer.step()
        current = float(value.detach().item())
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
    )


def library_quasi_newton(
    objective: Objective,
    initial_point: torch.Tensor,
    settings: SolverSettings | None = None,
) -> SolverResult:
    """Fast non-submission backend; the mathematical objective is unchanged."""
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

    def closure() -> torch.Tensor:
        nonlocal calls
        calls += 1
        optimizer.zero_grad(set_to_none=True)
        value = objective(point)
        value.backward()
        return value

    optimizer.step(closure)
    final = point.detach()
    return SolverResult(
        final,
        float(objective(final).detach().item()),
        calls,
        True,
        "library quasi-Newton backend completed",
    )
