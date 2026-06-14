from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from .model import ImageModelContext
from .objectives import (
    lagrange_relaxation,
    loss_threshold_violation,
    negative_loss,
    target_margin,
    weighted_sum,
)
from .solvers.efficient import library_quasi_newton
from .solvers.manual import (
    conjugate_gradient_linear,
    dfp_change_scale,
    external_point_method,
    levenberg_marquardt_newton,
    nonlinear_conjugate_gradient,
    project_l2_ball,
    projected_gradient,
    steepest_descent,
)
from .types import IterationRecord, SolverResult, SolverSettings


@dataclass
class AttackParameters:
    perturbation_threshold: float = 4.0
    weight: float = 10.0
    loss_threshold: float = 5.0
    target_margin_threshold: float = 0.1
    target_class: int = 999
    initial_multiplier: float = 1.0
    multiplier_step: float = 1.0
    outer_iterations: int = 10
    initial_penalty: float = 10.0
    penalty_growth: float = 10.0
    restarts: int = 1
    random_start_scale: float = 0.01
    random_seed: int = 0


@dataclass
class AttackOutcome:
    perturbation: torch.Tensor
    result: SolverResult
    decision_function: float
    decision_threshold: float | None
    perturbation_threshold: float | None


def zero_perturbation(context: ImageModelContext) -> torch.Tensor:
    return torch.zeros_like(context.raw_image)


def analytic_first_order(
    context: ImageModelContext,
    threshold: float,
) -> AttackOutcome:
    point = zero_perturbation(context).requires_grad_(True)
    loss = context.loss(point)
    gradient = torch.autograd.grad(loss, point)[0].detach()
    norm = torch.linalg.vector_norm(gradient).clamp_min(1e-30)
    perturbation = threshold * gradient / norm
    candidate_loss = float(context.loss(perturbation).detach().item())
    result = SolverResult(
        perturbation,
        -candidate_loss,
        1,
        True,
        "first-order Lagrange satisfactory solution",
        [
            IterationRecord(
                0,
                float((-loss).detach().item()),
                float(norm.item()),
                0.0,
                0.0,
            ),
            IterationRecord(
                1,
                -candidate_loss,
                0.0,
                float(torch.linalg.vector_norm(perturbation).item()),
                threshold,
            ),
        ],
    )
    return AttackOutcome(
        perturbation,
        result,
        candidate_loss,
        None,
        threshold,
    )


def second_order_ball_approximation(
    context: ImageModelContext,
    threshold: float,
    linear_iterations: int = 12,
    multiplier_iterations: int = 12,
) -> AttackOutcome:
    point = zero_perturbation(context).requires_grad_(True)
    loss = context.loss(point)
    gradient_graph = torch.autograd.grad(loss, point, create_graph=True)[0]
    gradient = gradient_graph.detach()

    def hessian_product(vector: torch.Tensor) -> torch.Tensor:
        return torch.autograd.grad(
            gradient_graph,
            point,
            grad_outputs=vector,
            retain_graph=True,
        )[0].detach()

    def solve(multiplier: float) -> torch.Tensor:
        def operator(vector: torch.Tensor) -> torch.Tensor:
            return 2.0 * multiplier * vector - hessian_product(vector)

        return conjugate_gradient_linear(operator, gradient, linear_iterations)

    upper = 1.0
    while (
        torch.dot(
            gradient.reshape(-1),
            (2.0 * upper * gradient - hessian_product(gradient)).reshape(-1),
        )
        <= 1e-12
        and upper < 1e8
    ):
        upper *= 2.0
    candidate = solve(upper)
    while torch.linalg.vector_norm(candidate) > threshold and upper < 1e8:
        upper *= 2.0
        candidate = solve(upper)
    lower = 0.0
    history: list[IterationRecord] = []
    for iteration in range(multiplier_iterations):
        multiplier = (lower + upper) / 2.0
        curvature = torch.dot(
            gradient.reshape(-1),
            (2.0 * multiplier * gradient - hessian_product(gradient)).reshape(-1),
        )
        if curvature <= 1e-12:
            lower = multiplier
            continue
        candidate = solve(multiplier)
        residual = 2.0 * multiplier * candidate - hessian_product(candidate) - gradient
        history.append(
            IterationRecord(
                iteration,
                float((-context.loss(candidate)).detach().item()),
                float(torch.linalg.vector_norm(residual).item()),
                float(torch.linalg.vector_norm(candidate).item()),
                multiplier,
            )
        )
        if torch.linalg.vector_norm(candidate) > threshold:
            lower = multiplier
        else:
            upper = multiplier
    perturbation = project_l2_ball(candidate, threshold)
    result = SolverResult(
        perturbation,
        float((-context.loss(perturbation)).detach().item()),
        multiplier_iterations,
        True,
        "second-order KKT approximation completed",
        history,
        extra={"lagrange_multiplier": upper},
    )
    return AttackOutcome(
        perturbation,
        result,
        float(context.loss(perturbation).detach().item()),
        None,
        threshold,
    )


def _unconstrained_solver(
    algorithm: str,
    backend: str,
):
    if backend == "efficient":
        return library_quasi_newton
    solvers = {
        "weighted_steepest": steepest_descent,
        "weighted_newton": lambda objective, point, settings: levenberg_marquardt_newton(
            objective, point, settings, initial_mu=0.0
        ),
        "weighted_newton_lm": levenberg_marquardt_newton,
        "weighted_dfp": dfp_change_scale,
        "weighted_conjugate_fr": lambda objective, point, settings: nonlinear_conjugate_gradient(
            objective, point, settings, beta_method="fletcher_reeves"
        ),
        "weighted_conjugate_pr_plus": nonlinear_conjugate_gradient,
    }
    return solvers[algorithm]


def weighted_attack(
    context: ImageModelContext,
    algorithm: str,
    backend: str,
    parameters: AttackParameters,
    settings: SolverSettings,
) -> AttackOutcome:
    objective = weighted_sum(context, parameters.weight)
    initial = zero_perturbation(context)
    result = _unconstrained_solver(algorithm, backend)(objective, initial, settings)
    return AttackOutcome(
        result.point,
        result,
        float(context.loss(result.point).detach().item()),
        None,
        None,
    )


def projection_attack(
    context: ImageModelContext,
    parameters: AttackParameters,
    settings: SolverSettings,
) -> AttackOutcome:
    threshold = parameters.perturbation_threshold

    def projector(point: torch.Tensor) -> torch.Tensor:
        return project_l2_ball(point, threshold)

    result = projected_gradient(
        negative_loss(context),
        zero_perturbation(context),
        projector,
        settings,
    )
    return AttackOutcome(
        result.point,
        result,
        float(context.loss(result.point).detach().item()),
        None,
        threshold,
    )


def external_point_attack(
    context: ImageModelContext,
    parameters: AttackParameters,
    settings: SolverSettings,
    backend: str,
) -> AttackOutcome:
    arguments = (
        lambda perturbation: perturbation.square().sum(),
        loss_threshold_violation(context, parameters.loss_threshold),
        zero_perturbation(context),
        settings,
    )
    options = {
        "initial_penalty": parameters.initial_penalty,
        "penalty_growth": parameters.penalty_growth,
        "outer_iterations": parameters.outer_iterations,
    }
    inner_solver = library_quasi_newton if backend == "efficient" else steepest_descent
    result = external_point_method(*arguments, **options, inner_solver=inner_solver)
    return AttackOutcome(
        result.point,
        result,
        float(context.loss(result.point).detach().item()),
        parameters.loss_threshold,
        None,
    )


def smooth_external_point_attack(
    context: ImageModelContext,
    parameters: AttackParameters,
    settings: SolverSettings,
    backend: str,
    alpha: float = 10.0,
) -> AttackOutcome:
    def violation(perturbation: torch.Tensor) -> torch.Tensor:
        return functional.softplus(
            alpha * (parameters.loss_threshold - context.loss(perturbation))
        ) / alpha

    arguments = (
        lambda perturbation: perturbation.square().sum(),
        violation,
        zero_perturbation(context),
        settings,
    )
    options = {
        "initial_penalty": parameters.initial_penalty,
        "penalty_growth": parameters.penalty_growth,
        "outer_iterations": parameters.outer_iterations,
    }
    inner_solver = library_quasi_newton if backend == "efficient" else steepest_descent
    result = external_point_method(*arguments, **options, inner_solver=inner_solver)
    return AttackOutcome(
        result.point,
        result,
        float(context.loss(result.point).detach().item()),
        parameters.loss_threshold,
        None,
    )


def approximate_primal_dual(
    context: ImageModelContext,
    measure,
    threshold: float,
    parameters: AttackParameters,
    settings: SolverSettings,
    backend: str,
) -> SolverResult:
    point = zero_perturbation(context)
    multiplier = parameters.initial_multiplier
    best_point: torch.Tensor | None = None
    best_norm = float("inf")
    history = []
    outer_trace: list[dict[str, float]] = []
    generator = torch.Generator(device=point.device)
    generator.manual_seed(parameters.random_seed)
    for outer in range(parameters.outer_iterations):
        objective = lagrange_relaxation(measure, multiplier)
        starts = [point]
        starts.extend(
            point
            + parameters.random_start_scale
            * torch.randn(
                point.shape,
                dtype=point.dtype,
                device=point.device,
                generator=generator,
            )
            for _ in range(max(parameters.restarts - 1, 0))
        )
        inner_results = []
        for start in starts:
            if backend == "efficient":
                inner_results.append(library_quasi_newton(objective, start, settings))
            else:
                inner_results.append(steepest_descent(objective, start, settings))
        result = min(inner_results, key=lambda candidate: candidate.value)
        point = result.point
        for candidate in inner_results:
            history.extend(candidate.history)
            candidate_measure = float(measure(candidate.point).detach().item())
            candidate_norm = float(candidate.point.square().sum().item())
            if candidate_measure >= threshold and candidate_norm < best_norm:
                best_point = candidate.point.detach().clone()
                best_norm = candidate_norm
        measure_value = float(measure(point).detach().item())
        outer_trace.append(
            {
                "outer": float(outer),
                "multiplier": float(multiplier),
                "measure": measure_value,
                "norm": float(torch.linalg.vector_norm(point).item()),
                "best_norm": float(best_norm**0.5) if best_point is not None else float("nan"),
                "violation": max(0.0, float(threshold - measure_value)),
            }
        )
        multiplier = max(
            0.0,
            multiplier + parameters.multiplier_step * (threshold - measure_value),
        )
    chosen = best_point if best_point is not None else point
    return SolverResult(
        chosen,
        float(chosen.square().sum().item()),
        len(history),
        best_point is not None,
        "feasible satisfactory solution retained" if best_point is not None else "no feasible candidate retained",
        history,
        {
            "lagrange_multiplier": multiplier,
            "outer_iterations": parameters.outer_iterations,
            "outer_trace": outer_trace,
        },
    )


def dual_loss_attack(
    context: ImageModelContext,
    parameters: AttackParameters,
    settings: SolverSettings,
    backend: str,
) -> AttackOutcome:
    def measure(perturbation: torch.Tensor) -> torch.Tensor:
        return context.loss(perturbation)

    result = approximate_primal_dual(
        context,
        measure,
        parameters.loss_threshold,
        parameters,
        settings,
        backend,
    )
    return AttackOutcome(
        result.point,
        result,
        float(measure(result.point).detach().item()),
        parameters.loss_threshold,
        None,
    )


def toilet_tissue_attack(
    context: ImageModelContext,
    parameters: AttackParameters,
    settings: SolverSettings,
    backend: str,
) -> AttackOutcome:
    def measure(perturbation: torch.Tensor) -> torch.Tensor:
        return target_margin(context, perturbation, parameters.target_class)
    result = approximate_primal_dual(
        context,
        measure,
        parameters.target_margin_threshold,
        parameters,
        settings,
        backend,
    )
    return AttackOutcome(
        result.point,
        result,
        float(measure(result.point).detach().item()),
        parameters.target_margin_threshold,
        None,
    )


def run_attack(
    context: ImageModelContext,
    algorithm: str,
    backend: str = "manual",
    parameters: AttackParameters | None = None,
    settings: SolverSettings | None = None,
) -> AttackOutcome:
    parameters = parameters or AttackParameters()
    settings = settings or SolverSettings()
    if algorithm == "analytic_first_order":
        return analytic_first_order(context, parameters.perturbation_threshold)
    if algorithm == "second_order_kkt":
        return second_order_ball_approximation(context, parameters.perturbation_threshold)
    if algorithm in {
        "weighted_steepest",
        "weighted_newton",
        "weighted_newton_lm",
        "weighted_dfp",
        "weighted_conjugate_fr",
        "weighted_conjugate_pr_plus",
    }:
        return weighted_attack(context, algorithm, backend, parameters, settings)
    if algorithm == "projected_gradient":
        return projection_attack(context, parameters, settings)
    if algorithm == "external_point":
        return external_point_attack(context, parameters, settings, backend)
    if algorithm == "external_point_softplus":
        return smooth_external_point_attack(context, parameters, settings, backend)
    if algorithm == "dual_loss":
        return dual_loss_attack(context, parameters, settings, backend)
    if algorithm == "toilet_tissue":
        return toilet_tissue_attack(context, parameters, settings, backend)
    raise ValueError(f"unknown algorithm: {algorithm}")


CONTINUOUS_ALGORITHMS = [
    "analytic_first_order",
    "second_order_kkt",
    "weighted_steepest",
    "weighted_newton",
    "weighted_newton_lm",
    "weighted_dfp",
    "weighted_conjugate_fr",
    "weighted_conjugate_pr_plus",
    "external_point",
    "external_point_softplus",
    "projected_gradient",
    "dual_loss",
    "toilet_tissue",
]
