from __future__ import annotations

import torch

from .model import ImageModelContext


def ball_optimality_diagnostics(
    context: ImageModelContext,
    perturbation: torch.Tensor,
    radius: float,
) -> dict[str, float]:
    """Fritz John and KKT diagnostics for the paper's ball-constrained model."""
    variable = perturbation.detach().clone().requires_grad_(True)
    loss = context.loss(variable)
    loss_gradient = torch.autograd.grad(loss, variable)[0].detach()
    norm_square = float(variable.detach().square().sum().item())
    denominator = max(2.0 * norm_square, 1e-30)
    multiplier = max(
        float(torch.dot(loss_gradient.reshape(-1), variable.detach().reshape(-1)).item())
        / denominator,
        0.0,
    )
    stationarity = -loss_gradient + 2.0 * multiplier * variable.detach()
    constraint = radius * radius - norm_square
    return {
        "lambda_0": 1.0,
        "lambda_1": multiplier,
        "primal_feasibility": constraint,
        "dual_feasibility": multiplier,
        "complementary_slackness_residual": abs(multiplier * constraint),
        "stationarity_residual": float(torch.linalg.vector_norm(stationarity).item()),
        "fritz_john_stationarity_residual": float(torch.linalg.vector_norm(stationarity).item()),
    }


def threshold_kkt_diagnostics(
    context: ImageModelContext,
    perturbation: torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    variable = perturbation.detach().clone().requires_grad_(True)
    loss = context.loss(variable)
    loss_gradient = torch.autograd.grad(loss, variable)[0].detach()
    denominator = float(loss_gradient.square().sum().item())
    multiplier = max(
        2.0
        * float(torch.dot(variable.detach().reshape(-1), loss_gradient.reshape(-1)).item())
        / max(denominator, 1e-30),
        0.0,
    )
    stationarity = 2.0 * variable.detach() - multiplier * loss_gradient
    constraint = float(loss.detach().item()) - threshold
    return {
        "multiplier": multiplier,
        "primal_feasibility": constraint,
        "dual_feasibility": multiplier,
        "complementary_slackness_residual": abs(multiplier * constraint),
        "stationarity_residual": float(torch.linalg.vector_norm(stationarity).item()),
    }
