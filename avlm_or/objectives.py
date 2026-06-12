from __future__ import annotations

from collections.abc import Callable

import torch

from .model import ImageModelContext

TensorFunction = Callable[[torch.Tensor], torch.Tensor]


def negative_loss(context: ImageModelContext) -> TensorFunction:
    return lambda perturbation: -context.loss(perturbation)


def weighted_sum(context: ImageModelContext, weight: float) -> TensorFunction:
    return lambda perturbation: perturbation.square().sum() - weight * context.loss(perturbation)


def loss_threshold_violation(context: ImageModelContext, threshold: float) -> TensorFunction:
    return lambda perturbation: torch.clamp(threshold - context.loss(perturbation), min=0.0)


def untargeted_margin(context: ImageModelContext, perturbation: torch.Tensor) -> torch.Tensor:
    logits = context.logits(perturbation)[0]
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[context.original_class] = False
    return logits[mask].max() - logits[context.original_class]


def target_margin(
    context: ImageModelContext,
    perturbation: torch.Tensor,
    target_class: int = 999,
) -> torch.Tensor:
    logits = context.logits(perturbation)[0]
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[target_class] = False
    return logits[target_class] - logits[mask].max()


def lagrange_relaxation(
    measure: TensorFunction,
    multiplier: float,
) -> TensorFunction:
    return lambda perturbation: perturbation.square().sum() - multiplier * measure(perturbation)

