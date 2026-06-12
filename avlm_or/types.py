from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class IterationRecord:
    iteration: int
    objective: float
    gradient_norm: float
    step_norm: float
    step_size: float


@dataclass
class SolverResult:
    point: torch.Tensor
    value: float
    iterations: int
    converged: bool
    message: str
    history: list[IterationRecord] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SolverSettings:
    max_iterations: int = 50
    gradient_tolerance: float = 1e-5
    point_tolerance: float = 1e-6
    objective_tolerance: float = 1e-8
    initial_step: float = 1.0
    line_search_reduction: float = 0.5
    sufficient_decrease: float = 1e-4
    line_search_trials: int = 12


@dataclass
class Decision:
    class_index: int
    class_name: str
    score: float
    loss: float

