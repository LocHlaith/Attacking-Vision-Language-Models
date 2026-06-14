from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import torch
from torchvision.transforms import functional as TF

from .model import ImageModelContext

CSV_COLUMNS = [
    "image",
    "algorithm",
    "backend",
    "original_class",
    "original_class_name",
    "decision_class",
    "decision_class_name",
    "decision_score",
    "decision_function",
    "decision_threshold",
    "perturbation_l2",
    "perturbation_threshold",
    "perturbation_area",
    "successful",
    "elapsed_seconds",
    "iterations",
    "converged",
    "message",
]


def save_tensor_image(tensor: torch.Tensor, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.detach().cpu().squeeze(0).clamp(0.0, 1.0)
    TF.to_pil_image(image).save(target)


def save_attack_images(
    context: ImageModelContext,
    perturbation: torch.Tensor,
    directory: str | Path,
) -> None:
    target = Path(directory)
    candidate = torch.clamp(context.raw_image + perturbation, 0.0, 1.0)
    effective_perturbation = candidate - context.raw_image
    magnitude = effective_perturbation.detach().abs()
    scale = magnitude.max().clamp_min(1e-12)
    save_tensor_image(candidate, target / "attacked.png")
    save_tensor_image(1.0 - magnitude / scale, target / "perturbation.png")


def write_csv(path: str | Path, rows: Iterable[dict[str, object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
