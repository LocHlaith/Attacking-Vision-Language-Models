from __future__ import annotations

import time
from pathlib import Path

import torch

from .attacks import AttackParameters, run_attack
from .io import save_attack_images, save_tensor_image, write_csv
from .model import load_context, load_model, load_reference_labels
from .types import SolverSettings


def dataset_images(dataset_directory: str | Path = "datasets") -> list[Path]:
    return sorted(Path(dataset_directory).glob("*.png"), key=lambda path: path.stem)


def record_originals(
    output_root: str | Path = "outputs",
    device: str = "auto",
    reference_labels_path: str | Path | None = None,
) -> list[dict[str, object]]:
    model = load_model(device=device)
    reference_labels = load_reference_labels(reference_labels_path)
    rows: list[dict[str, object]] = []
    output_root = Path(output_root)
    for image_path in dataset_images():
        context = load_context(image_path, model, reference_labels)
        decision = context.decision()
        save_tensor_image(context.raw_image, output_root / "originals" / image_path.name)
        rows.append(
            {
                "image": image_path.name,
                "algorithm": "original",
                "backend": "none",
                "original_class": context.original_class,
                "original_class_name": context.categories[context.original_class],
                "decision_class": decision.class_index,
                "decision_class_name": decision.class_name,
                "decision_score": decision.score,
                "decision_function": decision.loss,
                "decision_threshold": "",
                "perturbation_l2": 0.0,
                "perturbation_threshold": "",
                "perturbation_area": "",
                "successful": False,
                "elapsed_seconds": 0.0,
                "iterations": 0,
                "converged": True,
                "message": "original image",
            }
        )
    write_csv(output_root / "originals.csv", rows)
    return rows


def run_batch(
    algorithm: str,
    backend: str = "manual",
    parameters: AttackParameters | None = None,
    settings: SolverSettings | None = None,
    output_root: str | Path = "outputs",
    device: str = "auto",
    reference_labels_path: str | Path | None = None,
) -> list[dict[str, object]]:
    parameters = parameters or AttackParameters()
    settings = settings or SolverSettings()
    model = load_model(device=device)
    reference_labels = load_reference_labels(reference_labels_path)
    target = Path(output_root) / algorithm / backend
    rows: list[dict[str, object]] = []
    for image_path in dataset_images():
        context = load_context(image_path, model, reference_labels)
        start = time.perf_counter()
        outcome = run_attack(context, algorithm, backend, parameters, settings)
        if torch.cuda.is_available() and context.raw_image.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        applied_perturbation = (
            torch.clamp(context.raw_image + outcome.perturbation, 0.0, 1.0)
            - context.raw_image
        )
        decision = context.decision(applied_perturbation)
        perturbation_l2 = float(torch.linalg.vector_norm(applied_perturbation).item())
        successful = (
            decision.class_index == parameters.target_class
            if algorithm == "toilet_tissue"
            else decision.class_index != context.original_class
        )
        save_attack_images(context, applied_perturbation, target / image_path.stem)
        rows.append(
            {
                "image": image_path.name,
                "algorithm": algorithm,
                "backend": backend,
                "original_class": context.original_class,
                "original_class_name": context.categories[context.original_class],
                "decision_class": decision.class_index,
                "decision_class_name": decision.class_name,
                "decision_score": decision.score,
                "decision_function": outcome.decision_function,
                "decision_threshold": outcome.decision_threshold
                if outcome.decision_threshold is not None
                else "",
                "perturbation_area": "",
                "perturbation_l2": perturbation_l2,
                "perturbation_threshold": outcome.perturbation_threshold
                if outcome.perturbation_threshold is not None
                else "",
                "successful": successful,
                "elapsed_seconds": elapsed,
                "iterations": outcome.result.iterations,
                "converged": outcome.result.converged,
                "message": outcome.result.message,
            }
        )
    write_csv(target / "results.csv", rows)
    return rows
