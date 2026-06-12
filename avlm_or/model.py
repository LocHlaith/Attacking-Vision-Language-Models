from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from torchvision.transforms import functional as TF

from .types import Decision

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224
RANDOM_SEED = 0
ReferenceLabels = Mapping[str, int | str]


def configure_deterministic_execution(seed: int = RANDOM_SEED) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            try:
                torch.zeros(1, device="cuda")
                return torch.device("cuda")
            except RuntimeError:
                pass
        return torch.device("cpu")
    return torch.device(requested)


def load_model(
    model_path: str | Path = "models/mobilenet_v2-b0353104.pth",
    device: str = "auto",
) -> nn.Module:
    configure_deterministic_execution()
    target_device = choose_device(device)
    model = mobilenet_v2(weights=None)
    state = torch.load(Path(model_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval().to(target_device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_raw_image(path: str | Path, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = TF.resize(image, 256, antialias=True)
    image = TF.center_crop(image, [IMAGE_SIZE, IMAGE_SIZE])
    return TF.to_tensor(image).unsqueeze(0).to(device)


def normalize(raw_image: torch.Tensor) -> torch.Tensor:
    mean = raw_image.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = raw_image.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (raw_image - mean) / std


def load_reference_labels(path: str | Path | None) -> dict[str, int | str]:
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8") as file:
        labels = json.load(file)
    if not isinstance(labels, dict):
        raise ValueError("reference label file must contain a JSON object")
    return labels


def resolve_reference_class(
    image_path: str | Path,
    predicted_class: int,
    categories: list[str],
    reference_labels: ReferenceLabels | None = None,
) -> int:
    if not reference_labels or Path(image_path).name not in reference_labels:
        return predicted_class
    requested = reference_labels[Path(image_path).name]
    if isinstance(requested, int):
        if 0 <= requested < len(categories):
            return requested
        raise ValueError(f"reference class index out of range: {requested}")
    if isinstance(requested, str):
        matches = [
            index
            for index, category in enumerate(categories)
            if category.casefold() == requested.casefold()
        ]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(f"unknown reference class name: {requested}")
    raise ValueError("reference class must be an integer index or category name")


@dataclass
class ImageModelContext:
    model: nn.Module
    raw_image: torch.Tensor
    source_path: Path
    original_class: int
    categories: list[str]

    def logits(self, perturbation: torch.Tensor | None = None) -> torch.Tensor:
        if perturbation is None:
            candidate = self.raw_image
        else:
            candidate = torch.clamp(self.raw_image + perturbation, 0.0, 1.0)
        return self.model(normalize(candidate))

    def loss(self, perturbation: torch.Tensor | None = None) -> torch.Tensor:
        target = torch.tensor([self.original_class], device=self.raw_image.device)
        return torch.nn.functional.cross_entropy(self.logits(perturbation), target)

    def decision(self, perturbation: torch.Tensor | None = None) -> Decision:
        with torch.no_grad():
            logits = self.logits(perturbation)
            class_index = int(logits.argmax(dim=1).item())
            score = float(logits[0, class_index].item())
            loss = float(self.loss(perturbation).item())
        return Decision(class_index, self.categories[class_index], score, loss)


def load_context(
    image_path: str | Path,
    model: nn.Module,
    reference_labels: ReferenceLabels | None = None,
) -> ImageModelContext:
    device = next(model.parameters()).device
    raw_image = load_raw_image(image_path, device)
    categories = list(MobileNet_V2_Weights.DEFAULT.meta["categories"])
    with torch.no_grad():
        predicted_class = int(model(normalize(raw_image)).argmax(dim=1).item())
    original_class = resolve_reference_class(
        image_path,
        predicted_class,
        categories,
        reference_labels,
    )
    return ImageModelContext(
        model=model,
        raw_image=raw_image,
        source_path=Path(image_path),
        original_class=original_class,
        categories=categories,
    )
