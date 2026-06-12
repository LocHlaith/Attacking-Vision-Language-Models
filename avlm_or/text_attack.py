from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .io import save_attack_images, save_tensor_image, write_csv
from .model import (
    IMAGE_SIZE,
    ImageModelContext,
    load_context,
    load_model,
    load_reference_labels,
    normalize,
)
from .objectives import untargeted_margin
from .solvers.graph import (
    Vertex,
    connected_components,
    four_neighbours,
    is_connected,
    network_flow_connected,
)

DEFAULT_FONT = Path("C:/Windows/Fonts/simhei.ttf")
DEFAULT_TEXT = "\u5f20\u5929\u7fbd"
DEFAULT_FONT_SIZES = (10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 64)
DEFAULT_ANGLES = (-90, -60, -45, -30, 0, 30, 45, 60, 90)


@dataclass
class TextTemplate:
    name: str
    mask: np.ndarray
    strokes: list[set[Vertex]]
    roots: list[Vertex]


@dataclass
class TextAttackResult:
    perturbation: torch.Tensor
    mask: np.ndarray
    margin: float
    area: int
    method: str
    message: str


def render_text_template(
    text: str = DEFAULT_TEXT,
    font_size: int = 64,
    vertical_position: str = "center",
    angle: int = 0,
    font_path: str | Path = DEFAULT_FONT,
) -> TextTemplate:
    font = ImageFont.truetype(str(font_path), font_size)
    box = font.getbbox(text)
    width = box[2] - box[0]
    height = box[3] - box[1]
    padding = 2
    glyph = Image.new("L", (width + 2 * padding, height + 2 * padding), 0)
    draw = ImageDraw.Draw(glyph)
    draw.text((padding - box[0], padding - box[1]), text, fill=255, font=font)
    if angle % 360:
        glyph = glyph.rotate(
            angle,
            resample=Image.Resampling.NEAREST,
            expand=True,
            fillcolor=0,
        )
    image = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    left = (IMAGE_SIZE - glyph.width) // 2
    positions = {
        "top": 4,
        "center": (IMAGE_SIZE - glyph.height) // 2,
        "bottom": IMAGE_SIZE - glyph.height - 4,
    }
    image.paste(glyph, (left, positions[vertical_position]), glyph)
    mask = np.asarray(image) >= 64
    vertices = set(map(tuple, np.argwhere(mask)))
    strokes = [
        component
        for component in connected_components(vertices, IMAGE_SIZE, IMAGE_SIZE)
        if len(component) >= 2
    ]
    clean_mask = np.zeros_like(mask)
    for stroke in strokes:
        for row, column in stroke:
            clean_mask[row, column] = True
    roots = [min(stroke) for stroke in strokes]
    return TextTemplate(
        f"{font_size}_{vertical_position}_{angle:+d}deg",
        clean_mask,
        strokes,
        roots,
    )


def candidate_templates(
    font_sizes: tuple[int, ...] = DEFAULT_FONT_SIZES,
    angles: tuple[int, ...] = DEFAULT_ANGLES,
) -> list[TextTemplate]:
    return [
        render_text_template(
            font_size=size,
            vertical_position=position,
            angle=angle,
        )
        for size in font_sizes
        for position in ("top", "center", "bottom")
        for angle in angles
    ]


def mask_tensor(mask: np.ndarray, context: ImageModelContext) -> torch.Tensor:
    return torch.from_numpy(mask).to(context.raw_image.device).view(1, 1, IMAGE_SIZE, IMAGE_SIZE)


def black_perturbation(context: ImageModelContext, mask: np.ndarray) -> torch.Tensor:
    return -context.raw_image * mask_tensor(mask, context)


def true_margin(context: ImageModelContext, mask: np.ndarray) -> float:
    with torch.no_grad():
        return float(untargeted_margin(context, black_perturbation(context, mask)).item())


def linearized_gain(
    context: ImageModelContext,
    reference_mask: np.ndarray,
) -> tuple[float, np.ndarray]:
    candidate = torch.clamp(
        context.raw_image + black_perturbation(context, reference_mask),
        0.0,
        1.0,
    ).detach().requires_grad_(True)
    logits = context.model(normalize(candidate))[0]
    competitor_mask = torch.ones_like(logits, dtype=torch.bool)
    competitor_mask[context.original_class] = False
    competitor_indices = torch.arange(logits.numel(), device=logits.device)[competitor_mask]
    competitor = competitor_indices[logits[competitor_mask].argmax()]
    margin = logits[competitor] - logits[context.original_class]
    gradient = torch.autograd.grad(margin, candidate)[0]
    gain = -(gradient * context.raw_image).sum(dim=1).squeeze(0)
    return float(margin.detach().item()), gain.detach().cpu().numpy()


def _stroke_constraints_hold(
    template: TextTemplate,
    mask: np.ndarray,
    retention: float,
    use_network_flow: bool = False,
) -> bool:
    selected = set(map(tuple, np.argwhere(mask)))
    for stroke, root in zip(template.strokes, template.roots):
        chosen = stroke & selected
        if len(chosen) < math.ceil(retention * len(stroke)):
            return False
        if root not in chosen:
            return False
        if use_network_flow:
            if not network_flow_connected(chosen, root, IMAGE_SIZE, IMAGE_SIZE):
                return False
        elif not is_connected(chosen, IMAGE_SIZE, IMAGE_SIZE):
            return False
    return True


def reverse_greedy_pruning(
    context: ImageModelContext,
    template: TextTemplate,
    margin_threshold: float,
    retention: float = 0.5,
    max_checks: int | None = None,
) -> TextAttackResult | None:
    mask = template.mask.copy()
    if true_margin(context, mask) < margin_threshold:
        return None
    checks = 0
    while True:
        _, gain = linearized_gain(context, mask)
        candidates = sorted(
            map(tuple, np.argwhere(mask)),
            key=lambda vertex: float(gain[vertex]),
        )
        changed = False
        for vertex in candidates:
            if vertex in template.roots:
                continue
            trial = mask.copy()
            trial[vertex] = False
            checks += 1
            if _stroke_constraints_hold(template, trial, retention) and true_margin(context, trial) >= margin_threshold:
                mask = trial
                changed = True
            if max_checks is not None and checks >= max_checks:
                changed = False
                break
        if not changed:
            break
    perturbation = black_perturbation(context, mask)
    return TextAttackResult(
        perturbation,
        mask,
        true_margin(context, mask),
        int(mask.sum()),
        "reverse_greedy",
        f"{template.name}; accepted after {checks} deletion checks",
    )


def manual_linearized_selection(
    template: TextTemplate,
    gain: np.ndarray,
    right_hand_side: float,
    retention: float,
) -> np.ndarray | None:
    """Hand-written satisfactory solver for the local mixed 0-1 model."""
    mask = template.mask.copy()
    current_gain = float(gain[mask].sum())
    if current_gain < right_hand_side:
        return None
    for vertex in sorted(map(tuple, np.argwhere(mask)), key=lambda item: float(gain[item])):
        if vertex in template.roots:
            continue
        trial_gain = current_gain - float(gain[vertex])
        if trial_gain < right_hand_side:
            continue
        trial = mask.copy()
        trial[vertex] = False
        if _stroke_constraints_hold(template, trial, retention):
            mask = trial
            current_gain = trial_gain
    return mask


def efficient_linearized_milp(
    template: TextTemplate,
    gain: np.ndarray,
    right_hand_side: float,
    retention: float,
    time_limit: float = 60.0,
) -> np.ndarray | None:
    vertices = list(map(tuple, np.argwhere(template.mask)))
    index = {vertex: position for position, vertex in enumerate(vertices)}
    directed_arcs: list[tuple[int, Vertex, Vertex]] = []
    for stroke_index, stroke in enumerate(template.strokes):
        for start in stroke:
            for end in four_neighbours(start, IMAGE_SIZE, IMAGE_SIZE):
                if end in stroke:
                    directed_arcs.append((stroke_index, start, end))
    number_binary = len(vertices)
    number_variables = number_binary + len(directed_arcs)
    objective = np.zeros(number_variables)
    objective[:number_binary] = 1.0
    lower_bounds = np.zeros(number_variables)
    upper_bounds = np.ones(number_variables)
    for arc_position, (stroke_index, _, _) in enumerate(directed_arcs):
        upper_bounds[number_binary + arc_position] = max(len(template.strokes[stroke_index]) - 1, 1)
    integrality = np.zeros(number_variables, dtype=int)
    integrality[:number_binary] = 1
    rows: list[dict[int, float]] = []
    lower: list[float] = []
    upper: list[float] = []

    rows.append({index[vertex]: float(gain[vertex]) for vertex in vertices})
    lower.append(right_hand_side)
    upper.append(np.inf)

    for stroke, root in zip(template.strokes, template.roots):
        rows.append({index[vertex]: 1.0 for vertex in stroke})
        lower.append(math.ceil(retention * len(stroke)))
        upper.append(np.inf)
        rows.append({index[root]: 1.0})
        lower.append(1.0)
        upper.append(1.0)

    for stroke_index, (stroke, root) in enumerate(zip(template.strokes, template.roots)):
        arc_positions = [
            position
            for position, arc in enumerate(directed_arcs)
            if arc[0] == stroke_index
        ]
        for vertex in stroke:
            row: dict[int, float] = {}
            for position in arc_positions:
                _, start, end = directed_arcs[position]
                variable = number_binary + position
                if end == vertex:
                    row[variable] = row.get(variable, 0.0) + 1.0
                if start == vertex:
                    row[variable] = row.get(variable, 0.0) - 1.0
            if vertex == root:
                row = {key: -value for key, value in row.items()}
                for other in stroke - {root}:
                    row[index[other]] = row.get(index[other], 0.0) - 1.0
            else:
                row[index[vertex]] = row.get(index[vertex], 0.0) - 1.0
            rows.append(row)
            lower.append(0.0)
            upper.append(0.0)
        capacity = max(len(stroke) - 1, 1)
        for position in arc_positions:
            _, start, end = directed_arcs[position]
            flow_variable = number_binary + position
            rows.append({flow_variable: 1.0, index[start]: -capacity})
            lower.append(-np.inf)
            upper.append(0.0)
            rows.append({flow_variable: 1.0, index[end]: -capacity})
            lower.append(-np.inf)
            upper.append(0.0)

    matrix = lil_matrix((len(rows), number_variables), dtype=float)
    for row_index, row in enumerate(rows):
        for column, value in row.items():
            matrix[row_index, column] = value
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(matrix.tocsr(), np.asarray(lower), np.asarray(upper)),
        options={"time_limit": time_limit},
    )
    if result.x is None:
        return None
    mask = np.zeros_like(template.mask)
    selected = result.x[:number_binary] >= 0.5
    for vertex, chosen in zip(vertices, selected):
        mask[vertex] = bool(chosen)
    return mask


def linearized_then_prune(
    context: ImageModelContext,
    template: TextTemplate,
    margin_threshold: float,
    backend: str,
    retention: float = 0.5,
    linearization_iterations: int = 3,
    max_checks: int | None = None,
    milp_time_limit: float = 60.0,
) -> TextAttackResult | None:
    mask = template.mask.copy()
    for _ in range(linearization_iterations):
        reference_margin, gain = linearized_gain(context, mask)
        right_hand_side = (
            margin_threshold
            - reference_margin
            + float(gain[mask].sum())
        )
        if backend == "efficient":
            efficient_linearized_milp(
                template,
                gain,
                right_hand_side,
                retention,
                time_limit=milp_time_limit,
            )
        candidate = manual_linearized_selection(template, gain, right_hand_side, retention)
        if candidate is None:
            return None
        mask = candidate
        if true_margin(context, mask) >= margin_threshold:
            break
    if true_margin(context, mask) < margin_threshold:
        return None
    reduced_template = TextTemplate(template.name, mask, template.strokes, template.roots)
    result = reverse_greedy_pruning(
        context,
        reduced_template,
        margin_threshold,
        retention,
        max_checks,
    )
    if result is not None:
        result.method = f"linearized_milp_{backend}"
    return result


def run_text_batch(
    method: str = "reverse_greedy",
    backend: str = "manual",
    margin_threshold: float = 0.1,
    retention: float = 0.5,
    max_checks: int | None = 200,
    template_limit: int | None = None,
    milp_time_limit: float = 60.0,
    linearization_iterations: int = 3,
    output_root: str | Path = "outputs",
    device: str = "auto",
    reference_labels_path: str | Path | None = None,
    font_sizes: tuple[int, ...] = DEFAULT_FONT_SIZES,
    angles: tuple[int, ...] = DEFAULT_ANGLES,
) -> list[dict[str, object]]:
    model = load_model(device=device)
    reference_labels = load_reference_labels(reference_labels_path)
    templates = candidate_templates(font_sizes, angles)
    target = Path(output_root) / f"text_{method}" / backend
    rows: list[dict[str, object]] = []
    for image_path in sorted(Path("datasets").glob("*.png"), key=lambda path: path.stem):
        context = load_context(image_path, model, reference_labels)
        start = time.perf_counter()
        candidates: list[TextAttackResult] = []
        scored_templates = [
            (template, true_margin(context, template.mask))
            for template in templates
        ]
        ranked_templates = [
            template
            for template, _ in sorted(
                scored_templates,
                key=lambda item: (
                    item[1] < margin_threshold,
                    int(item[0].mask.sum()),
                    -item[1],
                ),
            )
        ]
        if template_limit is not None:
            ranked_templates = ranked_templates[:template_limit]
        for template in ranked_templates:
            if method == "reverse_greedy":
                result = reverse_greedy_pruning(
                    context,
                    template,
                    margin_threshold,
                    retention,
                    max_checks,
                )
            else:
                result = linearized_then_prune(
                    context,
                    template,
                    margin_threshold,
                    backend,
                    retention,
                    linearization_iterations,
                    max_checks=max_checks,
                    milp_time_limit=milp_time_limit,
                )
            if result is not None:
                candidates.append(result)
        if not candidates:
            elapsed = time.perf_counter() - start
            perturbation = torch.zeros_like(context.raw_image)
            decision = context.decision(perturbation)
            image_target = target / image_path.stem
            save_attack_images(context, perturbation, image_target)
            rows.append(
                {
                    "image": image_path.name,
                    "algorithm": f"text_{method}",
                    "backend": backend,
                    "original_class": context.original_class,
                    "original_class_name": context.categories[context.original_class],
                    "decision_class": decision.class_index,
                    "decision_class_name": decision.class_name,
                    "decision_score": decision.score,
                    "decision_function": true_margin(
                        context, np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
                    ),
                    "decision_threshold": margin_threshold,
                    "perturbation_l2": 0.0,
                    "perturbation_threshold": "",
                    "perturbation_area": 0,
                    "successful": False,
                    "elapsed_seconds": elapsed,
                    "iterations": 0,
                    "converged": False,
                    "message": "no feasible text template found",
                }
            )
            continue
        result = min(candidates, key=lambda item: item.area)
        elapsed = time.perf_counter() - start
        decision = context.decision(result.perturbation)
        image_target = target / image_path.stem
        save_attack_images(context, result.perturbation, image_target)
        save_tensor_image(mask_tensor(result.mask, context).float(), image_target / "mask.png")
        rows.append(
            {
                "image": image_path.name,
                "algorithm": f"text_{method}",
                "backend": backend,
                "original_class": context.original_class,
                "original_class_name": context.categories[context.original_class],
                "decision_class": decision.class_index,
                "decision_class_name": decision.class_name,
                "decision_score": decision.score,
                "decision_function": result.margin,
                "decision_threshold": margin_threshold,
                "perturbation_l2": float(torch.linalg.vector_norm(result.perturbation).item()),
                "perturbation_threshold": "",
                "perturbation_area": result.area,
                "successful": decision.class_index != context.original_class,
                "elapsed_seconds": elapsed,
                "iterations": result.area,
                "converged": True,
                "message": result.message,
            }
        )
    write_csv(target / "results.csv", rows)
    return rows
