from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import CubicSpline

from .model import IMAGE_SIZE
from .types import IterationRecord

PALETTE = ("#692F7C", "#B43970", "#D96558", "#EFA143", "#F6C63C")
MARKERS = ("o", "s", "^", "D", "v", "P", "X")
TIMES_FONT = Path("C:/Windows/Fonts/times.ttf")
TIMES_BOLD_FONT = Path("C:/Windows/Fonts/timesbd.ttf")


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))


def _rgb_to_hex(color: Iterable[float]) -> str:
    channels = [int(np.clip(round(value), 0, 255)) for value in color]
    return "#{:02X}{:02X}{:02X}".format(*channels)


def interpolated_palette(count: int) -> list[str]:
    if count <= 0:
        return []
    if count <= len(PALETTE):
        return list(PALETTE[:count])
    base_x = np.linspace(0.0, 1.0, len(PALETTE))
    target_x = np.linspace(0.0, 1.0, count)
    base_rgb = np.asarray([_hex_to_rgb(color) for color in PALETTE], dtype=float)
    channels = [
        CubicSpline(base_x, base_rgb[:, channel])(target_x)
        for channel in range(3)
    ]
    return [_rgb_to_hex(color) for color in np.stack(channels, axis=1)]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = TIMES_BOLD_FONT if bold else TIMES_FONT
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    return plt


def _normalized(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    if not finite.any():
        return [np.nan for _ in values]
    minimum = float(array[finite].min())
    maximum = float(array[finite].max())
    if abs(maximum - minimum) <= 1e-12:
        return [0.5 if np.isfinite(value) else np.nan for value in array]
    result = (array - minimum) / (maximum - minimum)
    return [float(value) if np.isfinite(value) else np.nan for value in result]


def _save_normalized_line_plot(
    rows: list[dict[str, float]],
    path: str | Path,
    labels: dict[str, str],
    x_key: str,
    x_label: str,
    y_label: str = "normalized value",
) -> None:
    if not rows or not labels:
        return
    plt = _configure_matplotlib()
    target = _ensure_parent(path)
    fig, axis = plt.subplots(figsize=(8.0, 6.0), dpi=180)
    axis.set_box_aspect(3 / 4)
    colors = interpolated_palette(len(labels))
    x_values = [float(row[x_key]) for row in rows]
    plotted = False
    marker_step = max(1, len(x_values) // 18)
    for index, (color, (key, label)) in enumerate(zip(colors, labels.items())):
        raw_values = [float(row.get(key, np.nan)) for row in rows]
        if not np.isfinite(raw_values).any():
            continue
        y_values = _normalized(raw_values)
        axis.plot(
            x_values,
            y_values,
            color=color,
            linewidth=2.0,
            label=label,
            marker=MARKERS[index % len(MARKERS)],
            markersize=6.0 if len(x_values) == 1 else 3.8,
            markerfacecolor="none",
            markeredgewidth=1.2,
            markevery=marker_step,
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    finite_x = np.asarray([value for value in x_values if np.isfinite(value)], dtype=float)
    if finite_x.size and abs(float(finite_x.max() - finite_x.min())) <= 1e-12:
        center = float(finite_x[0])
        pad = max(0.5, abs(center) * 0.05)
        axis.set_xlim(center - pad, center + pad)
    axis.set_ylim(-0.05, 1.05)
    axis.grid(True, color="#DDDDDD", linewidth=0.6)
    axis.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)


def save_solver_history(records: list[IterationRecord], path: str | Path) -> None:
    rows = [
        {
            "index": float(index),
            "objective": record.objective,
            "gradient_norm": record.gradient_norm,
            "step_norm": record.step_norm,
            "step_size": record.step_size,
        }
        for index, record in enumerate(records)
    ]
    _save_normalized_line_plot(
        rows,
        path,
        {
            "objective": r"$f_k$",
            "gradient_norm": r"$\|g_k\|_2$",
            "step_norm": r"$\|\Delta x_k\|_2$",
            "step_size": r"$\alpha_k$",
        },
        "index",
        r"$k$",
    )


def save_outer_trace(rows: list[dict[str, float]], path: str | Path) -> None:
    if not rows:
        return
    candidates = {
        "multiplier": r"$\lambda_k$",
        "penalty": r"$M_k$",
        "measure": r"$m(\delta_k)$",
        "violation": r"$v_k$",
        "norm": r"$\|\delta_k\|_2$",
        "best_norm": r"$\|\delta_k^*\|_2$",
    }
    labels = {
        key: label
        for key, label in candidates.items()
        if any(key in row for row in rows)
    }
    _save_normalized_line_plot(rows, path, labels, "outer", r"$k$")


def save_linearization_trace(rows: list[dict[str, float]], path: str | Path) -> None:
    if not rows:
        return
    _save_normalized_line_plot(
        rows,
        path,
        {
            "area": "area",
            "margin": "true margin",
            "reference_margin": "linear point",
            "right_hand_side": "right-hand side",
        },
        "iteration",
        r"$q$",
    )


def save_pruning_trace(rows: list[dict[str, float]], path: str | Path) -> None:
    if not rows:
        return
    _save_normalized_line_plot(
        rows,
        path,
        {
            "area": "area",
            "margin": "true margin",
        },
        "check",
        r"$k$",
    )


def _mask_center(mask: np.ndarray) -> tuple[int, int]:
    points = np.argwhere(mask)
    if len(points) == 0:
        return IMAGE_SIZE // 2, IMAGE_SIZE // 2
    row, column = points.mean(axis=0)
    return int(column), int(row)


def save_template_trials(
    trials: list[dict[str, Any]],
    path: str | Path,
) -> None:
    if not trials:
        return
    target = _ensure_parent(path)
    canvas = Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE), (255, 255, 255, 255))
    colors = [_hex_to_rgb(color) for color in interpolated_palette(len(trials))]
    centers: list[tuple[int, int]] = []
    for trial, color in zip(trials, colors):
        mask = trial["mask"]
        centers.append(_mask_center(mask))
        alpha = 34
        if trial.get("status") == "feasible":
            alpha = 72
        if trial.get("status") == "chosen":
            alpha = 155
        overlay = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 4), dtype=np.uint8)
        overlay[mask] = (*color, alpha)
        canvas = Image.alpha_composite(canvas, Image.fromarray(overlay, mode="RGBA"))

    draw = ImageDraw.Draw(canvas, "RGBA")
    for index, (start, end) in enumerate(zip(centers, centers[1:])):
        color = colors[min(index + 1, len(colors) - 1)]
        draw.line([start, end], fill=(*color, 115), width=1)
    for index in np.linspace(0, len(centers) - 1, min(12, len(centers)), dtype=int):
        color = colors[index]
        center = centers[index]
        radius = 2 if trials[index].get("status") != "chosen" else 4
        draw.ellipse(
            (
                center[0] - radius,
                center[1] - radius,
                center[0] + radius,
                center[1] + radius,
            ),
            fill=(*color, 230),
        )

    label_font = _font(12, bold=True)
    first = centers[0]
    last = centers[-1]
    draw.text((first[0] + 4, first[1] + 4), "1", fill=(*colors[0], 255), font=label_font)
    draw.text(
        (last[0] + 4, last[1] + 4),
        str(len(centers)),
        fill=(*colors[-1], 255),
        font=label_font,
    )
    for index, trial in enumerate(trials):
        if trial.get("status") == "chosen":
            center = centers[index]
            draw.text(
                (center[0] + 5, center[1] - 16),
                "best",
                fill=(*colors[index], 255),
                font=label_font,
            )
            break
    canvas.convert("RGB").save(target)
