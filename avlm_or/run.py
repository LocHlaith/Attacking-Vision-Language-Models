from __future__ import annotations

import argparse

from .attacks import AttackParameters, CONTINUOUS_ALGORITHMS
from .experiment import record_originals, run_batch
from .text_attack import DEFAULT_ANGLES, DEFAULT_FONT_SIZES, run_text_batch
from .types import SolverSettings


def comma_separated_integers(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run the operations-research attack experiments.")
    result.add_argument("command", choices=["baseline", "attack", "all-continuous", "text"])
    result.add_argument("--algorithm", choices=CONTINUOUS_ALGORITHMS)
    result.add_argument("--backend", choices=["manual", "efficient"], default="manual")
    result.add_argument("--device", default="auto")
    result.add_argument("--output-root", default="outputs")
    result.add_argument("--reference-labels")
    result.add_argument("--max-iterations", type=int, default=30)
    result.add_argument("--initial-step", type=float, default=0.05)
    result.add_argument("--perturbation-threshold", type=float, default=4.0)
    result.add_argument("--weight", type=float, default=10.0)
    result.add_argument("--loss-threshold", type=float, default=5.0)
    result.add_argument("--target-margin-threshold", type=float, default=0.1)
    result.add_argument("--outer-iterations", type=int, default=8)
    result.add_argument("--restarts", type=int, default=1)
    result.add_argument("--text-method", choices=["reverse_greedy", "linearized_milp"], default="reverse_greedy")
    result.add_argument("--text-margin-threshold", type=float, default=0.1)
    result.add_argument("--retention", type=float, default=0.5)
    result.add_argument("--max-checks", type=int, default=200)
    result.add_argument("--template-limit", type=int)
    result.add_argument("--milp-time-limit", type=float, default=60.0)
    result.add_argument("--linearization-iterations", type=int, default=3)
    result.add_argument("--text-font-sizes", type=comma_separated_integers)
    result.add_argument("--text-angles", type=comma_separated_integers)
    return result


def main() -> None:
    arguments = parser().parse_args()
    if arguments.command == "baseline":
        record_originals(
            arguments.output_root,
            arguments.device,
            arguments.reference_labels,
            arguments.text_font_sizes
            if arguments.text_font_sizes is not None
            else DEFAULT_FONT_SIZES,
            arguments.text_angles
            if arguments.text_angles is not None
            else DEFAULT_ANGLES,
        )
        return
    if arguments.command == "text":
        run_text_batch(
            arguments.text_method,
            arguments.backend,
            arguments.text_margin_threshold,
            arguments.retention,
            arguments.max_checks,
            arguments.template_limit,
            arguments.milp_time_limit,
            arguments.linearization_iterations,
            arguments.output_root,
            arguments.device,
            arguments.reference_labels,
        )
        return
    settings = SolverSettings(
        max_iterations=arguments.max_iterations,
        initial_step=arguments.initial_step,
    )
    parameters = AttackParameters(
        perturbation_threshold=arguments.perturbation_threshold,
        weight=arguments.weight,
        loss_threshold=arguments.loss_threshold,
        target_margin_threshold=arguments.target_margin_threshold,
        outer_iterations=arguments.outer_iterations,
        restarts=arguments.restarts,
    )
    algorithms = (
        CONTINUOUS_ALGORITHMS
        if arguments.command == "all-continuous"
        else [arguments.algorithm]
    )
    if algorithms == [None]:
        raise SystemExit("--algorithm is required for the attack command")
    for algorithm in algorithms:
        run_batch(
            algorithm,
            arguments.backend,
            parameters,
            settings,
            arguments.output_root,
            arguments.device,
            arguments.reference_labels,
        )


if __name__ == "__main__":
    main()
