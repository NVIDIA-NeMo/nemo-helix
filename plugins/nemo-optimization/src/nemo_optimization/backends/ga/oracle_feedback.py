# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Oracle feedback extraction for prompt GA operators."""

from __future__ import annotations

from dataclasses import dataclass

from nemo_optimization.backends.ga.config import GaPromptOptimizerConfig, MetricDirection
from nemo_optimization.backends.ga.individual import GaIndividual
from nemo_optimization.candidate import CandidateEvaluationResult


@dataclass(frozen=True)
class OracleFeedbackState:
    """Latched adaptive oracle-feedback state."""

    adaptive_enabled: bool = False


def should_use_oracle_feedback(
    *,
    config: GaPromptOptimizerConfig,
    parent: GaIndividual,
    state: OracleFeedbackState,
) -> bool:
    """Decide whether a transform should include evaluator reasoning feedback."""

    mode = config.oracle_feedback_mode
    if mode == "never":
        return False
    if mode == "always":
        return True
    if mode == "failing_only":
        return parent.fitness is not None and parent.fitness < config.oracle_feedback_fitness_threshold
    if mode == "adaptive":
        return state.adaptive_enabled
    return False


def adaptive_feedback_triggered(
    *,
    best_fitness_history: list[float],
    population: list[GaIndividual],
    config: GaPromptOptimizerConfig,
) -> bool:
    """Apply NAT's stagnation, variance, and diversity triggers."""

    window = config.oracle_feedback_stagnation_generations
    if len(best_fitness_history) >= window:
        recent = best_fitness_history[-window:]
        if max(recent) - min(recent) < 0.001:
            return True

    fitness_values = [individual.fitness or 0.0 for individual in population]
    if len(fitness_values) > 1:
        mean = sum(fitness_values) / len(fitness_values)
        sample_variance = sum((value - mean) ** 2 for value in fitness_values) / (len(fitness_values) - 1)
        if sample_variance < config.oracle_feedback_fitness_variance_threshold:
            return True

    if population:
        unique_ratio = len({individual.prompt_signature for individual in population}) / len(population)
        if unique_ratio < 1.0 - config.oracle_feedback_diversity_threshold:
            return True
    return False


def build_oracle_feedback(
    *,
    individual: GaIndividual,
    config: GaPromptOptimizerConfig,
) -> str | None:
    """Build compact row-level feedback from evaluator reasoning outputs."""

    if not individual.raw_scores:
        return None

    evaluation = CandidateEvaluationResult(
        aggregate_metrics=dict(individual.aggregate_metrics),
        scores=individual.raw_scores,
    )
    weighted_rows: list[tuple[float, str, str]] = []
    for metric in config.metrics:
        for row in evaluation.reasoning_for_metric(metric.name):
            weight = max(metric.weight, 0.01)
            if metric.direction is MetricDirection.MINIMIZE:
                priority = -row.objective_value * weight
            else:
                priority = row.objective_value / weight
            weighted_rows.append((priority, row.reasoning.strip(), metric.name))

    weighted_rows.sort(key=lambda item: item[0])
    reasoning = [
        f"[{metric_name}] {text}" for _, text, metric_name in weighted_rows[: config.oracle_feedback_worst_n] if text
    ]
    return _truncate_feedback(reasoning, max_chars=config.oracle_feedback_max_chars)


def _truncate_feedback(reasoning: list[str], *, max_chars: int) -> str | None:
    parts: list[str] = []
    current_length = 0
    truncated = False
    for index, text in enumerate(reasoning, 1):
        entry = f"{index}. {text}\n"
        if current_length + len(entry) > max_chars:
            remaining = max_chars - current_length
            if remaining > 20:
                parts.append(entry[: remaining - 3] + "...")
            else:
                truncated = True
            break
        parts.append(entry)
        current_length += len(entry)
    if not parts:
        return None
    result = "".join(parts)
    if truncated and not result.endswith("..."):
        result = result.rstrip("\n") + "...\n"
    return result


__all__ = [
    "OracleFeedbackState",
    "adaptive_feedback_triggered",
    "build_oracle_feedback",
    "should_use_oracle_feedback",
]
