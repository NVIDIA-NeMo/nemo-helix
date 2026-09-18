# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Oracle feedback extraction for prompt GA operators."""

from __future__ import annotations

from dataclasses import dataclass

from nemo_optimization.backends.ga.config import GaPromptOptimizerConfig, MetricDirection, MetricSpec
from nemo_optimization.backends.ga.individual import GaIndividual
from nemo_optimization.candidate import CandidateEvaluationResult, RowReasoning


@dataclass(frozen=True)
class OracleFeedbackState:
    """Generation-level signals for adaptive oracle feedback."""

    stagnation_generations: int
    fitness_variance: float
    duplicate_ratio: float


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
        return (
            state.stagnation_generations >= config.oracle_feedback_stagnation_generations
            or state.fitness_variance <= config.oracle_feedback_fitness_variance_threshold
            or state.duplicate_ratio >= config.oracle_feedback_diversity_threshold
        )
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
    sections: list[str] = []
    weighted_metrics = sorted(config.metrics, key=lambda metric: (metric.weight, metric.name), reverse=True)
    total_weight = sum(metric.weight for metric in weighted_metrics)
    remaining_chars = config.oracle_feedback_max_chars
    for metric in weighted_metrics:
        section = _metric_feedback_section(evaluation, individual=individual, metric=metric, config=config)
        if not section:
            continue
        separator_chars = 2 if sections else 0
        available_chars = remaining_chars - separator_chars
        if available_chars <= 0:
            break
        budget = int(config.oracle_feedback_max_chars * (metric.weight / total_weight))
        budget = max(1, min(budget, available_chars))
        sections.append(section[:budget])
        remaining_chars -= separator_chars + len(sections[-1])

    feedback = "\n\n".join(sections).strip()
    if not feedback:
        return None
    return feedback


def _metric_feedback_section(
    evaluation: CandidateEvaluationResult,
    *,
    individual: GaIndividual,
    metric: MetricSpec,
    config: GaPromptOptimizerConfig,
) -> str | None:
    if metric.name not in individual.aggregate_metrics:
        return None
    reasoning_rows = list(evaluation.reasoning_for_metric(metric.name))
    if not reasoning_rows:
        return None
    rows = _worst_rows(reasoning_rows, direction=metric.direction, limit=config.oracle_feedback_worst_n)
    lines = [
        f"Metric {metric.name} ({metric.direction.value}, weight={metric.weight:.6g}); "
        f"aggregate={individual.aggregate_metrics[metric.name]:.6g}:"
    ]
    for row in rows:
        lines.append(f"- row={row.task_id} score={row.objective_value:.6g} reasoning={row.reasoning.strip()}")
    return "\n".join(lines)


def _worst_rows(
    rows: list[RowReasoning],
    *,
    direction: MetricDirection,
    limit: int,
) -> list[RowReasoning]:
    reverse = direction is MetricDirection.MINIMIZE
    return sorted(rows, key=lambda row: row.objective_value, reverse=reverse)[:limit]


__all__ = [
    "OracleFeedbackState",
    "build_oracle_feedback",
    "should_use_oracle_feedback",
]
