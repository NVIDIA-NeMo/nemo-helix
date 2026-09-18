# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fitness normalization, scalarization, and ranking for prompt GA."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import pvariance

from nemo_optimization.backends.ga.config import MetricDirection, MetricSpec
from nemo_optimization.backends.ga.individual import GaIndividual

EPSILON = 1e-12


class GaFitnessError(RuntimeError):
    """Raised when GA fitness cannot be computed."""


@dataclass(frozen=True)
class FitnessSnapshot:
    """Aggregate fitness/diversity signals for one generation."""

    valid_count: int
    failed_count: int
    best_fitness: float
    fitness_variance: float
    duplicate_ratio: float


def assign_generation_fitness(
    population: Sequence[GaIndividual],
    *,
    metrics: Sequence[MetricSpec],
    mode: str,
    diversity_lambda: float,
) -> FitnessSnapshot:
    """Normalize completed individuals within one generation and assign higher-is-better fitness."""

    valid = _valid_for_metrics(population, metrics)
    if not valid:
        failed_count = len([individual for individual in population if individual.status == "failed"])
        raise GaFitnessError(
            f"Generation {population[0].generation if population else 0} has no valid individuals "
            f"({failed_count} failed)."
        )

    normalized = _normalized_metric_values(valid, metrics)
    scalar_scores = _scalar_scores(normalized, metrics=metrics, mode=mode)
    penalties = _diversity_penalties(valid, diversity_lambda=diversity_lambda)
    for index, individual in enumerate(valid):
        individual.normalized_metrics = normalized[index]
        individual.fitness = max(0.0, scalar_scores[index] - penalties[index])

    fitness_values = [individual.fitness for individual in valid if individual.fitness is not None]
    return FitnessSnapshot(
        valid_count=len(valid),
        failed_count=len(population) - len(valid),
        best_fitness=max(fitness_values),
        fitness_variance=pvariance(fitness_values) if len(fitness_values) > 1 else 0.0,
        duplicate_ratio=duplicate_ratio(valid),
    )


def rank_valid_individuals(population: Sequence[GaIndividual]) -> list[GaIndividual]:
    """Return selectable individuals best-first."""

    valid = [individual for individual in population if individual.is_valid]
    return sorted(valid, key=_rank_key, reverse=True)


def best_individual(
    individuals: Sequence[GaIndividual],
    *,
    metrics: Sequence[MetricSpec],
    mode: str,
) -> GaIndividual | None:
    """Pick the best completed individual across generations using global normalization."""

    valid = _valid_for_metrics(individuals, metrics)
    if not valid:
        return None
    normalized = _normalized_metric_values(valid, metrics)
    scalar_scores = _scalar_scores(normalized, metrics=metrics, mode=mode)
    ranked = sorted(
        zip(valid, scalar_scores, strict=True),
        key=lambda item: (
            _safe_score(item[1]),
            -_trial_number(item[0]),
            -item[0].generation,
            -item[0].individual_index,
        ),
        reverse=True,
    )
    return ranked[0][0]


def target_met(individual: GaIndividual, *, metrics: Sequence[MetricSpec], target: float | None) -> bool:
    """Return true when a single-objective target has been met."""

    if target is None or len(metrics) != 1:
        return False
    metric = metrics[0]
    if metric.name not in individual.aggregate_metrics:
        return False
    value = individual.aggregate_metrics[metric.name]
    if metric.direction is MetricDirection.MINIMIZE:
        return value <= target
    return value >= target


def duplicate_ratio(individuals: Sequence[GaIndividual]) -> float:
    """Return the share of valid individuals that duplicate another prompt signature."""

    if not individuals:
        return 0.0
    unique_count = len({individual.prompt_signature for individual in individuals})
    return 1.0 - (unique_count / len(individuals))


def _valid_for_metrics(individuals: Sequence[GaIndividual], metrics: Sequence[MetricSpec]) -> list[GaIndividual]:
    metric_names = {metric.name for metric in metrics}
    return [
        individual
        for individual in individuals
        if individual.status == "completed" and metric_names.issubset(individual.aggregate_metrics)
    ]


def _normalized_metric_values(
    individuals: Sequence[GaIndividual],
    metrics: Sequence[MetricSpec],
) -> list[dict[str, float]]:
    by_metric: dict[str, list[float]] = {
        metric.name: [float(individual.aggregate_metrics[metric.name]) for individual in individuals]
        for metric in metrics
    }
    normalized = [{metric.name: 0.0 for metric in metrics} for _ in individuals]
    for metric in metrics:
        values = by_metric[metric.name]
        low = min(values)
        high = max(values)
        if abs(high - low) < EPSILON:
            for row in normalized:
                row[metric.name] = 1.0
            continue
        for index, value in enumerate(values):
            if metric.direction is MetricDirection.MINIMIZE:
                normalized[index][metric.name] = (high - value) / (high - low)
            else:
                normalized[index][metric.name] = (value - low) / (high - low)
    return normalized


def _scalar_scores(
    normalized: Sequence[dict[str, float]],
    *,
    metrics: Sequence[MetricSpec],
    mode: str,
) -> list[float]:
    total_weight = sum(metric.weight for metric in metrics)
    weights = [metric.weight / total_weight for metric in metrics]
    scores: list[float] = []
    for row in normalized:
        values = [max(0.0, min(1.0, float(row[metric.name]))) for metric in metrics]
        if mode == "weighted_sum":
            score = sum(weight * value for weight, value in zip(weights, values, strict=True))
        elif mode == "chebyshev":
            score = 1.0 - max(weight * (1.0 - value) for weight, value in zip(weights, values, strict=True))
        elif mode == "harmonic":
            score = 1.0 / sum(weight / max(value, EPSILON) for weight, value in zip(weights, values, strict=True))
        else:
            raise GaFitnessError(f"Unsupported multi-objective combination mode: {mode!r}")
        scores.append(score)
    return scores


def _diversity_penalties(individuals: Sequence[GaIndividual], *, diversity_lambda: float) -> list[float]:
    if diversity_lambda <= 0 or len(individuals) <= 1:
        return [0.0 for _ in individuals]
    counts = Counter(individual.prompt_signature for individual in individuals)
    denominator = max(1, len(individuals) - 1)
    return [diversity_lambda * ((counts[individual.prompt_signature] - 1) / denominator) for individual in individuals]


def _rank_key(individual: GaIndividual) -> tuple[float, int, int, int]:
    return (
        _safe_score(individual.fitness),
        -_trial_number(individual),
        -individual.generation,
        -individual.individual_index,
    )


def _safe_score(score: float | None) -> float:
    return score if score is not None else float("-inf")


def _trial_number(individual: GaIndividual) -> int:
    if individual.global_trial_number is not None:
        return individual.global_trial_number
    if individual.phase_trial_number is not None:
        return individual.phase_trial_number
    return 1_000_000_000


__all__ = [
    "FitnessSnapshot",
    "GaFitnessError",
    "assign_generation_fitness",
    "best_individual",
    "duplicate_ratio",
    "rank_valid_individuals",
    "target_met",
]
