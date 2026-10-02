# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fitness normalization, scalarization, and ranking for prompt GA."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import variance

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
        individual.fitness = scalar_scores[index] - penalties[index]

    fitness_values = [individual.fitness for individual in valid if individual.fitness is not None]
    return FitnessSnapshot(
        valid_count=len(valid),
        failed_count=len(population) - len(valid),
        best_fitness=max(fitness_values),
        fitness_variance=variance(fitness_values) if len(fitness_values) > 1 else 0.0,
        duplicate_ratio=duplicate_ratio(valid),
    )


def rank_valid_individuals(population: Sequence[GaIndividual]) -> list[GaIndividual]:
    """Return selectable individuals best-first."""

    valid = [individual for individual in population if individual.is_valid]
    return sorted(valid, key=lambda individual: individual.fitness or 0.0, reverse=True)


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
                row[metric.name] = 0.5
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
    weights = [metric.weight for metric in metrics]
    scores: list[float] = []
    for row in normalized:
        values = [max(0.0, min(1.0, float(row[metric.name]))) for metric in metrics]
        if mode == "weighted_sum":
            score = sum(weight * value for weight, value in zip(weights, values, strict=True))
        elif mode == "chebyshev":
            score = min(values)
        elif mode == "harmonic":
            score = len(values) / sum(1.0 / max(value, EPSILON) for value in values)
        else:
            raise GaFitnessError(f"Unsupported multi-objective combination mode: {mode!r}")
        scores.append(score)
    return scores


def _diversity_penalties(individuals: Sequence[GaIndividual], *, diversity_lambda: float) -> list[float]:
    if diversity_lambda <= 0 or len(individuals) <= 1:
        return [0.0 for _ in individuals]
    counts = Counter(individual.prompt_signature for individual in individuals)
    return [diversity_lambda * (counts[individual.prompt_signature] - 1) for individual in individuals]


__all__ = [
    "FitnessSnapshot",
    "GaFitnessError",
    "assign_generation_fitness",
    "duplicate_ratio",
    "rank_valid_individuals",
]
