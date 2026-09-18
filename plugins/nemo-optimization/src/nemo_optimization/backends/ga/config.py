# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GA-specific view of the shared optimizer configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nemo_optimization.optimizer_config import (
    MetricDirection,
    MetricSpec,
    OptimizerConfigError,
    parse_optimizer_config,
)
from nemo_optimization.search_space import PromptSearchSpaceSpec

GaConfigError = OptimizerConfigError


@dataclass(frozen=True)
class GaPromptOptimizerConfig:
    backend: str
    model: str
    search_space: dict[str, PromptSearchSpaceSpec]
    metrics: tuple[MetricSpec, ...]
    population_size: int
    generations: int
    crossover_rate: float
    mutation_rate: float
    elitism: int
    selection_method: str
    tournament_size: int
    diversity_lambda: float
    seed: int | None
    target: float | None
    multi_objective_mode: str
    reps_per_param_set: int
    oracle_feedback_mode: str
    oracle_feedback_worst_n: int
    oracle_feedback_max_chars: int
    oracle_feedback_fitness_threshold: float
    oracle_feedback_stagnation_generations: int
    oracle_feedback_fitness_variance_threshold: float
    oracle_feedback_diversity_threshold: float

    @property
    def metric_names(self) -> tuple[str, ...]:
        return tuple(metric.name for metric in self.metrics)


def parse_ga_prompt_optimizer_config(payload: dict[str, Any]) -> GaPromptOptimizerConfig:
    config = parse_optimizer_config(payload)
    prompt = config.prompt
    if prompt is None or not prompt.enabled:
        raise GaConfigError("optimizer.prompt.enabled must be true.", phase="prompt")
    assert prompt.elitism is not None
    return GaPromptOptimizerConfig(
        backend=prompt.backend,
        model=prompt.model,
        search_space=config.prompt_search_space,
        metrics=config.metrics,
        population_size=prompt.population_size,
        generations=prompt.generations,
        crossover_rate=float(prompt.crossover_rate),
        mutation_rate=float(prompt.mutation_rate),
        elitism=prompt.elitism,
        selection_method=prompt.selection_method,
        tournament_size=prompt.tournament_size,
        diversity_lambda=float(prompt.diversity_lambda),
        seed=prompt.seed,
        target=config.target,
        multi_objective_mode=config.multi_objective_mode,
        reps_per_param_set=config.reps_per_param_set,
        oracle_feedback_mode=prompt.oracle_feedback_mode,
        oracle_feedback_worst_n=prompt.oracle_feedback_worst_n,
        oracle_feedback_max_chars=prompt.oracle_feedback_max_chars,
        oracle_feedback_fitness_threshold=float(prompt.oracle_feedback_fitness_threshold),
        oracle_feedback_stagnation_generations=prompt.oracle_feedback_stagnation_generations,
        oracle_feedback_fitness_variance_threshold=float(prompt.oracle_feedback_fitness_variance_threshold),
        oracle_feedback_diversity_threshold=float(prompt.oracle_feedback_diversity_threshold),
    )


__all__ = [
    "GaConfigError",
    "GaPromptOptimizerConfig",
    "MetricDirection",
    "MetricSpec",
    "parse_ga_prompt_optimizer_config",
]
