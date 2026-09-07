# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration shared by numeric and prompt optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictBool, ValidationError, model_validator

from nemo_optimization.search_space import (
    NumericSearchSpaceSpec,
    PromptSearchSpaceSpec,
    SearchSpaceError,
    parse_all_search_space,
    parse_prompt_search_space,
)

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
Rate = Annotated[FiniteFloat, Field(ge=0, le=1)]
NonNegativeFloat = Annotated[FiniteFloat, Field(ge=0)]


class OptimizerConfigError(ValueError):
    """Raised when the shared optimizer configuration is invalid."""

    def __init__(self, message: str, *, phase: str | None = None) -> None:
        super().__init__(message)
        self.phase = phase


class MetricDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: MetricDirection
    weight: float


class _MetricInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluator_name: str | None = None
    direction: MetricDirection = MetricDirection.MAXIMIZE
    weight: Annotated[FiniteFloat, Field(gt=0)] = 1.0


class NumericPhaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: StrictBool
    backend: str = "optuna"
    n_trials: PositiveInt = 20
    sampler: Literal["bayesian", "tpe", "grid"] | None = None


class PromptPhaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: StrictBool
    backend: str = "ga"
    model: str
    population_size: PositiveInt = 10
    generations: PositiveInt = 5
    crossover_rate: Rate = 0.7
    mutation_rate: Rate = 0.1
    elitism: NonNegativeInt | None = None
    selection_method: Literal["tournament", "roulette"] = "tournament"
    tournament_size: PositiveInt = 3
    diversity_lambda: NonNegativeFloat = 0.0
    seed: int | None = None
    oracle_feedback_mode: Literal["never", "always", "failing_only", "adaptive"] = "never"
    oracle_feedback_worst_n: PositiveInt = 5
    oracle_feedback_max_chars: PositiveInt = 4000
    oracle_feedback_fitness_threshold: NonNegativeFloat = 0.3
    oracle_feedback_stagnation_generations: PositiveInt = 3
    oracle_feedback_fitness_variance_threshold: NonNegativeFloat = 0.01
    oracle_feedback_diversity_threshold: NonNegativeFloat = 0.5

    @model_validator(mode="after")
    def _validate_elitism(self) -> PromptPhaseConfig:
        elitism = self.elitism
        if elitism is None:
            elitism = min(2, max(0, self.population_size - 1))
            object.__setattr__(self, "elitism", elitism)
        if elitism >= self.population_size and (self.population_size > 1 or elitism > 0):
            raise ValueError("elitism must be less than population_size")
        return self


class _OptimizerInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    numeric: NumericPhaseConfig | None = None
    prompt: PromptPhaseConfig | None = None
    reps_per_param_set: PositiveInt = 1
    target: FiniteFloat | None = None
    multi_objective_combination_mode: Literal["harmonic", "weighted_sum", "chebyshev"] = "harmonic"
    eval_metrics: dict[str, _MetricInput] = Field(default_factory=dict)
    search_space: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class OptimizerConfig:
    numeric: NumericPhaseConfig | None
    prompt: PromptPhaseConfig | None
    reps_per_param_set: int
    target: float | None
    multi_objective_mode: str
    metrics: tuple[MetricSpec, ...]
    numeric_search_space: dict[str, NumericSearchSpaceSpec]
    prompt_search_space: dict[str, PromptSearchSpaceSpec]


def parse_optimizer_config(payload: dict[str, Any]) -> OptimizerConfig:
    raw = payload.get("optimizer")
    if not isinstance(raw, dict):
        raise OptimizerConfigError("payload.optimizer must be a mapping.")
    parsed = _parse_optimizer_mapping(raw)

    try:
        all_space = parse_all_search_space(raw)
        prompt_space = (
            parse_prompt_search_space(raw, payload=payload) if parsed.prompt and parsed.prompt.enabled else {}
        )
    except SearchSpaceError as exc:
        raise OptimizerConfigError(str(exc), phase=_search_space_phase(raw, str(exc))) from exc
    numeric_space = {name: spec for name, spec in all_space.items() if isinstance(spec, NumericSearchSpaceSpec)}

    if parsed.numeric and parsed.numeric.enabled and not numeric_space:
        raise OptimizerConfigError(
            "optimizer.search_space must declare at least one non-prompt dimension.", phase="numeric"
        )
    if parsed.prompt and parsed.prompt.enabled:
        models = payload.get("models")
        if not isinstance(models, dict) or not isinstance(models.get(parsed.prompt.model), dict):
            raise OptimizerConfigError(
                f"optimizer.prompt.model references unknown payload model {parsed.prompt.model!r}.", phase="prompt"
            )
    if (
        parsed.numeric and parsed.numeric.enabled or parsed.prompt and parsed.prompt.enabled
    ) and not parsed.eval_metrics:
        raise OptimizerConfigError("optimizer.eval_metrics must declare at least one metric.")

    metrics = tuple(
        MetricSpec(
            name=(metric.evaluator_name or name).strip(),
            direction=metric.direction,
            weight=float(metric.weight),
        )
        for name, metric in parsed.eval_metrics.items()
    )
    if any(not metric.name for metric in metrics):
        raise OptimizerConfigError("optimizer.eval_metrics evaluator_name values must be non-empty strings.")

    return OptimizerConfig(
        numeric=parsed.numeric,
        prompt=parsed.prompt,
        reps_per_param_set=parsed.reps_per_param_set,
        target=float(parsed.target) if parsed.target is not None else None,
        multi_objective_mode=parsed.multi_objective_combination_mode,
        metrics=metrics,
        numeric_search_space=numeric_space,
        prompt_search_space=prompt_space,
    )


def _parse_optimizer_mapping(raw: dict[str, Any]) -> _OptimizerInput:
    try:
        return _OptimizerInput.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors(include_url=False)[0]
        location = ".".join(str(part) for part in first["loc"])
        phase = str(first["loc"][0]) if first["loc"] and first["loc"][0] in {"numeric", "prompt"} else None
        raise OptimizerConfigError(f"optimizer.{location}: {first['msg']}", phase=phase) from exc


def _search_space_phase(raw: dict[str, Any], message: str) -> str | None:
    search_space = raw.get("search_space")
    if not isinstance(search_space, dict):
        return None
    for name, value in search_space.items():
        if repr(name) in message and isinstance(value, dict):
            return "prompt" if value.get("is_prompt") is True else "numeric"
    return None


__all__ = [
    "MetricDirection",
    "MetricSpec",
    "NumericPhaseConfig",
    "OptimizerConfig",
    "OptimizerConfigError",
    "PromptPhaseConfig",
    "parse_optimizer_config",
]
