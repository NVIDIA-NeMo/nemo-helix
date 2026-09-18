# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt genetic-algorithm optimizer."""

from __future__ import annotations

import copy
import csv
import json
import logging
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from nemo_optimization.artifact_utils import sanitize_config_for_artifact
from nemo_optimization.backends.ga.config import GaPromptOptimizerConfig, parse_ga_prompt_optimizer_config
from nemo_optimization.backends.ga.fitness import (
    FitnessSnapshot,
    GaFitnessError,
    assign_generation_fitness,
    best_individual,
    rank_valid_individuals,
    target_met,
)
from nemo_optimization.backends.ga.individual import GaIndividual
from nemo_optimization.backends.ga.oracle_feedback import (
    OracleFeedbackState,
    build_oracle_feedback,
    should_use_oracle_feedback,
)
from nemo_optimization.backends.ga.transform import PromptTransformer, PromptTransformError
from nemo_optimization.candidate import CandidateEvaluationError, CandidateEvaluationResult, CandidateEvaluator
from nemo_optimization.config_overlay import apply_suggestions
from nemo_optimization.search_space import (
    PromptSearchSpaceSpec,
    SearchSpaceError,
    initial_prompt_for_spec,
    suggestions_by_path,
    validate_prompt_variables,
)

logger = logging.getLogger(__name__)


class GaPromptOptimizerError(RuntimeError):
    """Raised when prompt GA optimization cannot produce a valid candidate."""

    def __init__(
        self,
        message: str,
        *,
        optimized_payload: dict[str, Any] | None = None,
        trial_count: int = 0,
    ) -> None:
        super().__init__(message)
        self.optimized_payload = optimized_payload
        self.trial_count = trial_count


@dataclass(frozen=True)
class GaPromptOptimizationResult:
    """Completed prompt GA run details."""

    optimized_payload: dict[str, Any]
    best_individual: GaIndividual
    metric_names: tuple[str, ...]
    executed_trials: int
    generations_completed: int
    history: tuple[GaIndividual, ...]
    output_dir: Path


def run_ga_prompt_optimization(
    payload: Mapping[str, Any],
    output_dir: Path,
    evaluator: CandidateEvaluator,
    transformer: PromptTransformer,
    *,
    config: GaPromptOptimizerConfig | None = None,
    trial_number_offset: int = 0,
) -> GaPromptOptimizationResult:
    """Execute prompt GA against the prompt dimensions in ``optimizer.search_space``."""

    base_payload = copy.deepcopy(dict(payload))
    ga_config = config or parse_ga_prompt_optimizer_config(base_payload)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(ga_config.seed)
    next_phase_trial_number = 0
    population = _initial_population(base_payload, config=ga_config, transformer=transformer, rng=rng)
    _fail_if_systematic_transform_failure(
        population,
        generation=0,
        trial_count=next_phase_trial_number,
        optimized_payload=base_payload,
    )
    history: list[GaIndividual] = []
    best_so_far: GaIndividual | None = None
    stagnation_generations = 0
    generations_completed = 0

    for generation in range(ga_config.generations):
        logger.info("Evaluating prompt GA generation %d/%d", generation + 1, ga_config.generations)
        next_phase_trial_number = _evaluate_population(
            base_payload,
            population,
            config=ga_config,
            evaluator=evaluator,
            output_dir=output_dir,
            next_phase_trial_number=next_phase_trial_number,
            trial_number_offset=trial_number_offset,
        )

        try:
            snapshot = assign_generation_fitness(
                population,
                metrics=ga_config.metrics,
                mode=ga_config.multi_objective_mode,
                diversity_lambda=ga_config.diversity_lambda,
            )
        except GaFitnessError as exc:
            optimized_payload = _optimized_payload_for_best(base_payload, best_so_far, ga_config)
            _write_history_artifacts(output_dir, history=(*history, *population))
            raise GaPromptOptimizerError(
                str(exc),
                optimized_payload=optimized_payload,
                trial_count=next_phase_trial_number,
            ) from exc

        history.extend(copy.deepcopy(population))
        previous_best_id = best_so_far.individual_id if best_so_far is not None else None
        best_so_far = best_individual(
            history,
            metrics=ga_config.metrics,
            mode=ga_config.multi_objective_mode,
        )
        if best_so_far is None:
            message = "Prompt GA completed a generation but no valid best individual was available."
            raise GaPromptOptimizerError(
                message,
                optimized_payload=copy.deepcopy(base_payload),
                trial_count=next_phase_trial_number,
            )

        if previous_best_id is not None and best_so_far.individual_id == previous_best_id:
            stagnation_generations += 1
        else:
            stagnation_generations = 0
        generations_completed = generation + 1
        _write_generation_artifacts(
            output_dir,
            generation=generation,
            population=population,
            snapshot=snapshot,
            best=best_so_far,
            config=ga_config,
        )

        if target_met(best_so_far, metrics=ga_config.metrics, target=ga_config.target):
            logger.info("Prompt GA stopped early after meeting target")
            break
        if generation == ga_config.generations - 1:
            break

        population = _next_generation(
            population,
            generation=generation + 1,
            config=ga_config,
            transformer=transformer,
            rng=rng,
            oracle_state=OracleFeedbackState(
                stagnation_generations=stagnation_generations,
                fitness_variance=snapshot.fitness_variance,
                duplicate_ratio=snapshot.duplicate_ratio,
            ),
        )
        _fail_if_systematic_transform_failure(
            population,
            generation=generation + 1,
            trial_count=next_phase_trial_number,
            optimized_payload=_optimized_payload_for_best(base_payload, best_so_far, ga_config),
        )

    if best_so_far is None:
        message = "Prompt GA did not produce a valid individual."
        raise GaPromptOptimizerError(
            message,
            optimized_payload=copy.deepcopy(base_payload),
            trial_count=next_phase_trial_number,
        )

    optimized_payload = _optimized_payload_for_best(base_payload, best_so_far, ga_config)
    _write_final_artifacts(
        output_dir,
        base_payload=base_payload,
        best=best_so_far,
        config=ga_config,
        history=history,
    )
    return GaPromptOptimizationResult(
        optimized_payload=optimized_payload,
        best_individual=copy.deepcopy(best_so_far),
        metric_names=ga_config.metric_names,
        executed_trials=next_phase_trial_number,
        generations_completed=generations_completed,
        history=tuple(history),
        output_dir=output_dir,
    )


def _initial_population(
    payload: Mapping[str, Any],
    *,
    config: GaPromptOptimizerConfig,
    transformer: PromptTransformer,
    rng: random.Random,
) -> list[GaIndividual]:
    prompts = {
        name: _initial_prompt(payload, name=name, spec=prompt_spec)
        for name, prompt_spec in sorted(config.search_space.items())
    }
    population = [GaIndividual(prompts=copy.deepcopy(prompts), generation=0, individual_index=0)]
    for individual_index in range(1, config.population_size):
        mutated_prompts: dict[str, str] = {}
        failures: list[str] = []
        successes: list[str] = []
        for name, spec in sorted(config.search_space.items()):
            prompt = prompts[name]
            mutated_prompts[name] = _mutate_prompt(
                transformer,
                prompt_name=name,
                prompt=prompt,
                spec=spec,
                feedback=None,
                failures=failures,
                successes=successes,
                rng=rng,
            )
        population.append(
            GaIndividual(
                prompts=mutated_prompts,
                generation=0,
                individual_index=individual_index,
                parent_ids=(population[0].individual_id,),
                transform_failures=failures,
                transform_successes=successes,
            )
        )
    return population


def _next_generation(
    population: Sequence[GaIndividual],
    *,
    generation: int,
    config: GaPromptOptimizerConfig,
    transformer: PromptTransformer,
    rng: random.Random,
    oracle_state: OracleFeedbackState,
) -> list[GaIndividual]:
    parents = rank_valid_individuals(population)
    if not parents:
        raise GaPromptOptimizerError("Cannot create next generation without a valid parent.")

    next_population: list[GaIndividual] = [
        elite.clone_as_elite(generation=generation, individual_index=index)
        for index, elite in enumerate(parents[: config.elitism])
    ]
    while len(next_population) < config.population_size:
        parent_a = _select_parent(parents, config=config, rng=rng)
        parent_b = _select_parent(parents, config=config, rng=rng)
        if parent_b.individual_id == parent_a.individual_id and len(parents) > 1:
            alternatives = [parent for parent in parents if parent.individual_id != parent_a.individual_id]
            parent_b = rng.choice(alternatives)
        child_prompts: dict[str, str] = {}
        failures: list[str] = []
        successes: list[str] = []
        for name, spec in sorted(config.search_space.items()):
            prompt = _child_prompt(
                transformer,
                name=name,
                spec=spec,
                parent_a=parent_a,
                parent_b=parent_b,
                config=config,
                rng=rng,
                oracle_state=oracle_state,
                failures=failures,
                successes=successes,
            )
            child_prompts[name] = prompt
        next_population.append(
            GaIndividual(
                prompts=child_prompts,
                generation=generation,
                individual_index=len(next_population),
                parent_ids=(parent_a.individual_id, parent_b.individual_id),
                transform_failures=failures,
                transform_successes=successes,
            )
        )
    return next_population


def _child_prompt(
    transformer: PromptTransformer,
    *,
    name: str,
    spec: PromptSearchSpaceSpec,
    parent_a: GaIndividual,
    parent_b: GaIndividual,
    config: GaPromptOptimizerConfig,
    rng: random.Random,
    oracle_state: OracleFeedbackState,
    failures: list[str],
    successes: list[str],
) -> str:
    parent_prompt = parent_a.prompts[name]
    prompt_sources = (parent_a,)
    if rng.random() < config.crossover_rate and parent_a.individual_id != parent_b.individual_id:
        feedback = _feedback_for_sources((parent_a, parent_b), config=config, state=oracle_state)
        try:
            parent_prompt = transformer.recombine(
                prompt_name=name,
                parent_a=parent_a.prompts[name],
                parent_b=parent_b.prompts[name],
                purpose=spec.purpose,
                prompt_format=spec.format,
                feedback=feedback,
            )
            _validate_transform_output(name, parent_prompt, spec=spec)
            successes.append(f"recombine:{name}:{parent_a.individual_id},{parent_b.individual_id}")
            prompt_sources = (parent_a, parent_b)
        except PromptTransformError as exc:
            fallback_parent = rng.choice([parent_a, parent_b])
            parent_prompt = fallback_parent.prompts[name]
            prompt_sources = (fallback_parent,)
            failures.append(
                f"recombine:{name}:{parent_a.individual_id},{parent_b.individual_id}:"
                f"{exc}; used {fallback_parent.individual_id}"
            )
    elif rng.random() < 0.5:
        parent_prompt = parent_b.prompts[name]
        prompt_sources = (parent_b,)

    if rng.random() < config.mutation_rate:
        feedback = _feedback_for_sources(prompt_sources, config=config, state=oracle_state)
        parent_prompt = _mutate_prompt(
            transformer,
            prompt_name=name,
            prompt=parent_prompt,
            spec=spec,
            feedback=feedback,
            failures=failures,
            successes=successes,
            rng=rng,
        )
    return parent_prompt


def _mutate_prompt(
    transformer: PromptTransformer,
    *,
    prompt_name: str,
    prompt: str,
    spec: PromptSearchSpaceSpec,
    feedback: str | None,
    failures: list[str],
    successes: list[str],
    rng: random.Random,
) -> str:
    del rng
    try:
        mutated = transformer.mutate(
            prompt_name=prompt_name,
            prompt=prompt,
            purpose=spec.purpose,
            prompt_format=spec.format,
            feedback=feedback,
        )
        _validate_transform_output(prompt_name, mutated, spec=spec)
        successes.append(f"mutate:{prompt_name}")
        return mutated
    except PromptTransformError as exc:
        failures.append(f"mutate:{prompt_name}:{exc}; retained parent prompt")
        return prompt


def _validate_transform_output(prompt_name: str, prompt: str, *, spec: PromptSearchSpaceSpec) -> None:
    try:
        validate_prompt_variables(
            prompt_name=prompt_name,
            prompt=prompt,
            required_variables=spec.required_variables,
            context="transformed prompt",
        )
    except SearchSpaceError as exc:
        raise PromptTransformError(str(exc)) from exc


def _fail_if_systematic_transform_failure(
    population: Sequence[GaIndividual],
    *,
    generation: int,
    trial_count: int,
    optimized_payload: Mapping[str, Any],
) -> None:
    attempted = [
        individual
        for individual in population
        if individual.carried_from is None and (individual.transform_failures or individual.transform_successes)
    ]
    if not attempted or any(individual.transform_successes for individual in attempted):
        return
    message = f"Prompt GA model transforms failed for every generated individual in generation {generation}."
    raise GaPromptOptimizerError(
        message,
        optimized_payload=copy.deepcopy(dict(optimized_payload)),
        trial_count=trial_count,
    )


def _select_parent(
    parents: Sequence[GaIndividual],
    *,
    config: GaPromptOptimizerConfig,
    rng: random.Random,
) -> GaIndividual:
    if config.selection_method == "tournament":
        contenders = rng.sample(list(parents), k=min(config.tournament_size, len(parents)))
        return rank_valid_individuals(contenders)[0]

    positive_weights = [max(0.0, parent.fitness or 0.0) for parent in parents]
    total = sum(positive_weights)
    if total <= 0:
        return rng.choice(list(parents))
    needle = rng.random() * total
    cumulative = 0.0
    for parent, weight in zip(parents, positive_weights, strict=True):
        cumulative += weight
        if cumulative >= needle:
            return parent
    return parents[-1]


def _evaluate_population(
    payload: Mapping[str, Any],
    population: Sequence[GaIndividual],
    *,
    config: GaPromptOptimizerConfig,
    evaluator: CandidateEvaluator,
    output_dir: Path,
    next_phase_trial_number: int,
    trial_number_offset: int,
) -> int:
    pending = [
        individual
        for individual in population
        if not (individual.status == "completed" and individual.aggregate_metrics)
    ]
    for individual in pending:
        individual.phase_trial_number = next_phase_trial_number
        individual.global_trial_number = trial_number_offset + next_phase_trial_number
        next_phase_trial_number += 1

    for individual in pending:
        _evaluate_assigned_individual(
            payload,
            individual,
            config=config,
            evaluator=evaluator,
            output_dir=output_dir,
        )
    return next_phase_trial_number


def _evaluate_assigned_individual(
    payload: Mapping[str, Any],
    individual: GaIndividual,
    *,
    config: GaPromptOptimizerConfig,
    evaluator: CandidateEvaluator,
    output_dir: Path,
) -> None:
    if individual.global_trial_number is None:
        raise GaPromptOptimizerError("Prompt GA individual was evaluated before trial numbering.")
    path_suggestions = suggestions_by_path(config.search_space, individual.prompts)
    write_candidate_config(
        output_dir,
        trial_number=individual.global_trial_number,
        trial_config=apply_suggestions(payload, path_suggestions),
    )
    try:
        rep_results = [
            evaluator.evaluate(
                trial_number=individual.global_trial_number,
                suggestions=dict(path_suggestions),
                trial_overlay=_trial_overlay(individual),
                rep=rep,
            )
            for rep in range(config.reps_per_param_set)
        ]
        individual.aggregate_metrics = _average_rep_metrics(rep_results, config.metric_names)
        individual.raw_scores = tuple(score for result in rep_results for score in result.scores)
        individual.status = "completed"
        individual.failure_reason = None
    except CandidateEvaluationError as exc:
        individual.status = "failed"
        individual.failure_reason = str(exc)
        individual.aggregate_metrics = {}
        individual.raw_scores = ()
        individual.fitness = None
        individual.normalized_metrics = {}


def _average_rep_metrics(
    rep_results: Sequence[CandidateEvaluationResult],
    metric_names: Sequence[str],
) -> dict[str, float]:
    if not rep_results:
        raise CandidateEvaluationError("Cannot average prompt GA metrics from zero repetitions.")
    averaged: dict[str, float] = {}
    for metric_name in metric_names:
        values: list[float] = []
        for rep_index, result in enumerate(rep_results):
            if metric_name not in result.aggregate_metrics:
                raise CandidateEvaluationError(f"Prompt GA rep {rep_index} missing metric {metric_name!r}.")
            try:
                value = float(result.aggregate_metrics[metric_name])
            except (TypeError, ValueError) as exc:
                raise CandidateEvaluationError(
                    f"Prompt GA rep {rep_index} metric {metric_name!r} must be numeric."
                ) from exc
            if not math.isfinite(value):
                raise CandidateEvaluationError(f"Prompt GA rep {rep_index} metric {metric_name!r} must be finite.")
            values.append(value)
        averaged[metric_name] = sum(values) / len(values)
    return averaged


def _trial_overlay(individual: GaIndividual) -> dict[str, Any]:
    return {
        "metadata": {
            "nemo.optimizer.phase": "prompt",
            "nemo.optimizer.phase_trial_number": individual.phase_trial_number,
            "nemo.optimizer.global_trial_number": individual.global_trial_number,
            "nemo.optimizer.generation": individual.generation,
            "nemo.optimizer.individual_index": individual.individual_index,
        }
    }


def _feedback_for_sources(
    sources: Sequence[GaIndividual],
    *,
    config: GaPromptOptimizerConfig,
    state: OracleFeedbackState,
) -> str | None:
    sections: list[str] = []
    seen: set[str] = set()
    for source in sources:
        if source.individual_id in seen:
            continue
        seen.add(source.individual_id)
        if not should_use_oracle_feedback(config=config, parent=source, state=state):
            continue
        feedback = build_oracle_feedback(individual=source, config=config)
        if feedback:
            sections.append(f"Feedback from {source.individual_id}:\n{feedback}")
    if not sections:
        return None
    return "\n\n".join(sections)


def _initial_prompt(payload: Mapping[str, Any], *, name: str, spec: PromptSearchSpaceSpec) -> str:
    try:
        return initial_prompt_for_spec(payload, name=name, spec=spec)
    except SearchSpaceError as exc:
        raise GaPromptOptimizerError(str(exc)) from exc


def _optimized_payload_for_best(
    payload: Mapping[str, Any],
    best: GaIndividual | None,
    config: GaPromptOptimizerConfig,
) -> dict[str, Any]:
    if best is None:
        return copy.deepcopy(dict(payload))
    return apply_suggestions(
        payload,
        suggestions_by_path(config.search_space, best.prompts),
        strip_optimizer=False,
    )


def write_candidate_config(
    output_dir: Path,
    *,
    trial_number: int,
    trial_config: Mapping[str, Any],
) -> Path:
    path = output_dir / f"config_prompt_trial_{trial_number:03d}.yml"
    path.write_text(
        yaml.safe_dump(sanitize_config_for_artifact(trial_config), sort_keys=False),
        encoding="utf-8",
    )
    return path


def _write_generation_artifacts(
    output_dir: Path,
    *,
    generation: int,
    population: Sequence[GaIndividual],
    snapshot: FitnessSnapshot,
    best: GaIndividual,
    config: GaPromptOptimizerConfig,
) -> None:
    payload = {
        "generation": generation,
        "best_individual": _individual_payload(best),
        "fitness": {
            "valid_count": snapshot.valid_count,
            "failed_count": snapshot.failed_count,
            "best_fitness": snapshot.best_fitness,
            "fitness_variance": snapshot.fitness_variance,
            "duplicate_ratio": snapshot.duplicate_ratio,
        },
        "population": [_individual_payload(individual) for individual in population],
        "prompt_paths": {name: spec.path for name, spec in sorted(config.search_space.items())},
    }
    (output_dir / f"optimized_prompts_gen{generation}.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _write_final_artifacts(
    output_dir: Path,
    *,
    base_payload: Mapping[str, Any],
    best: GaIndividual,
    config: GaPromptOptimizerConfig,
    history: Sequence[GaIndividual],
) -> None:
    optimized_config = apply_suggestions(
        base_payload,
        suggestions_by_path(config.search_space, best.prompts),
    )
    (output_dir / "optimized_config.yml").write_text(
        yaml.safe_dump(sanitize_config_for_artifact(optimized_config), sort_keys=False),
        encoding="utf-8",
    )
    (output_dir / "optimized_prompts.json").write_text(
        json.dumps(
            {
                "best_individual": _individual_payload(best),
                "prompts": dict(best.prompts),
                "prompt_paths": {name: spec.path for name, spec in sorted(config.search_space.items())},
                "metrics": dict(best.aggregate_metrics),
                "fitness": best.fitness,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_history_artifacts(output_dir, history=history)


def _write_history_artifacts(output_dir: Path, *, history: Sequence[GaIndividual]) -> None:
    rows = [individual.to_history_row() for individual in history]
    (output_dir / "ga_history_prompts.json").write_text(
        json.dumps(rows, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (output_dir / "ga_score_records.json").write_text(
        json.dumps(_score_record_rows(history), indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    csv_path = output_dir / "ga_history_prompts.csv"
    fieldnames = _history_fieldnames(rows)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _history_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    prefix = [
        "individual_id",
        "phase",
        "generation",
        "individual_index",
        "phase_trial_number",
        "global_trial_number",
        "status",
        "fitness",
        "failure_reason",
        "parent_ids",
        "carried_from",
        "transform_failures",
        "transform_successes",
    ]
    fields = set().union(*(row.keys() for row in rows)) if rows else set()
    dynamic = sorted(field for field in fields if field not in prefix)
    return [*prefix, *dynamic]


def _score_record_rows(history: Sequence[GaIndividual]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for individual in history:
        if individual.carried_from is not None:
            continue
        metadata = {
            "individual_id": individual.individual_id,
            "phase": "prompt",
            "generation": individual.generation,
            "individual_index": individual.individual_index,
            "phase_trial_number": individual.phase_trial_number,
            "global_trial_number": individual.global_trial_number,
        }
        for score in individual.raw_scores:
            rows.append({**metadata, "score": score.model_dump(mode="json")})
    return rows


def _individual_payload(individual: GaIndividual) -> dict[str, Any]:
    return {
        "individual_id": individual.individual_id,
        "generation": individual.generation,
        "individual_index": individual.individual_index,
        "phase_trial_number": individual.phase_trial_number,
        "global_trial_number": individual.global_trial_number,
        "status": individual.status,
        "fitness": individual.fitness,
        "aggregate_metrics": dict(individual.aggregate_metrics),
        "normalized_metrics": dict(individual.normalized_metrics),
        "parent_ids": list(individual.parent_ids),
        "carried_from": individual.carried_from,
        "failure_reason": individual.failure_reason,
        "transform_failures": list(individual.transform_failures),
        "transform_successes": list(individual.transform_successes),
        "prompts": dict(individual.prompts),
    }


__all__ = [
    "GaPromptOptimizationResult",
    "GaPromptOptimizerError",
    "run_ga_prompt_optimization",
    "write_candidate_config",
]
