# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from nemo_evaluator_sdk.agent_eval.scores import AgentEvalScoreStatus, AgentEvalTaskScore
from nemo_evaluator_sdk.metrics.protocol import MetricOutput
from nemo_optimization.backends.ga.config import parse_ga_prompt_optimizer_config
from nemo_optimization.backends.ga.driver import GaPromptOptimizerError, _child_prompt, run_ga_prompt_optimization
from nemo_optimization.backends.ga.fitness import assign_generation_fitness, rank_valid_individuals
from nemo_optimization.backends.ga.individual import GaIndividual
from nemo_optimization.backends.ga.oracle_feedback import OracleFeedbackState, build_oracle_feedback
from nemo_optimization.backends.ga.transform import PromptTransformError
from nemo_optimization.candidate import CandidateEvaluationError, CandidateEvaluationResult


def test_ga_prompt_optimizer_runs_prompt_only_and_writes_artifacts(tmp_path: Path) -> None:
    payload = _payload(
        prompt={
            "population_size": 3,
            "generations": 2,
            "mutation_rate": 1.0,
            "crossover_rate": 1.0,
            "elitism": 1,
            "seed": 11,
        }
    )
    evaluator = RecordingEvaluator()

    result = run_ga_prompt_optimization(
        payload,
        tmp_path,
        evaluator,
        DeterministicTransformer(),
        trial_number_offset=10,
    )

    assert result.executed_trials == 5
    assert result.generations_completed == 2
    assert result.best_individual.global_trial_number in {10, 11, 12, 13, 14}
    assert "[system_prompt:" in result.optimized_payload["instructions"]["system"]["content"]
    assert "optimizer" in result.optimized_payload
    assert (tmp_path / "optimized_config.yml").is_file()
    assert (tmp_path / "optimized_prompts.json").is_file()
    assert (tmp_path / "optimized_prompts_gen0.json").is_file()
    assert (tmp_path / "optimized_prompts_gen1.json").is_file()
    assert (tmp_path / "ga_history_prompts.csv").is_file()
    assert (tmp_path / "ga_score_records.json").is_file()

    assert [call["metadata"]["nemo.optimizer.global_trial_number"] for call in evaluator.calls] == [
        10,
        11,
        12,
        13,
        14,
    ]
    assert {call["metadata"]["nemo.optimizer.phase"] for call in evaluator.calls} == {"prompt"}
    assert any(individual.carried_from for individual in result.history)


def test_ga_transform_failures_abort_when_no_variation_succeeds(tmp_path: Path) -> None:
    payload = _payload(
        prompt={
            "population_size": 3,
            "generations": 2,
            "mutation_rate": 0.0,
            "crossover_rate": 1.0,
            "elitism": 0,
            "seed": 7,
        }
    )

    with pytest.raises(GaPromptOptimizerError, match="failed for every generated individual") as exc_info:
        run_ga_prompt_optimization(
            payload,
            tmp_path,
            RecordingEvaluator(),
            FailingTransformer(),
        )

    assert exc_info.value.trial_count == 0
    assert not (tmp_path / "prompt_phase_failure.json").exists()


def test_ga_uses_explicit_initial_prompt_and_preserves_required_variables(tmp_path: Path) -> None:
    payload = _payload(prompt={"population_size": 2, "generations": 1})
    payload["instructions"]["system"]["content"] = "Payload prompt {question}."
    payload["optimizer"]["search_space"]["system_prompt"]["initial_prompt"] = "Seed prompt {question}."
    payload["optimizer"]["search_space"]["system_prompt"]["required_variables"] = ["question"]

    result = run_ga_prompt_optimization(
        payload,
        tmp_path,
        RecordingEvaluator(),
        DeterministicTransformer(),
    )

    assert "Seed prompt {question}." in result.optimized_payload["instructions"]["system"]["content"]


def test_ga_rejects_transform_that_drops_required_variables(tmp_path: Path) -> None:
    payload = _payload(prompt={"population_size": 2, "generations": 1})
    payload["instructions"]["system"]["content"] = "Answer {question}."
    payload["optimizer"]["search_space"]["system_prompt"]["required_variables"] = ["question"]

    with pytest.raises(GaPromptOptimizerError, match="failed for every generated individual"):
        run_ga_prompt_optimization(
            payload,
            tmp_path,
            RecordingEvaluator(),
            DroppingVariableTransformer(),
        )


def test_ga_excludes_failed_individuals_from_parent_selection(tmp_path: Path) -> None:
    payload = _payload(
        prompt={
            "population_size": 3,
            "generations": 2,
            "mutation_rate": 0.0,
            "crossover_rate": 0.0,
            "elitism": 0,
            "seed": 3,
        }
    )

    result = run_ga_prompt_optimization(
        payload,
        tmp_path,
        FailsBadPromptEvaluator(),
        OneBadMutationTransformer(),
    )

    failed = [individual for individual in result.history if individual.status == "failed"]
    assert [individual.individual_id for individual in failed] == ["g000-i001"]
    generation_one = [individual for individual in result.history if individual.generation == 1]
    assert generation_one
    assert all("g000-i001" not in individual.parent_ids for individual in generation_one)


def test_ga_fails_when_every_individual_in_generation_fails(tmp_path: Path) -> None:
    payload = _payload(prompt={"population_size": 3, "generations": 1, "seed": 5})

    with pytest.raises(GaPromptOptimizerError) as exc_info:
        run_ga_prompt_optimization(
            payload,
            tmp_path,
            AlwaysFailEvaluator(),
            DeterministicTransformer(),
        )

    assert exc_info.value.trial_count == 3
    history = json.loads((tmp_path / "ga_history_prompts.json").read_text(encoding="utf-8"))
    assert {row["status"] for row in history} == {"failed"}


def test_ga_fitness_supports_minimize_and_diversity_penalty() -> None:
    config = parse_ga_prompt_optimizer_config(
        _payload(
            optimizer={
                "multi_objective_combination_mode": "weighted_sum",
                "eval_metrics": {"latency": {"direction": "minimize", "weight": 1.0}},
            },
            prompt={"population_size": 3, "generations": 1, "diversity_lambda": 0.6},
        )
    )
    population = [
        _completed_individual(prompt="same", score=3.0, index=0, metric_name="latency"),
        _completed_individual(prompt="same", score=1.0, index=1, metric_name="latency"),
        _completed_individual(prompt="other", score=2.0, index=2, metric_name="latency"),
    ]

    snapshot = assign_generation_fitness(
        population,
        metrics=config.metrics,
        mode=config.multi_objective_mode,
        diversity_lambda=config.diversity_lambda,
    )

    assert snapshot.duplicate_ratio == pytest.approx(1 / 3)
    ranked = rank_valid_individuals(population)
    assert ranked[0].individual_index == 1
    assert ranked[0].normalized_metrics["latency"] == 1.0


@pytest.mark.parametrize("option", ["ga_population_size", "parallel_evaluations"])
def test_ga_config_rejects_unsupported_options(option: str) -> None:
    with pytest.raises(ValueError, match=option):
        parse_ga_prompt_optimizer_config(_payload(prompt={option: 4}))


def test_oracle_feedback_uses_worst_reasoning_rows_for_metric() -> None:
    config = parse_ga_prompt_optimizer_config(
        _payload(
            optimizer={"eval_metrics": {"average_score": {"direction": "maximize", "weight": 1.0}}},
            prompt={"oracle_feedback_worst_n": 1, "oracle_feedback_mode": "always"},
        )
    )
    individual = GaIndividual(
        prompts={"system_prompt": "Base prompt."},
        generation=0,
        individual_index=0,
        status="completed",
        aggregate_metrics={"average_score": 0.6},
        raw_scores=(
            _score(task_id="bad-row", metric_name="average_score", value=0.2, reasoning="Missed the key fact."),
            _score(task_id="good-row", metric_name="average_score", value=0.9, reasoning="Correct and complete."),
        ),
    )

    feedback = build_oracle_feedback(individual=individual, config=config)

    assert feedback is not None
    assert "bad-row" in feedback
    assert "Missed the key fact." in feedback
    assert "good-row" not in feedback


def test_oracle_feedback_prioritizes_high_weight_metrics_when_truncated() -> None:
    config = parse_ga_prompt_optimizer_config(
        _payload(
            optimizer={
                "eval_metrics": {
                    "low_value": {"direction": "maximize", "weight": 1.0},
                    "high_value": {"direction": "maximize", "weight": 9.0},
                }
            },
            prompt={"oracle_feedback_max_chars": 160, "oracle_feedback_worst_n": 1},
        )
    )
    individual = GaIndividual(
        prompts={"system_prompt": "Base prompt."},
        generation=0,
        individual_index=0,
        status="completed",
        aggregate_metrics={"low_value": 0.1, "high_value": 0.2},
        raw_scores=(
            _score(
                task_id="low-row",
                metric_name="low_value",
                value=0.1,
                reasoning="low-weight feedback should not lead the prompt context.",
            ),
            _score(
                task_id="high-row",
                metric_name="high_value",
                value=0.2,
                reasoning="high-weight feedback should lead the prompt context.",
            ),
        ),
    )

    feedback = build_oracle_feedback(individual=individual, config=config)

    assert feedback is not None
    assert feedback.startswith("Metric high_value")
    assert len(feedback) <= 160


def test_mutation_feedback_follows_selected_prompt_source() -> None:
    config = parse_ga_prompt_optimizer_config(
        _payload(
            prompt={
                "population_size": 2,
                "generations": 2,
                "crossover_rate": 0.0,
                "mutation_rate": 1.0,
                "oracle_feedback_mode": "always",
            }
        )
    )
    transformer = FeedbackRecordingTransformer()
    parent_a = _completed_parent(prompt="Parent A", index=0, reasoning="reasoning from A")
    parent_b = _completed_parent(prompt="Parent B", index=1, reasoning="reasoning from B")

    prompt = _child_prompt(
        transformer,
        name="system_prompt",
        spec=config.search_space["system_prompt"],
        parent_a=parent_a,
        parent_b=parent_b,
        config=config,
        rng=FixedRandom([0.99, 0.0, 0.0]),
        oracle_state=OracleFeedbackState(stagnation_generations=0, fitness_variance=0.0, duplicate_ratio=0.0),
        failures=[],
        successes=[],
    )

    assert prompt == "Parent B"
    assert transformer.mutation_feedback is not None
    assert "Feedback from g000-i001" in transformer.mutation_feedback
    assert "reasoning from B" in transformer.mutation_feedback
    assert "reasoning from A" not in transformer.mutation_feedback


def test_recombination_feedback_combines_both_parents() -> None:
    config = parse_ga_prompt_optimizer_config(
        _payload(
            prompt={
                "population_size": 2,
                "generations": 2,
                "crossover_rate": 1.0,
                "mutation_rate": 0.0,
                "oracle_feedback_mode": "always",
            }
        )
    )
    transformer = FeedbackRecordingTransformer()
    parent_a = _completed_parent(prompt="Parent A", index=0, reasoning="reasoning from A")
    parent_b = _completed_parent(prompt="Parent B", index=1, reasoning="reasoning from B")

    prompt = _child_prompt(
        transformer,
        name="system_prompt",
        spec=config.search_space["system_prompt"],
        parent_a=parent_a,
        parent_b=parent_b,
        config=config,
        rng=FixedRandom([0.0, 0.99]),
        oracle_state=OracleFeedbackState(stagnation_generations=0, fitness_variance=0.0, duplicate_ratio=0.0),
        failures=[],
        successes=[],
    )

    assert prompt == "Parent A + Parent B"
    assert transformer.recombination_feedback is not None
    assert "reasoning from A" in transformer.recombination_feedback
    assert "reasoning from B" in transformer.recombination_feedback


def test_raw_score_records_skip_carried_elites(tmp_path: Path) -> None:
    payload = _payload(
        prompt={
            "population_size": 2,
            "generations": 2,
            "mutation_rate": 1.0,
            "crossover_rate": 1.0,
            "elitism": 1,
        }
    )

    result = run_ga_prompt_optimization(
        payload,
        tmp_path,
        RawScoreEvaluator(),
        DeterministicTransformer(),
    )

    records = json.loads((tmp_path / "ga_score_records.json").read_text(encoding="utf-8"))
    assert len(records) == result.executed_trials
    carried_ids = {individual.individual_id for individual in result.history if individual.carried_from}
    assert carried_ids
    assert carried_ids.isdisjoint({record["individual_id"] for record in records})


class RecordingEvaluator:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def evaluate(
        self,
        *,
        trial_number: int,
        suggestions: dict[str, Any],
        trial_overlay: dict[str, Any],
        rep: int,
    ) -> CandidateEvaluationResult:
        self.calls.append(
            {
                "trial_number": trial_number,
                "suggestions": dict(suggestions),
                "metadata": dict(trial_overlay["metadata"]),
                "rep": rep,
            }
        )
        prompt = str(suggestions["instructions.system.content"])
        return CandidateEvaluationResult(aggregate_metrics={"average_score": float(len(prompt))})


class DeterministicTransformer:
    def mutate(
        self,
        *,
        prompt_name: str,
        prompt: str,
        purpose: str,
        prompt_format: str | None,
        feedback: str | None,
    ) -> str:
        del purpose, prompt_format, feedback
        return f"{prompt}\n\n[{prompt_name}: mutation]"

    def recombine(
        self,
        *,
        prompt_name: str,
        parent_a: str,
        parent_b: str,
        purpose: str,
        prompt_format: str | None,
        feedback: str | None,
    ) -> str:
        del purpose, prompt_format, feedback
        return f"{parent_a}\n\n[{prompt_name}: recombined]\n{parent_b}"


class FeedbackRecordingTransformer:
    def __init__(self) -> None:
        self.mutation_feedback: str | None = None
        self.recombination_feedback: str | None = None

    def mutate(
        self,
        *,
        prompt_name: str,
        prompt: str,
        purpose: str,
        prompt_format: str | None,
        feedback: str | None,
    ) -> str:
        del prompt_name, purpose, prompt_format
        self.mutation_feedback = feedback
        return prompt

    def recombine(
        self,
        *,
        prompt_name: str,
        parent_a: str,
        parent_b: str,
        purpose: str,
        prompt_format: str | None,
        feedback: str | None,
    ) -> str:
        del prompt_name, purpose, prompt_format
        self.recombination_feedback = feedback
        return f"{parent_a} + {parent_b}"


class FixedRandom:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def random(self) -> float:
        return next(self._values)


class FailsBadPromptEvaluator:
    def evaluate(
        self,
        *,
        trial_number: int,
        suggestions: dict[str, Any],
        trial_overlay: dict[str, Any],
        rep: int,
    ) -> CandidateEvaluationResult:
        del trial_number, trial_overlay, rep
        prompt = str(suggestions["instructions.system.content"])
        if "bad candidate" in prompt:
            raise CandidateEvaluationError("candidate crashed")
        return CandidateEvaluationResult(aggregate_metrics={"average_score": float(len(prompt))})


class AlwaysFailEvaluator:
    def evaluate(
        self,
        *,
        trial_number: int,
        suggestions: dict[str, Any],
        trial_overlay: dict[str, Any],
        rep: int,
    ) -> CandidateEvaluationResult:
        del trial_number, suggestions, trial_overlay, rep
        raise CandidateEvaluationError("all candidates failed")


class RawScoreEvaluator:
    def evaluate(
        self,
        *,
        trial_number: int,
        suggestions: dict[str, Any],
        trial_overlay: dict[str, Any],
        rep: int,
    ) -> CandidateEvaluationResult:
        del trial_overlay, rep
        prompt = str(suggestions["instructions.system.content"])
        score_value = float(len(prompt))
        return CandidateEvaluationResult(
            aggregate_metrics={"average_score": score_value},
            scores=(
                _score(
                    task_id=f"row-{trial_number}",
                    metric_name="average_score",
                    value=score_value,
                    reasoning=f"reasoning for trial {trial_number}",
                ),
            ),
        )


class FailingTransformer:
    def mutate(
        self,
        *,
        prompt_name: str,
        prompt: str,
        purpose: str,
        prompt_format: str | None,
        feedback: str | None,
    ) -> str:
        del prompt_name, prompt, purpose, prompt_format, feedback
        raise PromptTransformError("mutation model failed")

    def recombine(
        self,
        *,
        prompt_name: str,
        parent_a: str,
        parent_b: str,
        purpose: str,
        prompt_format: str | None,
        feedback: str | None,
    ) -> str:
        del prompt_name, parent_a, parent_b, purpose, prompt_format, feedback
        raise PromptTransformError("recombination model failed")


class DroppingVariableTransformer:
    def mutate(
        self,
        *,
        prompt_name: str,
        prompt: str,
        purpose: str,
        prompt_format: str | None,
        feedback: str | None,
    ) -> str:
        del prompt_name, prompt, purpose, prompt_format, feedback
        return "Answer clearly."

    def recombine(
        self,
        *,
        prompt_name: str,
        parent_a: str,
        parent_b: str,
        purpose: str,
        prompt_format: str | None,
        feedback: str | None,
    ) -> str:
        del prompt_name, parent_a, parent_b, purpose, prompt_format, feedback
        return "Answer clearly."


class OneBadMutationTransformer:
    def __init__(self) -> None:
        self._mutation_count = 0

    def mutate(
        self,
        *,
        prompt_name: str,
        prompt: str,
        purpose: str,
        prompt_format: str | None,
        feedback: str | None,
    ) -> str:
        del prompt_name, purpose, prompt_format, feedback
        self._mutation_count += 1
        if self._mutation_count == 1:
            return f"{prompt}\n\nbad candidate"
        return f"{prompt}\n\nbetter candidate"

    def recombine(
        self,
        *,
        prompt_name: str,
        parent_a: str,
        parent_b: str,
        purpose: str,
        prompt_format: str | None,
        feedback: str | None,
    ) -> str:
        del prompt_name, parent_b, purpose, prompt_format, feedback
        return parent_a


def _payload(
    *,
    optimizer: dict[str, Any] | None = None,
    prompt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    optimizer_config: dict[str, Any] = {
        "eval_metrics": {"average_score": {"direction": "maximize", "weight": 1.0}},
    }
    if optimizer:
        optimizer_config.update(optimizer)
    prompt_config = {"enabled": True, "backend": "ga", "model": "prompt_optimizer"}
    if prompt:
        prompt_config.update(prompt)
    optimizer_config["prompt"] = prompt_config
    optimizer_config["search_space"] = {
        "system_prompt": {
            "type": "fabric",
            "path": "instructions.system.content",
            "is_prompt": True,
            "purpose": "Answer accurately.",
        }
    }
    return {
        "schema_version": "fabric.agent/v1alpha1",
        "metadata": {"name": "demo"},
        "models": {"prompt_optimizer": {"provider": "openai", "model": "gpt-5-mini"}},
        "instructions": {"system": {"content": "Base prompt."}},
        "optimizer": optimizer_config,
    }


def _completed_individual(*, prompt: str, score: float, index: int, metric_name: str) -> GaIndividual:
    return GaIndividual(
        prompts={"system_prompt": prompt},
        generation=0,
        individual_index=index,
        status="completed",
        aggregate_metrics={metric_name: score},
    )


def _completed_parent(*, prompt: str, index: int, reasoning: str) -> GaIndividual:
    return GaIndividual(
        prompts={"system_prompt": prompt},
        generation=0,
        individual_index=index,
        status="completed",
        aggregate_metrics={"average_score": 0.5},
        fitness=1.0,
        raw_scores=(
            _score(
                task_id=f"row-{index}",
                metric_name="average_score",
                value=0.5,
                reasoning=reasoning,
            ),
        ),
    )


def _score(*, task_id: str, metric_name: str, value: float, reasoning: str) -> AgentEvalTaskScore:
    return AgentEvalTaskScore(
        id=f"score-{task_id}",
        run_id="run",
        task_id=task_id,
        trial_id=f"trial-{task_id}",
        metric_type="tunable-rag-evaluator",
        status=AgentEvalScoreStatus.COMPLETED,
        outputs=[
            MetricOutput(name=metric_name, value=value),
            MetricOutput(name="reasoning", value=reasoning),
        ],
    )
