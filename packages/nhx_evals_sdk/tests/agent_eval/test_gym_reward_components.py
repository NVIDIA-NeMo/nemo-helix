# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reward components from a multi-reward Gym resources-server, scored per trial.

A resources-server built on Gym's ``BaseMultiRewardVerifyResponse`` returns ``reward_components`` beside its
scalar ``reward``. The runner keeps them on each trial, adds the ones it found to the task's
``GymRewardMetric`` as optional outputs, and leaves them out of the ``runner.gym.*`` aggregates it imports,
since the SDK now scores them itself.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import pytest
from nhx_evals_sdk.agent_eval.evaluator import AgentEvaluator
from nhx_evals_sdk.agent_eval.runtimes.gym import GymRewardMetric, discover_gym_tasks
from nhx_evals_sdk.agent_eval.runtimes.gym.dataset import materialize_dataset
from nhx_evals_sdk.agent_eval.runtimes.gym.records import NG_ROLLOUT_INDEX, NG_TASK_INDEX
from nhx_evals_sdk.agent_eval.runtimes.gym.results import (
    aggregate_scores_from_gym,
    gym_scoring_metrics,
    trials_from_rollouts,
)
from nhx_evals_sdk.agent_eval.runtimes.gym.sandboxed import SandboxedGymAgentTaskRunner, SandboxedGymRuntimeConfig
from nhx_evals_sdk.agent_eval.tasks import AgentEvalRunConfig
from nhx_evals_sdk.metrics.protocol import CandidateOutput, DatasetRow, MetricInput
from nhx_evals_sdk.metrics.string_check import StringCheckMetric
from pydantic import ValidationError

EXAMPLE = Path(__file__).parent / "fixtures" / "gym_mcqa_example.jsonl"
ROLLOUT_URL = "http://gym-host.invalid/rollouts"


def _tasks(count: int = 2):
    return discover_gym_tasks(EXAMPLE, metrics=[GymRewardMetric()])[:count]


def _trials(records: list[dict[str, Any]], tasks, tmp_path: Path):
    index_map = materialize_dataset(tasks, tmp_path / "gym_input.jsonl")
    bundle = tmp_path / "rollouts.jsonl"
    bundle.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    return trials_from_rollouts(bundle, tasks, index_map, run_id="run")


def _input(metadata: dict[str, Any]) -> MetricInput:
    return MetricInput(row=DatasetRow(data={}), candidate=CandidateOutput(metadata=metadata))


def test_a_rollouts_reward_components_reach_its_trial(tmp_path: Path) -> None:
    tasks = _tasks(1)
    record = {
        NG_TASK_INDEX: 0,
        NG_ROLLOUT_INDEX: 0,
        "reward": 1.0,
        "reward_components": {"exact": 1, "relaxed": 0.5},
    }
    (trial,) = _trials([record], tasks, tmp_path)
    assert trial.metadata["reward_components"] == {"exact": 1.0, "relaxed": 0.5}


def test_unusable_components_are_left_out_and_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    tasks = _tasks(1)
    components = {"kept": 0.25, "text": "high", "flag": True, "nan": math.nan, "x.pass@1": 1.0}
    record = {NG_TASK_INDEX: 0, NG_ROLLOUT_INDEX: 0, "reward": 1.0, "reward_components": components}
    with caplog.at_level(logging.WARNING):
        (trial,) = _trials([record], tasks, tmp_path)
    # A boolean isn't read as 1/0, a non-finite value isn't a measurement, and a reserved name would
    # collide with the SDK's pass@k aggregates.
    assert trial.metadata["reward_components"] == {"kept": 0.25}
    assert caplog.text.count("unusable reward component") == 4


def test_components_that_are_not_a_mapping_are_ignored(tmp_path: Path) -> None:
    tasks = _tasks(1)
    record = {NG_TASK_INDEX: 0, NG_ROLLOUT_INDEX: 0, "reward": 1.0, "reward_components": [1.0, 0.0]}
    (trial,) = _trials([record], tasks, tmp_path)
    assert "reward_components" not in trial.metadata


def test_a_trial_without_components_has_none(tmp_path: Path) -> None:
    (trial,) = _trials([{NG_TASK_INDEX: 0, NG_ROLLOUT_INDEX: 0, "reward": 1.0}], _tasks(1), tmp_path)
    assert "reward_components" not in trial.metadata


def test_an_agent_that_never_ran_reports_no_components(tmp_path: Path) -> None:
    # Like its reward, an agent that never called the model earned none of its components. (A second
    # rollout that did run keeps this from being a run where no agent ran at all, which raises.)
    never_ran = {
        NG_TASK_INDEX: 0,
        NG_ROLLOUT_INDEX: 0,
        "reward": 0.0,
        "reward_components": {"exact": 0.0},
        "response": {"output": [], "usage": {"input_tokens": 0, "output_tokens": 0}},
    }
    ran = {NG_TASK_INDEX: 1, NG_ROLLOUT_INDEX: 0, "reward": 1.0, "reward_components": {"exact": 1.0}}
    trials = {trial.metadata[NG_TASK_INDEX]: trial for trial in _trials([never_ran, ran], _tasks(2), tmp_path)}
    assert trials[0].metadata["reward"] is None
    assert "reward_components" not in trials[0].metadata
    assert trials[1].metadata["reward_components"] == {"exact": 1.0}


async def test_the_metric_scores_each_named_component_as_an_optional_output() -> None:
    metric = GymRewardMetric(reward_components=("exact", "relaxed"))
    spec = {output.name: output.required for output in metric.output_spec()}
    assert spec == {"reward": True, "exact": False, "relaxed": False}

    result = await metric.compute_scores(_input({"reward": 1.0, "reward_components": {"exact": 1.0, "relaxed": 0.0}}))
    assert [(output.name, output.value) for output in result.outputs] == [
        ("reward", 1.0),
        ("exact", 1.0),
        ("relaxed", 0.0),
    ]


async def test_a_component_the_trial_did_not_report_is_omitted() -> None:
    metric = GymRewardMetric(reward_components=("exact", "relaxed"))
    result = await metric.compute_scores(_input({"reward": 0.0, "reward_components": {"exact": 0.0}}))
    assert [output.name for output in result.outputs] == ["reward", "exact"]
    bare = await metric.compute_scores(_input({"reward": 0.0}))
    assert [output.name for output in bare.outputs] == ["reward"]


def test_a_component_named_like_the_primary_output_is_declared_once() -> None:
    metric = GymRewardMetric(reward_components=("reward", "exact"))
    assert [output.name for output in metric.output_spec()] == ["reward", "exact"]


@pytest.mark.parametrize("names", [("x.pass@1",), ("",), ("exact", "exact")])
def test_unusable_component_names_are_refused(names: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError):
        GymRewardMetric(reward_components=names)


def test_scoring_metrics_add_the_components_found_in_the_trials(tmp_path: Path) -> None:
    check = StringCheckMetric(operation="equals", left_template="a", right_template="a")
    task = _tasks(1)[0].model_copy(update={"metrics": [GymRewardMetric(reward_components=("declared",)), check]})
    records = [
        {NG_TASK_INDEX: 0, NG_ROLLOUT_INDEX: 0, "reward": 1.0, "reward_components": {"relaxed": 1.0}},
        {NG_TASK_INDEX: 0, NG_ROLLOUT_INDEX: 1, "reward": 0.0, "reward_components": {"exact": 0.0}},
    ]
    trials = _trials(records, [task], tmp_path)

    reward, kept = gym_scoring_metrics(task, trials)
    assert isinstance(reward, GymRewardMetric)
    assert reward.reward_components == ("declared", "exact", "relaxed")
    assert kept is check


def test_scoring_metrics_leave_a_task_without_components_unchanged(tmp_path: Path) -> None:
    task = _tasks(1)[0]
    trials = _trials([{NG_TASK_INDEX: 0, NG_ROLLOUT_INDEX: 0, "reward": 1.0}], [task], tmp_path)
    assert gym_scoring_metrics(task, trials) == list(task.metrics)


def test_components_scored_natively_are_not_imported_again() -> None:
    family = {"mean": 0.5, "max": 1.0, "min": 0.0, "median": 0.5, "std": 0.5}
    agent_metrics = {f"{stat}/{metric}": value for metric in ("exact", "steps") for stat, value in family.items()}
    aggregations = {"simple_agent": {"agent_metrics": agent_metrics}}

    assert {score.name for score in aggregate_scores_from_gym(aggregations)} == {
        "runner.gym.exact",
        "runner.gym.steps",
    }
    imported = aggregate_scores_from_gym(aggregations, scored_natively=frozenset({"exact"}))
    assert {score.name for score in imported} == {"runner.gym.steps"}


async def test_an_evaluation_scores_components_per_trial(tmp_path: Path) -> None:
    # End to end through AgentEvaluator: tasks carry a plain GymRewardMetric, and the runner's scoring
    # hook adds the components its rollouts reported.
    tasks = _tasks(2)
    components = {0: {"exact": 1.0, "relaxed": 1.0}, 1: {"exact": 0.0, "relaxed": 1.0}}

    async def collect(examples: list[dict[str, Any]]) -> list[Any]:
        return [
            {
                NG_TASK_INDEX: example[NG_TASK_INDEX],
                NG_ROLLOUT_INDEX: example[NG_ROLLOUT_INDEX],
                "reward": components[example[NG_TASK_INDEX]]["exact"],
                "reward_components": components[example[NG_TASK_INDEX]],
                "response": f"answer-{example[NG_TASK_INDEX]}",
            }
            for example in examples
        ]

    runner = SandboxedGymAgentTaskRunner(config=SandboxedGymRuntimeConfig(rollout_url=ROLLOUT_URL), collect=collect)
    result = await AgentEvaluator().run(tasks=tasks, target=runner, config=AgentEvalRunConfig(work_dir=tmp_path))

    means = {score.name: getattr(score, "mean", None) for score in result.summary.scores.scores}
    assert means["gym_reward.reward"] == 0.5
    assert means["gym_reward.exact"] == 0.5
    assert means["gym_reward.relaxed"] == 1.0
    per_trial = {(score.task_id, output.name): output.value for score in result.scores for output in score.outputs}
    assert per_trial[(tasks[0].id, "exact")] == 1.0
    assert per_trial[(tasks[1].id, "exact")] == 0.0
