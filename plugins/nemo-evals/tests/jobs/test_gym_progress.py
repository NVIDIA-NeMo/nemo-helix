# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A Gym job's status details count its rollouts as they finish: total, finished, failed and their reward."""

from __future__ import annotations

import logging
from typing import Any

import pytest
from nemo_evals.jobs.gym_progress import GymTrialCounter
from nemo_evals.jobs.gym_sandbox import collect_reporting
from nemo_evals.taskset_runs.trials import TRIAL_COUNTS_STATUS_KEY
from nhx_evals_sdk.agent_eval.runtimes.gym.records import NG_FAILURE_CLASS, NG_ROLLOUT_INDEX, NG_TASK_INDEX
from nhx_evals_sdk.agent_eval.runtimes.gym.results import AGENT_NEVER_RAN_ERROR_TYPE


class Reports:
    def __init__(self) -> None:
        self.published: list[dict[str, Any]] = []

    async def __call__(self, details: dict[str, Any]) -> None:
        self.published.append(details)

    def last(self) -> dict[str, Any]:
        return self.published[-1][TRIAL_COUNTS_STATUS_KEY]


def _rollout(task: int, rollout: int, **fields: Any) -> dict[str, Any]:
    return {NG_TASK_INDEX: task, NG_ROLLOUT_INDEX: rollout, **fields}


async def test_planned_rollouts_start_pending() -> None:
    reports = Reports()
    await GymTrialCounter(reports, reward_key="reward").planned(6)

    assert reports.last() == {"total": 6, "phases": {"pending": 6}, "rewarded": 0, "reward": None, "errors": {}}


async def test_finished_rollouts_count_as_succeeded_or_errored_with_their_reward() -> None:
    reports = Reports()
    counter = GymTrialCounter(reports, reward_key="score")
    await counter.planned(4)

    await counter.finished([_rollout(0, 0, score=1.0), _rollout(0, 1, score=0.0)])
    await counter.finished([_rollout(1, 0, **{NG_FAILURE_CLASS: "agent_error"})])

    assert reports.last() == {
        "total": 4,
        "phases": {"succeeded": 2, "errored": 1, "pending": 1},
        "rewarded": 2,
        "reward": 0.5,
        "errors": {"agent_error": 1},
    }


async def test_a_rollout_whose_agent_never_ran_counts_as_failed_like_its_final_trial() -> None:
    reports = Reports()
    never_ran = _rollout(0, 0, reward=0.0, response={"output": [], "usage": {"input_tokens": 0}})

    await GymTrialCounter(reports, reward_key="reward").finished([never_ran])

    assert reports.last()["phases"] == {"errored": 1}
    assert reports.last()["errors"] == {AGENT_NEVER_RAN_ERROR_TYPE: 1}
    assert reports.last()["rewarded"] == 0


async def test_every_record_counts_as_a_trial_like_the_final_evaluation() -> None:
    """A sandboxed host restamps rollout indices per chunk, so repeats share them: identity can't dedupe records.

    A retried rollout therefore counts once per attempt, and the total can pass the plan, as the final
    evaluation counts it.
    """
    reports = Reports()
    counter = GymTrialCounter(reports, reward_key="reward")
    await counter.planned(2)

    await counter.finished([_rollout(0, 0, reward=1.0), _rollout(0, 0, reward=0.0)])
    await counter.finished([_rollout(1, 0, **{NG_FAILURE_CLASS: "transient"}), _rollout(1, 0, reward=1.0)])

    assert reports.last()["total"] == 4
    assert reports.last()["phases"] == {"succeeded": 3, "errored": 1}


async def test_rollouts_unfinished_when_collection_stops_count_as_cancelled() -> None:
    reports = Reports()
    counter = GymTrialCounter(reports, reward_key="reward")
    await counter.planned(3)
    await counter.finished([_rollout(0, 0, reward=1.0)])

    await counter.stopped()

    assert reports.last()["phases"] == {"succeeded": 1, "cancelled": 2}


@pytest.mark.parametrize("reward", [True, "1.0", float("nan"), None])
async def test_a_reward_that_is_not_a_finite_number_leaves_the_rollout_unrewarded(reward: object) -> None:
    reports = Reports()
    counter = GymTrialCounter(reports, reward_key="reward")

    await counter.finished([_rollout(0, 0, reward=reward)])

    assert (reports.last()["rewarded"], reports.last()["reward"]) == (0, None)


async def test_a_report_that_fails_never_fails_the_run(caplog: pytest.LogCaptureFixture) -> None:
    async def unreachable(details: dict[str, Any]) -> None:
        raise RuntimeError("jobs service down")

    with caplog.at_level(logging.WARNING):
        await GymTrialCounter(unreachable, reward_key="reward").finished([_rollout(0, 0, reward=1.0)])

    assert "Could not report progress" in caplog.text


async def test_sandboxed_collection_plans_the_batch_and_reports_each_chunk() -> None:
    reports = Reports()
    counter = GymTrialCounter(reports, reward_key="reward")
    examples = [{NG_TASK_INDEX: 0, NG_ROLLOUT_INDEX: index} for index in range(3)]

    async def arun_rollouts(batch: list[dict[str, Any]], *, on_results: Any) -> list[Any]:
        assert reports.last()["phases"] == {"pending": 3}
        results = [{**row, "reward": 1.0} for row in batch]
        await on_results(results[:2])
        assert reports.last()["phases"] == {"succeeded": 2, "pending": 1}
        await on_results(results[2:])
        return results

    results = await collect_reporting(arun_rollouts, examples, progress=counter)

    assert len(results) == 3
    assert reports.last()["phases"] == {"succeeded": 3}


async def test_sandboxed_collection_that_fails_cancels_the_rollouts_it_never_returned() -> None:
    reports = Reports()
    counter = GymTrialCounter(reports, reward_key="reward")

    async def arun_rollouts(batch: list[dict[str, Any]], *, on_results: Any) -> list[Any]:
        await on_results([{**batch[0], "reward": 1.0}])
        raise TimeoutError("collection_timeout_s exceeded")

    with pytest.raises(TimeoutError):
        await collect_reporting(
            arun_rollouts, [{NG_TASK_INDEX: 0, NG_ROLLOUT_INDEX: index} for index in range(2)], progress=counter
        )

    assert reports.last()["phases"] == {"succeeded": 1, "cancelled": 1}


async def test_sandboxed_collection_without_progress_calls_collect_as_before() -> None:
    async def arun_rollouts(batch: list[dict[str, Any]]) -> list[Any]:
        return batch

    assert await collect_reporting(arun_rollouts, [{"n": 1}], progress=None) == [{"n": 1}]
