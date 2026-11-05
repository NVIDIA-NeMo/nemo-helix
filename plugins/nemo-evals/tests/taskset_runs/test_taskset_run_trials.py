# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trial progress read from a task Job's status details, counted per task and per run."""

import pytest
from nemo_evals.taskset_runs.trials import (
    TRIALS_STATUS_KEY,
    TrialCounts,
    TrialPhase,
    TrialProgress,
    count_trials,
    reported_trials,
    sum_trial_counts,
)


def trial(name: str, phase: TrialPhase, *, reward: float | None = None, error_type: str | None = None) -> TrialProgress:
    return TrialProgress(trial=name, phase=phase, reward=reward, error_type=error_type)


def test_count_trials_counts_phases_errors_and_mean_reward() -> None:
    counts = count_trials(
        [
            trial("a", TrialPhase.SUCCEEDED, reward=1.0),
            trial("b", TrialPhase.ERRORED, reward=0.0, error_type="AgentTimeoutError"),
            trial("c", TrialPhase.ERRORED),
            trial("d", TrialPhase.AGENT_RUNNING),
            trial("e", TrialPhase.RETRYING),
            trial("f", TrialPhase.PENDING),
            trial("g", TrialPhase.CANCELLED),
        ]
    )
    assert counts == TrialCounts(
        total=7,
        pending=1,
        running=2,
        succeeded=1,
        errored=2,
        cancelled=1,
        rewarded=2,
        reward=0.5,
        errors={"AgentTimeoutError": 1, "error": 1},
    )
    assert count_trials([trial("a", TrialPhase.PENDING)]).reward is None


def test_reported_trials_reads_the_status_details_list() -> None:
    details = {TRIALS_STATUS_KEY: [{"trial": "a", "phase": "succeeded", "reward": 1, "future_field": True}]}
    assert reported_trials(details) == [trial("a", TrialPhase.SUCCEEDED, reward=1.0)]


@pytest.mark.parametrize(
    "details",
    [None, [], {}, {TRIALS_STATUS_KEY: "nope"}, {TRIALS_STATUS_KEY: [{"trial": "a", "phase": "unknown"}]}],
    ids=["none", "not a dict", "no trials", "not a list", "unreadable trial"],
)
def test_reported_trials_ignores_what_it_cant_read(details: object) -> None:
    assert reported_trials(details) is None


def test_run_totals_sum_counts_and_average_task_rewards() -> None:
    """Each task weighs the same in the run's reward, whatever its trial count."""
    one = TrialCounts(total=4, succeeded=4, rewarded=4, reward=1.0)
    other = TrialCounts(total=1, errored=1, rewarded=1, reward=0.0, errors={"AgentTimeoutError": 1})
    unscored = TrialCounts(total=3, pending=1, running=2)
    assert sum_trial_counts([one, other, unscored]) == TrialCounts(
        total=8,
        pending=1,
        running=2,
        succeeded=4,
        errored=1,
        rewarded=5,
        reward=0.5,
        errors={"AgentTimeoutError": 1},
    )
    assert sum_trial_counts([]) is None
