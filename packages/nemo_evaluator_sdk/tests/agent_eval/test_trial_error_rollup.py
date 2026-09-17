# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AALGO-428: ``AgentEvalSummary.error_trial_ids`` — Harbor's ``exception_stats`` shape."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import pytest
from nemo_evaluator_sdk.agent_eval.results import AgentEvalSummary, _error_trial_ids
from nemo_evaluator_sdk.agent_eval.scores import AgentEvalScoreStatus, AgentEvalTaskScore
from nemo_evaluator_sdk.agent_eval.trials import (
    AgentEvalTrial,
    AgentEvalTrialStatus,
    AgentOutput,
    TrialError,
)
from nemo_evaluator_sdk.metrics.protocol import MetricOutput
from pydantic import ValidationError


def _vendored_module(name: str) -> Any:
    return import_module(f"nemo_platform.beta.evaluator.agent_eval.{name}")


def _trial(
    trial_id: str,
    *,
    task_id: str = "task-a",
    error: str | None = None,
    status: AgentEvalTrialStatus = AgentEvalTrialStatus.PARTIAL,
) -> AgentEvalTrial:
    return AgentEvalTrial(
        id=trial_id,
        task_id=task_id,
        status=status,
        # COMPLETED requires an output; the others tolerate None.
        output=AgentOutput(output_text="done") if status is AgentEvalTrialStatus.COMPLETED else None,
        error=None if error is None else TrialError(type=error),
    )


def _score(task_id: str, trial_id: str, value: float) -> AgentEvalTaskScore:
    return AgentEvalTaskScore(
        id=f"run:{task_id}:{trial_id}:reward",
        run_id="run",
        task_id=task_id,
        trial_id=trial_id,
        metric_type="reward",
        status=AgentEvalScoreStatus.COMPLETED,
        outputs=[MetricOutput(name="score", value=value)],
    )


def test_omitting_trials_leaves_the_rollup_empty_rather_than_raising() -> None:
    # Same silent-skip contract `tasks=None` already has for pass@k: a caller that only has scores
    # gets a summary, not an error.
    summary = AgentEvalSummary.from_scores([_score("task-a", "t0", 1.0)])

    assert summary.error_trial_ids == {}
    assert summary.error_count == 0


def test_errors_group_by_type_with_ids_in_trial_order() -> None:
    trials = [
        _trial("t0", error="RuntimeError"),
        _trial("t1"),  # no error
        _trial("t2", error="RuntimeError"),
        _trial("t3", error="TimeoutError"),
    ]

    summary = AgentEvalSummary.from_scores([], trials=trials)

    assert summary.error_trial_ids == {"RuntimeError": ["t0", "t2"], "TimeoutError": ["t3"]}
    assert summary.error_count == 3


def test_membership_ignores_trial_status() -> None:
    # An errored Harbor trial is PARTIAL rather than FAILED precisely so it is still scored, so a
    # status filter here would drop the trials this rollup exists to name. Harbor keys on the
    # presence of exception_info alone.
    trials = [
        _trial("done", error="RuntimeError", status=AgentEvalTrialStatus.COMPLETED),
        _trial("partial", error="RuntimeError", status=AgentEvalTrialStatus.PARTIAL),
        _trial("failed", error="RuntimeError", status=AgentEvalTrialStatus.FAILED),
    ]

    summary = AgentEvalSummary.from_scores([], trials=trials)

    assert summary.error_trial_ids == {"RuntimeError": ["done", "partial", "failed"]}


def test_duplicate_trial_ids_stay_two_entries() -> None:
    # Nothing enforces trial-id uniqueness (Gym derives ids from a rollout index in two separate
    # loops), so the rollup must append rather than collect into a set — collapsing them would
    # understate the error count. trial_count is len(trials) for the same reason.
    summary = AgentEvalSummary.from_scores([], trials=[_trial("dup", error="E"), _trial("dup", error="E")])

    assert summary.error_trial_ids == {"E": ["dup", "dup"]}
    assert summary.error_count == 2
    assert summary.trial_count == 2


def test_trials_may_be_wider_than_the_scores() -> None:
    # A caller re-aggregating a subset can hand over more trials than scores. The rollup names them
    # regardless: it reads trials, not scores, so the two need not line up. trial_count is
    # len(trials), so the unmeasured t1 still contributes even though it has no score.
    unmeasured = _trial("t1", task_id="task-b", error="RuntimeError")
    summary = AgentEvalSummary.from_scores(
        [_score("task-a", "t0", 1.0)],
        trials=[_trial("t0"), unmeasured],
    )

    assert summary.error_trial_ids == {"RuntimeError": ["t1"]}
    assert summary.trial_count == 2
    assert summary.score_count == 1
    assert "task-b" not in summary.task_metric_values


def test_trial_count_without_trials_is_distinct_score_pairs() -> None:
    # Same trial_id on two tasks is two attempts. Two score rows for one pair are one attempt.
    two_tasks = AgentEvalSummary.from_scores([_score("task-a", "shared", 1.0), _score("task-b", "shared", 0.0)])
    two_metrics = AgentEvalSummary.from_scores([_score("task-a", "t0", 1.0), _score("task-a", "t0", 0.0)])

    assert two_tasks.trial_count == 2
    assert two_metrics.trial_count == 1


def test_trial_count_with_trials_is_the_supplied_list_length() -> None:
    summary = AgentEvalSummary.from_scores(
        [_score("task-a", "shared", 1.0), _score("task-b", "shared", 0.0)],
        trials=[_trial("shared"), _trial("shared", task_id="task-b")],
    )

    assert summary.trial_count == 2


def test_error_count_must_agree_with_the_rollup() -> None:
    # The model is public and directly constructible; a count contradicting the rollup beside it is
    # worse than no count at all.
    with pytest.raises(ValidationError, match="does not match"):
        AgentEvalSummary(error_trial_ids={"RuntimeError": ["t0", "t1"]}, error_count=1)

    ok = AgentEvalSummary(error_trial_ids={"RuntimeError": ["t0", "t1"]}, error_count=2)
    assert ok.error_count == 2


def test_helper_returns_an_empty_rollup_for_no_trials() -> None:
    assert _error_trial_ids(None) == {}
    assert _error_trial_ids([]) == {}


def test_summary_round_trips_the_rollup_through_json() -> None:
    summary = AgentEvalSummary.from_scores([], trials=[_trial("t0", error="RuntimeError")])

    reloaded = AgentEvalSummary.model_validate(summary.model_dump(mode="json"))

    assert reloaded.error_trial_ids == {"RuntimeError": ["t0"]}
    assert reloaded.error_count == 1


def test_vendored_module_exposes_the_error_rollup_surface() -> None:
    # The byte-copy pin proves file parity, not that these names are importable through the shipped
    # package -- which is the path a nemo-platform consumer actually uses.
    vendored_results = _vendored_module("results")
    vendored_trials = _vendored_module("trials")
    VendoredSummary = vendored_results.AgentEvalSummary
    VendoredTrial = vendored_trials.AgentEvalTrial
    VendoredStatus = vendored_trials.AgentEvalTrialStatus
    VendoredError = vendored_trials.TrialError

    trial = VendoredTrial(
        id="t0",
        task_id="task-a",
        status=VendoredStatus.PARTIAL,
        error=VendoredError(type="RuntimeError", message="boom"),
    )
    summary = VendoredSummary.from_scores([], trials=[trial])

    assert summary.error_trial_ids == {"RuntimeError": ["t0"]}
    assert summary.error_count == 1
    assert trial.get_evidence("result") is None
