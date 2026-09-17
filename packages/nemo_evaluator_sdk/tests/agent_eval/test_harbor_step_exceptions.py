# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A multi-step Harbor trial records its exception on the step, not on the trial."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nemo_evaluator_sdk.agent_eval.runtimes.harbor_trial_adapter import _trial_from_harbor_result
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus

_STEP_EXCEPTION = {
    "exception_type": "RuntimeError",
    "exception_message": "available adapters: []",
    "exception_traceback": "Traceback (most recent call last): ...",
    "occurred_at": "2026-09-16T12:00:00+00:00",
}


def _trial(tmp_path: Path, **overrides: Any) -> AgentEvalTrial:
    data: dict[str, Any] = {
        "task_name": "task-1",
        "trial_name": "trial-1",
        "verifier_result": {"rewards": {"reward": 0.0}},
        **overrides,
    }
    return _trial_from_harbor_result(tmp_path, data, reward_key="reward")


def test_a_step_level_exception_becomes_the_trials_error(tmp_path: Path) -> None:
    # Without this the trial normalizes clean: reward 0.0, no error, status COMPLETED — a crashed
    # agent reported as a score of zero, which is the failure this guards.
    trial = _trial(tmp_path, step_results=[{"step_name": "agent", "exception_info": _STEP_EXCEPTION}])

    assert trial.error is not None
    assert trial.error.type == "RuntimeError"
    assert trial.error.message == "available adapters: []"
    assert trial.status is AgentEvalTrialStatus.PARTIAL


def test_the_first_step_that_raised_is_the_one_reported(tmp_path: Path) -> None:
    trial = _trial(
        tmp_path,
        step_results=[
            {"step_name": "setup", "exception_info": None},
            {"step_name": "agent", "exception_info": _STEP_EXCEPTION},
            {"step_name": "verify", "exception_info": {"exception_type": "TimeoutError"}},
        ],
    )

    assert trial.error is not None and trial.error.type == "RuntimeError"


def test_a_trial_level_exception_wins_over_a_step_level_one(tmp_path: Path) -> None:
    # Harbor sets the trial-level field for the failure that ended the trial; a step's exception may
    # have been retried past. Preferring the step would rename the error a caller sees.
    trial = _trial(
        tmp_path,
        exception_info={"exception_type": "TimeoutError"},
        step_results=[{"step_name": "agent", "exception_info": _STEP_EXCEPTION}],
    )

    assert trial.error is not None and trial.error.type == "TimeoutError"


def test_steps_that_all_succeeded_leave_the_trial_without_an_error(tmp_path: Path) -> None:
    trial = _trial(tmp_path, step_results=[{"step_name": "agent", "exception_info": None}])

    assert trial.error is None
    assert trial.status is AgentEvalTrialStatus.COMPLETED


def test_a_malformed_step_results_payload_is_ignored_rather_than_raising(tmp_path: Path) -> None:
    # Normalization never raises on a shape it did not expect; the adapter's other readers follow
    # the same rule, and a crash here would lose the whole run over one odd field.
    for steps in ("not-a-list", [None, 7], [{"step_name": "agent"}], []):
        trial = _trial(tmp_path, step_results=steps)

        assert trial.error is None
