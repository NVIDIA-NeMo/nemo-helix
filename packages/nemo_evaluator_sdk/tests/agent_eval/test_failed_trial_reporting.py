# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AALGO-648: a run whose trials failed must not read as a run that scored zero."""

from __future__ import annotations

from pathlib import Path

from nemo_evaluator_sdk.agent_eval.results import AgentEvalResult, AgentEvalSummary
from nemo_evaluator_sdk.agent_eval.scores import AgentEvalScoreStatus, AgentEvalTaskScore
from nemo_evaluator_sdk.agent_eval.trials import (
    AgentEvalTrial,
    AgentEvalTrialStatus,
    TrialError,
)
from nemo_evaluator_sdk.metrics.protocol import MetricOutput
from nemo_evaluator_sdk.values.results import AggregatedMetricResult


def _trial(trial_id: str, *, error_type: str | None = None, message: str | None = None) -> AgentEvalTrial:
    return AgentEvalTrial(
        id=trial_id,
        task_id="task-a",
        status=AgentEvalTrialStatus.PARTIAL,
        error=None if error_type is None else TrialError(type=error_type, message=message),
    )


def _score(trial_id: str, value: float) -> AgentEvalTaskScore:
    return AgentEvalTaskScore(
        id=f"run:task-a:{trial_id}:reward",
        run_id="run",
        task_id="task-a",
        trial_id=trial_id,
        metric_type="reward",
        status=AgentEvalScoreStatus.COMPLETED,
        outputs=[MetricOutput(name="reward", value=value)],
    )


def _result(*trials: AgentEvalTrial, work_dir: Path | None = None) -> AgentEvalResult:
    scores = [_score(trial.id, 0.0) for trial in trials]
    return AgentEvalResult(
        run_id="run",
        tasks=[],
        trials=list(trials),
        scores=scores,
        summary=AgentEvalSummary.from_scores(scores, trials=trials),
        work_dir=work_dir,
    )


def test_the_header_states_the_error_count_beside_the_score_counts() -> None:
    # The signal already existed on the summary and was never rendered: every failed run printed a
    # header indistinguishable from a clean run that happened to score zero.
    result = _result(_trial("t0", error_type="RuntimeError"), _trial("t1"))

    assert "errors=1" in result.format_summary()


def test_a_clean_run_has_no_error_field_in_its_header() -> None:
    result = _result(_trial("t0"), _trial("t1"))

    assert "errors=" not in result.format_summary()


def test_failed_trials_are_named_by_error_type_with_their_ids() -> None:
    result = _result(
        _trial("t0", error_type="RuntimeError"),
        _trial("t1", error_type="TimeoutError"),
        _trial("t2", error_type="RuntimeError"),
    )

    rendered = result.format_summary()

    assert "Failed trials (3 of 3)" in rendered
    assert "RuntimeError (2): t0, t2" in rendered
    assert "TimeoutError (1): t1" in rendered


def test_a_failure_message_is_shown_so_the_cause_is_readable_without_the_artifacts() -> None:
    result = _result(_trial("t0", error_type="RuntimeError", message="available adapters: []"))

    assert "t0: available adapters: []" in result.format_summary()


def test_a_clean_run_has_no_failed_trials_section() -> None:
    assert "Failed trials" not in _result(_trial("t0")).format_summary()


def test_a_wide_failure_truncates_the_ids_and_says_how_many_it_dropped() -> None:
    result = _result(*(_trial(f"t{index}", error_type="RuntimeError") for index in range(8)))

    rendered = result.format_summary()

    assert "RuntimeError (8): t0, t1, t2, t3, t4, ... (3 more)" in rendered


def test_the_section_points_at_the_directory_holding_the_trial_evidence(tmp_path: Path) -> None:
    # The cause is already on disk when a run fails this way; the ticket's complaint is that nothing
    # printed says so, leaving a zero score as the only thing a reader sees.
    result = _result(_trial("t0", error_type="RuntimeError"), work_dir=tmp_path)

    assert f"Trial evidence: {tmp_path}" in result.format_summary()


def test_an_in_memory_run_omits_the_evidence_pointer_rather_than_naming_nothing() -> None:
    result = _result(_trial("t0", error_type="RuntimeError"))

    assert "Trial evidence:" not in result.format_summary()


def test_a_summary_without_its_trials_states_the_count_without_a_false_denominator() -> None:
    # A persisted summary can outlive the trial list it was rolled up from. "3 of 0" would read as
    # a broken run rather than an absent population.
    result = AgentEvalResult(
        run_id="run",
        tasks=[],
        trials=[],
        scores=[],
        summary=AgentEvalSummary(
            scores=AggregatedMetricResult(scores=[]),
            error_trial_ids={"RuntimeError": ["a", "b", "c"]},
            error_count=3,
        ),
    )

    rendered = result.format_summary()

    assert "Failed trials (3)" in rendered
    assert "of 0" not in rendered


def test_trials_sharing_an_id_each_keep_their_own_message_and_error_type() -> None:
    # Trial ids are not unique (Gym derives them from a rollout index in two separate loops) and the
    # rollup preserves every occurrence. Keying messages by id alone let the last duplicate's text be
    # printed for an earlier one, under an error type that trial never had.
    result = _result(
        _trial("t0", error_type="RuntimeError", message="adapter missing"),
        _trial("t0", error_type="TimeoutError", message="timed out"),
    )

    rendered = result.format_summary()

    assert "RuntimeError (1): t0\n    t0: adapter missing" in rendered
    assert "TimeoutError (1): t0\n    t0: timed out" in rendered


def test_repeated_occurrences_of_one_error_type_are_consumed_in_trial_order() -> None:
    result = _result(
        _trial("t0", error_type="RuntimeError", message="first"),
        _trial("t0", error_type="RuntimeError", message="second"),
    )

    rendered = result.format_summary()

    assert "RuntimeError (2): t0, t0\n    t0: first\n    t0: second" in rendered
