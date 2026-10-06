# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The run-outcome rollup: when a tolerated-failure run has in fact produced nothing usable."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from nemo_evaluator.jobs.run_outcome import (
    STATUS_DETAILS_KEY,
    RunOutcome,
    agent_eval_outcome,
    report_run_outcome,
    row_eval_outcome,
)
from nemo_evaluator_sdk.agent_eval.results import AgentEvalResult, AgentEvalSummary
from nemo_evaluator_sdk.agent_eval.scores import AgentEvalScoreStatus, AgentEvalTaskScore
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus, AgentOutput, TrialError
from nemo_evaluator_sdk.values.protocol import MetricOutput
from nemo_evaluator_sdk.values.results import AggregatedMetricResult, EvaluationResult, RowScore
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.job_context import JobContext, StoragePaths
from nemo_helix_plugin.job_results import LocalJobResults
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from pytest_mock import MockerFixture


def _trial(trial_id: str, *, status: AgentEvalTrialStatus, error: TrialError | None = None) -> AgentEvalTrial:
    return AgentEvalTrial(
        id=trial_id,
        task_id="task-1",
        status=status,
        output=None if status is AgentEvalTrialStatus.FAILED else AgentOutput(output_text="4"),
        error=error,
    )


def _score(trial_id: str, *, status: AgentEvalScoreStatus, value: float = 1.0) -> AgentEvalTaskScore:
    return AgentEvalTaskScore(
        id=f"{trial_id}:m",
        run_id="run-1",
        task_id="task-1",
        trial_id=trial_id,
        metric_type="m",
        status=status,
        outputs=[] if status is AgentEvalScoreStatus.FAILED else [MetricOutput(name="score", value=value)],
    )


def _agent_result(trials: list[AgentEvalTrial], scores: list[AgentEvalTaskScore]) -> AgentEvalResult:
    return AgentEvalResult(run_id="run-1", tasks=[], trials=trials, scores=scores, summary=AgentEvalSummary())


def _row(*, value: float | None, inference_error: str | None = None, metric_error: str | None = None) -> RowScore:
    sample: dict[str, object] = {"output_text": None if inference_error else "4"}
    if inference_error is not None:
        sample["inference_error"] = inference_error
    return RowScore(
        item={},
        sample=sample,
        metrics={"m": [MetricOutput(name="m", value=value)]} if value is not None else {"m": []},
        requests=[],
        metric_errors={"m": metric_error} if metric_error is not None else None,
    )


def _row_result(rows: list[RowScore]) -> EvaluationResult:
    return EvaluationResult(row_scores=rows, aggregate_scores=AggregatedMetricResult(scores=[]))


class TestVerdict:
    def test_fails_only_when_units_exist_and_none_scored(self) -> None:
        assert RunOutcome(unit="trials", total=3, errored=3, scored=0).failed
        assert not RunOutcome(unit="trials", total=3, errored=2, scored=1).failed
        # An empty run is not a failed run: there was nothing to score, which is a different problem.
        assert not RunOutcome(unit="rows", total=0, errored=0, scored=0).failed

    def test_details_carry_the_verdict_and_message_under_the_status_key(self) -> None:
        outcome = RunOutcome(unit="rows", total=28, errored=28, scored=0)

        details = outcome.status_details()

        assert details == {
            STATUS_DETAILS_KEY: {
                "unit": "rows",
                "total": 28,
                "errored": 28,
                "scored": 0,
                "failed": True,
                "message": (
                    "No usable scores across 28 rows (28 reported errors). "
                    "See the job logs for the underlying request or metric failures."
                ),
            }
        }


class TestAgentEvalOutcome:
    def test_every_trial_failed_to_generate(self) -> None:
        """The Studio path: ``ignore_request_failure`` turns each dead agent call into a FAILED trial."""
        trials = [
            _trial("t1", status=AgentEvalTrialStatus.FAILED, error=TrialError(type="ConnectError")),
            _trial("t2", status=AgentEvalTrialStatus.FAILED, error=TrialError(type="ConnectError")),
        ]
        scores = [_score("t1", status=AgentEvalScoreStatus.FAILED), _score("t2", status=AgentEvalScoreStatus.FAILED)]

        outcome = agent_eval_outcome(_agent_result(trials, scores))

        assert outcome == RunOutcome(unit="trials", total=2, errored=2, scored=0)
        assert outcome.failed

    def test_trials_generated_but_every_judge_call_failed(self) -> None:
        """Trials completed, so nothing is 'errored', yet no score exists: still a failed run."""
        trials = [_trial("t1", status=AgentEvalTrialStatus.COMPLETED)]
        scores = [_score("t1", status=AgentEvalScoreStatus.FAILED)]

        outcome = agent_eval_outcome(_agent_result(trials, scores))

        assert outcome == RunOutcome(unit="trials", total=1, errored=0, scored=0)
        assert outcome.failed

    def test_a_completed_score_holding_only_nan_is_not_scored(self) -> None:
        """An LLM judge under ``ignore_request_failure`` returns NaN outputs with a COMPLETED status."""
        trials = [_trial("t1", status=AgentEvalTrialStatus.COMPLETED)]
        scores = [_score("t1", status=AgentEvalScoreStatus.COMPLETED, value=float("nan"))]

        outcome = agent_eval_outcome(_agent_result(trials, scores))

        assert outcome == RunOutcome(unit="trials", total=1, errored=0, scored=0)
        assert outcome.failed

    def test_one_scored_trial_keeps_the_run_completed(self) -> None:
        trials = [
            _trial("t1", status=AgentEvalTrialStatus.FAILED, error=TrialError(type="ConnectError")),
            _trial("t2", status=AgentEvalTrialStatus.COMPLETED),
        ]
        scores = [
            _score("t1", status=AgentEvalScoreStatus.FAILED),
            _score("t2", status=AgentEvalScoreStatus.PARTIAL),
        ]

        outcome = agent_eval_outcome(_agent_result(trials, scores))

        assert outcome == RunOutcome(unit="trials", total=2, errored=1, scored=1)
        assert not outcome.failed

    def test_duplicate_trial_ids_do_not_inflate_the_scored_count(self) -> None:
        """Gym can emit two trials with one id; a single usable score must not mark both as scored."""
        trials = [
            _trial("t1", status=AgentEvalTrialStatus.COMPLETED),
            _trial("t1", status=AgentEvalTrialStatus.FAILED, error=TrialError(type="ConnectError")),
        ]
        scores = [_score("t1", status=AgentEvalScoreStatus.COMPLETED), _score("t1", status=AgentEvalScoreStatus.FAILED)]

        outcome = agent_eval_outcome(_agent_result(trials, scores))

        assert outcome == RunOutcome(unit="trials", total=2, errored=1, scored=1)

    def test_an_errored_partial_trial_counts_as_errored_and_scored(self) -> None:
        """Harbor marks an errored trial PARTIAL so it still scores; it belongs in both counts."""
        trials = [_trial("t1", status=AgentEvalTrialStatus.PARTIAL, error=TrialError(type="TimeoutError"))]
        scores = [_score("t1", status=AgentEvalScoreStatus.COMPLETED)]

        outcome = agent_eval_outcome(_agent_result(trials, scores))

        assert outcome == RunOutcome(unit="trials", total=1, errored=1, scored=1)


class TestRowEvalOutcome:
    def test_every_row_is_a_nan_placeholder(self) -> None:
        rows = [
            _row(value=float("nan"), inference_error="boom", metric_error="boom"),
            _row(value=float("nan"), metric_error="judge down"),
        ]

        outcome = row_eval_outcome(_row_result(rows))

        assert outcome == RunOutcome(unit="rows", total=2, errored=2, scored=0)
        assert outcome.failed

    def test_a_row_with_a_finite_value_is_scored(self) -> None:
        rows = [_row(value=float("nan"), metric_error="boom"), _row(value=0.0)]

        outcome = row_eval_outcome(_row_result(rows))

        assert outcome == RunOutcome(unit="rows", total=2, errored=1, scored=1)
        assert not outcome.failed

    def test_non_numeric_outputs_count_as_scored(self) -> None:
        rows = [
            RowScore(item={}, sample={}, metrics={"label": [MetricOutput(name="label", value="spam")]}, requests=[])
        ]

        assert row_eval_outcome(_row_result(rows)).scored == 1

    def test_empty_result_is_not_failed(self) -> None:
        assert not row_eval_outcome(_row_result([])).failed


def _ctx(tmp_path: Path, job_id: str | None) -> JobContext:
    storage = StoragePaths(ephemeral=tmp_path / "e", persistent=tmp_path / "p")
    storage.ephemeral.mkdir()
    storage.persistent.mkdir()
    return JobContext(
        workspace="dev", storage=storage, results=LocalJobResults(root=storage.persistent / "r"), job_id=job_id
    )


def _async_client() -> AsyncNemoClient:
    return AsyncNemoClient(base_url="http://platform.test", workspace="dev", http_client=httpx.AsyncClient())


class TestReportRunOutcome:
    def test_patches_the_job_status_details(self, tmp_path: Path, mocker: MockerFixture) -> None:
        update = mocker.patch.object(AsyncJobsClient, "update_status_details", AsyncMock())
        outcome = RunOutcome(unit="trials", total=2, errored=2, scored=0)

        report_run_outcome(outcome, ctx=_ctx(tmp_path, "job-1"), async_client=_async_client())

        update.assert_awaited_once_with("job-1", workspace="dev", body=outcome.status_details())

    @pytest.mark.parametrize("job_id", [None, "job-1"])
    def test_skips_without_a_platform_job_or_client(self, tmp_path: Path, mocker: MockerFixture, job_id) -> None:
        update = mocker.patch.object(AsyncJobsClient, "update_status_details", AsyncMock())
        outcome = RunOutcome(unit="rows", total=1, errored=0, scored=1)

        report_run_outcome(outcome, ctx=_ctx(tmp_path, job_id), async_client=None if job_id else _async_client())

        update.assert_not_awaited()

    def test_is_best_effort(self, tmp_path: Path, mocker: MockerFixture) -> None:
        mocker.patch.object(AsyncJobsClient, "update_status_details", AsyncMock(side_effect=RuntimeError("down")))

        report_run_outcome(
            RunOutcome(unit="rows", total=1, errored=0, scored=1),
            ctx=_ctx(tmp_path, "job-1"),
            async_client=_async_client(),
        )
