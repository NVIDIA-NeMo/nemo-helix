# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for evaluator job token-usage aggregation."""

from __future__ import annotations

from datetime import UTC, datetime

from nemo_evaluator.jobs.token_usage import (
    capture_evaluator_request_logs,
    report_agent_evaluation_usage,
    report_row_evaluation_usage,
)
from nemo_evaluator_sdk.agent_eval.results import AgentEvalResult, AgentEvalSummary, RunMetadata
from nemo_evaluator_sdk.agent_eval.trials import (
    AgentEvalTrial,
    AgentEvalTrialStatus,
    AgentOutput,
    TrialMeasurements,
)
from nemo_evaluator_sdk.inference import requests_log_var
from nemo_evaluator_sdk.values.results import AggregatedMetricResult, EvaluationResult, RowScore
from nemo_platform_plugin.job_usage import LocalJobUsageReporter


def _request(input_tokens: object = 10, output_tokens: object = 4) -> dict:
    return {"response": {"usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens}}}


def _row_result(requests: list[dict]) -> EvaluationResult:
    return EvaluationResult(
        row_scores=[RowScore(item={}, sample={}, metrics={}, requests=requests)],
        aggregate_scores=AggregatedMetricResult(scores=[]),
    )


def _agent_result(*measurements: TrialMeasurements) -> AgentEvalResult:
    trials = [
        AgentEvalTrial(
            id=f"trial-{index}",
            task_id="task",
            status=AgentEvalTrialStatus.COMPLETED,
            output=AgentOutput(output_text="ok"),
            measurements=value,
        )
        for index, value in enumerate(measurements)
    ]
    return AgentEvalResult(
        run_id="run",
        tasks=[],
        trials=trials,
        scores=[],
        summary=AgentEvalSummary(),
        metadata=RunMetadata(started_at=datetime.now(UTC)),
    )


def test_row_usage_sums_target_and_metric_requests() -> None:
    reporter = LocalJobUsageReporter()

    report_row_evaluation_usage(_row_result([_request(10, 4), _request(7, 3)]), reporter)

    assert reporter.latest is not None
    assert reporter.latest.input_tokens == 17
    assert reporter.latest.output_tokens == 7


def test_missing_request_dimension_keeps_that_total_unknown() -> None:
    reporter = LocalJobUsageReporter()

    report_row_evaluation_usage(_row_result([_request(10, 4), _request(None, 3)]), reporter)

    assert reporter.latest is not None
    assert reporter.latest.input_tokens is None
    assert reporter.latest.output_tokens == 7


def test_invalid_counts_are_missing_not_zero() -> None:
    reporter = LocalJobUsageReporter()

    report_row_evaluation_usage(_row_result([_request(True, 4), _request(-1, 3)]), reporter)

    assert reporter.latest is not None
    assert reporter.latest.input_tokens is None
    assert reporter.latest.output_tokens == 7


def test_agent_runner_usage_combines_trials_and_judge_requests() -> None:
    reporter = LocalJobUsageReporter()
    result = _agent_result(
        TrialMeasurements(prompt_tokens=20, completion_tokens=5),
        TrialMeasurements(prompt_tokens=30, completion_tokens=8),
    )

    report_agent_evaluation_usage(
        result,
        [_request(11, 2)],
        reporter,
        include_trial_measurements=True,
    )

    assert reporter.latest is not None
    assert reporter.latest.input_tokens == 61
    assert reporter.latest.output_tokens == 15


def test_agent_http_target_does_not_double_count_trial_measurements() -> None:
    reporter = LocalJobUsageReporter()
    result = _agent_result(TrialMeasurements(prompt_tokens=20, completion_tokens=5))

    report_agent_evaluation_usage(
        result,
        [_request(20, 5), _request(11, 2)],
        reporter,
        include_trial_measurements=False,
    )

    assert reporter.latest is not None
    assert reporter.latest.input_tokens == 31
    assert reporter.latest.output_tokens == 7


def test_request_capture_restores_outer_log() -> None:
    outer: list[dict] = []
    token = requests_log_var.set(outer)
    try:
        with capture_evaluator_request_logs() as captured:
            assert requests_log_var.get() is captured
            captured.append(_request())
        assert requests_log_var.get() is outer
    finally:
        requests_log_var.reset(token)
