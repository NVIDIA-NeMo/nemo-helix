# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate evaluator request and trial token usage for platform jobs."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

from nemo_evaluator_sdk.agent_eval.results import AgentEvalResult
from nemo_evaluator_sdk.inference import requests_log_var
from nemo_evaluator_sdk.values.multi_metric_results import BenchmarkEvaluationResult
from nemo_evaluator_sdk.values.results import EvaluationResult
from nemo_platform_plugin.job_usage import JobUsageReporter

_INPUT_TOKEN_KEYS = ("prompt_tokens", "input_tokens", "inputTokens")
_OUTPUT_TOKEN_KEYS = ("completion_tokens", "output_tokens", "outputTokens")


def _token_count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _first_token_count(usage: Mapping[str, object], keys: Sequence[str]) -> int | None:
    for key in keys:
        if (value := _token_count(usage.get(key))) is not None:
            return value
    return None


def _request_usage(request_log: Mapping[str, object]) -> tuple[int | None, int | None]:
    response = request_log.get("response")
    if not isinstance(response, Mapping):
        return None, None
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        return None, None
    typed_usage = cast(Mapping[str, object], usage)
    return _first_token_count(typed_usage, _INPUT_TOKEN_KEYS), _first_token_count(typed_usage, _OUTPUT_TOKEN_KEYS)


@dataclass
class _UsageAccumulator:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    input_complete: bool = True
    output_complete: bool = True

    def add(self, input_tokens: int | None, output_tokens: int | None) -> None:
        self.calls += 1
        if input_tokens is None:
            self.input_complete = False
        else:
            self.input_tokens += input_tokens
        if output_tokens is None:
            self.output_complete = False
        else:
            self.output_tokens += output_tokens

    def report(self, reporter: JobUsageReporter) -> None:
        if self.calls == 0:
            return
        input_tokens = self.input_tokens if self.input_complete else None
        output_tokens = self.output_tokens if self.output_complete else None
        if input_tokens is None and output_tokens is None:
            return
        reporter.report_totals(input_tokens=input_tokens, output_tokens=output_tokens)


def _add_request_logs(accumulator: _UsageAccumulator, request_logs: Sequence[Mapping[str, object]]) -> None:
    for request_log in request_logs:
        accumulator.add(*_request_usage(request_log))


def report_row_evaluation_usage(
    result: EvaluationResult | BenchmarkEvaluationResult,
    reporter: JobUsageReporter,
) -> None:
    """Report all target and metric calls captured in row-evaluation results."""
    accumulator = _UsageAccumulator()
    for row in result.row_scores:
        _add_request_logs(accumulator, row.requests)
    accumulator.report(reporter)


def report_agent_evaluation_usage(
    result: AgentEvalResult,
    request_logs: Sequence[Mapping[str, object]],
    reporter: JobUsageReporter,
    *,
    include_trial_measurements: bool,
) -> None:
    """Report judge calls and, for runner targets, their trial measurements."""
    accumulator = _UsageAccumulator()
    _add_request_logs(accumulator, request_logs)
    if include_trial_measurements:
        for trial in result.trials:
            accumulator.add(trial.measurements.prompt_tokens, trial.measurements.completion_tokens)
    accumulator.report(reporter)


@contextmanager
def capture_evaluator_request_logs() -> Iterator[list[dict[str, Any]]]:
    """Capture evaluator inference calls that are not persisted on agent scores."""
    request_logs: list[dict[str, Any]] = []
    token = requests_log_var.set(request_logs)
    try:
        yield request_logs
    finally:
        requests_log_var.reset(token)


__all__ = [
    "capture_evaluator_request_logs",
    "report_agent_evaluation_usage",
    "report_row_evaluation_usage",
]
