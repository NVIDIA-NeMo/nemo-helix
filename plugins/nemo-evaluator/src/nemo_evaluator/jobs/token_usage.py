# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate evaluator request and trial token usage for platform jobs."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

from nemo_evaluator_sdk.agent_eval.results import AgentEvalResult
from nemo_evaluator_sdk.inference import requests_log_var
from nemo_evaluator_sdk.values.multi_metric_results import BenchmarkEvaluationResult
from nemo_evaluator_sdk.values.results import EvaluationResult
from nemo_platform_plugin.job_usage import JobTokenUsage, JobUsageReporter

logger = logging.getLogger(__name__)

T = TypeVar("T")
_NO_RESULT = object()

_INPUT_TOKEN_KEYS = ("prompt_tokens", "input_tokens", "inputTokens")
_INPUT_CACHE_TOKEN_KEYS = ("cache_read_input_tokens", "cache_creation_input_tokens")
_OUTPUT_TOKEN_KEYS = ("completion_tokens", "output_tokens", "outputTokens")


def _token_count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _first_token_count(usage: Mapping[str, object], keys: Sequence[str]) -> int | None:
    for key in keys:
        if (value := _token_count(usage.get(key))) is not None:
            return value
    return None


def _optional_token_count(usage: Mapping[str, object], key: str) -> int | None:
    if key not in usage:
        return 0
    return _token_count(usage.get(key))


def _input_token_count(usage: Mapping[str, object]) -> int | None:
    base_input_tokens = _first_token_count(usage, _INPUT_TOKEN_KEYS)
    if base_input_tokens is None:
        return None
    cache_token_counts = [_optional_token_count(usage, key) for key in _INPUT_CACHE_TOKEN_KEYS]
    if any(count is None for count in cache_token_counts):
        return None
    return base_input_tokens + sum(count for count in cache_token_counts if count is not None)


def _request_usage(request_log: Mapping[str, object]) -> tuple[int | None, int | None]:
    response = request_log.get("response")
    if not isinstance(response, Mapping):
        return None, None
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        return None, None
    typed_usage = cast(Mapping[str, object], usage)
    return _input_token_count(typed_usage), _first_token_count(typed_usage, _OUTPUT_TOKEN_KEYS)


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

    def usage(self) -> JobTokenUsage | None:
        if self.calls == 0:
            return None
        input_tokens = self.input_tokens if self.input_complete else None
        output_tokens = self.output_tokens if self.output_complete else None
        if input_tokens is None and output_tokens is None:
            return None
        return JobTokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)

    def report(self, reporter: JobUsageReporter) -> None:
        usage = self.usage()
        if usage is not None:
            reporter.report_totals(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)


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


class _ResultCapture(Generic[T]):
    """Evaluator-local holder for a result that may not exist if run_sync raises."""

    def __init__(self) -> None:
        self._result: T | object = _NO_RESULT

    @property
    def has_result(self) -> bool:
        return self._result is not _NO_RESULT

    @property
    def result(self) -> T:
        if self._result is _NO_RESULT:
            raise RuntimeError("No evaluator result was recorded")
        return cast(T, self._result)

    def record(self, result: T) -> T:
        self._result = result
        return result


def _agent_evaluation_usage(
    capture: _ResultCapture[AgentEvalResult],
    request_logs: Sequence[Mapping[str, object]],
    *,
    include_trial_measurements: bool,
) -> JobTokenUsage | None:
    accumulator = _UsageAccumulator()
    _add_request_logs(accumulator, request_logs)
    if include_trial_measurements and capture.has_result:
        for trial in capture.result.trials:
            accumulator.add(trial.measurements.prompt_tokens, trial.measurements.completion_tokens)
    return accumulator.usage()


def report_agent_evaluation_usage(
    result: AgentEvalResult,
    request_logs: Sequence[Mapping[str, object]],
    reporter: JobUsageReporter,
    *,
    include_trial_measurements: bool,
) -> None:
    """Report judge calls and, for runner targets, their trial measurements."""
    capture: _ResultCapture[AgentEvalResult] = _ResultCapture()
    capture.record(result)
    usage = _agent_evaluation_usage(capture, request_logs, include_trial_measurements=include_trial_measurements)
    if usage is not None:
        reporter.report_totals(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)


@contextmanager
def capture_agent_evaluation_usage(
    reporter: JobUsageReporter,
    request_logs: Sequence[Mapping[str, object]],
    *,
    include_trial_measurements: bool,
) -> Iterator[_ResultCapture[AgentEvalResult]]:
    """Report agent-evaluation usage in a source-local finalizer.

    Request logs are side-channel data and can be reported even if the evaluator
    raises before returning a result. Trial measurements are result-dependent,
    so they are included only after ``record`` stores an ``AgentEvalResult``.
    """
    capture: _ResultCapture[AgentEvalResult] = _ResultCapture()
    try:
        yield capture
    finally:
        try:
            usage = _agent_evaluation_usage(
                capture,
                request_logs,
                include_trial_measurements=include_trial_measurements,
            )
            if usage is not None:
                reporter.report_totals(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)
        except Exception:
            logger.warning("Failed to report evaluator agent token usage", exc_info=True)


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
    "capture_agent_evaluation_usage",
    "capture_evaluator_request_logs",
    "report_agent_evaluation_usage",
    "report_row_evaluation_usage",
]
