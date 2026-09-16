# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Data Designer job token-usage aggregation."""

from __future__ import annotations

import pytest
from data_designer.engine.models.usage_events import TokenUsageEvent, emit_token_usage_event
from data_designer.engine.observability import RuntimeCorrelation, runtime_correlation_provider
from nemo_data_designer_plugin.jobs.token_usage import capture_token_usage
from nemo_platform_plugin.job_usage import LocalJobUsageReporter


def _event(*, input_tokens: int, output_tokens: int, correlation: RuntimeCorrelation | None) -> TokenUsageEvent:
    return TokenUsageEvent(
        model_alias="model",
        model_name="provider/model",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        correlation=correlation,
    )


def test_capture_token_usage_reports_cumulative_totals() -> None:
    reporter = LocalJobUsageReporter()

    with capture_token_usage(reporter):
        correlation = runtime_correlation_provider.current()
        emit_token_usage_event(_event(input_tokens=10, output_tokens=4, correlation=correlation))
        emit_token_usage_event(_event(input_tokens=7, output_tokens=3, correlation=correlation))

    assert reporter.latest is not None
    assert reporter.latest.input_tokens == 17
    assert reporter.latest.output_tokens == 7


def test_capture_token_usage_ignores_other_runs() -> None:
    reporter = LocalJobUsageReporter()

    with capture_token_usage(reporter):
        current = runtime_correlation_provider.current()
        assert current is not None
        unrelated = RuntimeCorrelation(
            run_id="another-run",
            row_group=None,
            task_column=None,
            task_type=None,
            scheduling_group_kind=None,
            scheduling_group_identity_hash=None,
            task_execution_id=None,
        )
        emit_token_usage_event(_event(input_tokens=100, output_tokens=100, correlation=unrelated))
        emit_token_usage_event(_event(input_tokens=2, output_tokens=1, correlation=current))

    assert reporter.latest is not None
    assert reporter.latest.input_tokens == 2
    assert reporter.latest.output_tokens == 1


def test_capture_token_usage_leaves_no_report_without_events() -> None:
    reporter = LocalJobUsageReporter()

    with capture_token_usage(reporter):
        pass

    assert reporter.latest is None


def test_capture_token_usage_reports_before_propagating_failure() -> None:
    reporter = LocalJobUsageReporter()

    with pytest.raises(RuntimeError, match="generation failed"):
        with capture_token_usage(reporter):
            correlation = runtime_correlation_provider.current()
            emit_token_usage_event(_event(input_tokens=5, output_tokens=2, correlation=correlation))
            raise RuntimeError("generation failed")

    assert reporter.latest is not None
    assert reporter.latest.input_tokens == 5
    assert reporter.latest.output_tokens == 2


def test_capture_token_usage_restores_outer_correlation() -> None:
    reporter = LocalJobUsageReporter()
    outer = RuntimeCorrelation(
        run_id="outer",
        row_group=None,
        task_column=None,
        task_type=None,
        scheduling_group_kind=None,
        scheduling_group_identity_hash=None,
        task_execution_id=None,
    )
    token = runtime_correlation_provider.set(outer)
    try:
        with capture_token_usage(reporter):
            assert runtime_correlation_provider.current() != outer
        assert runtime_correlation_provider.current() == outer
    finally:
        runtime_correlation_provider.reset(token)
