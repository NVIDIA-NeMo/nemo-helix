# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data Designer token-usage aggregation for NeMo Platform jobs."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock
from uuid import uuid4

from data_designer.engine.models.usage_events import TokenUsageEvent, subscribe_token_usage
from data_designer.engine.observability import RuntimeCorrelation, runtime_correlation_provider
from nemo_platform_plugin.job_usage import JobUsageReporter


class _TokenUsageAccumulator:
    """Thread-safe totals for one Data Designer runtime-correlation run."""

    def __init__(self, run_id: str) -> None:
        """Create an accumulator scoped to one generated Data Designer run id."""
        self._run_id = run_id
        self._lock = Lock()
        self._input_tokens = 0
        self._output_tokens = 0
        self._event_count = 0

    def record(self, event: TokenUsageEvent) -> None:
        """Add matching token-usage events and ignore unrelated runs."""
        if event.correlation is None or event.correlation.run_id != self._run_id:
            return
        with self._lock:
            self._input_tokens += event.input_tokens
            self._output_tokens += event.output_tokens
            self._event_count += 1

    def report(self, reporter: JobUsageReporter) -> None:
        """Publish accumulated totals when at least one event was captured."""
        with self._lock:
            if self._event_count == 0:
                return
            input_tokens = self._input_tokens
            output_tokens = self._output_tokens
        reporter.report_totals(input_tokens=input_tokens, output_tokens=output_tokens)


@contextmanager
def capture_token_usage(reporter: JobUsageReporter) -> Iterator[None]:
    """Capture Data Designer model-token events and report them on exit.

    The wrapper owns the full setup/teardown sequence shared by Data Designer
    and Anonymizer tasks: create a fresh runtime correlation, subscribe to the
    Data Designer token-event bus, restore the previous correlation, unsubscribe,
    and report cumulative totals even when the wrapped generation raises.
    """
    run_id = f"nemo-platform-{uuid4().hex}"
    accumulator = _TokenUsageAccumulator(run_id)
    unsubscribe = subscribe_token_usage(accumulator.record)
    correlation_token = runtime_correlation_provider.set(
        RuntimeCorrelation(
            run_id=run_id,
            row_group=None,
            task_column=None,
            task_type=None,
            scheduling_group_kind=None,
            scheduling_group_identity_hash=None,
            task_execution_id=None,
        )
    )
    try:
        yield
    finally:
        runtime_correlation_provider.reset(correlation_token)
        unsubscribe()
        accumulator.report(reporter)


__all__ = ["capture_token_usage"]
