# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time a target spends serving one sample, excluding admission waits, retried failures, and backoff."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class TargetRuntime:
    """Accumulates the time one sample's target calls spent in flight.

    Attempts dispatched by the resilience scheduler are preferred: each scheduler operation
    contributes the duration of its final attempt, so earlier attempts that were retried and the
    backoff between them are excluded. Calls that bypass the scheduler (a custom inference function) fall back to
    the wall-clock duration of the target call.
    """

    _succeeded_attempt_seconds: float = 0.0
    _last_failure_seconds_by_operation: dict[object, float] = field(default_factory=dict)
    _scheduled_attempts: int = 0
    _target_call_seconds: float = 0.0
    _target_calls: int = 0

    @property
    def seconds(self) -> float | None:
        if self._scheduled_attempts:
            return self._succeeded_attempt_seconds + sum(self._last_failure_seconds_by_operation.values())
        if self._target_calls:
            return self._target_call_seconds
        return None

    def record_attempt(self, seconds: float, *, succeeded: bool, operation: object) -> None:
        self._scheduled_attempts += 1
        if succeeded:
            self._succeeded_attempt_seconds += seconds
            self._last_failure_seconds_by_operation.pop(operation, None)
        else:
            self._last_failure_seconds_by_operation[operation] = seconds

    def record_target_call(self, seconds: float) -> None:
        self._target_calls += 1
        self._target_call_seconds += seconds


_current_runtime: ContextVar[TargetRuntime | None] = ContextVar("resilience_target_runtime", default=None)


@contextmanager
def measure_target_runtime() -> Iterator[TargetRuntime]:
    """Collect target timing for everything awaited inside the block, whether it succeeds or raises."""
    runtime = TargetRuntime()
    token = _current_runtime.set(runtime)
    try:
        yield runtime
    finally:
        _current_runtime.reset(token)


@contextmanager
def timed_target_call() -> Iterator[None]:
    """Record the wall-clock duration of one call into the target as the fallback runtime."""
    started = time.monotonic()
    try:
        yield
    finally:
        runtime = _current_runtime.get()
        if runtime is not None:
            runtime.record_target_call(max(0.0, time.monotonic() - started))


@contextmanager
def untimed() -> Iterator[None]:
    """Keep calls inside the block out of the enclosing target runtime."""
    token = _current_runtime.set(None)
    try:
        yield
    finally:
        _current_runtime.reset(token)


def record_attempt(seconds: float, *, succeeded: bool, operation: object) -> None:
    runtime = _current_runtime.get()
    if runtime is not None:
        runtime.record_attempt(seconds, succeeded=succeeded, operation=operation)
