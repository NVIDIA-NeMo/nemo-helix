# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wall-clock duration of the attempt that succeeded, excluding admission waits, failed attempts, and backoff."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class AttemptTiming:
    seconds: float | None = None


_current_timing: ContextVar[AttemptTiming | None] = ContextVar("resilience_attempt_timing", default=None)


@contextmanager
def time_successful_attempt() -> Iterator[AttemptTiming]:
    """Capture how long the last successful resilience attempt inside the block took."""
    timing = AttemptTiming()
    token = _current_timing.set(timing)
    try:
        yield timing
    finally:
        _current_timing.reset(token)


def record_successful_attempt(seconds: float) -> None:
    timing = _current_timing.get()
    if timing is not None:
        timing.seconds = seconds
