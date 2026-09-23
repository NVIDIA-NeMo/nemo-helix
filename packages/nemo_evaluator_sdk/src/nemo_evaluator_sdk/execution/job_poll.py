# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async job polling aligned with ``nhx.testing.e2e.jobs.poll_until_terminal``.

The E2E helper lives in ``nhx_testing``; the evaluator SDK cannot depend on that
package at runtime, so this module mirrors its timeout semantics (pending time
excluded from the job timeout, separate image-pull cap).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import TypeVar

log = logging.getLogger(__name__)

_StatusT = TypeVar("_StatusT")


async def _async_pause(seconds: float) -> None:
    await asyncio.sleep(seconds)


async def async_poll_until_terminal(
    get_status: Callable[[], Awaitable[_StatusT]],
    *,
    status_value: Callable[[_StatusT], str],
    details_value: Callable[[_StatusT], Mapping[str, object] | None] | None = None,
    job_name: str,
    terminal: frozenset[str],
    timeout: float,
    pending_timeout: float,
    poll_interval: float,
) -> _StatusT:
    """Poll *get_status* until it returns a value in *terminal* or a timeout fires.

    *status_value* must return a **lowercase** status string for each response
    returned by *get_status*.

    *details_value* may return status details to include as a nested JSON object
    in each progress log line.

    Time spent in ``pending`` status is not counted against *timeout*; it is
    instead capped by the separate *pending_timeout*.

    Returns:
        The terminal response returned by *get_status*.

    Raises:
        TimeoutError: When *timeout* is exceeded (excluding pending time) or
            *pending_timeout* is exceeded while in pending status.
    """
    elapsed = 0.0
    pending_elapsed = 0.0

    while True:
        poll_start = time.monotonic()
        status_response = await get_status()
        status = status_value(status_response)
        log_payload: dict[str, object] = {
            "job_name": job_name,
            "status": status or "<empty>",
            "elapsed_s": round(elapsed, 1),
            "poll_interval_s": poll_interval,
        }
        if status == "pending":
            log_payload["pending_elapsed_s"] = round(pending_elapsed, 1)
        if details_value is not None:
            details = details_value(status_response)
            if details:
                log_payload["status_details"] = dict(details)
        log.info(json.dumps(log_payload, separators=(",", ":")))

        if status in terminal:
            return status_response

        poll_duration = time.monotonic() - poll_start
        if status == "pending":
            pending_elapsed += poll_duration
            if pending_elapsed >= pending_timeout:
                raise TimeoutError(f"'{job_name}' stuck in pending after {pending_timeout}s.")
        else:
            elapsed += poll_duration
            if elapsed >= timeout:
                raise TimeoutError(f"'{job_name}' timed out after {timeout}s. Status: {status}")

        sleep_start = time.monotonic()
        await _async_pause(poll_interval)
        sleep_duration = time.monotonic() - sleep_start
        if status == "pending":
            pending_elapsed += sleep_duration
        else:
            elapsed += sleep_duration
