# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pyleak utilities for detecting event loop blocking.

This module provides utilities for detecting synchronous/blocking operations
that block the asyncio event loop.

Usage:
    from nhx.common.pyleak import detect_blocking

    # Wrap specific code sections with a threshold
    async with detect_blocking(threshold=0.5):
        result = await db.execute(query)

Only blocking with useful stack traces (containing application code) will be
logged. Shallow stacks from framework/stdlib code are filtered out.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from pyleak import no_event_loop_blocking
from pyleak.eventloop import EventLoopBlockError

logger = logging.getLogger(__name__)

# Path patterns that indicate application code (not framework/stdlib)
_APPLICATION_PATH_PATTERNS = frozenset(
    {
        "/services/",
        "/packages/",
    }
)


def _has_useful_stack(event) -> bool:
    """Check if a blocking event has a useful (non-shallow) stack trace.

    A stack is considered useful if ANY frame is in application code (see
    _APPLICATION_PATH_PATTERNS). This catches cases where our code calls a
    blocking library (e.g., sync HTTP client) - the library is the deepest
    frame, but our code is in the call chain.

    Shallow stacks only have framework/stdlib code - not actionable.
    Also filters out self-referential blocking from our own logging.
    """
    stack = getattr(event, "blocking_stack", None)
    if not stack:
        return False

    # Check if blocking is from our own logging (feedback loop)
    for frame in stack:
        filename = getattr(frame, "filename", "")
        if "pyleak.py" in filename and "detect_blocking" in getattr(frame, "name", ""):
            return False

    # Check if ANY frame is from application code
    for frame in stack:
        filename = getattr(frame, "filename", "")

        # Skip framework and stdlib code
        if "site-packages" in filename or ".venv/bin" in filename:
            continue
        if "/lib/python" in filename:
            continue

        # Found a frame in application code - stack is useful
        if any(pattern in filename for pattern in _APPLICATION_PATH_PATTERNS):
            return True

    return False


def _filter_useful_events(blocking_events: list) -> list:
    """Filter blocking events to only include those with useful stacks."""
    return [event for event in blocking_events if _has_useful_stack(event)]


@asynccontextmanager
async def detect_blocking(threshold: float) -> AsyncIterator[None]:
    """Context manager to detect event loop blocking in a code section.

    Args:
        threshold: Minimum blocking duration in seconds to trigger detection.

    Usage:
        async with detect_blocking(threshold=0.5):
            result = await db.execute(query)

    When blocking is detected, logs the blocking duration and stack trace.
    Only logs if the stack trace is useful (contains application code, not just
    event loop internals). Shallow stacks from background tasks are suppressed.

    Note: We use action="raise" and catch the exception to log it. The exception's
    string representation includes the blocking stack traces from pyleak.
    We avoid calling traceback.format_stack() which causes additional blocking.
    """
    try:
        async with no_event_loop_blocking(
            action="raise",  # type: ignore[arg-type]
            threshold=threshold,
        ):
            yield
    except EventLoopBlockError as e:
        # Filter to only events with useful stacks (containing application code)
        # Shallow stacks that only show event loop internals are not actionable
        useful_events = _filter_useful_events(e.blocking_events)
        if useful_events:
            # Format only the useful events for logging
            event_strs = [str(event) for event in useful_events]
            logger.warning(
                "Event loop blocking detected (%d of %d blocks have useful stacks):\n%s",
                len(useful_events),
                len(e.blocking_events),
                "\n".join(event_strs),
            )
        else:
            logger.debug(
                "Event loop blocking detected but all stacks are shallow (background tasks): %d block(s)",
                len(e.blocking_events),
            )
