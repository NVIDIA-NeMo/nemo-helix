# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic utilities for the Files service."""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)

# Threshold for logging slow operation warnings (in seconds)
SLOW_OPERATION_THRESHOLD_SECONDS = 1.0


def _to_log_value(value: Any) -> str | int | float | bool | None:
    """Convert a value to a JSON-safe type for structured logging."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@asynccontextmanager
async def warn_if_slow(
    operation: str,
    threshold_seconds: float = SLOW_OPERATION_THRESHOLD_SECONDS,
    **metadata,
) -> AsyncIterator[None]:
    """Context manager that logs a warning if the wrapped operation is slow."""
    start = time.monotonic()
    yield
    duration = time.monotonic() - start
    if duration >= threshold_seconds:
        logger.warning(
            f"Slow {operation}",
            extra={
                "duration_seconds": round(duration, 3),
                **{k: _to_log_value(v) for k, v in metadata.items()},
            },
        )
