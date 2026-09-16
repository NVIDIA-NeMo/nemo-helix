# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retry policy for NIM embedding and ranking HTTP calls."""

from __future__ import annotations

import httpx

RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


def backoff_seconds(attempt: int) -> float:
    """Exponential backoff: 0.5s, 1s, 2s, capped at 8s."""
    return min(0.5 * 2**attempt, 8.0)


def is_retryable_status(status_code: int) -> bool:
    return status_code in RETRYABLE_STATUS_CODES


def is_retryable_error(error: BaseException) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        body = error.response.text or ""
        if "image inputs require VLM" in body:
            return False
        return is_retryable_status(error.response.status_code)
    return isinstance(error, httpx.TransportError)
