# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retry rules shared by the typed client and callers that rebuild request bodies.

The client retries a request by sending it again unchanged. A caller that streams
a one-shot body, such as ``FilesetFileSystem.put_file``, rebuilds the body and
retries on its own, under these same rules.
"""

from __future__ import annotations

import email.utils
from datetime import timezone
from time import time as wall_time

import httpx
from nemo_helix_plugin.client.types import RetryPolicy

# Transport failures httpx raises before it starts reading the request body: the
# connection has to exist before any body byte can be written. Everything else, such
# as a read/write timeout, a dropped connection or a protocol error, can land mid-body.
_PRE_BODY_TRANSPORT_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    httpx.ProxyError,
    httpx.UnsupportedProtocol,
)

_MAX_RETRY_AFTER_SECONDS = 60


def body_was_sent(error: httpx.TransportError) -> bool:
    """Whether any of the request body may have been written before ``error``."""
    return not isinstance(error, _PRE_BODY_TRANSPORT_ERRORS)


def is_retryable(
    policy: RetryPolicy,
    failure: httpx.Response | httpx.TransportError,
    *,
    exist_ok: bool = False,
) -> bool:
    """Whether ``policy`` allows another attempt after ``failure``.

    A transport failure is retryable unless the policy sets
    ``resend_after_body_sent=False`` and the body may have been sent (see
    :func:`body_was_sent`); a failure before any body byte was sent is always
    retryable. Whether a one-shot body can be sent again is still the caller's
    concern. ``exist_ok`` marks a 409 as an
    expected outcome the caller resolves, so it is not retried.
    """
    if isinstance(failure, httpx.TransportError):
        return policy.resend_after_body_sent or not body_was_sent(failure)
    status = failure.status_code
    if status == 409 and exist_ok:
        return False
    if policy.respect_retry_decision_headers:
        if status < 400:
            return False
        decision = failure.headers.get("x-should-retry")
        if decision in ("true", "false"):
            return decision == "true"
    return status in policy.retryable_status_codes or (policy.retry_all_server_errors and status >= 500)


def retry_delay(policy: RetryPolicy, attempt: int, failure: httpx.Response | httpx.TransportError) -> float:
    """Seconds to wait after the failed attempt ``attempt`` (counting from 0).

    Exponential backoff, or the server's ``Retry-After`` when the policy honors it
    and the value is reasonable.
    """
    backoff = policy.backoff_base * (2**attempt)
    if isinstance(failure, httpx.Response) and policy.respect_retry_after_headers:
        return _retry_after(failure) or backoff
    return backoff


def _retry_after(response: httpx.Response) -> float | None:
    """Parse a reasonable server-requested retry delay in seconds."""
    retry_after_ms = response.headers.get("retry-after-ms")
    try:
        delay = float(retry_after_ms) / 1000
    except (TypeError, ValueError):
        retry_after = response.headers.get("retry-after")
        try:
            delay = float(retry_after)
        except (TypeError, ValueError):
            # Retry-After may instead be an RFC 5322 / HTTP-date, e.g.
            # "Fri, 31 Dec 2027 23:59:59 GMT"; convert that to a delta.
            try:
                retry_date = email.utils.parsedate_to_datetime(retry_after)
            except (TypeError, ValueError, OverflowError):
                return None
            if retry_date.tzinfo is None:
                retry_date = retry_date.replace(tzinfo=timezone.utc)
            try:
                delay = retry_date.timestamp() - wall_time()
            except (OverflowError, OSError, ValueError):
                return None
    return delay if 0 < delay <= _MAX_RETRY_AFTER_SECONDS else None
