# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

FailureCategory = Literal["infrastructure", "provider", "task", "unknown"]

_INFRASTRUCTURE_CODES = frozenset(
    {
        "runner_disappeared",
        "runner_oomkilled",
        "runner_evicted",
        "runner_node_lost",
        "runner_deadline_exceeded",
        "runner_handoff_lost",
        "object_store_unavailable",
        "poll_timeout",
        "ConnectionError",
        "TimeoutError",
        "OSError",
        "KubernetesJobError",
        "SandboxExecutionError",
        "SwitchyardReadinessError",
    }
)
_PROVIDER_CODES = frozenset(
    {
        "provider_unavailable",
        "provider_rate_limited",
        "provider_timeout",
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "ServiceUnavailableError",
    }
)
_RETRYABLE_INFRASTRUCTURE_CODES = frozenset(
    {
        "runner_disappeared",
        "runner_oomkilled",
        "runner_evicted",
        "runner_node_lost",
        "runner_deadline_exceeded",
        "runner_handoff_lost",
        "object_store_unavailable",
        "SwitchyardReadinessError",
    }
)
_RETRYABLE_PROVIDER_CODES = frozenset(
    {
        "provider_unavailable",
        "provider_rate_limited",
        "provider_timeout",
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "ServiceUnavailableError",
    }
)
_RETRYABLE_TASK_CODES = frozenset({"NonZeroAgentExitCodeError"})
# Task codes whose detail text names the runtime (e.g. "harbor_opensandbox"), so they would
# otherwise match an infrastructure pattern such as "sandbox".
_UNAMBIGUOUS_TASK_CODES = frozenset({"IncompatibleTaskError"})
_TASK_CODES = frozenset(
    {
        "NonZeroAgentExitCodeError",
        "task_object_missing",
        "evaluation_already_terminal",
        "InvalidReference",
        "ValidationError",
        "ValueError",
        "IncompatibleTaskError",
    }
)
_INFRASTRUCTURE_PATTERNS = (
    "sandbox",
    "kubernetes",
    "dispatch",
    "object store",
    "rustfs",
    "s3",
    "timed out",
    "timeout",
    "connection",
)
_PROVIDER_PATTERNS = (
    "provider",
    "rate limit",
    "429",
    "api timeout",
    "api connection",
    "service unavailable",
    "503",
)
_TASK_PATTERNS = ("task object", "task pack", "not runnable", "invalid reference")


def failure_category_for_code(
    failure_code: str | None,
    detail: str | None = None,
    *,
    default: FailureCategory = "unknown",
) -> FailureCategory:
    """Return the user-visible category for a terminal member failure."""

    code = str(failure_code or "").strip()
    text = f"{code} {detail or ''}".lower()
    if code in _UNAMBIGUOUS_TASK_CODES:
        return "task"
    if code in _PROVIDER_CODES or any(pattern in text for pattern in _PROVIDER_PATTERNS):
        return "provider"
    if code in _INFRASTRUCTURE_CODES or any(pattern in text for pattern in _INFRASTRUCTURE_PATTERNS):
        return "infrastructure"
    if code in _TASK_CODES or any(pattern in text for pattern in _TASK_PATTERNS):
        return "task"
    return default


def is_retryable_failure(
    failure_code: str | None,
    detail: str | None = None,
) -> bool:
    """True for policy-bounded automatic retries of selected transient failures."""

    category = failure_category_for_code(failure_code, detail)
    code = str(failure_code or "").strip()
    if category == "infrastructure":
        return code in _RETRYABLE_INFRASTRUCTURE_CODES
    if category == "provider":
        return code in _RETRYABLE_PROVIDER_CODES
    if category == "task":
        return code in _RETRYABLE_TASK_CODES
    return False
