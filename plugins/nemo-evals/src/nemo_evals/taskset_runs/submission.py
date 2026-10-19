# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure budget checks and deadline calculation for taskset run submission."""

from datetime import datetime, timedelta, timezone

from nemo_evals.config import TasksetRunsConfig


class TimeoutAboveMaximumError(ValueError):
    """Signal that a requested timeout exceeds the operator's maximum run budget.

    The exception carries the requested and maximum values in its message so submit
    can report a field-specific validation error for timeout_seconds.
    """


def check_timeout(timeout_seconds: int | None, runs: TasksetRunsConfig) -> int:
    """Select the caller's timeout or operator default and enforce the maximum.

    Args:
        timeout_seconds: Validated request budget, or None for the operator default.
        runs: Operator settings for taskset run budgets.

    Returns:
        The effective execution budget in seconds.

    Raises:
        TimeoutAboveMaximumError: The selected budget exceeds the operator maximum.
    """
    timeout = timeout_seconds if timeout_seconds is not None else runs.default_timeout_seconds
    if timeout > runs.max_timeout_seconds:
        raise TimeoutAboveMaximumError(
            f"timeout_seconds {timeout} exceeds the operator maximum {runs.max_timeout_seconds}"
        )
    return timeout


def resolve_deadlines(accepted_at: datetime, timeout: int, runs: TasksetRunsConfig) -> tuple[datetime, datetime]:
    """Calculate elapsed-time deadlines without consulting a clock.

    Args:
        accepted_at: Timezone-aware acceptance timestamp supplied by submit.
        timeout: Effective execution budget in seconds.
        runs: Operator settings supplying the finalization grace period.

    Returns:
        Execution deadline and finalization deadline, preserving the input timezone.

    Raises:
        ValueError: accepted_at is naive, including a tzinfo that supplies no UTC offset.
    """
    if accepted_at.utcoffset() is None:
        raise ValueError("accepted_at must be timezone-aware")
    execution = accepted_at.astimezone(timezone.utc) + timedelta(seconds=timeout)
    finalization = execution + timedelta(seconds=runs.finalization_grace_seconds)
    return execution.astimezone(accepted_at.tzinfo), finalization.astimezone(accepted_at.tzinfo)
