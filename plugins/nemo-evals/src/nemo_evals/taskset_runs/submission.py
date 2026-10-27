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


def resolve_deadlines(start: datetime, budget: int, grace: int) -> tuple[datetime, datetime]:
    """Calculate elapsed-time deadlines without consulting a clock.

    Submit calls it with ``accepted_at`` and the queue timeout; the run controller calls it again with the
    run's first slot and its execution budget (see ``TasksetRunStore.start``).

    Args:
        start: Timezone-aware time the budget counts from.
        budget: Seconds until the execution deadline.
        grace: Seconds after the execution deadline to save the result.

    Returns:
        Execution deadline and finalization deadline, preserving the input timezone.

    Raises:
        ValueError: start is naive, including a tzinfo that supplies no UTC offset.
    """
    if start.utcoffset() is None:
        raise ValueError("start must be timezone-aware")
    execution = start.astimezone(timezone.utc) + timedelta(seconds=budget)
    finalization = execution + timedelta(seconds=grace)
    return execution.astimezone(start.tzinfo), finalization.astimezone(start.tzinfo)
