# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure wall-clock schedule math for periodic insights analysis.

This module is intentionally dependency-free (stdlib only) so the scheduling
logic is unit-testable in isolation. The operator configures an intended local
hour plus an IANA timezone; conversion to UTC happens here at evaluation time so
runs fire at the intended local hour even across DST transitions.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from nemo_compass_plugin.config import Frequency


def previous_scheduled(
    now_utc: datetime,
    *,
    frequency: Frequency,
    run_at_hour: int,
    run_on_weekday: int,
    tz: ZoneInfo,
) -> datetime:
    """Return the most recent scheduled instant (UTC) at or before ``now_utc``.

    ``run_on_weekday`` follows Python's ``0=Monday`` convention and is only
    consulted for weekly schedules.
    """
    now_local = now_utc.astimezone(tz)
    candidate = now_local.replace(hour=run_at_hour, minute=0, second=0, microsecond=0)
    if frequency == Frequency.WEEKLY:
        candidate -= timedelta(days=(now_local.weekday() - run_on_weekday) % 7)
        if candidate > now_local:
            candidate -= timedelta(days=7)
    else:
        if candidate > now_local:
            candidate -= timedelta(days=1)
    return candidate.astimezone(timezone.utc)


def is_due(
    now_utc: datetime,
    anchor: datetime | None,
    *,
    frequency: Frequency,
    run_at_hour: int,
    run_on_weekday: int,
    tz: ZoneInfo,
) -> bool:
    """Whether a run is due given the last run time ``anchor`` (UTC or naive-UTC).

    A run is due when no prior run exists, or when the last run happened before
    the most recent scheduled instant.
    """
    if anchor is None:
        return True
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=timezone.utc)
    return anchor < previous_scheduled(
        now_utc,
        frequency=frequency,
        run_at_hour=run_at_hour,
        run_on_weekday=run_on_weekday,
        tz=tz,
    )
