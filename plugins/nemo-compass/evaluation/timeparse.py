# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parse a lookback or ISO date into a UTC-aware datetime."""

import re
import sys
from datetime import date, datetime, timedelta, timezone

_LOOKBACK = re.compile(r"(\d+)([dhm])")


def parse_since(value: str | datetime | date | None) -> datetime | None:
    """Lower bound for trace reads → UTC datetime (or ``None`` for no bound).

    Accepts ``Nd``/``Nh``/``Nm`` (days/hours/minutes ago) or an ISO string, plus
    the ``datetime``/``date`` objects ``tomllib`` yields for unquoted date/datetime
    literals in evaluations.toml (those are honored, not crashed on). Any other type
    (e.g. a bare integer) exits cleanly rather than raising ``AttributeError``.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if not isinstance(value, str):
        sys.exit(f"Invalid since value {value!r}: use Nd/Nh/Nm (days/hours/minutes) or an ISO date.")
    text = value.strip()
    match = _LOOKBACK.fullmatch(text)
    if match:
        n, unit = int(match.group(1)), match.group(2)
        delta = {
            "d": timedelta(days=n),
            "h": timedelta(hours=n),
            "m": timedelta(minutes=n),
        }[unit]
        return datetime.now(timezone.utc) - delta
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        sys.exit(f"Invalid --since '{value}': use Nd/Nh/Nm (days/hours/minutes) or an ISO date.")
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
