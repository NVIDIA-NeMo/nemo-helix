# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Datetime utilities shared across controllers and reconcilers."""

from datetime import datetime, timezone


def ensure_utc(dt: datetime | None) -> datetime | None:
    """Normalize a datetime to UTC, handling naive timestamps from the entity store.

    The entity store may return naive datetimes (no tzinfo) that are implicitly UTC.
    This function makes them explicitly UTC-aware so arithmetic with other UTC
    datetimes works correctly.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt
