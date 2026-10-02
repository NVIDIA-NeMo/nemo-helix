# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Timestamp formatting utilities for the NeMo CLI."""

from __future__ import annotations

from datetime import datetime, timezone


def parse_timestamp(timestamp_str: str) -> datetime | None:
    """
    Parse an ISO timestamp string to a datetime object.

    Args:
        timestamp_str: ISO format timestamp string

    Returns:
        datetime object or None if parsing fails
    """
    if not timestamp_str:
        return None

    try:
        # Try parsing ISO format with various formats
        # Handle formats like: 2025-01-01T10:00:00.123456Z or 2025-01-01T10:00:00
        if timestamp_str.endswith("Z"):
            timestamp_str = timestamp_str[:-1] + "+00:00"

        return datetime.fromisoformat(timestamp_str)
    except (ValueError, AttributeError):
        return None


def format_relative_time(dt: datetime) -> str:
    """
    Format a datetime as relative time (e.g., "2 hours ago", "3 days ago").

    Args:
        dt: datetime object to format

    Returns:
        Relative time string
    """
    # Get current time
    # If dt has timezone info, use UTC now; otherwise use naive now
    if dt.tzinfo is not None:
        now = datetime.now(timezone.utc)
        # Make dt timezone-aware if it isn't already
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    else:
        # Both naive - use naive comparison
        now = datetime.now()

    diff = now - dt
    seconds = diff.total_seconds()

    if seconds < 0:
        # Future time
        return "in the future"
    elif seconds < 60:
        return "just now"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes}m ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours}h ago"
    elif seconds < 2592000:  # 30 days
        days = int(seconds / 86400)
        return f"{days}d ago"
    elif seconds < 31536000:  # 365 days
        months = int(seconds / 2592000)
        return f"{months}mo ago"
    else:
        years = int(seconds / 31536000)
        return f"{years}y ago"


def format_simple_datetime(dt: datetime) -> str:
    """
    Format a datetime in a simple, readable format without seconds/milliseconds.

    Args:
        dt: datetime object to format

    Returns:
        Formatted datetime string (e.g., "2025-01-07 14:30")
    """
    return dt.strftime("%Y-%m-%d %H:%M")


def format_timestamp(
    timestamp_str: str,
    format_type: str = "iso",
) -> str:
    """
    Format a timestamp string according to the specified format type.

    Args:
        timestamp_str: ISO format timestamp string
        format_type: Format type ("iso", "relative", "datetime")

    Returns:
        Formatted timestamp string
    """
    if not timestamp_str:
        return ""

    if format_type == "iso":
        # Return as-is
        return timestamp_str

    # Parse the timestamp
    dt = parse_timestamp(timestamp_str)
    if dt is None:
        # If parsing fails, return as-is
        return timestamp_str

    if format_type == "relative":
        return format_relative_time(dt)
    elif format_type == "datetime":
        return format_simple_datetime(dt)
    else:
        # Default to ISO
        return timestamp_str
