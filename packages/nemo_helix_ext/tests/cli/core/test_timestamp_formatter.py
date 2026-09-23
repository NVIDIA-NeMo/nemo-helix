# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for timestamp formatting."""

from datetime import datetime, timedelta, timezone

from nemo_helix_ext.cli.core.timestamp_formatter import (
    format_relative_time,
    format_simple_datetime,
    format_timestamp,
    parse_timestamp,
)


def test_parse_timestamp_iso_format():
    """Test parsing ISO format timestamps."""
    timestamp_str = "2025-01-07T14:30:00"
    result = parse_timestamp(timestamp_str)

    assert result is not None
    assert isinstance(result, datetime)
    assert result.year == 2025
    assert result.month == 1
    assert result.day == 7


def test_parse_timestamp_with_z():
    """Test parsing timestamp with Z suffix."""
    timestamp_str = "2025-01-07T14:30:00.123456Z"
    result = parse_timestamp(timestamp_str)

    assert result is not None
    assert result.year == 2025


def test_parse_timestamp_invalid():
    """Test parsing invalid timestamp."""
    result = parse_timestamp("not-a-timestamp")
    assert result is None


def test_format_relative_time_just_now():
    """Test relative time formatting for recent timestamps."""
    now = datetime.now(timezone.utc)
    recent = now - timedelta(seconds=30)

    result = format_relative_time(recent)
    assert result == "just now"


def test_format_relative_time_minutes():
    """Test relative time formatting for minutes ago."""
    now = datetime.now(timezone.utc)
    past = now - timedelta(minutes=5)

    result = format_relative_time(past)
    assert "5m ago" in result


def test_format_relative_time_hours():
    """Test relative time formatting for hours ago."""
    now = datetime.now(timezone.utc)
    past = now - timedelta(hours=3)

    result = format_relative_time(past)
    assert "3h ago" in result


def test_format_relative_time_days():
    """Test relative time formatting for days ago."""
    now = datetime.now(timezone.utc)
    past = now - timedelta(days=5)

    result = format_relative_time(past)
    assert "5d ago" in result


def test_format_simple_datetime():
    """Test simple datetime formatting."""
    dt = datetime(2025, 1, 7, 14, 30, 45, 123456)
    result = format_simple_datetime(dt)

    # Should not include seconds or milliseconds
    assert result == "2025-01-07 14:30"
    assert ":45" not in result


def test_format_timestamp_iso():
    """Test formatting timestamp in ISO format."""
    timestamp_str = "2025-01-07T14:30:00"
    result = format_timestamp(timestamp_str, format_type="iso")

    # Should return as-is
    assert result == timestamp_str


def test_format_timestamp_datetime():
    """Test formatting timestamp in datetime format."""
    timestamp_str = "2025-01-07T14:30:45.123456"
    result = format_timestamp(timestamp_str, format_type="datetime")

    # Should be simplified
    assert result == "2025-01-07 14:30"


def test_format_timestamp_empty():
    """Test formatting empty timestamp."""
    result = format_timestamp("", format_type="relative")
    assert result == ""


def test_format_timestamp_invalid():
    """Test formatting invalid timestamp."""
    result = format_timestamp("invalid", format_type="relative")
    # Should return as-is when parsing fails
    assert result == "invalid"


def test_format_timestamp_months_ago():
    """Test formatting a timestamp from several months ago.

    Anchored relative to `now` so the test doesn't drift past a year boundary.
    """
    six_months_ago = datetime.now(timezone.utc) - timedelta(days=180)
    timestamp_str = six_months_ago.isoformat()
    result = format_timestamp(timestamp_str, format_type="relative")

    # Should show months, not minutes
    assert "mo ago" in result or "d ago" in result
    assert "5m ago" not in result  # Should NOT be 5 minutes


def test_format_timestamp_datetime_simple():
    """Test datetime format strips seconds and milliseconds."""
    timestamp_str = "2025-05-19T20:25:01.698084"
    result = format_timestamp(timestamp_str, format_type="datetime")

    # Should be simplified to YYYY-MM-DD HH:MM
    assert result == "2025-05-19 20:25"
    assert ".698084" not in result
    assert ":01" not in result  # No seconds
