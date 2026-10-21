# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from datetime import date, datetime, timedelta, timezone
from typing import cast

import pytest
from evaluation.timeparse import parse_since


def test_none_returns_none():
    assert parse_since(None) is None


def test_empty_string_returns_none():
    assert parse_since("") is None


def test_toml_date_becomes_utc_midnight():
    # tomllib yields a `date` for an unquoted `since = 2026-06-01`.
    parsed = parse_since(date(2026, 6, 1))
    assert parsed is not None
    assert parsed.isoformat() == "2026-06-01T00:00:00+00:00"


def test_toml_datetime_normalized_to_utc():
    # tomllib yields an aware `datetime` for `since = 2026-06-01T00:00:00-07:00`.
    aware = datetime(2026, 6, 1, tzinfo=timezone(timedelta(hours=-7)))
    parsed = parse_since(aware)
    assert parsed is not None
    assert parsed.isoformat() == "2026-06-01T07:00:00+00:00"


def test_non_string_non_date_exits():
    # `since = 7` (TOML int) is ambiguous → clean exit, not AttributeError.
    with pytest.raises(SystemExit):
        parse_since(cast(str, 7))


def test_lookback_days_is_utc_aware():
    dt = parse_since("7d")
    assert dt is not None
    offset = dt.utcoffset()
    assert offset is not None and offset.total_seconds() == 0


def test_naive_iso_assumed_utc():
    parsed = parse_since("2026-06-01")
    assert parsed is not None
    assert parsed.isoformat() == "2026-06-01T00:00:00+00:00"


def test_offset_iso_normalized_to_utc():
    parsed = parse_since("2026-06-01T00:00:00-07:00")
    assert parsed is not None
    assert parsed.isoformat() == "2026-06-01T07:00:00+00:00"


def test_bad_value_exits():
    with pytest.raises(SystemExit):
        parse_since("7D")
