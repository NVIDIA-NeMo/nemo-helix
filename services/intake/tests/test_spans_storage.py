# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Span storage helper tests."""

from datetime import datetime, timezone

from nhx.intake.spans.storage import (
    created_after_bound_micros,
    json_loads_or_none,
    stable_id,
    text_for_mode,
    text_query_parameters,
    text_select_for_mode,
    to_unix_micros,
    truncated_text_select,
)


def test_stable_id_uses_unambiguous_part_boundaries():
    assert stable_id("a", "b\x1fc") != stable_id("a\x1fb", "c")


def test_json_loads_or_none_returns_none_for_malformed_json():
    assert json_loads_or_none('{"unterminated"') is None


def test_json_loads_or_none_passes_through_non_string_values():
    value = {"already": "decoded"}

    assert json_loads_or_none(value) == value


def test_truncated_text_select_builds_bounded_utf8_expression() -> None:
    assert (
        truncated_text_select("trace_roots.root_input", alias="input")
        == "substringUTF8(trace_roots.root_input, 1, %(payload_char_limit)s) AS input"
    )


def test_text_projection_for_response_modes() -> None:
    assert text_select_for_mode("root_input", alias="input", mode="summary") == "'' AS input"
    assert text_select_for_mode("root_input", alias="input", mode="preview") == (
        "substringUTF8(root_input, 1, %(payload_char_limit)s) AS input"
    )
    assert text_select_for_mode("root_input", alias="input", mode="detailed") == "root_input AS input"
    assert text_query_parameters("summary") == {}
    assert text_query_parameters("preview") == {"payload_char_limit": 300}
    assert text_query_parameters("detailed") == {}


def test_text_for_mode_bounds_api_values() -> None:
    assert text_for_mode("abcdef", mode="summary") is None
    assert text_for_mode("x" * 350, mode="preview") == "x" * 300
    assert text_for_mode("abcdef", mode="detailed") == "abcdef"
    assert text_for_mode(None, mode="preview") is None


def test_to_unix_micros_keeps_microsecond_resolution_and_reads_naive_as_utc() -> None:
    aware = datetime(2026, 3, 1, 12, 0, 0, 123456, tzinfo=timezone.utc)
    naive = datetime(2026, 3, 1, 12, 0, 0, 123456)

    # The entity store keeps naive UTC timestamps; both spellings of the same instant must
    # convert identically, with the microseconds intact (no float rounding).
    assert to_unix_micros(naive) == to_unix_micros(aware)
    assert to_unix_micros(aware) % 1_000_000 == 123456


def test_created_after_bound_floors_to_whole_second() -> None:
    value = datetime(2026, 3, 1, 12, 0, 0, 123456, tzinfo=timezone.utc)

    assert created_after_bound_micros(value) == to_unix_micros(value.replace(microsecond=0))
