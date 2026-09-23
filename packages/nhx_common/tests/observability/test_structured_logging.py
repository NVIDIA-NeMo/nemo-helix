# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the structured-logging processors."""

from __future__ import annotations

import logging

import pytest
from nhx.common.observability.structured_logging import _sanitize_log_strings


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("admin\n[ERROR] forged log line", "admin [ERROR] forged log line"),
        ("with\r\ncrlf", "with  crlf"),
        ("tab\tis fine", "tab\tis fine"),
        ("plain string", "plain string"),
        ("nel\x85next", "nel next"),
        ("ls\u2028lsep", "ls lsep"),
        ("ps\u2029psep", "ps psep"),
    ],
)
def test_sanitize_log_strings_replaces_newline_variants(raw: str, expected: str) -> None:
    event = {"event": "test", "user": raw}
    result = _sanitize_log_strings(logging.getLogger(), "info", event)
    assert result["user"] == expected


def test_sanitize_log_strings_leaves_non_string_values_alone() -> None:
    event = {"event": "test", "count": 5, "ok": True, "items": [1, 2, 3]}
    result = _sanitize_log_strings(logging.getLogger(), "info", event)
    assert result == event


def test_sanitize_log_strings_skips_clean_strings() -> None:
    event = {"event": "test", "name": "default/workspace"}
    result = _sanitize_log_strings(logging.getLogger(), "info", event)
    assert result["name"] == "default/workspace"


def test_sanitize_log_strings_sanitizes_event_key() -> None:
    """The 'event' key carries the primary log message — must be sanitized too."""
    event = {"event": "user input was: bad\ninjected line"}
    result = _sanitize_log_strings(logging.getLogger(), "info", event)
    assert result["event"] == "user input was: bad injected line"


def test_sanitize_log_strings_sanitizes_exception_field() -> None:
    """structlog.format_exc_info writes a multi-line traceback into 'exception'.

    The sanitizer must run after format_exc_info so attacker-controlled
    exception messages can't forge log entries.
    """
    event = {"event": "boom", "exception": "Traceback (most recent call last):\n  ...\nValueError: pwn"}
    result = _sanitize_log_strings(logging.getLogger(), "info", event)
    assert "\n" not in result["exception"]


def test_initialize_logging_wires_sanitizer_into_chain() -> None:
    """Regression guard: 177 dismissed py/log-injection alerts depend on the sanitizer being in the chain."""
    import logging as stdlib_logging

    import structlog
    from nhx.common.observability.structured_logging import _sanitize_log_strings, initialize_logging

    root = stdlib_logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    try:
        root.handlers.clear()
        initialize_logging()
        # Locate the ProcessorFormatter by type, not handler index, so a future
        # refactor that reorders or adds handlers doesn't silently false-pass.
        formatters = [
            h.formatter for h in root.handlers if isinstance(h.formatter, structlog.stdlib.ProcessorFormatter)
        ]
        assert formatters, "initialize_logging() did not attach a structlog ProcessorFormatter"
        chain = getattr(formatters[0], "foreign_pre_chain", None) or []
        assert _sanitize_log_strings in chain, (
            "_sanitize_log_strings missing from structlog chain — log-injection defense is disabled"
        )
    finally:
        root.handlers.clear()
        root.handlers.extend(saved_handlers)
        root.setLevel(saved_level)
