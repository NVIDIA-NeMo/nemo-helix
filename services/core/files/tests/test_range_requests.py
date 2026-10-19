# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for range request parsing and header generation."""

import pytest
from nhx.core.files.app.backends.base import ByteRange
from nhx.core.files.app.range_requests import (
    download_response_status_and_headers,
    parse_range_header,
)
from nhx.core.files.exceptions import InvalidRangeError


@pytest.mark.parametrize(
    "range_header,file_size,expected_start,expected_end,should_raise",
    [
        # Valid ranges
        ("bytes=0-99", 1000, 0, 99, False),
        ("bytes=100-200", 1000, 100, 200, False),
        ("bytes=500-", 1000, 500, 999, False),
        ("bytes=-500", 1000, 500, 999, False),
        ("bytes=-2000", 1000, 0, 999, False),  # Suffix larger than file
        (None, 1000, None, None, False),  # No range
        ("", 1000, None, None, False),  # Empty range
        # Invalid ranges
        ("invalid", 1000, None, None, True),
        ("bytes=-", 1000, None, None, True),
        ("bytes=1000-2000", 1000, None, None, True),  # Start beyond file size
        ("bytes=500-100", 1000, None, None, True),  # End before start
    ],
)
def test_parse_range_header(range_header, file_size, expected_start, expected_end, should_raise):
    """Test parse_range_header with various inputs."""
    if should_raise:
        with pytest.raises(InvalidRangeError):
            parse_range_header(range_header, file_size)
    else:
        result = parse_range_header(range_header, file_size)
        if expected_start is None:
            assert result is None
        else:
            assert result == ByteRange(start=expected_start, end=expected_end)


@pytest.mark.parametrize(
    "byte_range,file_size,expected_status,expected_content_range,expected_content_length",
    [
        # Partial content - first 100 bytes
        (ByteRange(start=0, end=99), 1000, 206, "bytes 0-99/1000", "100"),
        # Partial content - suffix range (last 100 bytes)
        (ByteRange(start=900, end=999), 1000, 206, "bytes 900-999/1000", "100"),
        # Partial content - single byte
        (ByteRange(start=500, end=500), 1000, 206, "bytes 500-500/1000", "1"),
        # Full content (no range) - still includes content-length
        (None, 1000, 200, None, "1000"),
    ],
)
def test_download_response_status_and_headers(
    byte_range,
    file_size,
    expected_status,
    expected_content_range,
    expected_content_length,
):
    """Test headers and status codes for download responses with various byte ranges."""
    status, headers = download_response_status_and_headers(byte_range, file_size)

    assert status == expected_status
    assert headers["accept-ranges"] == "bytes"
    assert headers["content-length"] == expected_content_length

    if expected_content_range is not None:
        assert headers["content-range"] == expected_content_range
    else:
        assert "content-range" not in headers
