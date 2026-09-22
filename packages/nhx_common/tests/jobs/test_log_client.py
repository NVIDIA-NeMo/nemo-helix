# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for JobLogsClient SDK wrapper and PageCursor."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import base58
import httpx
import pytest
from nemo_helix_plugin.client.errors import NotFoundError, UnprocessableEntityError
from nhx.common.jobs.log_client import JobLogsClient
from nhx.common.jobs.schemas import (
    InvalidPageCursorError,
    LogPageCursorV1,
    PageCursor,
    PaginationDirection,
    HelixJobLog,
    HelixJobLogPage,
    decode_log_page_cursor,
)

# =============================================================================
# PageCursor Tests
# =============================================================================


def test_encode_decode_forward():
    """Test encoding and decoding a forward pagination cursor."""
    cursor = PageCursor(start_id=5, direction=PaginationDirection.FORWARD)
    encoded = cursor.encode()
    decoded = PageCursor.decode(encoded)

    assert decoded.start_id == 5
    assert decoded.direction == PaginationDirection.FORWARD


def test_encode_decode_backward():
    """Test encoding and decoding a backward pagination cursor."""
    cursor = PageCursor(start_id=3, direction=PaginationDirection.BACKWARD)
    encoded = cursor.encode()
    decoded = PageCursor.decode(encoded)

    assert decoded.start_id == 3
    assert decoded.direction == PaginationDirection.BACKWARD


def test_decode_invalid_cursor():
    """Test decoding an invalid cursor raises InvalidPageCursorError."""
    with pytest.raises(InvalidPageCursorError, match="Invalid page cursor"):
        PageCursor.decode("invalid_cursor_string")


def test_decode_log_page_cursor_accepts_existing_page_cursor():
    """Log cursor decoding preserves the legacy page-number cursor format."""
    cursor = PageCursor(start_id=2, direction=PaginationDirection.FORWARD).encode()

    decoded = decode_log_page_cursor(cursor)

    assert isinstance(decoded, PageCursor)
    assert decoded.start_id == 2
    assert decoded.direction == PaginationDirection.FORWARD


def test_log_page_cursor_v1_round_trip():
    """V1 cursor encoding stays compact while validating typed fields."""
    cursor = LogPageCursorV1(
        boundary_timestamp=datetime(2026, 9, 2, 12, 0, 1),
        boundary_row_hash="a" * 32,
        query_scope_hash="b" * 32,
        emitted_boundary_rows=2,
    )

    decoded = decode_log_page_cursor(cursor.encode())

    assert decoded == cursor


def test_decode_log_page_cursor_rejects_invalid_v1_hash_length():
    """V1 cursors reject malformed compact hash fields."""
    bad_payload = {
        "v": 1,
        "t": "2026-09-02T12:00:01",
        "r": "short",
        "q": "b" * 32,
        "e": 0,
    }
    encoded = base58.b58encode(json.dumps(bad_payload).encode()).decode()

    with pytest.raises(InvalidPageCursorError, match="Invalid page cursor"):
        decode_log_page_cursor(encoded)


# =============================================================================
# JobLogsClient Tests
# =============================================================================


def _make_response_mock(log_page: HelixJobLogPage) -> MagicMock:
    mock = MagicMock()
    mock.data.return_value = log_page
    return mock


@pytest.fixture
def mock_files_client():
    """Create a mock AsyncFilesClient for testing."""
    client = AsyncMock()
    return client


@pytest.fixture
def log_client(mock_files_client):
    """Create a JobLogsClient with a mock files client."""
    with patch("nhx.common.jobs.log_client.client_from_platform", return_value=mock_files_client):
        client = JobLogsClient(sdk=MagicMock())
    return client, mock_files_client


async def test_query_logs_success(log_client):
    """Test successful log query via FilesClient."""
    client, mock_files = log_client

    page = HelixJobLogPage(
        data=[
            HelixJobLog(
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                job="job-123",
                job_step="step1",
                job_task="task1",
                message="Test log message",
            )
        ],
        total=1,
        next_page=None,
        prev_page=None,
    )
    mock_files.query_otlp_logs.return_value = _make_response_mock(page)

    result = await client.query_logs(
        fileset="logs",
        workspace="test-workspace",
        filters={"job": "job-123"},
        page_size=100,
    )

    assert isinstance(result, HelixJobLogPage)
    assert len(result.data) == 1
    assert result.total == 1

    mock_files.query_otlp_logs.assert_called_once()
    call_kwargs = mock_files.query_otlp_logs.call_args.kwargs
    assert call_kwargs["name"] == "logs"
    assert call_kwargs["workspace"] == "test-workspace"


async def test_query_logs_with_pagination_cursor(log_client):
    """Test query_logs passes pagination cursor correctly."""
    client, mock_files = log_client

    mock_files.query_otlp_logs.return_value = _make_response_mock(
        HelixJobLogPage(data=[], total=0, next_page=None, prev_page=None)
    )

    cursor = PageCursor(start_id=2, direction=PaginationDirection.FORWARD).encode()

    await client.query_logs(
        fileset="logs",
        workspace="test-workspace",
        page_cursor=cursor,
    )

    call_kwargs = mock_files.query_otlp_logs.call_args.kwargs
    assert call_kwargs["body"].page_cursor == cursor


async def test_query_logs_with_tail(log_client):
    """Test query_logs passes tail without also setting limit."""
    client, mock_files = log_client

    mock_files.query_otlp_logs.return_value = _make_response_mock(
        HelixJobLogPage(data=[], total=0, next_page=None, prev_page=None)
    )

    await client.query_logs(
        fileset="logs",
        workspace="test-workspace",
        filters={"job": "job-123"},
        tail=500,
    )

    call_kwargs = mock_files.query_otlp_logs.call_args.kwargs
    body = call_kwargs["body"]
    assert body.tail == 500
    assert body.limit is None


async def test_query_logs_404_returns_empty_page(log_client):
    """Test that NotFoundError returns an empty page."""
    client, mock_files = log_client

    mock_files.query_otlp_logs.side_effect = NotFoundError(
        httpx.Response(status_code=404),
    )

    result = await client.query_logs(
        fileset="logs",
        workspace="test-workspace",
    )

    assert result.data == []
    assert result.total == 0
    assert result.next_page is None
    assert result.prev_page is None


async def test_query_logs_invalid_page_cursor_error_is_preserved(log_client):
    """Downstream Files cursor validation errors stay client-visible to the Jobs endpoint."""
    client, mock_files = log_client

    mock_files.query_otlp_logs.side_effect = UnprocessableEntityError(
        httpx.Response(status_code=422, json={"detail": "Invalid page cursor"}),
    )

    with pytest.raises(InvalidPageCursorError, match="Invalid page cursor"):
        await client.query_logs(
            fileset="logs",
            workspace="test-workspace",
            page_cursor="garbage",
        )


async def test_query_logs_page_cursor_scope_error_is_preserved(log_client):
    """Cursor scope mismatches keep their specific diagnostic for callers."""
    client, mock_files = log_client

    mock_files.query_otlp_logs.side_effect = UnprocessableEntityError(
        httpx.Response(status_code=422, json={"detail": "page_cursor does not match the current log filters."}),
    )

    with pytest.raises(InvalidPageCursorError, match="page_cursor does not match the current log filters"):
        await client.query_logs(
            fileset="logs",
            workspace="test-workspace",
            page_cursor="old",
        )


async def test_query_logs_other_error_raises(log_client):
    """Test that other errors are raised to the caller."""
    client, mock_files = log_client

    mock_files.query_otlp_logs.side_effect = RuntimeError("Unexpected error")

    with pytest.raises(RuntimeError, match="Unexpected error"):
        await client.query_logs(
            fileset="logs",
            workspace="test-workspace",
        )


async def test_query_logs_empty_filters(log_client):
    """Test query_logs with no filters."""
    client, mock_files = log_client

    mock_files.query_otlp_logs.return_value = _make_response_mock(
        HelixJobLogPage(data=[], total=0, next_page=None, prev_page=None)
    )

    await client.query_logs(
        fileset="logs",
        workspace="test-workspace",
        filters=None,
    )

    call_kwargs = mock_files.query_otlp_logs.call_args.kwargs
    assert call_kwargs["body"].filters == {}


def test_sdk_created_in_constructor():
    """Test that SDK is set in constructor."""
    sdk = MagicMock()
    with patch("nhx.common.jobs.log_client.client_from_platform") as mock_adapter:
        client = JobLogsClient(sdk=sdk)
    assert client._sdk is sdk
    mock_adapter.assert_called_once()
