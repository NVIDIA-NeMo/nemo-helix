# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from nmp.common.files.storage_config import LocalStorageConfig, S3StorageConfig
from nmp.common.jobs.schemas import InvalidPageCursorError, LogPageCursorV1, PaginationDirection, decode_log_page_cursor
from nmp.core.files.app.backends.local import LocalStorageImpl
from nmp.core.files.app.backends.s3 import S3StorageImpl
from nmp.core.files.app.log_db import DuckDBLogRepository
from nmp.core.files.app.log_storage import LogEntry, LogStorage
from nmp.core.files.exceptions import InvalidFilterError, InvalidPathError


@pytest.fixture
def log_storage():
    """Create a LogStorage instance for testing."""
    return LogStorage()


@pytest.fixture
def local_storage(tmp_path):
    """Create a LocalStorageImpl backed by a temp directory."""
    config = LocalStorageConfig(path=str(tmp_path))
    return LocalStorageImpl(config=config)


def get_path(storage: LocalStorageImpl) -> Path:
    """Get the underlying pathlib.Path from a LocalStorageImpl."""
    return Path(storage.config.path)


@pytest.fixture
def sample_log_entries():
    """Create sample log entries for testing."""
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    return [
        LogEntry(
            workspace="test-workspace",
            job="job-123",
            job_attempt="attempt-1",
            job_step="step1",
            job_task="task1",
            log_message=f"Log message {i}",
            timestamp=base_time + timedelta(seconds=i),
        )
        for i in range(25)
    ]


async def test_insert_logs_creates_logs_prefix(log_storage, local_storage, sample_log_entries):
    """Test that insert_logs creates files under the 'logs' prefix."""
    count = await log_storage.insert_logs(local_storage, sample_log_entries[:5])
    assert count == 5

    # Check files were created under logs/
    base_path = get_path(local_storage)
    logs_path = base_path / "logs"
    assert logs_path.exists()

    parquet_files = list(logs_path.rglob("*.parquet"))
    assert len(parquet_files) > 0


async def test_insert_logs_maintains_partition_structure(log_storage, local_storage):
    """Test that insert_logs maintains Hive partition structure."""
    diverse_logs = [
        LogEntry(
            workspace="test-workspace",
            job="job-123",
            job_attempt="attempt-1",
            job_step="step1",
            job_task="task1",
            log_message="Message 1",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
        ),
        LogEntry(
            workspace="test-workspace",
            job="job-123",
            job_attempt="attempt-1",
            job_step="step2",
            job_task="task2",
            log_message="Message 2",
            timestamp=datetime(2024, 1, 1, 12, 0, 1),
        ),
    ]

    count = await log_storage.insert_logs(local_storage, diverse_logs)
    assert count == 2

    # Check partition structure exists
    base_path = get_path(local_storage)
    logs_path = base_path / "logs"
    parquet_files = list(logs_path.rglob("*.parquet"))

    # Should have files in partition directories
    assert len(parquet_files) > 0

    # Check that Hive partitioning is used
    for pq_file in parquet_files:
        rel_path = str(pq_file.relative_to(logs_path))
        assert "job=" in rel_path
        assert "job_attempt=" in rel_path
        assert "job_step=" in rel_path
        assert "job_task=" in rel_path


async def test_insert_and_query_logs_roundtrip(log_storage, local_storage, sample_log_entries):
    """Test the full flow of inserting and querying logs."""
    # Insert logs
    insert_count = await log_storage.insert_logs(local_storage, sample_log_entries)
    assert insert_count == 25

    # Query logs back
    result = await log_storage.query_logs(
        local_storage,
        filters={"job": "job-123"},
        page_size=100,
    )

    assert result.total == 25
    assert len(result.data) == 25

    # Verify log content
    messages = {log.message for log in result.data}
    for i in range(25):
        assert f"Log message {i}" in messages


async def test_insert_and_query_logs_with_artifact_base_path_roundtrip(log_storage, local_storage, sample_log_entries):
    """Logs written under a per-job artifact_base_path are nested there and only readable with the same artifact_base_path."""
    insert_count = await log_storage.insert_logs(local_storage, sample_log_entries, artifact_base_path="jobs/job-123")
    assert insert_count == 25

    # Files land under <base>/jobs/job-123/logs/, not the root logs/ prefix.
    base_path = get_path(local_storage)
    assert (base_path / "jobs" / "job-123" / "logs").exists()
    assert not (base_path / "logs").exists()

    # Readable only with the matching artifact_base_path (symmetry invariant).
    nested = await log_storage.query_logs(
        local_storage, filters={"job": "job-123"}, page_size=100, artifact_base_path="jobs/job-123"
    )
    assert nested.total == 25

    # Querying the root (no artifact_base_path) finds nothing — proves the nesting isolates the logs.
    flat = await log_storage.query_logs(local_storage, filters={"job": "job-123"}, page_size=100)
    assert flat.total == 0


@pytest.mark.parametrize("artifact_base_path", ["../escape", "jobs/../../escape", "/abs/path", "jobs/./here"])
async def test_artifact_base_path_traversal_is_rejected(
    log_storage, local_storage, sample_log_entries, artifact_base_path
):
    """A artifact_base_path that could escape the fileset is refused on both write and read."""
    with pytest.raises(InvalidPathError):
        await log_storage.insert_logs(local_storage, sample_log_entries, artifact_base_path=artifact_base_path)
    with pytest.raises(InvalidPathError):
        await log_storage.query_logs(local_storage, filters={}, page_size=10, artifact_base_path=artifact_base_path)


async def test_query_logs_with_filters(log_storage, local_storage):
    """Test querying logs with different filter combinations."""
    # Insert logs with different partitions
    logs = [
        LogEntry(
            workspace="test-workspace",
            job="job-A",
            job_attempt="attempt-1",
            job_step="step1",
            job_task="task1",
            log_message="Job A message",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
        ),
        LogEntry(
            workspace="test-workspace",
            job="job-B",
            job_attempt="attempt-1",
            job_step="step1",
            job_task="task1",
            log_message="Job B message",
            timestamp=datetime(2024, 1, 1, 12, 0, 1),
        ),
    ]

    await log_storage.insert_logs(local_storage, logs)

    # Query for job-A only
    result_a = await log_storage.query_logs(
        local_storage,
        filters={"job": "job-A"},
    )
    assert result_a.total == 1
    assert result_a.data[0].message == "Job A message"

    # Query for job-B only
    result_b = await log_storage.query_logs(
        local_storage,
        filters={"job": "job-B"},
    )
    assert result_b.total == 1
    assert result_b.data[0].message == "Job B message"

    # Query all logs (no filter)
    result_all = await log_storage.query_logs(local_storage)
    assert result_all.total == 2


async def test_query_logs_pagination(log_storage, local_storage, sample_log_entries):
    """Test pagination of log queries."""
    await log_storage.insert_logs(local_storage, sample_log_entries)

    # Get first page
    page1 = await log_storage.query_logs(
        local_storage,
        filters={"job": "job-123"},
        page_size=10,
    )

    assert len(page1.data) == 10
    assert page1.total == 25
    assert page1.next_page is not None
    assert page1.prev_page is None

    # Get second page
    page2 = await log_storage.query_logs(
        local_storage,
        filters={"job": "job-123"},
        page_size=10,
        page_cursor=page1.next_page,
    )

    assert len(page2.data) == 10
    assert page2.total == 25
    assert page2.next_page is not None
    assert page2.prev_page is not None

    # Get third page (partial)
    page3 = await log_storage.query_logs(
        local_storage,
        filters={"job": "job-123"},
        page_size=10,
        page_cursor=page2.next_page,
    )

    assert len(page3.data) == 5
    assert page3.total == 25
    assert page3.next_page is None


async def test_query_logs_tail_returns_latest_rows_oldest_to_newest(log_storage, local_storage, sample_log_entries):
    """Tail returns the latest window in normal oldest-to-newest display order."""
    await log_storage.insert_logs(local_storage, sample_log_entries)

    result = await log_storage.query_logs(
        local_storage,
        filters={"job": "job-123", "job_attempt": "attempt-1"},
        page_size=3,
        tail=3,
    )

    assert [log.message for log in result.data] == ["Log message 22", "Log message 23", "Log message 24"]
    assert result.total == 25
    assert result.next_page is None
    assert result.prev_page is not None


async def test_query_logs_tail_larger_than_log_returns_everything(log_storage, local_storage, sample_log_entries):
    """Tail larger than the available rows returns the complete log without a previous cursor."""
    await log_storage.insert_logs(local_storage, sample_log_entries)

    result = await log_storage.query_logs(
        local_storage,
        filters={"job": "job-123", "job_attempt": "attempt-1"},
        page_size=100,
        tail=100,
    )

    assert len(result.data) == 25
    assert result.total == 25
    assert result.next_page is None
    assert result.prev_page is None


async def test_query_logs_prev_page_deduplicates_tail_boundary(log_storage, local_storage):
    """The previous-page cursor avoids re-serving a non-unique timestamp boundary row."""
    timestamp = datetime(2026, 9, 2, 12, 0, 0)
    entries = [
        LogEntry(
            workspace="default",
            job="j",
            job_attempt="1",
            job_step="s",
            job_task="t",
            log_message=message,
            timestamp=timestamp,
        )
        for message in ["a", "b", "c", "d"]
    ]
    await log_storage.insert_logs(local_storage, entries)

    tail_page = await log_storage.query_logs(
        local_storage,
        filters={"job": "j", "job_attempt": "1"},
        page_size=2,
        tail=2,
    )
    previous_page = await log_storage.query_logs(
        local_storage,
        filters={"job": "j", "job_attempt": "1"},
        page_size=2,
        page_cursor=tail_page.prev_page,
    )

    combined_messages = [log.message for log in previous_page.data + tail_page.data]
    assert sorted(combined_messages) == ["a", "b", "c", "d"]
    assert len(combined_messages) == len(set(combined_messages))


async def test_query_logs_prev_page_preserves_identical_boundary_row_count(log_storage, local_storage):
    """The emitted-boundary count preserves identical rows without returning the same count twice."""
    timestamp = datetime(2026, 9, 2, 12, 0, 0)
    entries = [
        LogEntry(
            workspace="default",
            job="j",
            job_attempt="1",
            job_step="s",
            job_task="t",
            log_message="same",
            timestamp=timestamp,
        )
        for _ in range(3)
    ]
    await log_storage.insert_logs(local_storage, entries)

    tail_page = await log_storage.query_logs(
        local_storage,
        filters={"job": "j", "job_attempt": "1"},
        page_size=1,
        tail=1,
    )
    previous_page = await log_storage.query_logs(
        local_storage,
        filters={"job": "j", "job_attempt": "1"},
        page_size=2,
        page_cursor=tail_page.prev_page,
    )

    assert [log.message for log in previous_page.data + tail_page.data] == ["same", "same", "same"]
    assert previous_page.prev_page is None


async def test_query_logs_prev_page_rejects_different_filters(log_storage, local_storage):
    """A cursor is only valid for the effective query scope that produced it."""
    base_time = datetime(2026, 9, 2, 12, 0, 0)
    entries = [
        LogEntry(
            workspace="default",
            job="j",
            job_attempt="1",
            job_step="s",
            job_task="t",
            log_message=f"message {i}",
            timestamp=base_time + timedelta(seconds=i),
        )
        for i in range(3)
    ]
    await log_storage.insert_logs(local_storage, entries)

    tail_page = await log_storage.query_logs(
        local_storage,
        filters={"job": "j", "job_attempt": "1"},
        page_size=1,
        tail=1,
    )

    with pytest.raises(InvalidPageCursorError, match="page_cursor does not match the current log filters"):
        await log_storage.query_logs(
            local_storage,
            filters={"job": "j"},
            page_size=1,
            page_cursor=tail_page.prev_page,
        )


def test_tail_sql_does_not_use_offset():
    """Tail queries avoid the current page-number OFFSET path."""
    query = DuckDBLogRepository._build_tail_query("read_parquet('/tmp/logs/**/*.parquet')", "job = ?")

    assert "OFFSET" not in query.upper()
    assert "SHA256" in query.upper()


def test_cursor_v1_sql_does_not_use_offset():
    """V1 cursor queries avoid the current page-number OFFSET path."""
    query = DuckDBLogRepository._build_cursor_v1_query("read_parquet('/tmp/logs/**/*.parquet')", "job = ?")

    assert "OFFSET" not in query.upper()
    assert "SHA256" in query.upper()


async def test_tail_prev_page_cursor_uses_compact_anchor_payload(log_storage, local_storage, sample_log_entries):
    """Tail previous-page cursors carry compact typed anchor fields."""
    await log_storage.insert_logs(local_storage, sample_log_entries)

    page = await log_storage.query_logs(
        local_storage,
        filters={"job": "job-123"},
        page_size=1,
        tail=1,
    )

    assert page.prev_page is not None
    cursor = decode_log_page_cursor(page.prev_page)
    assert isinstance(cursor, LogPageCursorV1)
    assert cursor.version == 1
    assert len(cursor.boundary_row_hash) == 32
    assert len(cursor.query_scope_hash) == 32
    assert cursor.emitted_boundary_rows == 1


async def test_query_logs_rejects_malformed_page_cursor(log_storage, local_storage):
    """Malformed cursor values are reported as client cursor errors."""
    with pytest.raises(InvalidPageCursorError, match="Invalid page cursor"):
        await log_storage.query_logs(local_storage, page_cursor="garbage")


async def test_query_logs_no_files(log_storage, local_storage):
    """Test querying when no log files exist."""
    result = await log_storage.query_logs(local_storage)

    assert result.total == 0
    assert len(result.data) == 0
    assert result.next_page is None
    assert result.prev_page is None


async def test_insert_logs_empty_list(log_storage, local_storage):
    """Test that insert_logs handles empty list correctly."""
    count = await log_storage.insert_logs(local_storage, [])
    assert count == 0

    # No files should be created
    base_path = get_path(local_storage)
    logs_path = base_path / "logs"
    assert not logs_path.exists()


async def test_multiple_inserts_to_same_partition(log_storage, local_storage):
    """Test multiple insert operations to the same partition."""
    base_time = datetime(2024, 1, 1, 12, 0, 0)

    # First batch
    batch1 = [
        LogEntry(
            workspace="test-workspace",
            job="job-123",
            job_attempt="attempt-1",
            job_step="step1",
            job_task="task1",
            log_message="Batch 1 message",
            timestamp=base_time,
        )
    ]
    await log_storage.insert_logs(local_storage, batch1)

    # Second batch to same partition
    batch2 = [
        LogEntry(
            workspace="test-workspace",
            job="job-123",
            job_attempt="attempt-1",
            job_step="step1",
            job_task="task1",
            log_message="Batch 2 message",
            timestamp=base_time + timedelta(seconds=1),
        )
    ]
    await log_storage.insert_logs(local_storage, batch2)

    # Query should return both
    result = await log_storage.query_logs(
        local_storage,
        filters={"job": "job-123"},
    )

    assert result.total == 2
    messages = {log.message for log in result.data}
    assert "Batch 1 message" in messages
    assert "Batch 2 message" in messages


async def test_query_logs_rejects_invalid_filter_key(log_storage, local_storage):
    """Test that query_logs rejects unsupported filter keys."""
    with pytest.raises(InvalidFilterError, match="Invalid filter key"):
        await log_storage.query_logs(
            local_storage,
            filters={"invalid_key": "value"},
        )


@pytest.mark.parametrize(
    "filter_value",
    [
        "bad'; SELECT 1; --",
        "../escape",
        "contains space",
    ],
)
async def test_query_logs_rejects_invalid_partition_filter_values(log_storage, local_storage, filter_value):
    """Test that unsafe partition filter values are rejected."""
    with pytest.raises(InvalidFilterError, match="Invalid partition value"):
        await log_storage.query_logs(
            local_storage,
            filters={"job": filter_value},
        )


async def test_query_logs_allows_apostrophe_in_log_message_filter(log_storage, local_storage):
    """Test that log_message filtering still accepts apostrophes via bind params."""
    log_entry = LogEntry(
        workspace="test-workspace",
        job="job-apostrophe",
        job_attempt="attempt-1",
        job_step="step1",
        job_task="task1",
        log_message="I'm testing apostrophes",
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
    )
    await log_storage.insert_logs(local_storage, [log_entry])

    result = await log_storage.query_logs(
        local_storage,
        filters={"log_message": "I'm testing apostrophes"},
    )

    assert result.total == 1
    assert result.data[0].message == "I'm testing apostrophes"


def test_log_repository_rejects_multi_statement_query(local_storage, monkeypatch):
    """Test that query SQL containing injected multi-statements is rejected."""

    def _injected_path(cls, base_path, filters):  # noqa: ARG001
        return "logs'; SELECT 1; --"

    monkeypatch.setattr(DuckDBLogRepository, "_build_query_path", classmethod(_injected_path))

    with pytest.raises(ValueError, match="Multiple SQL statements detected in query"):
        DuckDBLogRepository._query_offset_page(
            base_path=local_storage.get_duckdb_path("logs"),
            filters={},
            page_size=10,
            current_page=1,
            direction=PaginationDirection.FORWARD,
            storage=local_storage,
        )


def test_log_repository_uses_hardened_duckdb_config(local_storage, monkeypatch):
    """Test that query connection is created with hardened DuckDB settings."""
    captured_database: str | None = None
    captured_config: dict[str, str] | None = None

    def _fake_connect(database: str, *, config: dict[str, str]):
        nonlocal captured_database, captured_config
        captured_database = database
        captured_config = config
        raise RuntimeError("connect intercepted")

    monkeypatch.setattr("nmp.core.files.app.log_db.duckdb.connect", _fake_connect)

    with pytest.raises(RuntimeError, match="connect intercepted"):
        DuckDBLogRepository._query_offset_page(
            base_path=local_storage.get_duckdb_path("logs"),
            filters={},
            page_size=10,
            current_page=1,
            direction=PaginationDirection.FORWARD,
            storage=local_storage,
        )

    assert captured_database == ":memory:"
    assert captured_config == {
        "autoload_known_extensions": "false",
        "autoinstall_known_extensions": "false",
    }


# S3 Storage Tests


class TestS3LogStorage:
    """Tests for S3 storage backend support in LogStorage."""

    @pytest.mark.parametrize(
        "prefix,expected_path",
        [
            ("test-prefix", "s3://test-bucket/test-prefix/logs"),
            ("", "s3://test-bucket/logs"),
        ],
    )
    def test_get_duckdb_path(self, prefix, expected_path):
        """Test S3 DuckDB path generation with and without prefix."""
        config = S3StorageConfig(
            bucket="test-bucket",
            prefix=prefix,
            use_sdk_auth=True,
        )
        storage = S3StorageImpl(config=config, secrets={})
        assert storage.get_duckdb_path("logs") == expected_path

    @pytest.mark.parametrize(
        "region,endpoint_url,expected_fragments",
        [
            ("us-east-1", None, ["REGION 'us-east-1'"]),
            (None, None, []),  # No region, no endpoint
            (
                "us-chicago-1",
                "https://example.objectstorage.oraclecloud.com",
                [
                    "REGION 'us-chicago-1'",
                    "ENDPOINT 'example.objectstorage.oraclecloud.com'",
                    "URL_STYLE 'path'",
                ],
            ),
        ],
    )
    def test_configure_s3_secret(self, region, endpoint_url, expected_fragments):
        """Test DuckDB S3 secret configuration."""
        config = S3StorageConfig(
            bucket="test-bucket",
            region=region,
            endpoint_url=endpoint_url,
            use_sdk_auth=True,
        )

        mock_conn = MagicMock()
        DuckDBLogRepository._configure_s3_secret(mock_conn, config)

        call_args = mock_conn.execute.call_args[0][0]
        assert "TYPE s3" in call_args
        assert "PROVIDER credential_chain" in call_args
        for fragment in expected_fragments:
            assert fragment in call_args
