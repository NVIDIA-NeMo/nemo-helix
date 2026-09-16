# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DuckDB repository for persisted job logs.

This module provides local DuckDB-based query and insert operations for job logs,
avoiding the cross-service HTTP overhead of the previous FilesetFileSystem approach.

NOTE: pandas and S3StorageImpl are intentionally imported inside methods
for startup performance. Do not hoist them to module level.

TODO: Right now, this is very Jobs logs specific; in a future MR we should try to make it more generic.
"""

import hashlib
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from urllib.parse import urlparse

import duckdb
from nhx.common.files.storage_config import S3StorageConfig
from nhx.common.jobs.schemas import (
    InvalidPageCursorError,
    LogPageCursor,
    LogPageCursorV0,
    LogPageCursorV1,
    PageCursor,
    PaginationDirection,
    PlatformJobLog,
    PlatformJobLogPage,
)
from nhx.core.files.app.backends.base import StorageImpl
from nhx.core.files.exceptions import InvalidFilterError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DuckDBQueryParam = str | int | datetime


class LogEntry(BaseModel):
    """Internal representation of a log entry for storage."""

    workspace: str
    job: str
    job_attempt: str
    job_step: str
    job_task: str
    log_message: str
    timestamp: datetime


class QueriedLogEntry(LogEntry):
    """Internal representation of a queried log row."""

    total_count: int = Field(alias="_total_count")
    row_hash: str | None = None


@dataclass(frozen=True)
class LogQuery:
    """Repository-level log query input."""

    base_path: str
    filters: dict[str, str]
    page_size: int
    cursor: LogPageCursor | None = None
    tail: int | None = None
    artifact_base_path: str | None = None


class LogRepository(Protocol):
    """Persistence boundary for job logs."""

    def query_logs(self, query: LogQuery, storage: StorageImpl) -> PlatformJobLogPage:
        """Read a page of logs from the repository."""
        ...

    def insert_logs(self, log_entries: list[LogEntry], base_path: str, storage: StorageImpl) -> int:
        """Persist log entries into the repository."""
        ...


class DuckDBLogRepository:
    """DuckDB-backed repository for Hive-partitioned parquet log storage.

    This class is stateless - each operation creates its own DuckDB connection.
    This ensures thread safety when operations run concurrently in the thread pool.

    For local storage: Uses direct path access with DuckDB.
    For S3 storage: Uses s3:// URIs with DuckDB's httpfs extension.
    """

    # Partition columns in Hive directory order
    PARTITION_COLUMNS = ("job", "job_attempt", "job_step", "job_task")
    SAFE_PARTITION_VALUE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
    SELECT_LOG_COLUMNS_SQL = (
        "workspace, CAST(job AS VARCHAR) AS job, CAST(job_attempt AS VARCHAR) AS job_attempt, "
        "CAST(job_step AS VARCHAR) AS job_step, CAST(job_task AS VARCHAR) AS job_task, log_message, timestamp"
    )
    ROW_HASH_SQL = (
        "left(sha256(concat_ws(chr(31), CAST(timestamp AS VARCHAR), CAST(workspace AS VARCHAR), "
        "CAST(job AS VARCHAR), CAST(job_attempt AS VARCHAR), CAST(job_step AS VARCHAR), "
        "CAST(job_task AS VARCHAR), log_message)), 32)"
    )

    def query_logs(self, query: LogQuery, storage: StorageImpl) -> PlatformJobLogPage:
        """Query logs from parquet files using direct storage access.

        Supports both legacy page-number cursors and v1 boundary cursors used by
        tail pagination. This method is synchronous; ``LogStorage`` runs it in a
        thread pool to avoid blocking the event loop.
        """
        try:
            if query.tail is not None:
                return self._query_tail_window(
                    base_path=query.base_path,
                    filters=query.filters,
                    page_size=query.tail,
                    storage=storage,
                    artifact_base_path=query.artifact_base_path,
                )

            if isinstance(query.cursor, LogPageCursorV1):
                return self._query_cursor_v1_window(
                    base_path=query.base_path,
                    filters=query.filters,
                    page_size=query.page_size,
                    cursor=query.cursor,
                    storage=storage,
                    artifact_base_path=query.artifact_base_path,
                )

            current_page = query.cursor.start_id if isinstance(query.cursor, LogPageCursorV0) else 1
            direction = (
                query.cursor.direction if isinstance(query.cursor, LogPageCursorV0) else PaginationDirection.FORWARD
            )
            return self._query_offset_page(
                base_path=query.base_path,
                filters=query.filters,
                page_size=query.page_size,
                current_page=current_page,
                direction=direction,
                storage=storage,
            )
        except duckdb.IOException as e:
            if "No files found that match the pattern" in str(e):
                return PlatformJobLogPage(data=[], total=0, next_page=None, prev_page=None)
            logger.exception("IO error when querying logs")
            raise

    @classmethod
    def _connect(cls, storage: StorageImpl) -> duckdb.DuckDBPyConnection:
        conn = duckdb.connect(
            ":memory:",
            config={
                "autoload_known_extensions": "false",
                "autoinstall_known_extensions": "false",
            },
        )
        from nhx.core.files.app.backends.s3 import S3StorageImpl

        if isinstance(storage, S3StorageImpl):
            cls._load_s3_extensions(conn)
            cls._configure_s3_secret(conn, storage.config)
        return conn

    @staticmethod
    def _load_s3_extensions(conn: duckdb.DuckDBPyConnection) -> None:
        """Load DuckDB extensions required for S3 (aws, httpfs).

        Load explicitly so DuckDB does not try to auto-install at runtime, which
        can fail in environments without outbound network. Extensions should be
        pre-installed at image build time (see Dockerfile.nhx-core).
        """
        conn.execute("LOAD aws")
        conn.execute("LOAD httpfs")

    @staticmethod
    def _configure_s3_secret(conn: duckdb.DuckDBPyConnection, config: S3StorageConfig) -> None:
        """Configure DuckDB S3 secret for accessing the storage backend.

        Only supports credential_chain provider (use_sdk_auth=True) since log storage
        is only used with the platform's default_storage_config.
        """
        # Build secret parameters - always use credential_chain
        params = ["TYPE s3", "PROVIDER credential_chain"]

        if config.region:
            # Escape single quotes to prevent SQL syntax issues
            region = config.region.replace("'", "''")
            params.append(f"REGION '{region}'")

        if config.endpoint_url:
            # Extract just the host for DuckDB ENDPOINT
            parsed = urlparse(config.endpoint_url)
            endpoint = parsed.netloc.replace("'", "''")
            params.append(f"ENDPOINT '{endpoint}'")
            # OCI and some S3-compatible services need path-style URLs
            params.append("URL_STYLE 'path'")
            # Disable SSL for HTTP endpoints
            if parsed.scheme == "http":
                params.append("USE_SSL 'false'")

        secret_sql = f"CREATE OR REPLACE SECRET nhx_s3 ({', '.join(params)})"
        conn.execute(secret_sql)

    @classmethod
    def _build_query_path(cls, base_path: str, filters: dict[str, str]) -> str:
        """Build an optimized parquet path pattern using partition filters.

        Hive partitioning uses directory structure: job=X/job_attempt=Y/job_step=Z/job_task=W/
        By pushing filters into the path, we avoid scanning unrelated partitions.

        Args:
            base_path: Base logs directory path.
            filters: Query filters, which may include partition and non-partition keys.

        Returns:
            Optimized glob pattern for read_parquet.
        """
        path_parts = [base_path]

        # Add partition filters to path in order (must be contiguous from root)
        for col in cls.PARTITION_COLUMNS:
            if col in filters:
                path_parts.append(f"{col}={filters[col]}")
            else:
                # Can't skip partition levels in Hive, stop here
                break

        # Add glob for remaining levels
        path_parts.append("**/*.parquet")
        return "/".join(path_parts)

    @staticmethod
    def _ensure_single_statement(query: str) -> None:
        sql = query.strip()
        if ";" in sql.rstrip(" ;\n\t"):
            logger.error(f"Multiple SQL statements detected in query: {sql}")
            raise ValueError("Multiple SQL statements detected in query")

    @classmethod
    def _prepare_query(
        cls,
        base_path: str,
        filters: dict[str, str],
    ) -> tuple[str, str, list[str]]:
        if filters:
            for key, value in filters.items():
                if key in cls.PARTITION_COLUMNS and not cls.SAFE_PARTITION_VALUE.fullmatch(str(value)):
                    raise InvalidFilterError(f"Invalid partition value: {value} for key: {key}.")
        query_path = cls._build_query_path(base_path, filters)
        base_table = f"read_parquet('{query_path}', hive_partitioning=1)"

        # Determine which filters were pushed into path vs need WHERE clause.
        # Filters are pushed contiguously from the start of PARTITION_COLUMNS.
        path_filters: set[str] = set()
        for col in cls.PARTITION_COLUMNS:
            if col in filters:
                path_filters.add(col)
            else:
                break

        # Build WHERE clause for remaining filters not in path.
        where_clauses: list[str] = []
        params: list[str] = []
        for key, value in filters.items():
            if key not in path_filters:
                # Only allow filters that are in the LogEntry model and nothing else.
                if key not in LogEntry.model_fields.keys():
                    raise InvalidFilterError(
                        f"Invalid filter key: {key}. Allowed keys are: {LogEntry.model_fields.keys()}"
                    )
                where_clauses.append(f"{key} = ?")
                params.append(value)
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        return base_table, where_clause, params

    @staticmethod
    def _fetch_queried_logs(result: duckdb.DuckDBPyConnection) -> list[QueriedLogEntry]:
        if result.description is None:
            raise RuntimeError("Cannot read description from DuckDB result")

        columns = [desc[0] for desc in result.description]
        return [QueriedLogEntry.model_validate(dict(zip(columns, row))) for row in result.fetchall()]

    @staticmethod
    def _platform_logs(rows: list[QueriedLogEntry]) -> list[PlatformJobLog]:
        return [
            PlatformJobLog(
                timestamp=row.timestamp,
                job=row.job,
                job_step=row.job_step,
                job_task=row.job_task,
                message=row.log_message,
            )
            for row in rows
        ]

    @staticmethod
    def _query_scope_hash(filters: dict[str, str], artifact_base_path: str | None) -> str:
        scope = {"artifact_base_path": artifact_base_path, "filters": filters}
        encoded = json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:32]

    @classmethod
    def _validate_cursor_scope(
        cls, cursor: LogPageCursorV1, filters: dict[str, str], artifact_base_path: str | None
    ) -> None:
        if cursor.query_scope_hash != cls._query_scope_hash(filters, artifact_base_path):
            raise InvalidPageCursorError("page_cursor does not match the current log filters.")

    @staticmethod
    def _same_boundary(row: QueriedLogEntry, cursor: LogPageCursorV1) -> bool:
        return row.timestamp == cursor.boundary_timestamp and row.row_hash == cursor.boundary_row_hash

    @classmethod
    def _drop_emitted_boundary_rows(cls, rows: list[QueriedLogEntry], cursor: LogPageCursorV1) -> list[QueriedLogEntry]:
        remaining = cursor.emitted_boundary_rows
        kept: list[QueriedLogEntry] = []
        for row in rows:
            if remaining > 0 and cls._same_boundary(row, cursor):
                remaining -= 1
                continue
            kept.append(row)
        return kept

    @classmethod
    def _emitted_count_for_boundary(
        cls,
        rows: list[QueriedLogEntry],
        prior_cursor: LogPageCursorV1 | None = None,
    ) -> int:
        if not rows:
            return 0
        boundary = rows[0]
        count = sum(1 for row in rows if row.timestamp == boundary.timestamp and row.row_hash == boundary.row_hash)
        if prior_cursor is not None and cls._same_boundary(boundary, prior_cursor):
            count += prior_cursor.emitted_boundary_rows
        return count

    @classmethod
    def _cursor_v1_from_boundary(
        cls,
        boundary_row: QueriedLogEntry,
        filters: dict[str, str],
        artifact_base_path: str | None,
        emitted_boundary_rows: int,
    ) -> str:
        if boundary_row.row_hash is None:
            raise RuntimeError("Cannot create a log cursor without a row hash.")
        return LogPageCursorV1(
            boundary_timestamp=boundary_row.timestamp,
            boundary_row_hash=boundary_row.row_hash,
            query_scope_hash=cls._query_scope_hash(filters, artifact_base_path),
            emitted_boundary_rows=emitted_boundary_rows,
        ).encode()

    @classmethod
    def _build_tail_query(cls, base_table: str, where_clause: str) -> str:
        return f"""
            SELECT
                {cls.SELECT_LOG_COLUMNS_SQL},
                {cls.ROW_HASH_SQL} AS row_hash,
                COUNT(*) OVER() as _total_count
            FROM {base_table}
            WHERE {where_clause}
            ORDER BY timestamp DESC, row_hash DESC
            LIMIT ?
        """

    @classmethod
    def _build_cursor_v1_query(cls, base_table: str, where_clause: str) -> str:
        return f"""
            WITH filtered AS (
                SELECT
                    {cls.SELECT_LOG_COLUMNS_SQL},
                    {cls.ROW_HASH_SQL} AS row_hash,
                    COUNT(*) OVER() as _total_count
                FROM {base_table}
                WHERE {where_clause}
            ),
            eligible AS (
                SELECT *
                FROM filtered
                WHERE timestamp < ? OR (timestamp = ? AND row_hash <= ?)
            )
            SELECT *
            FROM eligible
            ORDER BY timestamp DESC, row_hash DESC
            LIMIT ?
        """

    @classmethod
    def _query_offset_page(
        cls,
        base_path: str,
        filters: dict[str, str],
        page_size: int,
        current_page: int,
        direction: PaginationDirection,
        storage: StorageImpl,
    ) -> PlatformJobLogPage:
        """Query the legacy page-number cursor mode.

        Uses a window function (COUNT(*) OVER()) to get total count in a single
        query pass, avoiding the overhead of a separate COUNT query.

        Optimizes glob pattern by pushing partition filters into the path,
        reducing filesystem scanning when filtering by job/attempt/step/task.
        """
        conn = cls._connect(storage)
        try:
            base_table, where_clause, params = cls._prepare_query(base_path, filters)

            query_direction = "ASC" if direction == PaginationDirection.FORWARD else "DESC"
            offset = (current_page - 1) * page_size

            # Single query with window function for total count.
            # Fetch page_size + 1 to check if there are more results.
            query = f"""
                SELECT
                    {cls.SELECT_LOG_COLUMNS_SQL},
                    COUNT(*) OVER() as _total_count
                FROM {base_table}
                WHERE {where_clause}
                ORDER BY timestamp {query_direction}
                LIMIT ? OFFSET ?
            """
            cls._ensure_single_statement(query)
            query_params: list[DuckDBQueryParam] = [*params, page_size + 1, offset]
            logs = cls._fetch_queried_logs(conn.execute(query, query_params))

            # Extract total count from first row (all rows have same _total_count).
            total_count = logs[0].total_count if logs else 0

            has_more = len(logs) > page_size
            if has_more:
                logs = logs[:page_size]

            log_lines = cls._platform_logs(logs)

            # Calculate pagination cursors.
            next_page: str | None = None
            prev_page: str | None = None

            if direction == PaginationDirection.FORWARD:
                if has_more:
                    next_page = PageCursor(start_id=current_page + 1, direction=PaginationDirection.FORWARD).encode()
                if current_page > 1:
                    prev_page = PageCursor(start_id=current_page - 1, direction=PaginationDirection.FORWARD).encode()
            else:
                if has_more:
                    prev_page = PageCursor(
                        start_id=current_page + 1,
                        direction=PaginationDirection.BACKWARD,
                    ).encode()
                if current_page > 1:
                    next_page = PageCursor(
                        start_id=current_page - 1,
                        direction=PaginationDirection.BACKWARD,
                    ).encode()

            return PlatformJobLogPage(
                data=log_lines,
                total=total_count,
                next_page=next_page,
                prev_page=prev_page,
            )
        finally:
            conn.close()

    @classmethod
    def _query_tail_window(
        cls,
        base_path: str,
        filters: dict[str, str],
        page_size: int,
        storage: StorageImpl,
        artifact_base_path: str | None,
    ) -> PlatformJobLogPage:
        conn = cls._connect(storage)
        try:
            base_table, where_clause, params = cls._prepare_query(base_path, filters)
            query = cls._build_tail_query(base_table, where_clause)
            cls._ensure_single_statement(query)
            query_params: list[DuckDBQueryParam] = [*params, page_size + 1]
            rows = cls._fetch_queried_logs(conn.execute(query, query_params))

            total_count = rows[0].total_count if rows else 0
            has_previous_window = len(rows) > page_size
            returned_rows = list(reversed(rows[:page_size]))
            prev_page = None
            if has_previous_window and returned_rows:
                prev_page = cls._cursor_v1_from_boundary(
                    returned_rows[0],
                    filters,
                    artifact_base_path,
                    cls._emitted_count_for_boundary(returned_rows),
                )

            return PlatformJobLogPage(
                data=cls._platform_logs(returned_rows),
                total=total_count,
                next_page=None,
                prev_page=prev_page,
            )
        finally:
            conn.close()

    @classmethod
    def _query_cursor_v1_window(
        cls,
        base_path: str,
        filters: dict[str, str],
        page_size: int,
        cursor: LogPageCursorV1,
        storage: StorageImpl,
        artifact_base_path: str | None,
    ) -> PlatformJobLogPage:
        cls._validate_cursor_scope(cursor, filters, artifact_base_path)
        conn = cls._connect(storage)
        try:
            base_table, where_clause, params = cls._prepare_query(base_path, filters)
            query = cls._build_cursor_v1_query(base_table, where_clause)
            cls._ensure_single_statement(query)
            rows = cls._fetch_queried_logs(
                conn.execute(
                    query,
                    [
                        *params,
                        cursor.boundary_timestamp,
                        cursor.boundary_timestamp,
                        cursor.boundary_row_hash,
                        page_size + cursor.emitted_boundary_rows + 1,
                    ],
                )
            )

            total_count = rows[0].total_count if rows else 0
            usable_rows = cls._drop_emitted_boundary_rows(rows, cursor)
            has_previous_window = len(usable_rows) > page_size
            returned_rows = list(reversed(usable_rows[:page_size]))
            prev_page = None
            if has_previous_window and returned_rows:
                prev_page = cls._cursor_v1_from_boundary(
                    returned_rows[0],
                    filters,
                    artifact_base_path,
                    cls._emitted_count_for_boundary(returned_rows, prior_cursor=cursor),
                )

            return PlatformJobLogPage(
                data=cls._platform_logs(returned_rows),
                total=total_count,
                next_page=None,
                prev_page=prev_page,
            )
        finally:
            conn.close()

    @classmethod
    def insert_logs(cls, log_entries: list[LogEntry], base_path: str, storage: StorageImpl) -> int:
        """Insert logs into Hive-partitioned parquet files.

        Uses DuckDB to write directly to the resolved storage path. ``base_path`` is
        resolved by the async facade and may point at root logs or an artifact-scoped
        nested logs directory. This method is synchronous; ``LogStorage`` runs it in
        a thread pool to avoid blocking the event loop.
        """
        conn = duckdb.connect(":memory:")
        table_name = f"temp_logs_{uuid.uuid4().hex[:8]}"

        try:
            from nhx.core.files.app.backends.local import LocalStorageImpl
            from nhx.core.files.app.backends.s3 import S3StorageImpl

            # DuckDB COPY creates the leaf + partition dirs but not intermediate parents, so a
            # nested base path (e.g. <root>/jobs/<job>/logs) needs its parents created first.
            if isinstance(storage, LocalStorageImpl):
                os.makedirs(base_path, exist_ok=True)

            if isinstance(storage, S3StorageImpl):
                cls._load_s3_extensions(conn)
                cls._configure_s3_secret(conn, storage.config)

            import pandas as pd

            df = pd.DataFrame([entry.model_dump() for entry in log_entries])
            conn.register(table_name, df)

            insert_query = f"""
                COPY (
                    SELECT workspace, job, job_attempt, job_step, job_task, log_message, timestamp
                    FROM {table_name}
                    ORDER BY timestamp
                ) TO '{base_path}' (
                    FORMAT PARQUET,
                    PARTITION_BY (job, job_attempt, job_step, job_task),
                    APPEND
                )
            """
            conn.execute(insert_query)
            logger.debug(f"Successfully inserted {len(log_entries)} log entries")
            return len(log_entries)
        except Exception:
            logger.exception("Failed to insert log entries")
            raise
        finally:
            conn.close()
