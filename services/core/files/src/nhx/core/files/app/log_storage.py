# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async log storage facade over the log repository."""

import re

from anyio import to_thread
from nhx.common.jobs.schemas import HelixJobLogPage, decode_log_page_cursor
from nhx.core.files.app.backends.base import StorageImpl
from nhx.core.files.app.log_db import DuckDBLogRepository, LogEntry, LogQuery, LogRepository
from nhx.core.files.exceptions import InvalidPathError

_PATH_SEGMENT_RE = re.compile(r"^[\w\-.]+$")


def logs_base_path(storage: StorageImpl, artifact_base_path: str | None = None) -> str:
    """Hive root for job logs, optionally nested under a per-job ``artifact_base_path``.

    ``artifact_base_path`` is caller-supplied, and ``get_duckdb_path`` is a plain join with no
    traversal check of its own, so validate the segments here.
    """
    if not artifact_base_path:
        return storage.get_duckdb_path("logs")
    segments = artifact_base_path.split("/")
    if any(segment in {".", ".."} or not _PATH_SEGMENT_RE.match(segment) for segment in segments):
        raise InvalidPathError(
            f"Artifact base path '{artifact_base_path}' is not a relative path within the fileset. "
            "Ensure that paths such as ../.. are not used in the path."
        )
    return storage.get_duckdb_path(f"{artifact_base_path}/logs")


class LogStorage:
    """Coordinates async log reads and writes through a repository implementation."""

    def __init__(self, repository: LogRepository | None = None) -> None:
        self._repository = repository or DuckDBLogRepository()

    async def query_logs(
        self,
        storage: StorageImpl,
        filters: dict[str, str] | None = None,
        page_size: int = 100,
        page_cursor: str | None = None,
        artifact_base_path: str | None = None,
        tail: int | None = None,
    ) -> HelixJobLogPage:
        """Query logs from storage.

        Runs repository work in a thread pool because the DuckDB implementation is synchronous.
        When ``artifact_base_path`` is set, logs are read from ``<artifact_base_path>/logs``
        (must match the value used at insert time).
        """
        cursor = None
        if page_cursor:
            cursor = decode_log_page_cursor(page_cursor)

        query = LogQuery(
            base_path=logs_base_path(storage, artifact_base_path),
            filters=filters or {},
            page_size=page_size,
            cursor=cursor,
            tail=tail,
            artifact_base_path=artifact_base_path,
        )
        return await to_thread.run_sync(self._repository.query_logs, query, storage)

    async def insert_logs(
        self, storage: StorageImpl, log_entries: list[LogEntry], artifact_base_path: str | None = None
    ) -> int:
        """Insert log entries into storage.

        Runs repository work in a thread pool because the DuckDB implementation is synchronous.
        When ``artifact_base_path`` is set, logs nest under ``<artifact_base_path>/logs``.
        """
        if not log_entries:
            return 0

        base_path = logs_base_path(storage, artifact_base_path)
        return await to_thread.run_sync(self._repository.insert_logs, log_entries, base_path, storage)


def dep_log_storage() -> LogStorage:
    """FastAPI dependency for LogStorage."""
    return LogStorage()
