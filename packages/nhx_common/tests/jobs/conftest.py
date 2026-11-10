# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest
from nhx.common.entities import DEFAULT_WORKSPACE
from nhx.common.jobs.file_manager import FilesetFileManager

NHX_URL = "http://localhost:8080"


@pytest.fixture
def mock_connection():
    """Create a mock DuckDB connection."""
    conn = MagicMock()
    conn.description = [
        ("workspace",),
        ("job",),
        ("job_attempt",),
        ("step",),
        ("task",),
        ("log_message",),
        ("timestamp",),
    ]
    return conn


@pytest.fixture
def mock_fileset_fs():
    """Mock FilesetFileSystem for testing."""
    fs = MagicMock()
    fs.info = MagicMock(return_value={"name": "test", "size": 0, "type": "directory"})
    fs.put_file = MagicMock()
    fs.get = MagicMock()
    fs.get_file = MagicMock()
    fs._info = AsyncMock(return_value={"name": "test", "size": 0, "type": "directory"})
    fs._put_file = AsyncMock()
    fs._get = AsyncMock()
    fs._get_file = AsyncMock()
    fs._client = MagicMock()
    return fs


@pytest.fixture
def fileset_manager(mock_fileset_fs) -> FilesetFileManager:
    """Create FilesetFileManager with mocked FilesetFileSystem.

    The FilesetFileManager only needs the concrete filesystem.
    """
    return FilesetFileManager(
        workspace=DEFAULT_WORKSPACE,
        fileset_name="job-results-jobid-123",
        filesystem=mock_fileset_fs,
    )
