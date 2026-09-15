# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from nemo_platform import AsyncNeMoPlatform
from nmp.common.entities import DEFAULT_WORKSPACE
from nmp.common.entities.utils import get_random_bytes
from nmp.common.jobs.file_manager import FilesetFileManager
from nmp.common.jobs.schemas import FileStorageType

NMP_URL = "http://localhost:8080"


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
def mock_nmp_sdk():
    """Mock sync NeMoPlatform SDK for jobs operations."""
    m = MagicMock()

    def _create(**kwargs):
        return SimpleNamespace(id=f"jobresult-{get_random_bytes()}", **kwargs)

    m.jobs.results.create.side_effect = _create
    return m


@pytest.fixture
def mock_async_nmp_sdk():
    """Mock async NeMoPlatform SDK for jobs operations."""
    m = AsyncMock(spec=AsyncNeMoPlatform)

    async def _create(**kwargs):
        return SimpleNamespace(id=f"jobresult-{get_random_bytes()}", **kwargs)

    m.jobs.results.create.side_effect = _create
    return m


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
def mock_sdk():
    """Mock NeMoPlatform SDK for legacy jobs tests."""

    from nemo_platform import NeMoPlatform

    sdk = MagicMock(spec=NeMoPlatform)
    sdk.base_url = "http://localhost:8080"
    sdk._custom_headers = None
    sdk._client = MagicMock()
    sdk.files = MagicMock()
    sdk.files.upload_content = MagicMock()

    # Mock list to return ListFilesResponse with empty data by default
    mock_list_response = MagicMock()
    mock_list_response.data = []
    sdk.files.list = MagicMock(return_value=mock_list_response)

    # Mock download_content to return bytes
    sdk.files.download_content = MagicMock(return_value=b"test content")

    return sdk


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


@pytest.fixture
def mock_sync_file_manager():
    """Mock sync file manager for ResultManager tests."""
    mock_fm = MagicMock()
    mock_fm.validate_storage.return_value = None
    mock_fm.upload.return_value = "test-ws/test-fileset#results/att-123/my-result"
    mock_fm.storage_type.return_value = FileStorageType.FILESET
    return mock_fm


@pytest.fixture
def mock_async_file_manager():
    """Mock async file manager for AsyncResultManager tests.

    Note: storage_type() is a sync method, so we use MagicMock for it.
    """
    mock_fm = MagicMock()
    mock_fm.validate_storage = AsyncMock(return_value=None)
    mock_fm.upload = AsyncMock(return_value="test-ws/test-fileset#results/att-123/my-result")
    mock_fm.storage_type.return_value = FileStorageType.FILESET
    return mock_fm
