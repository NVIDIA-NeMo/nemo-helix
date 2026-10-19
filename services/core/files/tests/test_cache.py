# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for cache utilities."""

from contextlib import asynccontextmanager
from dataclasses import dataclass
from unittest.mock import AsyncMock

import anyio
import pytest
from nhx.core.files.app.cache import (
    cache_file_directly,
    reset_background_cache_limiter,
    warm_fileset_cache,
)
from nhx.core.files.exceptions import NotFoundError


@pytest.fixture(autouse=True)
def reset_limiter():
    """Reset the global background cache limiter before and after each test."""
    reset_background_cache_limiter()
    yield
    reset_background_cache_limiter()


def create_mock_lock_manager(acquired: bool = True):
    """Create a mock lock manager that yields the specified acquired value."""
    mock = AsyncMock()

    @asynccontextmanager
    async def mock_acquire(path):
        yield acquired

    mock.acquire = mock_acquire
    return mock


@dataclass
class MockFileInfo:
    path: str
    size: int = 100


def create_mock_source_storage(
    cache_path_key: str | None = "cache/test/path.bin",
    download_data: list[bytes] | None = None,
):
    """Create a mock source storage backend."""
    mock = AsyncMock()
    mock.get_cache_path_key = AsyncMock(return_value=cache_path_key)

    if download_data is None:
        download_data = [b"data"]

    async def mock_download(path, byte_range):
        for chunk in download_data:
            yield chunk

    mock.download.return_value = mock_download(None, None)

    # get_file is needed for cache_file_directly to get content_length
    async def mock_get_file(path):
        return MockFileInfo(path=path, size=sum(len(d) for d in download_data))

    mock.get_file = mock_get_file
    return mock


def create_mock_cache_storage(file_exists: bool = False):
    """Create a mock cache storage backend."""
    mock = AsyncMock()

    async def mock_get_file(path):
        if file_exists:
            return MockFileInfo(path=path)
        raise NotFoundError(f"File not found: {path}")

    mock.get_file = mock_get_file
    return mock


@pytest.mark.parametrize(
    "scenario,cache_key,file_exists,lock_acquired,expected",
    [
        ("not_cacheable", None, False, True, False),
        ("already_cached", "cache/path", True, True, False),
        ("lock_not_acquired", "cache/path", False, False, False),
        ("success", "cache/path", False, True, True),
    ],
)
async def test_cache_file_directly_scenarios(scenario, cache_key, file_exists, lock_acquired, expected):
    """Test cache_file_directly returns correct result for various scenarios."""
    source_storage = create_mock_source_storage(cache_path_key=cache_key)
    cache_storage = create_mock_cache_storage(file_exists=file_exists)
    lock_manager = create_mock_lock_manager(acquired=lock_acquired)

    result = await cache_file_directly(source_storage, cache_storage, "test/file.bin", lock_manager)

    assert result == expected


async def test_cache_file_directly_uploads_to_cache():
    """Test that cache_file_directly actually uploads data to cache."""
    download_data = [b"chunk1", b"chunk2"]
    source_storage = create_mock_source_storage(download_data=download_data)
    cache_storage = create_mock_cache_storage(file_exists=False)
    lock_manager = create_mock_lock_manager(acquired=True)

    result = await cache_file_directly(source_storage, cache_storage, "test/file.bin", lock_manager)

    assert result is True
    cache_storage.upload.assert_called_once()
    call_args = cache_storage.upload.call_args
    assert call_args[0][0] == "cache/test/path.bin"


async def test_cache_file_directly_handles_download_error():
    """Test that download errors propagate."""
    source_storage = create_mock_source_storage()
    source_storage.download.side_effect = RuntimeError("Download failed")
    cache_storage = create_mock_cache_storage(file_exists=False)
    lock_manager = create_mock_lock_manager(acquired=True)

    with pytest.raises(RuntimeError, match="Download failed"):
        await cache_file_directly(source_storage, cache_storage, "test/file.bin", lock_manager)


async def test_cache_file_directly_handles_upload_error():
    """Test that upload errors propagate."""
    source_storage = create_mock_source_storage()
    cache_storage = create_mock_cache_storage(file_exists=False)
    cache_storage.upload.side_effect = RuntimeError("Upload failed")
    lock_manager = create_mock_lock_manager(acquired=True)

    with pytest.raises(RuntimeError, match="Upload failed"):
        await cache_file_directly(source_storage, cache_storage, "test/file.bin", lock_manager)


async def test_warm_fileset_cache_caches_all_files():
    """Test that warm_fileset_cache processes all files."""
    files = [MockFileInfo(path="file1.txt"), MockFileInfo(path="file2.txt")]

    source_storage = AsyncMock()
    source_storage.get_cache_path_key = AsyncMock(side_effect=lambda p=None: f"cache/{p}" if p else "cache/")
    source_storage.list_files.return_value = files

    async def mock_download(path, byte_range):
        yield b"data"

    source_storage.download.side_effect = lambda p, r: mock_download(p, r)

    # get_file is needed for content_length lookup
    source_storage.get_file = AsyncMock(return_value=MockFileInfo(path="any", size=4))

    cache_storage = create_mock_cache_storage(file_exists=False)
    lock_manager = create_mock_lock_manager(acquired=True)

    await warm_fileset_cache(source_storage, cache_storage, lock_manager)

    # Should have uploaded both files
    assert cache_storage.upload.call_count == 2


async def test_warm_fileset_cache_skips_non_cacheable_storage():
    """Test that warm_fileset_cache skips when storage is not cacheable."""
    source_storage = AsyncMock()
    source_storage.get_cache_path_key = AsyncMock(return_value=None)  # Not cacheable
    cache_storage = AsyncMock()
    lock_manager = AsyncMock()

    await warm_fileset_cache(source_storage, cache_storage, lock_manager)

    # Should not list files or upload anything
    source_storage.list_files.assert_not_called()
    cache_storage.upload.assert_not_called()


async def test_warm_fileset_cache_continues_on_error():
    """Test that warm_fileset_cache continues processing after individual file errors."""
    files = [
        MockFileInfo(path="file1.txt"),
        MockFileInfo(path="file2.txt"),
        MockFileInfo(path="file3.txt"),
    ]

    source_storage = AsyncMock()
    source_storage.get_cache_path_key = AsyncMock(side_effect=lambda p=None: f"cache/{p}" if p else "cache/")
    source_storage.list_files.return_value = files

    download_attempts = []

    async def mock_download(path, byte_range):
        download_attempts.append(path)
        if "file2" in path:
            raise RuntimeError("Download failed for file2")
        yield b"data"

    source_storage.download.side_effect = mock_download

    # Cache storage that consumes the stream on upload
    cache_storage = AsyncMock()

    async def mock_get_file(path):
        raise NotFoundError(f"File not found: {path}")

    async def mock_upload(path, stream, content_length=None):
        async for _ in stream:
            pass

    cache_storage.get_file = mock_get_file
    cache_storage.upload = mock_upload

    # Source needs get_file for content_length lookup
    source_storage.get_file = AsyncMock(return_value=MockFileInfo(path="any", size=100))

    lock_manager = create_mock_lock_manager(acquired=True)

    # Should not raise, just log and continue
    await warm_fileset_cache(source_storage, cache_storage, lock_manager)

    # All three files should have been attempted
    assert len(download_attempts) == 3
    assert set(download_attempts) == {"file1.txt", "file2.txt", "file3.txt"}


async def test_warm_fileset_cache_respects_concurrency_limit():
    """Test that warm_fileset_cache respects the concurrency limit.

    The global limiter uses cache_warming_max_concurrent from config (default=4).
    """
    files = [MockFileInfo(path=f"file{i}.txt") for i in range(10)]

    source_storage = AsyncMock()
    source_storage.get_cache_path_key = AsyncMock(side_effect=lambda p=None: f"cache/{p}" if p else "cache/")
    source_storage.list_files.return_value = files

    concurrent_count = 0
    max_concurrent_seen = 0
    lock = anyio.Lock()

    async def mock_download(path, byte_range):
        nonlocal concurrent_count, max_concurrent_seen
        async with lock:
            concurrent_count += 1
            max_concurrent_seen = max(max_concurrent_seen, concurrent_count)
        await anyio.sleep(0.01)  # Simulate some work
        async with lock:
            concurrent_count -= 1
        yield b"data"

    source_storage.download.side_effect = mock_download

    # get_file is needed for content_length lookup
    source_storage.get_file = AsyncMock(return_value=MockFileInfo(path="any", size=4))

    cache_storage = create_mock_cache_storage(file_exists=False)
    lock_manager = create_mock_lock_manager(acquired=True)

    await warm_fileset_cache(source_storage, cache_storage, lock_manager)

    # Concurrency should never exceed the config limit (default=4)
    assert max_concurrent_seen <= 4
    # All files should have been processed
    assert cache_storage.upload.call_count == 10
