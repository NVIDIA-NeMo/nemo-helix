# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for download_with_cache orchestration."""

from unittest.mock import AsyncMock, MagicMock

from nhx.core.files.api.endpoint_helpers import CacheContext, download_with_cache
from nhx.core.files.app.backends.base import ByteRange
from nhx.core.files.exceptions import NotFoundError


async def async_iter_from_list(items: list[bytes]):
    """Helper to create an async iterator from a list of bytes."""
    for item in items:
        yield item


def create_mock_cache_context(cache_storage: MagicMock) -> CacheContext:
    """Create a CacheContext with mocked storage and lock manager."""
    return CacheContext(
        storage=cache_storage,
        lock_manager=AsyncMock(),  # Lock manager only used in background task
    )


class MockBackgroundTasks:
    """Mock FastAPI BackgroundTasks for testing."""

    def __init__(self):
        self.tasks: list[tuple] = []

    def add_task(self, func, *args, **kwargs):
        self.tasks.append((func, args, kwargs))


async def test_cache_hit_serves_from_cache():
    """Test that cache hits are served directly from cache storage."""
    # Mock storages
    source_storage = MagicMock()
    source_storage.get_cache_path_key = AsyncMock(return_value="test/cache/key")
    source_storage.download = AsyncMock()  # Should NOT be called

    cache_storage = MagicMock()
    cached_data = [b"cached", b"data"]
    cache_storage.download = AsyncMock(return_value=async_iter_from_list(cached_data))

    cache_ctx = create_mock_cache_context(cache_storage)
    background_tasks = MockBackgroundTasks()

    # Download with cache hit
    result_chunks = []
    async for chunk in download_with_cache(source_storage, "file.txt", cache_ctx, background_tasks):
        result_chunks.append(chunk)

    # Should get cached data
    assert result_chunks == cached_data

    # Should NOT have called source download
    source_storage.download.assert_not_called()

    # Should have checked cache
    cache_storage.download.assert_called_once_with("test/cache/key", None)

    # Should NOT have scheduled any background task (cache hit)
    assert len(background_tasks.tasks) == 0


async def test_cache_miss_schedules_background_task():
    """Test that cache misses schedule a background caching task and stream from source."""
    from nhx.core.files.app.cache import cache_file_directly

    # Mock storages
    source_storage = MagicMock()
    source_storage.get_cache_path_key = AsyncMock(return_value="test/cache/key")
    source_data = [b"source", b"data"]
    source_storage.download = AsyncMock(return_value=async_iter_from_list(source_data))

    cache_storage = MagicMock()
    cache_storage.download = AsyncMock(side_effect=NotFoundError("Not in cache"))

    cache_ctx = create_mock_cache_context(cache_storage)
    background_tasks = MockBackgroundTasks()

    # Download with cache miss
    result_chunks = []
    async for chunk in download_with_cache(source_storage, "file.txt", cache_ctx, background_tasks):
        result_chunks.append(chunk)

    # Should get source data directly
    assert result_chunks == source_data

    # Should have called source download
    source_storage.download.assert_called_once_with("file.txt", None)

    # Should have tried cache first
    cache_storage.download.assert_called_once_with("test/cache/key", None)

    # Should have scheduled a background task for caching
    assert len(background_tasks.tasks) == 1
    task_func, task_args, task_kwargs = background_tasks.tasks[0]
    assert task_func is cache_file_directly
    assert task_args == (
        source_storage,
        cache_storage,
        "file.txt",
        cache_ctx.lock_manager,
    )


async def test_non_cacheable_storage_bypasses_cache():
    """Test that storage without cache_path_key bypasses caching entirely."""
    # Mock storage that doesn't support caching
    source_storage = MagicMock()
    source_storage.get_cache_path_key = AsyncMock(return_value=None)  # Not cacheable
    source_data = [b"direct", b"data"]
    source_storage.download = AsyncMock(return_value=async_iter_from_list(source_data))

    cache_storage = MagicMock()
    cache_ctx = create_mock_cache_context(cache_storage)
    background_tasks = MockBackgroundTasks()

    # Download non-cacheable content
    result_chunks = []
    async for chunk in download_with_cache(source_storage, "file.txt", cache_ctx, background_tasks):
        result_chunks.append(chunk)

    # Should get source data
    assert result_chunks == source_data

    # Should have called source download
    source_storage.download.assert_called_once_with("file.txt", None)

    # Should NOT have touched cache at all
    cache_storage.download.assert_not_called()

    # Should NOT have scheduled any background task (not cacheable)
    assert len(background_tasks.tasks) == 0


async def test_byte_range_requests_check_cache_and_schedule_background_task():
    """Test that byte range requests check cache and schedule background caching on miss."""
    from nhx.core.files.app.cache import cache_file_directly

    # Mock storages
    source_storage = MagicMock()
    source_storage.get_cache_path_key = AsyncMock(return_value="test/cache/key")
    source_data = [b"partial"]
    source_storage.download = AsyncMock(return_value=async_iter_from_list(source_data))

    cache_storage = MagicMock()
    cache_storage.download = AsyncMock(side_effect=NotFoundError("Not in cache"))

    cache_ctx = create_mock_cache_context(cache_storage)
    background_tasks = MockBackgroundTasks()
    byte_range = ByteRange(start=100, end=200)

    # Download with byte range
    result_chunks = []
    async for chunk in download_with_cache(source_storage, "file.txt", cache_ctx, background_tasks, byte_range):
        result_chunks.append(chunk)

    # Should get source data
    assert result_chunks == source_data

    # Should have tried cache first (with byte range)
    cache_storage.download.assert_called_once_with("test/cache/key", byte_range)

    # Should have called source with byte range
    source_storage.download.assert_called_once_with("file.txt", byte_range)

    # Should schedule background caching of full file for future requests
    assert len(background_tasks.tasks) == 1
    task_func, task_args, task_kwargs = background_tasks.tasks[0]
    assert task_func is cache_file_directly


async def test_byte_range_cache_hit_serves_from_cache():
    """Test that byte range requests can be served from cache when cached."""
    # Mock storages
    source_storage = MagicMock()
    source_storage.get_cache_path_key = AsyncMock(return_value="test/cache/key")
    source_storage.download = AsyncMock()  # Should NOT be called

    cache_storage = MagicMock()
    cached_data = [b"partial_cached"]
    cache_storage.download = AsyncMock(return_value=async_iter_from_list(cached_data))

    cache_ctx = create_mock_cache_context(cache_storage)
    background_tasks = MockBackgroundTasks()
    byte_range = ByteRange(start=100, end=200)

    # Download with byte range - cache hit
    result_chunks = []
    async for chunk in download_with_cache(source_storage, "file.txt", cache_ctx, background_tasks, byte_range):
        result_chunks.append(chunk)

    # Should get cached data
    assert result_chunks == cached_data

    # Should NOT have called source
    source_storage.download.assert_not_called()

    # Should have served from cache with byte range
    cache_storage.download.assert_called_once_with("test/cache/key", byte_range)

    # Should NOT schedule background task (cache hit)
    assert len(background_tasks.tasks) == 0
