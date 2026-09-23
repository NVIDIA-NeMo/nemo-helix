# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cache utilities for downloading files to cache storage."""

import logging

import anyio
from nemo_helix_plugin.files.types import CacheStatus as CacheStatus
from nhx.core.files.app.backends.base import StorageImpl
from nhx.core.files.app.file_lock import FileLockManager
from nhx.core.files.exceptions import NotFoundError

logger = logging.getLogger(__name__)

# Global capacity limiter for background cache operations.
_background_cache_limiter: anyio.CapacityLimiter | None = None


def _get_background_cache_limiter() -> anyio.CapacityLimiter:
    """Get the shared background cache limiter, creating it if necessary.

    Returns:
        The shared anyio.CapacityLimiter instance.
    """
    global _background_cache_limiter

    if _background_cache_limiter is None:
        from nhx.core.files.config import files_config

        config = files_config()
        _background_cache_limiter = anyio.CapacityLimiter(config.cache_warming_max_concurrent)
        logger.info(
            "Created background cache limiter",
            extra={"max_concurrent": config.cache_warming_max_concurrent},
        )
    return _background_cache_limiter


def reset_background_cache_limiter() -> None:
    """Reset the background cache limiter for testing purposes."""
    global _background_cache_limiter
    _background_cache_limiter = None


async def cache_file_directly(
    source_storage: StorageImpl,
    cache_storage: StorageImpl,
    path: str,
    lock_manager: FileLockManager,
) -> bool:
    """
    Download a file directly to cache storage.

    Unlike cache_stream which tees data to both cache and client, this function
    downloads directly to cache without any client response. Used for proactive
    cache warming.

    Acquires a global capacity limiter to prevent too many concurrent cache
    operations.

    Args:
        source_storage: The external storage backend (HuggingFace, NGC)
        cache_storage: The cache storage backend (default/local storage)
        path: Path of the file in the source storage
        lock_manager: Lock manager for cross-request coordination

    Returns:
        True if file was cached, False if skipped (not cacheable, already cached,
        or lock not acquired)
    """
    limiter = _get_background_cache_limiter()
    async with limiter:
        cache_path_key = await source_storage.get_cache_path_key(path)
        if cache_path_key is None:
            logger.debug("File not cacheable", extra={"path": path})
            return False

        # Check if already cached
        try:
            await cache_storage.get_file(cache_path_key)
            logger.debug("File already cached", extra={"path": path})
            return False
        except NotFoundError:
            pass  # Not cached, proceed

        # Acquire lock and cache
        async with lock_manager.acquire(cache_path_key) as acquired:
            if not acquired:
                logger.debug("Lock not acquired, skipping cache", extra={"path": path})
                return False

            # Double-check after acquiring lock (another request may have just cached it)
            try:
                await cache_storage.get_file(cache_path_key)
                logger.debug("File cached by another request", extra={"path": path})
                return False
            except NotFoundError:
                pass

            file_info = await source_storage.get_file(path)
            logger.info(
                "Starting file cache",
                extra={"path": path, "size_bytes": file_info.size},
            )
            source_stream = await source_storage.download(path, None)
            await cache_storage.upload(cache_path_key, source_stream, content_length=file_info.size)
            logger.info(
                "File cached successfully",
                extra={"path": path, "size_bytes": file_info.size},
            )
            return True


async def warm_fileset_cache(
    source_storage: StorageImpl,
    cache_storage: StorageImpl,
    lock_manager: FileLockManager,
) -> None:
    """
    Cache all files from a fileset in the background.

    This is a best-effort operation - failures are logged but not raised.
    Used for proactive cache warming when creating filesets with cache=True.

    Concurrency is controlled by cache_file_directly's global limiter.

    Args:
        source_storage: The external storage backend (HuggingFace, NGC)
        cache_storage: The cache storage backend (default/local storage)
        lock_manager: Lock manager for cross-request coordination
    """
    # Check if source is cacheable at all
    if await source_storage.get_cache_path_key() is None:
        logger.debug("Storage backend not cacheable, skipping cache warming")
        return

    try:
        files = await source_storage.list_files()
    except Exception:
        logger.warning("Failed to list files for cache warming", exc_info=True)
        return

    if not files:
        logger.debug("No files to cache")
        return

    async def cache_with_logging(path: str) -> None:
        try:
            await cache_file_directly(source_storage, cache_storage, path, lock_manager)
        except Exception:
            logger.warning("Failed to cache file", extra={"path": path}, exc_info=True)

    async with anyio.create_task_group() as tg:
        for file_info in files:
            tg.start_soon(cache_with_logging, file_info.path)

    logger.info("Cache warming complete", extra={"file_count": len(files)})
