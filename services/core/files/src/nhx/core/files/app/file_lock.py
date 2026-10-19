# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Database-backed file locking for coordinating cache writes across requests."""

import hashlib
import json
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import AsyncIterator

from nhx.common.entities import EntityClient, EntityConflictError, EntityNotFoundError
from nhx.core.files.entities import FileLock

logger = logging.getLogger(__name__)


def _path_to_lock_name(path: str) -> str:
    """Convert a file path to a valid entity name.

    Entity names must start with a lowercase letter and contain only lowercase
    letters, digits, and hyphens. We hash the path and prefix with 'lock-'.
    Uses SHA256 truncated to 32 chars for reasonable uniqueness.
    """
    return f"lock-{hashlib.sha256(path.encode()).hexdigest()[:32]}"


class FileLockManager:
    """Manages file locks via the entity store.

    Uses database unique constraints for atomic lock acquisition.
    Handles stale lock cleanup when locks expire.
    """

    def __init__(
        self,
        entity_client: EntityClient,
        workspace: str,
        lock_ttl_seconds: int = 300,
    ):
        """Initialize the lock manager.

        Args:
            entity_client: Entity client for database operations
            workspace: Workspace for lock entities (should match fileset workspace)
            lock_ttl_seconds: Time-to-live for locks in seconds (default 5 minutes)
        """
        self.entity_client = entity_client
        self.workspace = workspace
        self.lock_ttl = timedelta(seconds=lock_ttl_seconds)

    @asynccontextmanager
    async def acquire(self, path: str) -> AsyncIterator[bool]:
        """Acquire a lock for the given path.

        Yields True if the lock was acquired, False if another request holds a fresh lock.
        The lock is automatically released when the context manager exits.

        Args:
            path: The file path to lock (used as the lock entity name)

        Yields:
            True if lock acquired and caller should proceed with write
            False if lock not acquired and caller should skip write
        """
        lock = await self._try_acquire(path)
        if lock is None:
            logger.debug("Lock not acquired, another request is writing")
            yield False
            return

        logger.debug("Lock acquired")
        try:
            yield True
        finally:
            await self._release(lock)

    async def _try_acquire(self, path: str, max_attempts: int = 3) -> FileLock | None:
        """Attempt to acquire the lock, handling conflicts and expiry.

        Args:
            path: The file path to lock
            max_attempts: Maximum number of attempts before giving up

        Returns the acquired lock entity if lock acquired, None if should skip.
        """
        lock_name = _path_to_lock_name(path)

        for _ in range(max_attempts):
            # Try to create the lock
            lock = FileLock(
                name=lock_name,
                workspace=self.workspace,
                path=path,
                acquired_at=datetime.now(UTC),
            )
            try:
                created = await self.entity_client.create(lock)
                return created or lock
            except EntityConflictError:
                pass  # Lock exists, check if we should retry

            # Lock exists - check if it's stale
            try:
                existing = await self.entity_client.get(FileLock, lock_name, workspace=self.workspace)
            except EntityNotFoundError:
                # Lock was deleted, retry on next iteration
                continue

            expires_at = existing.acquired_at + self.lock_ttl
            if expires_at >= datetime.now(UTC):
                # Lock is fresh - another request is actively writing
                logger.debug("Fresh lock held by another request")
                return None

            # Lock is stale - atomically take it over by updating with version check.
            # Only succeeds if db_version hasn't changed (no one else took it).
            logger.debug("Found expired lock, attempting atomic takeover")
            try:
                # Update the entity - db_version is automatically included for optimistic locking
                existing.acquired_at = datetime.now(UTC)
                updated = await self.entity_client.update(existing)
                # Successfully took over the stale lock
                logger.debug("Successfully took over stale lock")
                return updated or existing
            except EntityConflictError:
                # Version changed - someone else took the lock, retry on next iteration
                logger.debug("Update failed, lock was modified by another request")
            except EntityNotFoundError:
                pass  # Someone else deleted it, that's fine

        logger.debug("Lock acquisition attempts exhausted")
        return None

    async def _release(self, lock: FileLock) -> None:
        """Release the lock."""
        try:
            await self.entity_client.delete(
                FileLock,
                lock.name,
                workspace=self.workspace,
                expected_db_version=lock.db_version,
            )
            logger.debug("Lock released")
        except EntityNotFoundError:
            # Lock already gone (expired and cleaned up by another request)
            logger.debug("Lock already released (expired)")
        except EntityConflictError:
            # Another request reacquired the lock after this holder lost ownership.
            logger.debug("Lock version changed before release; leaving current lock in place")

    async def get_active_locks(self, paths: list[str]) -> set[str]:
        """Get paths that have active (non-expired) locks.

        Args:
            paths: File paths to check for locks

        Returns:
            Set of paths that have active locks (caching in progress)
        """
        if not paths:
            return set()

        # Build reverse lookup: lock_name -> path
        lock_to_path = {_path_to_lock_name(p): p for p in paths}
        lock_names = list(lock_to_path.keys())

        # Query all locks at once using $in operator
        response = await self.entity_client.list(
            FileLock,
            workspace=self.workspace,
            filter_str=json.dumps({"name": {"$in": ",".join(lock_names)}}),
            page_size=len(lock_names),
        )

        # Filter to non-expired locks and return original paths
        now = datetime.now(UTC)
        active_lock_paths: set[str] = set()

        for lock in response.data:
            if lock.acquired_at + self.lock_ttl >= now:
                # Lock is still active
                if lock.name in lock_to_path:
                    active_lock_paths.add(lock_to_path[lock.name])

        return active_lock_paths
