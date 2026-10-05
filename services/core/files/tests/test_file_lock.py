# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for FileLockManager and cache locking."""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from nhx.common.entities import EntityConflictError, EntityNotFoundError
from nhx.core.files.app.file_lock import FileLockManager, _path_to_lock_name
from nhx.core.files.entities import FileLock


@pytest.fixture
def mock_entity_client():
    """Create a mock entity client."""
    return AsyncMock()


@pytest.fixture
def lock_manager(mock_entity_client):
    """Create a FileLockManager with mocked entity client."""
    return FileLockManager(
        entity_client=mock_entity_client,
        workspace="test-workspace",
        lock_ttl_seconds=300,
    )


async def test_acquire_succeeds_when_no_existing_lock(lock_manager, mock_entity_client):
    """Test that lock acquisition succeeds when no existing lock."""
    mock_entity_client.create.return_value = None
    path = "test/file.bin"
    expected_lock_name = _path_to_lock_name(path)

    async with lock_manager.acquire(path) as acquired:
        assert acquired is True

    # Should have created a lock with hashed name
    mock_entity_client.create.assert_called_once()
    created_lock = mock_entity_client.create.call_args[0][0]
    assert isinstance(created_lock, FileLock)
    assert created_lock.name == expected_lock_name
    assert created_lock.workspace == "test-workspace"

    # Should have released the lock
    mock_entity_client.delete.assert_called_once_with(
        FileLock,
        expected_lock_name,
        workspace="test-workspace",
        expected_db_version=1,
    )


async def test_acquire_fails_when_fresh_lock_exists(lock_manager, mock_entity_client):
    """Test that lock acquisition fails when another pod has a fresh lock."""
    path = "test/file.bin"
    lock_name = _path_to_lock_name(path)

    # First create attempt fails with conflict
    mock_entity_client.create.side_effect = EntityConflictError("conflict")

    # Get returns a fresh lock (not expired - acquired recently)
    fresh_lock = FileLock(
        name=lock_name,
        workspace="test-workspace",
        path=path,
        acquired_at=datetime.now(UTC),
    )
    mock_entity_client.get.return_value = fresh_lock

    async with lock_manager.acquire(path) as acquired:
        assert acquired is False

    # Should not have released anything (we don't own the lock)
    mock_entity_client.delete.assert_not_called()


async def test_acquire_succeeds_after_cleaning_stale_lock(lock_manager, mock_entity_client):
    """Test that stale locks are atomically taken over using update with automatic version check."""
    path = "test/file.bin"
    lock_name = _path_to_lock_name(path)

    # First create attempt fails with conflict (lock exists)
    mock_entity_client.create.side_effect = EntityConflictError("conflict")

    # Get returns a stale lock (acquired > TTL ago, so expired) with db_version 2
    stale_time = datetime.now(UTC) - timedelta(minutes=10)  # Older than 5min TTL
    stale_lock = FileLock(
        name=lock_name,
        workspace="test-workspace",
        path=path,
        acquired_at=stale_time,
    )
    # Set db_version on the stale lock
    stale_lock._db_version = 2
    mock_entity_client.get.return_value = stale_lock

    # Update succeeds (atomically takes over the stale lock)
    updated_lock = FileLock(
        name=lock_name,
        workspace="test-workspace",
        path=path,
        acquired_at=datetime.now(UTC),  # Updated timestamp
    )
    updated_lock._db_version = 3  # Version incremented after update
    mock_entity_client.update.return_value = updated_lock

    async with lock_manager.acquire(path) as acquired:
        assert acquired is True

    # Should have called update with the modified entity (db_version automatically included)
    mock_entity_client.update.assert_called_once()
    call_args = mock_entity_client.update.call_args
    updated_entity = call_args[0][0]  # First positional arg is the entity
    assert isinstance(updated_entity, FileLock)
    assert updated_entity.name == lock_name
    assert updated_entity.workspace == "test-workspace"
    # acquired_at should be updated (check it's recent, not the old stale time)
    # Note: The code modifies the same object, so we compare to the original stale_time
    assert updated_entity.acquired_at > stale_time
    assert updated_entity._db_version == 2  # db_version from stale lock is preserved
    # Lock should be released when context manager exits (delete is called in _release)
    mock_entity_client.delete.assert_called_once_with(
        FileLock,
        lock_name,
        workspace="test-workspace",
        expected_db_version=3,
    )


async def test_acquire_handles_update_version_conflict(lock_manager, mock_entity_client):
    """Test that version conflict on update is handled gracefully."""
    path = "test/file.bin"
    lock_name = _path_to_lock_name(path)

    # First create attempt fails with conflict
    mock_entity_client.create.side_effect = EntityConflictError("conflict")

    # Get returns a stale lock with db_version 2
    stale_lock = FileLock(
        name=lock_name,
        workspace="test-workspace",
        path=path,
        acquired_at=datetime.now(UTC) - timedelta(minutes=10),
    )
    stale_lock._db_version = 2
    mock_entity_client.get.return_value = stale_lock

    # Update fails with version conflict (someone else took the lock)
    mock_entity_client.update.side_effect = EntityConflictError("Version mismatch")

    # On retry, get returns a fresh lock (someone else successfully took it)
    fresh_lock = FileLock(
        name=lock_name,
        workspace="test-workspace",
        path=path,
        acquired_at=datetime.now(UTC),  # Fresh lock
    )
    fresh_lock._db_version = 3
    # On second iteration, get returns fresh lock
    mock_entity_client.get.side_effect = [stale_lock, fresh_lock]

    async with lock_manager.acquire(path) as acquired:
        # Should fail because lock is now fresh (held by another request)
        assert acquired is False

    # Should have called update once (failed due to version conflict)
    mock_entity_client.update.assert_called_once()
    # Should not have deleted anything
    mock_entity_client.delete.assert_not_called()


async def test_acquire_handles_update_not_found(lock_manager, mock_entity_client):
    """Test that EntityNotFoundError from update is handled (someone else deleted it)."""
    path = "test/file.bin"
    lock_name = _path_to_lock_name(path)

    # First create attempt fails with conflict
    mock_entity_client.create.side_effect = EntityConflictError("conflict")

    # Get returns a stale lock
    stale_lock = FileLock(
        name=lock_name,
        workspace="test-workspace",
        path=path,
        acquired_at=datetime.now(UTC) - timedelta(minutes=10),
    )
    stale_lock._db_version = 2
    mock_entity_client.get.return_value = stale_lock

    # Update fails because someone else deleted the lock
    mock_entity_client.update.side_effect = EntityNotFoundError("not found")

    # On retry, create succeeds (lock is gone, we can create new one)
    mock_entity_client.create.side_effect = [
        EntityConflictError("conflict"),  # First attempt
        None,  # Second attempt succeeds
    ]
    # Second get call returns not found (lock was deleted)
    mock_entity_client.get.side_effect = [
        stale_lock,  # First get returns stale lock
        EntityNotFoundError("not found"),  # Second get returns not found
    ]

    async with lock_manager.acquire(path) as acquired:
        # Should succeed after retry
        assert acquired is True

    # Should have called update once (failed, but that's fine)
    mock_entity_client.update.assert_called_once()
    # Should have created new lock after retry
    assert mock_entity_client.create.call_count == 2


async def test_race_condition_multiple_requests_takeover_stale_lock(lock_manager, mock_entity_client):
    """Test race condition: multiple requests trying to take over stale lock simultaneously."""
    path = "test/file.bin"
    lock_name = _path_to_lock_name(path)

    # First create attempt fails with conflict
    mock_entity_client.create.side_effect = EntityConflictError("conflict")

    # Get returns a stale lock with db_version 2
    stale_lock = FileLock(
        name=lock_name,
        workspace="test-workspace",
        path=path,
        acquired_at=datetime.now(UTC) - timedelta(minutes=10),
    )
    stale_lock._db_version = 2
    mock_entity_client.get.return_value = stale_lock

    # Simulate race condition: update fails because another request
    # already updated the lock (version changed from 2 to 3)
    mock_entity_client.update.side_effect = EntityConflictError("Version mismatch: expected 2, got 3")

    # On retry, get returns the lock with updated version (another request won)
    updated_lock = FileLock(
        name=lock_name,
        workspace="test-workspace",
        path=path,
        acquired_at=datetime.now(UTC),  # Updated by another request
    )
    updated_lock._db_version = 3
    # First get returns stale, second get returns updated (fresh) lock
    mock_entity_client.get.side_effect = [stale_lock, updated_lock]

    async with lock_manager.acquire(path) as acquired:
        # Should fail because another request successfully took the lock
        assert acquired is False

    # Should have attempted update once
    mock_entity_client.update.assert_called_once()
    # Verify it was called with the entity that has db_version 2 from stale lock
    call_args = mock_entity_client.update.call_args
    updated_entity = call_args[0][0]
    assert updated_entity._db_version == 2


async def test_lock_released_on_success(lock_manager, mock_entity_client):
    """Test that lock is released after successful operation."""
    mock_entity_client.create.return_value = None
    path = "test/file.bin"
    expected_lock_name = _path_to_lock_name(path)

    async with lock_manager.acquire(path) as acquired:
        assert acquired is True
        # Do some work
        await asyncio.sleep(0.01)

    # Lock should be released
    mock_entity_client.delete.assert_called_with(
        FileLock,
        expected_lock_name,
        workspace="test-workspace",
        expected_db_version=1,
    )


async def test_lock_released_on_exception(lock_manager, mock_entity_client):
    """Test that lock is released even if an exception occurs."""
    mock_entity_client.create.return_value = None
    path = "test/file.bin"
    expected_lock_name = _path_to_lock_name(path)

    with pytest.raises(ValueError, match="test error"):
        async with lock_manager.acquire(path) as acquired:
            assert acquired is True
            raise ValueError("test error")

    # Lock should still be released
    mock_entity_client.delete.assert_called_with(
        FileLock,
        expected_lock_name,
        workspace="test-workspace",
        expected_db_version=1,
    )


async def test_lock_not_released_if_already_gone(lock_manager, mock_entity_client):
    """Test that missing lock during release doesn't cause an error."""
    mock_entity_client.create.return_value = None
    mock_entity_client.delete.side_effect = EntityNotFoundError("not found")

    # Should not raise
    async with lock_manager.acquire("test/file.bin") as acquired:
        assert acquired is True

    # Delete was called but didn't raise despite EntityNotFoundError
    mock_entity_client.delete.assert_called_once()


async def test_lock_not_released_if_version_changed(lock_manager, mock_entity_client):
    """Test that a lock reacquired by another request is not deleted by stale release."""
    mock_entity_client.create.return_value = None
    mock_entity_client.delete.side_effect = EntityConflictError("changed")

    async with lock_manager.acquire("test/file.bin") as acquired:
        assert acquired is True

    mock_entity_client.delete.assert_called_once()


async def test_conflict_then_lock_deleted_before_get(lock_manager, mock_entity_client):
    """Test handling when lock is deleted between create and get."""
    # Track create calls - first fails, second (after retry) succeeds
    create_call_count = 0

    async def create_side_effect(lock):
        nonlocal create_call_count
        create_call_count += 1
        if create_call_count == 1:
            raise EntityConflictError("conflict")
        return None

    mock_entity_client.create.side_effect = create_side_effect

    # Get fails because lock was deleted
    mock_entity_client.get.side_effect = EntityNotFoundError("not found")

    async with lock_manager.acquire("test/file.bin") as acquired:
        assert acquired is True

    # Should have retried and succeeded
    assert mock_entity_client.create.call_count == 2


async def test_retries_are_bounded(lock_manager, mock_entity_client):
    """Test that lock acquisition gives up after max retries to prevent infinite loops."""
    # Every create fails with conflict
    mock_entity_client.create.side_effect = EntityConflictError("conflict")
    # Every get fails with not found (simulating race condition)
    mock_entity_client.get.side_effect = EntityNotFoundError("not found")

    async with lock_manager.acquire("test/file.bin") as acquired:
        # Should give up after retries exhausted
        assert acquired is False

    # Should have tried a bounded number of times (initial + retries)
    assert mock_entity_client.create.call_count <= 3


# Tests for get_active_locks


async def test_get_active_locks_empty_paths(lock_manager, mock_entity_client):
    """Test get_active_locks returns empty set for empty paths list."""
    result = await lock_manager.get_active_locks([])

    assert result == set()
    mock_entity_client.list.assert_not_called()


async def test_get_active_locks_no_locks_exist(lock_manager, mock_entity_client):
    """Test get_active_locks returns empty set when no locks exist."""
    mock_entity_client.list.return_value = AsyncMock(data=[])

    paths = ["cache/hf/repo/main/file1.txt", "cache/hf/repo/main/file2.txt"]
    result = await lock_manager.get_active_locks(paths)

    assert result == set()
    mock_entity_client.list.assert_called_once()


async def test_get_active_locks_with_active_locks(lock_manager, mock_entity_client):
    """Test get_active_locks returns paths with active (non-expired) locks."""
    path1 = "cache/hf/repo/main/file1.txt"
    path2 = "cache/hf/repo/main/file2.txt"

    # Create active lock for path1
    active_lock = FileLock(
        name=_path_to_lock_name(path1),
        workspace="test-workspace",
        path=path1,
        acquired_at=datetime.now(UTC),  # Fresh lock
    )
    mock_entity_client.list.return_value = AsyncMock(data=[active_lock])

    result = await lock_manager.get_active_locks([path1, path2])

    assert result == {path1}


async def test_get_active_locks_ignores_expired_locks(lock_manager, mock_entity_client):
    """Test get_active_locks ignores expired locks."""
    path1 = "cache/hf/repo/main/file1.txt"

    # Create expired lock (older than TTL)
    expired_lock = FileLock(
        name=_path_to_lock_name(path1),
        workspace="test-workspace",
        path=path1,
        acquired_at=datetime.now(UTC) - timedelta(minutes=10),  # Older than 5min TTL
    )
    mock_entity_client.list.return_value = AsyncMock(data=[expired_lock])

    result = await lock_manager.get_active_locks([path1])

    assert result == set()  # Expired lock should not be returned


async def test_get_active_locks_mixed_active_and_expired(lock_manager, mock_entity_client):
    """Test get_active_locks correctly filters mixed active and expired locks."""
    path1 = "cache/hf/repo/main/file1.txt"
    path2 = "cache/hf/repo/main/file2.txt"

    # Active lock for path1
    active_lock = FileLock(
        name=_path_to_lock_name(path1),
        workspace="test-workspace",
        path=path1,
        acquired_at=datetime.now(UTC),
    )

    # Expired lock for path2
    expired_lock = FileLock(
        name=_path_to_lock_name(path2),
        workspace="test-workspace",
        path=path2,
        acquired_at=datetime.now(UTC) - timedelta(minutes=10),
    )

    mock_entity_client.list.return_value = AsyncMock(data=[active_lock, expired_lock])

    result = await lock_manager.get_active_locks([path1, path2])

    assert result == {path1}  # Only active lock returned
