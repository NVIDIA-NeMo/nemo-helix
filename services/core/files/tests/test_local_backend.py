# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for LocalStorageImpl backend."""

from pathlib import Path

import anyio
import pytest
from nmp.core.files.app.backends.base import ByteRange
from nmp.core.files.app.backends.local import LocalStorageConfig, LocalStorageImpl
from nmp.core.files.exceptions import InvalidPathError, NotFoundError


@pytest.fixture
def storage_path(tmp_path: Path) -> Path:
    """Create a temporary storage directory."""
    path = tmp_path / "storage"
    path.mkdir()
    return path


@pytest.fixture
def storage_impl(storage_path: Path) -> LocalStorageImpl:
    """Create a LocalStorageImpl instance with temp storage."""
    config = LocalStorageConfig(path=str(storage_path))
    return LocalStorageImpl(config)


async def test_upload_cleanup_on_cancellation(storage_impl: LocalStorageImpl, storage_path: Path):
    """Test that partial files are cleaned up when upload is cancelled."""
    test_file_path = storage_path / "test.txt"

    # Create an async generator that yields chunks but gets cancelled mid-way
    async def chunked_data_with_cancellation():
        """Yield chunks and then simulate cancellation."""
        yield b"chunk1"
        yield b"chunk2"
        # Simulate cancellation after a few chunks
        raise anyio.get_cancelled_exc_class()()

    # Attempt upload that will be cancelled
    with pytest.raises(anyio.get_cancelled_exc_class()):
        await storage_impl.upload("test.txt", chunked_data_with_cancellation())

    # Verify the partial file was cleaned up
    assert not test_file_path.exists(), "Partial file should have been cleaned up after cancellation"


async def test_upload_success_keeps_file(storage_impl: LocalStorageImpl, storage_path: Path):
    """Test that successful uploads keep the file."""
    test_file_path = storage_path / "test.txt"
    test_content = b"chunk1chunk2chunk3"

    async def chunked_data():
        """Yield all chunks successfully."""
        yield b"chunk1"
        yield b"chunk2"
        yield b"chunk3"

    result = await storage_impl.upload("test.txt", chunked_data())
    assert result.path == "test.txt"

    assert test_file_path.exists(), "File should exist after successful upload"
    with open(test_file_path, "rb") as f:
        assert f.read() == test_content

    assert (storage_path / "test.txt").stat().st_size == result.size


async def test_upload_cleanup_on_exception(storage_impl: LocalStorageImpl, storage_path: Path):
    """Test that partial files are cleaned up when upload fails with an exception."""
    test_file_path = storage_path / "test.txt"

    async def chunked_data_with_error():
        """Yield chunks and then raise an error."""
        yield b"chunk1"
        yield b"chunk2"
        raise RuntimeError("Simulated upload error")

    with pytest.raises(RuntimeError, match="Simulated upload error"):
        await storage_impl.upload("test.txt", chunked_data_with_error())

    assert not test_file_path.exists(), "Partial file should have been cleaned up after error"


async def test_upload_creates_parent_directories(storage_impl: LocalStorageImpl, storage_path: Path):
    """Test that upload creates parent directories if they don't exist."""
    nested_path = "dir1/dir2/test.txt"
    test_file_path = storage_path / nested_path

    async def chunked_data():
        yield b"content"

    result = await storage_impl.upload(nested_path, chunked_data())
    assert result.path == nested_path
    assert test_file_path.exists()
    assert test_file_path.parent.parent.name == "dir1"
    assert test_file_path.parent.name == "dir2"


async def test_get_file_returns_file_info(storage_impl: LocalStorageImpl, storage_path: Path):
    """Test get_file returns FileInfo with path and size."""
    content = b"Hello, World!"
    test_file = storage_path / "test.txt"
    test_file.write_bytes(content)

    file_info = await storage_impl.get_file("test.txt")

    assert file_info.path == "test.txt"
    assert file_info.size == len(content)


async def test_get_file_not_found_raises_error(storage_impl: LocalStorageImpl):
    """Test get_file raises NotFoundError when file doesn't exist."""
    with pytest.raises(NotFoundError, match="File not found"):
        await storage_impl.get_file("nonexistent.txt")


async def test_list_files_with_path(storage_impl: LocalStorageImpl, storage_path: Path):
    """Test list_files with specific path filters results."""
    # Create files in different directories
    (storage_path / "root.txt").write_bytes(b"root")
    (storage_path / "dir1").mkdir()
    (storage_path / "dir1" / "file1.txt").write_bytes(b"file1")
    (storage_path / "dir2").mkdir()
    (storage_path / "dir2" / "file2.txt").write_bytes(b"file2")

    # List files in dir1 only
    files = await storage_impl.list_files("dir1")

    assert len(files) == 1
    assert files[0].path == "dir1/file1.txt"
    assert files[0].size == 5


async def test_list_files_includes_size(storage_impl: LocalStorageImpl, storage_path: Path):
    """Test list_files returns FileInfo with size information."""
    content1 = b"Hello"
    content2 = b"World!"
    (storage_path / "file1.txt").write_bytes(content1)
    (storage_path / "file2.txt").write_bytes(content2)

    files = await storage_impl.list_files()

    assert len(files) == 2
    sizes = {f.path: f.size for f in files}
    assert sizes["file1.txt"] == len(content1)
    assert sizes["file2.txt"] == len(content2)


async def test_download_with_byte_range(storage_impl: LocalStorageImpl, storage_path: Path):
    """Test download with ByteRange returns only requested bytes."""
    content = b"0123456789ABCDEFGHIJ"
    (storage_path / "test.txt").write_bytes(content)

    # Download bytes 5-14 (10 bytes)
    byte_range = ByteRange(start=5, end=14)
    chunks = []
    async for chunk in await storage_impl.download("test.txt", byte_range):
        chunks.append(chunk)

    result = b"".join(chunks)
    assert result == content[5:15]  # end is inclusive, so 5-14 means indices 5-14


async def test_download_with_suffix_byte_range(storage_impl: LocalStorageImpl, storage_path: Path):
    """Test download with suffix range (last N bytes)."""
    content = b"0123456789ABCDEFGHIJ"
    (storage_path / "test.txt").write_bytes(content)

    # Download last 5 bytes
    byte_range = ByteRange(start=len(content) - 5, end=len(content) - 1)
    chunks = []
    async for chunk in await storage_impl.download("test.txt", byte_range):
        chunks.append(chunk)

    result = b"".join(chunks)
    assert result == content[-5:]


async def test_download_without_byte_range(storage_impl: LocalStorageImpl, storage_path: Path):
    """Test download without byte range returns entire file."""
    content = b"Full file content here"
    (storage_path / "test.txt").write_bytes(content)

    chunks = []
    async for chunk in await storage_impl.download("test.txt", None):
        chunks.append(chunk)

    result = b"".join(chunks)
    assert result == content


async def test_download_nonexistent_file_raises_error(storage_impl: LocalStorageImpl):
    """Test download raises NotFoundError when file doesn't exist."""
    with pytest.raises(NotFoundError, match="does not exist"):
        async for _ in await storage_impl.download("nonexistent.txt", None):
            pass


async def test_delete_file_success(storage_impl: LocalStorageImpl, storage_path: Path):
    """Test successful file deletion."""
    test_file = storage_path / "to_delete.txt"
    test_file.write_bytes(b"delete me")
    assert test_file.exists()

    # Delete the file
    await storage_impl.delete("to_delete.txt")

    # Verify file is gone
    assert not test_file.exists()


async def test_delete_file_not_found(storage_impl: LocalStorageImpl):
    """Test deleting non-existent file raises NotFoundError."""
    with pytest.raises(NotFoundError, match="does not exist"):
        await storage_impl.delete("nonexistent.txt")


async def test_delete_directory_raises_error(storage_impl: LocalStorageImpl, storage_path: Path):
    """Test that attempting to delete a directory raises RuntimeError."""
    # Create a directory
    test_dir = storage_path / "directory"
    test_dir.mkdir()

    # Attempting to delete a directory should raise RuntimeError
    with pytest.raises(RuntimeError, match="is not a file"):
        await storage_impl.delete("directory")


async def test_validate_storage_creates_directory_if_missing(tmp_path: Path):
    """Test validate_storage creates directory if it doesn't exist."""
    new_storage_path = tmp_path / "new_storage"
    assert not new_storage_path.exists()

    config = LocalStorageConfig(path=str(new_storage_path))
    storage = LocalStorageImpl(config)

    await storage.validate_storage()

    assert new_storage_path.exists()
    assert new_storage_path.is_dir()


def test_local_config_owns_storage_data(storage_path: Path):
    """Local is platform-owned source storage."""
    config = LocalStorageConfig(path=str(storage_path))
    assert config.owns_storage_data is True


def test_local_tracks_no_revision(storage_impl: LocalStorageImpl):
    """Uploaded files have no upstream ref, so there is nothing to refresh them against."""
    assert storage_impl.tracked_revision is None

    with pytest.raises(NotImplementedError):
        storage_impl.config_at_tracked_revision()


async def test_delete_all_removes_directory_and_contents(storage_impl: LocalStorageImpl, storage_path: Path):
    """Test delete_all removes the entire storage directory and all contents."""
    # Create some files and nested directories
    (storage_path / "file1.txt").write_bytes(b"content1")
    (storage_path / "subdir").mkdir()
    (storage_path / "subdir" / "file2.txt").write_bytes(b"content2")
    (storage_path / "subdir" / "nested").mkdir()
    (storage_path / "subdir" / "nested" / "file3.txt").write_bytes(b"content3")

    assert storage_path.exists()

    await storage_impl.delete_all()

    assert not storage_path.exists()


async def test_delete_all_on_nonexistent_directory_is_noop(tmp_path: Path):
    """Test delete_all does not raise error when directory doesn't exist."""
    nonexistent_path = tmp_path / "nonexistent"
    config = LocalStorageConfig(path=str(nonexistent_path))
    storage = LocalStorageImpl(config)

    assert not nonexistent_path.exists()

    # Should not raise any error
    await storage.delete_all()

    assert not nonexistent_path.exists()


async def test_get_cache_path_key_returns_none(storage_impl: LocalStorageImpl):
    """Test get_cache_path_key returns None for local storage (not cacheable)."""
    # With path
    assert await storage_impl.get_cache_path_key("file.txt") is None

    # Without path
    assert await storage_impl.get_cache_path_key() is None


async def test_upload_rejects_parent_traversal_path(storage_impl: LocalStorageImpl):
    """Upload rejects traversal paths that escape storage root."""

    async def chunked_data():
        yield b"blocked"

    with pytest.raises(InvalidPathError, match="path traversal attack"):
        await storage_impl.upload("../outside.txt", chunked_data())


async def test_download_rejects_absolute_path(storage_impl: LocalStorageImpl):
    """Download rejects absolute paths outside storage root."""
    with pytest.raises(InvalidPathError, match="path traversal attack"):
        async for _ in await storage_impl.download("/tmp/outside.txt", None):
            pass


async def test_delete_rejects_parent_traversal_path(storage_impl: LocalStorageImpl):
    """Delete rejects traversal paths that escape storage root."""
    with pytest.raises(InvalidPathError, match="path traversal attack"):
        await storage_impl.delete("../../outside.txt")


async def test_upload_rejects_symlink_escape(tmp_path: Path, storage_path: Path):
    """Upload rejects paths that escape through a symlink."""
    outside_dir = tmp_path / "outside_upload"
    outside_dir.mkdir()
    (storage_path / "escape").symlink_to(outside_dir, target_is_directory=True)

    storage = LocalStorageImpl(LocalStorageConfig(path=str(storage_path)))

    async def chunked_data():
        yield b"blocked"

    with pytest.raises(InvalidPathError, match="path traversal attack"):
        await storage.upload("escape/new.txt", chunked_data())


async def test_download_rejects_symlink_escape(tmp_path: Path, storage_path: Path):
    """Download rejects paths that escape through a symlink."""
    outside_dir = tmp_path / "outside_download"
    outside_dir.mkdir()
    (outside_dir / "secret.txt").write_bytes(b"secret")
    (storage_path / "escape").symlink_to(outside_dir, target_is_directory=True)

    storage = LocalStorageImpl(LocalStorageConfig(path=str(storage_path)))

    with pytest.raises(InvalidPathError, match="path traversal attack"):
        async for _ in await storage.download("escape/secret.txt", None):
            pass


async def test_delete_rejects_symlink_escape(tmp_path: Path, storage_path: Path):
    """Delete rejects paths that escape through a symlink."""
    outside_dir = tmp_path / "outside_delete"
    outside_dir.mkdir()
    (outside_dir / "victim.txt").write_bytes(b"victim")
    (storage_path / "escape").symlink_to(outside_dir, target_is_directory=True)

    storage = LocalStorageImpl(LocalStorageConfig(path=str(storage_path)))

    with pytest.raises(InvalidPathError, match="path traversal attack"):
        await storage.delete("escape/victim.txt")
