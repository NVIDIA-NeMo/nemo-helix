# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local filesystem storage backend."""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field
from stat import S_ISREG
from typing import (
    AsyncIterator,
)

import anyio
from anyio import to_thread
from nmp.common.files.storage_config import LocalStorageConfig as LocalStorageConfig
from nmp.core.files.app.backends.base import (
    ByteRange,
    FileInfo,
    StorageImpl,
)
from nmp.core.files.app.utils import warn_if_slow
from nmp.core.files.exceptions import InvalidPathError, NotFoundError

logger = logging.getLogger(__name__)


@dataclass
class LocalStorageImpl(StorageImpl):
    config: LocalStorageConfig

    _path: anyio.Path = field(init=False)

    def __post_init__(self):
        self._path = anyio.Path(self.config.path)

    def _is_file(self, stat_info: os.stat_result) -> bool:
        """Simple helper that avoids another sys call if we already have stat info."""
        return S_ISREG(stat_info.st_mode)

    async def _validate_local_path(self, path: str) -> anyio.Path:
        # Resolve symlinks for consistent path comparison (e.g., /var -> /private/var on macOS)
        root = await self._path.resolve()
        resolved_path = await (root / path).resolve()

        if not resolved_path.is_relative_to(root):
            raise InvalidPathError(
                f"Path '{path}' resolves outside of the storage root. "
                "This may indicate a path traversal attack. "
                "Ensure that paths such as ../.. are not used in the path.",
            )

        return resolved_path

    async def _validate_file_exists(self, path: str, file_path: anyio.Path) -> None:
        """Validate that a file exists and is a regular file.

        Uses open() instead of stat() to avoid NFS negative dentry cache.
        NFS caches "file not found" responses, but open() triggers close-to-open
        consistency which forces revalidation with the server.

        Args:
            path: The logical path (for error messages)
            file_path: The resolved filesystem path to check

        Raises:
            NotFoundError: If the file does not exist
            RuntimeError: If the path is a directory, not a file
        """
        try:
            async with await anyio.open_file(file_path, mode="rb"):
                pass
        except FileNotFoundError as exc:
            raise NotFoundError(f"File {path} does not exist in local storage at {file_path}") from exc
        except IsADirectoryError:
            raise RuntimeError(f"Path {path} is not a file in local storage")

    async def list_files(self, path: str | None = None) -> list[FileInfo]:
        # Resolve symlinks for consistent path comparison (e.g., /var -> /private/var on macOS)
        resolved_path = await self._path.resolve()
        full_path = resolved_path
        if path is not None:
            full_path = await self._validate_local_path(path)

        async def _paths_to_traverse():
            yield full_path
            async for path_object in full_path.rglob("*"):
                yield path_object

        files = []
        async for path_object in _paths_to_traverse():
            try:
                stat = await path_object.stat()
            except FileNotFoundError:
                continue
            if self._is_file(stat):
                relative_path = str(path_object.relative_to(resolved_path))
                files.append(
                    FileInfo(
                        path=relative_path,
                        size=stat.st_size,
                    )
                )
        return files

    async def download(self, path: str, byte_range: ByteRange | None) -> AsyncIterator[bytes]:
        file_path = await self._validate_local_path(path)
        await self._validate_file_exists(path, file_path)

        if byte_range is not None:

            async def _gen() -> AsyncIterator[bytes]:
                async with await anyio.open_file(file_path, mode="rb") as file:
                    start = byte_range.start
                    await file.seek(byte_range.start)
                    while start <= byte_range.end:
                        chunk = await file.read(min(self.config.read_chunk_size, byte_range.end - start + 1))
                        if not chunk:
                            break
                        start += len(chunk)
                        yield chunk
        else:

            async def _gen() -> AsyncIterator[bytes]:
                async with await anyio.open_file(file_path, mode="rb") as file:
                    while chunk := await file.read(self.config.read_chunk_size):
                        yield chunk

        return _gen()

    async def upload(
        self,
        path: str,
        fstream: AsyncIterator[bytes],
        content_length: int | None = None,
        *,
        if_absent: bool = False,
    ) -> FileInfo:
        """Upload file to local storage path.

        Writes to a temporary file first, then atomically moves to final destination.
        This prevents partial/corrupt files if process crashes during write.

        Args:
            path: Path within storage to write file
            fstream: Async iterator of bytes to write
        """
        file_path = await self._validate_local_path(path)

        # Create parent directories if they don't exist
        await file_path.parent.mkdir(parents=True, exist_ok=True)

        # Use temp file in same directory to ensure atomic rename on same filesystem
        # UUID prevents collisions from concurrent writes
        temp_suffix = f".tmp.{uuid.uuid4().hex[:8]}"
        temp_path = file_path.with_suffix(file_path.suffix + temp_suffix)

        success = False
        try:
            # Pre-allocate buffer to avoid repeated reallocations during extend().
            # We use slice assignment to copy chunks in, which is O(chunk_size)
            # rather than O(buffer_size) like extend() can be when reallocating.
            buffer_size = self.config.write_buffer_size
            write_buffer = bytearray(buffer_size)
            buffer_pos = 0
            total_bytes = 0

            async with await anyio.open_file(temp_path, "wb") as dest_file:
                async for chunk in fstream:
                    chunk_len = len(chunk)
                    total_bytes += chunk_len

                    # If chunk would overflow buffer, flush first
                    if buffer_pos + chunk_len > buffer_size:
                        if buffer_pos > 0:
                            async with warn_if_slow("write", path=path, bytes=buffer_pos):
                                await dest_file.write(memoryview(write_buffer)[:buffer_pos])
                            buffer_pos = 0

                        # If chunk is larger than buffer, write it directly
                        if chunk_len > buffer_size:
                            async with warn_if_slow("write", path=path, bytes=chunk_len):
                                await dest_file.write(chunk)
                            continue

                    # Copy chunk into buffer using slice assignment (no reallocation)
                    write_buffer[buffer_pos : buffer_pos + chunk_len] = chunk
                    buffer_pos += chunk_len

                # Write any remaining buffered data
                if buffer_pos > 0:
                    async with warn_if_slow("write", path=path, bytes=buffer_pos):
                        await dest_file.write(memoryview(write_buffer)[:buffer_pos])

                # Flush and fsync to ensure data is on the NFS server before rename.
                # This helps with NFS consistency - other clients are more likely to
                # see the file after rename if the data has been flushed to server.
                await dest_file.flush()
                await to_thread.run_sync(os.fsync, dest_file.wrapped.fileno())

            # Atomic rename to final destination
            # This works on POSIX systems including NFS (rename is atomic)
            if content_length is not None and total_bytes != content_length:
                raise ValueError("Upload size does not match Content-Length")
            if if_absent:
                # Hard-link promotion is atomic and fails if the destination exists.
                await to_thread.run_sync(os.link, str(temp_path), str(file_path))
                await temp_path.unlink()
            else:
                await temp_path.rename(file_path)

            # Fsync the parent directory to ensure the rename is durable and visible.
            # This is especially important for NFS where directory entries may be cached.
            def _fsync_dir() -> None:
                dir_fd = os.open(str(file_path.parent), os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)

            await to_thread.run_sync(_fsync_dir)

            success = True
            return FileInfo(path=path, size=total_bytes)
        except Exception:
            logger.exception(f"Failed to upload file {path} to local storage")
            raise
        finally:
            # If we didn't use this separate CancelScope, the `await` calls
            # within the cleanup would immediately fail due to anyio's level-cancellation.
            with anyio.CancelScope(shield=True):
                try:
                    # Clean up temp file if it still exists (upload failed before rename)
                    if not success and await temp_path.exists():
                        await temp_path.unlink()
                        logger.warning(f"Cleaned up temporary file for {path}")
                except Exception:
                    logger.exception(f"Failed to clean up temporary file for {path}")

    async def upload_if_absent(
        self, path: str, fstream: AsyncIterator[bytes], content_length: int | None = None
    ) -> FileInfo:
        return await self.upload(path, fstream, content_length, if_absent=True)

    async def validate_storage(self):
        """Validate that the local storage path exists and is accessible. Creates the directory if it doesn't exist."""
        storage_path = anyio.Path(self.config.path)

        if not await storage_path.exists():
            logger.info(f"Local storage path {storage_path} does not exist. Creating directory.")
            await storage_path.mkdir(parents=True, exist_ok=True)

        if not await storage_path.is_dir():
            raise RuntimeError(f"Local storage path {storage_path} is not a directory.")

        # For write access check, we still need to use os.access since anyio.Path doesn't have this
        # But we can convert to str for the check
        if not os.access(str(storage_path), os.W_OK):
            raise RuntimeError(f"Local storage path {storage_path} is not writable.")

    async def create_storage(self, exist_ok: bool = False):
        await self._path.mkdir(parents=True, exist_ok=exist_ok)

    async def delete(self, path: str) -> FileInfo:
        """Delete a file from local storage."""
        file_path = await self._validate_local_path(path)
        await self._validate_file_exists(path, file_path)

        stat = await file_path.stat()

        # Capture file info before deletion
        file_info = FileInfo(path=path, size=stat.st_size)

        await file_path.unlink()
        logger.info(f"Deleted file {path} from local storage")

        return file_info

    async def delete_all(self) -> None:
        """Delete the entire storage directory and all its contents."""

        if await self._path.exists():
            await to_thread.run_sync(shutil.rmtree, str(self._path))
            logger.info(f"Deleted storage directory {self._path}")

    def get_duckdb_path(self, path: str) -> str:
        """Return filesystem path for DuckDB access."""
        return str(self._path / path)
