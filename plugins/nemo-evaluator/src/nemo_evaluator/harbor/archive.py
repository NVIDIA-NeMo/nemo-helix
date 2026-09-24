# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded self-contained task capture, deterministic packaging and verified extraction."""

import asyncio
import gzip
import hashlib
import os
import shutil
import stat
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO, cast

import anyio
from anyio.lowlevel import RunVar
from anyio.to_thread import run_sync
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_archive import (
    CHUNK_BYTES,
    MAX_CONFIG_BYTES,
    MAX_ENTRIES,
    MAX_FILE_BYTES,
    MAX_IGNORE_BYTES,
    MAX_INSTRUCTION_BYTES,
    MAX_PAYLOAD_BYTES,
    TASK_CONFIG_FILENAME,
    TASK_TEMPLATE_DIRNAME,
    NativeTask,
    check_internal_symlink,
    inventory_task_entries,
    validate_archive_path,
    validate_native_task_inputs,
)

MAX_STREAM_BYTES = MAX_PAYLOAD_BYTES + 256 * 1024**2
MAX_ARCHIVE_BYTES = MAX_STREAM_BYTES + 64 * 1024**2
MAX_EXTENSION_BYTES = 1024**2
MAX_EXTENSION_CHAIN = 8
# Each operation can expand up to MAX_PAYLOAD_BYTES while doing blocking filesystem and native
# Harbor work. Two permits allow limited overlap across API requests while bounding an event loop
# to two simultaneous archive workloads.
ARCHIVE_OPERATION_CONCURRENCY = 2


async def run_blocking_archive_operation[T](function: Callable[..., T], *args: Any) -> T:
    """Run a blocking archive operation in a concurrency-limited worker thread.

    Algorithm:
        - Start the operation under the event loop's shared archive-work limiter.
        - Shield the worker so caller cancellation cannot stop filesystem work midway.
        - If cancelled, drain the worker before propagating cancellation so cleanup is safe.

    Args:
        function: Blocking callable to run outside the event loop.
        *args: Positional arguments passed to ``function``.

    Returns:
        The callable's result.
    """
    work = asyncio.create_task(run_sync(function, *args, abandon_on_cancel=False, limiter=_archive_operation_limiter()))
    try:
        return await asyncio.shield(work)
    except asyncio.CancelledError:
        # asyncio task cancellation can bypass AnyIO cancellation scopes.
        while not work.done():
            try:
                await asyncio.shield(work)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not work.cancelled():
            work.exception()
        raise


def _archive_operation_limiter() -> anyio.CapacityLimiter:
    """Share the archive-operation concurrency limit within the current async event loop."""
    # AnyIO's run-local variable keeps limiters on their owning event loop.
    try:
        return _limiter.get()
    except LookupError:
        limiter = anyio.CapacityLimiter(ARCHIVE_OPERATION_CONCURRENCY)
        _limiter.set(limiter)
        return limiter


_limiter = RunVar[anyio.CapacityLimiter]("harbor_archive_operation_limiter")


def pack_task(root: Path, output: Path) -> str:
    """Create a deterministic ``tar.gz`` archive for one validated Harbor task.

    Algorithm:
        - Inventory and validate the task before opening the destination exclusively.
        - Write stable PAX headers, modes, ordering, and a zero gzip timestamp.
        - Hash the exact completed archive bytes.

    Args:
        root: Captured Harbor task directory to package.
        output: New archive path; an existing file is never overwritten.

    Returns:
        Lowercase SHA-256 digest of the final archive bytes.

    Raises:
        ValueError: The task is unsafe, invalid, or exceeds archive limits.
        FileExistsError: ``output`` already exists.
    """
    entries = inventory_task_entries(root)
    validate_native_task_inputs(root)
    with output.open("xb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            for path, info in entries:
                header = tarfile.TarInfo(path.relative_to(root.parent).as_posix())
                header.mode = stat.S_IMODE(info.st_mode)
                if stat.S_ISDIR(info.st_mode):
                    header.type = tarfile.DIRTYPE
                    archive.addfile(header)
                elif stat.S_ISLNK(info.st_mode):
                    header.type = tarfile.SYMTYPE
                    header.mode = 0o777
                    header.linkname = check_internal_symlink(path, root)
                    archive.addfile(header)
                else:
                    header.size = info.st_size
                    with path.open("rb") as source:
                        archive.addfile(header, source)
    with output.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class _BoundedReader:
    def __init__(self, stream: BinaryIO):
        self.stream = stream
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        """Bound each read and reject archives whose total decompressed data exceeds the stream limit."""
        if size < 0:
            raise ValueError("Unbounded archive read")
        value = self.stream.read(min(size, MAX_STREAM_BYTES - self.size + 1))
        self.size += len(value)
        if self.size > MAX_STREAM_BYTES:
            raise ValueError("Expanded archive exceeds limit")
        return value


class _SafeTarInfo(tarfile.TarInfo):
    """Reject GNU sparse maps before tarfile allocates them, blocking a memory-exhaustion attack."""

    def _proc_gnusparse_00(self, next, raw_headers):
        raise ValueError("Unsupported sparse tar metadata")

    def _proc_gnusparse_01(self, next, pax_headers):
        raise ValueError("Unsupported sparse tar metadata")

    def _proc_gnusparse_10(self, next, pax_headers, archive):
        raise ValueError("Unsupported sparse tar metadata")

    def _proc_member(self, archive):
        """Reject unsafe tar records and excessive metadata before the standard library processes them."""
        # Called before stdlib allocates PAX/long-name bodies or recurses to the next header.
        archive.physical_count = getattr(archive, "physical_count", 0) + 1
        if archive.physical_count > MAX_ENTRIES * 2:
            raise ValueError("Too many physical tar records")
        if self.type == tarfile.XHDTYPE:
            depth = getattr(archive, "extension_depth", 0) + 1
            if self.size > MAX_EXTENSION_BYTES or depth > MAX_EXTENSION_CHAIN:
                raise ValueError("Excessive tar metadata")
            archive.extension_depth = depth
            try:
                return super()._proc_member(archive)  # ty: ignore[unresolved-attribute] -- stdlib private pre-allocation hook
            finally:
                archive.extension_depth -= 1
        if self.type not in {tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE, tarfile.SYMTYPE}:
            raise ValueError("Unsupported tar entry type")
        return super()._proc_member(archive)  # ty: ignore[unresolved-attribute] -- stdlib private pre-allocation hook


def extract_task(archive_path: Path, parent: Path) -> tuple[Path, NativeTask]:
    """Safely extract and validate one checksum-verified Harbor archive.

    Algorithm:
        - Stream the gzip and tar layers while enforcing compressed metadata and payload limits.
        - Require one ordered root tree of regular files, directories, and symlinks.
        - Validate the gzip trailer, then require every symlink in the complete tree to stay inside the root.
        - Restore recorded modes and load native task metadata.

    Args:
        archive_path: Archive whose external digest has already been verified.
        parent: Empty private directory that will receive the task root.

    Returns:
        Extracted root path and its validated native task metadata.

    Raises:
        ValueError: The archive is unsafe, malformed, truncated, excessive, or not a valid task.
    """
    seen: set[str] = set()
    directories: dict[str, int] = {}
    total = 0
    links: list[Path] = []
    root: Path | None = None
    with archive_path.open("rb") as raw, gzip.GzipFile(fileobj=raw, mode="rb") as zipped:
        bounded = _BoundedReader(cast(BinaryIO, zipped))
        with tarfile.open(fileobj=cast(BinaryIO, bounded), mode="r|", tarinfo=_SafeTarInfo) as archive:
            for entry in archive:
                name = entry.name.rstrip("/") if entry.isdir() else entry.name
                validate_archive_path(name)
                if name.casefold() in seen or len(seen) >= MAX_ENTRIES:
                    raise ValueError("Duplicate or excessive archive entries")
                if entry.pax_headers.keys() - {"path", "linkpath"} or entry.sparse is not None:
                    raise ValueError("Unsupported tar metadata")
                seen.add(name.casefold())
                parts = name.split("/")
                if root is None:
                    if len(parts) != 1 or not entry.isdir() or name.casefold() == TASK_TEMPLATE_DIRNAME:
                        raise ValueError("Archive must start with one named root directory")
                    root = parent / name
                if parts[0] != root.name or any("/".join(parts[:i]) not in directories for i in range(1, len(parts))):
                    raise ValueError("Missing parent or multiple archive roots")
                path = parent / name
                if entry.mode & ~0o7777:
                    raise ValueError("Unsupported mode bits")
                if entry.isdir():
                    if entry.size:
                        raise ValueError("Directory has a payload")
                    path.mkdir(mode=0o700)
                    directories[name] = entry.mode
                elif entry.issym():
                    # Parents must be recorded real directories, so nothing is written through a
                    # link; targets are checked only after the whole tree exists.
                    if entry.size or not entry.linkname:
                        raise ValueError("Invalid symlink entry")
                    os.symlink(entry.linkname, path)
                    links.append(path)
                else:
                    total += entry.size
                    if entry.size < 0 or entry.size > MAX_FILE_BYTES or total > MAX_PAYLOAD_BYTES:
                        raise ValueError("Archive payload exceeds limits")
                    limit = MAX_FILE_BYTES
                    if len(parts) == 2 and parts[1] == TASK_CONFIG_FILENAME:
                        limit = MAX_CONFIG_BYTES
                    elif len(parts) == 2 and parts[1] == ".gitignore":
                        limit = MAX_IGNORE_BYTES
                    elif parts[-1] == "instruction.md":
                        limit = MAX_INSTRUCTION_BYTES
                    if entry.size > limit:
                        raise ValueError("Metadata exceeds byte limit")
                    source = archive.extractfile(entry)
                    if source is None:
                        raise ValueError("Missing tar file body")
                    with source, path.open("xb") as output:
                        shutil.copyfileobj(source, output, CHUNK_BYTES)
                    if path.stat().st_size != entry.size:
                        raise ValueError("Truncated tar file")
                    path.chmod(entry.mode)
                    if stat.S_IMODE(path.stat().st_mode) != entry.mode:
                        raise ValueError("Unable to restore archive file permissions")
            # Drain gzip to validate CRC/trailer and expanded-byte limits.
            while chunk := archive.fileobj.read(CHUNK_BYTES):
                if any(chunk):
                    raise ValueError("Unexpected trailing archive content")
    if root is None:
        raise ValueError("Empty archive")
    # Check links on the complete tree while its directories are still owner-traversable.
    for link in links:
        check_internal_symlink(link, root)
    for name, mode in reversed(list(directories.items())):
        (parent / name).chmod(mode)
        if stat.S_IMODE((parent / name).stat().st_mode) != mode:
            raise ValueError("Unable to restore archive directory permissions")
    return root, validate_native_task_inputs(root)
