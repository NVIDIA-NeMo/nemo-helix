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
import tomllib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast

import anyio
from anyio.lowlevel import RunVar
from anyio.to_thread import run_sync
from nemo_evaluator.api.task_definitions.harbor import validate_archive_path

CHUNK_BYTES = 1024 * 1024
MAX_ENTRIES = 100_000
MAX_FILE_BYTES = 4 * 1024**3
MAX_PAYLOAD_BYTES = 16 * 1024**3
MAX_STREAM_BYTES = MAX_PAYLOAD_BYTES + 256 * 1024**2
MAX_ARCHIVE_BYTES = MAX_STREAM_BYTES + 64 * 1024**2
MAX_CONFIG_BYTES = 1024**2
MAX_IGNORE_BYTES = 256 * 1024
MAX_INSTRUCTION_BYTES = 1024**2
MAX_METADATA_BYTES = 16 * 1024**2
MAX_STEPS = 128
MAX_EXTENSION_BYTES = 1024**2
MAX_EXTENSION_CHAIN = 8


@dataclass(frozen=True)
class NativeTask:
    task_id: str
    instruction: str | None
    config: dict[str, Any]


def _read_metadata(path: Path, limit: int) -> bytes:
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"Metadata must be a regular file: {path.name}")
    with path.open("rb") as stream:
        value = stream.read(limit + 1)
    if len(value) > limit:
        raise ValueError(f"Metadata exceeds byte limit: {path.name}")
    return value


def validate_native_task_inputs(root: Path) -> NativeTask:
    """Guard host paths and whole-file reads before calling native Harbor APIs."""
    config_bytes = _read_metadata(root / "task.toml", MAX_CONFIG_BYTES)
    config = tomllib.loads(config_bytes.decode("utf-8"))
    steps = config.get("steps", [])
    if not isinstance(steps, list) or len(steps) > MAX_STEPS:
        raise ValueError("Invalid or excessive Harbor steps")
    names: set[str] = set()
    instructions = [root / "instruction.md"] if (root / "instruction.md").exists() else []
    for step in steps:
        name = step.get("name") if isinstance(step, dict) else None
        if not isinstance(name, str) or "/" in name:
            raise ValueError("Step name must be a single directory component")
        validate_archive_path(name)
        if name.casefold() in names:
            raise ValueError("Duplicate step name")
        names.add(name.casefold())
        directory = root / "steps" / name
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError("Missing packaged step directory")
        if not directory.resolve().is_relative_to(root.resolve()):
            raise ValueError("Step escapes task root")
        instructions.append(directory / "instruction.md")
    total = len(config_bytes)
    ignore = root / ".gitignore"
    if ignore.exists():
        total += len(_read_metadata(ignore, MAX_IGNORE_BYTES))
    instruction = None
    for path in instructions:
        value = _read_metadata(path, MAX_INSTRUCTION_BYTES)
        total += len(value)
        if total > MAX_METADATA_BYTES:
            raise ValueError("Task metadata exceeds aggregate byte limit")
        if path == root / "instruction.md":
            instruction = value.decode("utf-8")
    if total > MAX_METADATA_BYTES:
        raise ValueError("Task metadata exceeds aggregate byte limit")
    from harbor.models.task.task import Task

    try:
        task = Task(root)
        if not (root / "environment").is_dir() or not Task.is_valid_dir(root):
            raise ValueError("Invalid native Harbor task")
    except OSError as exc:
        raise ValueError("Missing or unreadable native Harbor artifact") from exc
    return NativeTask(task.name, instruction, config)


def remove_owned_tree(root: Path) -> None:
    """Remove only invocation-owned staging, including restrictive directory modes."""
    if not root.exists():
        return
    root.chmod(stat.S_IMODE(root.stat().st_mode) | 0o700)
    for directory, _, _ in os.walk(root, topdown=True, followlinks=False):
        os.chmod(directory, stat.S_IMODE(os.stat(directory).st_mode) | 0o700)
        for child in Path(directory).iterdir():
            if not child.is_symlink() and child.is_dir():
                child.chmod(stat.S_IMODE(child.stat().st_mode) | 0o700)
    shutil.rmtree(root)


@contextmanager
def private_directory(parent: Path | None = None) -> Iterator[Path]:
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="harbor-", dir=parent))
    try:
        yield root
    finally:
        remove_owned_tree(root)


async def run_validation[T](function: Callable[..., T], *args: Any) -> T:
    """Bound blocking validation and drain it before cancellation removes its inputs."""
    work = asyncio.create_task(run_sync(function, *args, abandon_on_cancel=False, limiter=_validation_limiter()))
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


def _validation_limiter() -> anyio.CapacityLimiter:
    # AnyIO's run-local variable keeps limiters on their owning event loop.
    try:
        return _limiter.get()
    except LookupError:
        limiter = anyio.CapacityLimiter(2)
        _limiter.set(limiter)
        return limiter


_limiter = RunVar[anyio.CapacityLimiter]("harbor_validation_limiter")


def _raise_walk_error(error: OSError) -> None:
    raise error


def _inventory(root: Path) -> list[tuple[Path, os.stat_result]]:
    validate_archive_path(root.name)
    if root.name.casefold() == "task_template" or not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError("Task root must be a real, non-reserved directory")
    entries = [(root, root.lstat())]
    seen = {root.name.casefold()}
    total = 0
    # Walk without following links; reject each link before descending/reading.
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=_raise_walk_error):
        for name in sorted(dirs + files):
            path = Path(directory) / name
            relative = path.relative_to(root.parent).as_posix()
            validate_archive_path(relative)
            if relative.casefold() in seen:
                raise ValueError("Duplicate case-folded archive path")
            seen.add(relative.casefold())
            info = path.lstat()
            if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                raise ValueError(f"Unsupported task entry: {relative}")
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
                if info.st_size > MAX_FILE_BYTES or total > MAX_PAYLOAD_BYTES:
                    raise ValueError("Task payload exceeds limits")
            entries.append((path, info))
            if len(entries) > MAX_ENTRIES:
                raise ValueError("Too many task entries")
    return sorted(entries, key=lambda pair: pair[0].relative_to(root.parent).as_posix())


def capture_task(root: Path, parent: Path) -> Path:
    entries = _inventory(root)
    target = parent / root.name
    for path, info in entries:
        destination = target / path.relative_to(root)
        if stat.S_ISDIR(info.st_mode):
            destination.mkdir(mode=0o700)
        else:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, "rb") as source, destination.open("xb") as output:
                before = os.fstat(source.fileno())
                if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != (info.st_dev, info.st_ino):
                    raise ValueError("Task changed during capture")
                size = 0
                while chunk := source.read(CHUNK_BYTES):
                    size += len(chunk)
                    if size > info.st_size:
                        raise ValueError("Task grew during capture")
                    output.write(chunk)
                after = os.fstat(source.fileno())
                if size != info.st_size or (before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ):
                    raise ValueError("Task changed during capture")
            destination.chmod(stat.S_IMODE(info.st_mode))
    for path, info in reversed(entries):
        if stat.S_ISDIR(info.st_mode):
            (target / path.relative_to(root)).chmod(stat.S_IMODE(info.st_mode))
    validate_native_task_inputs(target)
    return target


def pack_task(root: Path, output: Path) -> str:
    entries = _inventory(root)
    validate_native_task_inputs(root)
    with output.open("xb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            for path, info in entries:
                header = tarfile.TarInfo(path.relative_to(root.parent).as_posix())
                header.mode = stat.S_IMODE(info.st_mode)
                if stat.S_ISDIR(info.st_mode):
                    header.type = tarfile.DIRTYPE
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
        if size < 0:
            raise ValueError("Unbounded archive read")
        value = self.stream.read(min(size, MAX_STREAM_BYTES - self.size + 1))
        self.size += len(value)
        if self.size > MAX_STREAM_BYTES:
            raise ValueError("Expanded archive exceeds limit")
        return value


class _SafeTarInfo(tarfile.TarInfo):
    # PAX processing invokes these before yielding a member to extract_task.
    # Reject here so sparse maps cannot allocate memory before our metadata checks.
    def _proc_gnusparse_00(self, next, raw_headers):
        raise ValueError("Unsupported sparse tar metadata")

    def _proc_gnusparse_01(self, next, pax_headers):
        raise ValueError("Unsupported sparse tar metadata")

    def _proc_gnusparse_10(self, next, pax_headers, archive):
        raise ValueError("Unsupported sparse tar metadata")

    def _proc_member(self, archive):
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
        if self.type not in {tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE}:
            raise ValueError("Unsupported tar entry type")
        return super()._proc_member(archive)  # ty: ignore[unresolved-attribute] -- stdlib private pre-allocation hook


def extract_task(archive_path: Path, parent: Path) -> tuple[Path, NativeTask]:
    """Extract an already checksum-verified archive into a fresh private directory."""
    seen: set[str] = set()
    directories: dict[str, int] = {}
    total = 0
    root: Path | None = None
    with archive_path.open("rb") as raw, gzip.GzipFile(fileobj=raw, mode="rb") as zipped:
        bounded = _BoundedReader(cast(BinaryIO, zipped))
        with tarfile.open(fileobj=cast(BinaryIO, bounded), mode="r|", tarinfo=_SafeTarInfo) as archive:
            for entry in archive:
                name = entry.name.rstrip("/") if entry.isdir() else entry.name
                validate_archive_path(name)
                if name.casefold() in seen or len(seen) >= MAX_ENTRIES:
                    raise ValueError("Duplicate or excessive archive entries")
                if entry.pax_headers.keys() - {"path"} or entry.sparse is not None:
                    raise ValueError("Unsupported tar metadata")
                seen.add(name.casefold())
                parts = name.split("/")
                if root is None:
                    if len(parts) != 1 or not entry.isdir() or name.casefold() == "task_template":
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
                else:
                    total += entry.size
                    if entry.size < 0 or entry.size > MAX_FILE_BYTES or total > MAX_PAYLOAD_BYTES:
                        raise ValueError("Archive payload exceeds limits")
                    limit = MAX_FILE_BYTES
                    if len(parts) == 2 and parts[1] == "task.toml":
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
    for name, mode in reversed(list(directories.items())):
        (parent / name).chmod(mode)
        if stat.S_IMODE((parent / name).stat().st_mode) != mode:
            raise ValueError("Unable to restore archive directory permissions")
    return root, validate_native_task_inputs(root)
