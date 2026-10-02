# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable agent source: one inspected selection policy and a verified archive descriptor."""

import fnmatch
import gzip
import hashlib
import logging
import os
import re
import stat
import tarfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Literal, cast
from uuid import uuid4

from filesets import parse_fileset_ref
from nemo_evaluator.api.task_definitions.harbor import ArchiveDigest, validate_archive_path
from nemo_evaluator.harbor.archive import CHUNK_BYTES, private_directory
from nemo_evaluator.harbor.archive_io import download_verified, download_verified_async
from nemo_evaluator.jobs.utils import run_with_isolated_async_client
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.client.errors import NemoTransportError
from nemo_platform_plugin.files.client import AsyncFilesClient, FilesClient
from nemo_platform_plugin.files.types import CreateFilesetRequest
from nemo_platform_plugin.refs import FILESET_REF_PATTERN
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)
MAX_ENTRIES = 10_000
MAX_FILE_BYTES = 256 * 1024**2
MAX_PAYLOAD_BYTES = 1024**3
MAX_STREAM_BYTES = 11 * 1024**3 // 10
MAX_EXTENSION_BYTES = 8192
_EXCLUDED_DIRS = {".git", ".venv", ".uv", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".cache"}
_SECRET_DIRS = {".aws", ".ssh"}
_SECRET_PATTERNS = (
    ".env",
    ".env.*",
    ".git-credentials",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa*",
    "id_ed25519*",
    "id_ecdsa*",
    "id_dsa*",
)


class HarborAgentSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["fileset-archive"] = "fileset-archive"
    fileset_ref: str = Field(pattern=FILESET_REF_PATTERN)
    sha256: ArchiveDigest
    archive_format: Literal["harbor-agent-tar-gzip-v1"] = "harbor-agent-tar-gzip-v1"

    @field_validator("fileset_ref")
    @classmethod
    def _reference(cls, value: str) -> str:
        workspace, name, path = parse_fileset_ref(value, workspace_fallback=None)
        if not workspace or not name or value != f"{workspace}/{name}#{path}":
            raise ValueError("Agent source must reference a canonical qualified Fileset object")
        for part in (workspace, name, path):
            validate_archive_path(part)
        return value


def validate_import_path(value: str) -> None:
    if re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", value, flags=re.ASCII) is None:
        raise ValueError("agent_import_path must be module.path:ClassName")


class AgentSourceOptions(BaseModel):
    """Exact file overrides may include filename-excluded assets, never links or tool caches."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    exclude_globs: tuple[str, ...] = ()
    include_files: tuple[str, ...] = ()

    @field_validator("include_files")
    @classmethod
    def _paths(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            validate_archive_path(value)
        return values


@dataclass(frozen=True)
class SourceEntry:
    path: str
    size: int
    mode: int
    directory: bool
    overridden: bool
    fingerprint: tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class AgentSourceInventory:
    root: Path
    entries: tuple[SourceEntry, ...]
    excluded: tuple[str, ...]

    @property
    def total_bytes(self) -> int:
        return sum(entry.size for entry in self.entries)


def _fingerprint(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_mode


def inspect_agent_source(
    root: Path, *, options: AgentSourceOptions | None = None, jobs_dir: Path | None = None
) -> AgentSourceInventory:
    """Read metadata only; never import source or apply implicit gitignore rules."""
    selection = options or AgentSourceOptions()
    root = root.expanduser()
    if not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError("Agent source root must be a directory, not a symlink")
    root = root.resolve()
    jobs = jobs_dir.expanduser().resolve() if jobs_dir is not None else None
    if jobs == root:
        raise ValueError("jobs_dir cannot be the agent source root")
    entries: list[SourceEntry] = []
    excluded: list[str] = []
    seen: set[str] = set()
    total = 0
    included: set[str] = set()

    def walk(directory: Path, inherited_exclusion: bool = False) -> None:
        nonlocal total
        for path in sorted(directory.iterdir()):
            relative = path.relative_to(root).as_posix()
            parts = relative.split("/")
            override = relative in selection.include_files
            hard_excluded = path == jobs or any(part.casefold() in _EXCLUDED_DIRS for part in parts)
            secret = (
                any(part.casefold() in _SECRET_DIRS for part in parts)
                or any(fnmatch.fnmatchcase(path.name.casefold(), pattern) for pattern in _SECRET_PATTERNS)
                or relative.casefold().endswith(".codex/auth.json")
            )
            selected_out = (
                inherited_exclusion
                or secret
                or any(
                    fnmatch.fnmatchcase("/".join(parts[:i]), pattern)
                    for i in range(1, len(parts) + 1)
                    for pattern in selection.exclude_globs
                )
            )
            # Walk excluded directories only when an exact requested file is below them.
            descendant_override = any(value.startswith(relative + "/") for value in selection.include_files)
            if hard_excluded or (selected_out and not override and not descendant_override):
                excluded.append(relative)
                continue
            validate_archive_path(relative)
            info = path.lstat()
            is_dir = stat.S_ISDIR(info.st_mode)
            if not is_dir and not stat.S_ISREG(info.st_mode):
                raise ValueError(f"Agent source contains a symlink or special file: {relative}")
            if relative.casefold() in seen:
                raise ValueError(f"Case-colliding agent source path: {relative}")
            seen.add(relative.casefold())
            size = 0 if is_dir else info.st_size
            total += size
            if size > MAX_FILE_BYTES or total > MAX_PAYLOAD_BYTES or len(entries) >= MAX_ENTRIES:
                raise ValueError("Agent source exceeds entry or payload limits")
            if override:
                if is_dir:
                    raise ValueError("include_files overrides must name regular files")
                included.add(relative)
            entries.append(
                SourceEntry(
                    relative,
                    size,
                    0o755 if is_dir or info.st_mode & stat.S_IXUSR else 0o644,
                    is_dir,
                    override,
                    _fingerprint(info),
                )
            )
            if is_dir:
                walk(path, inherited_exclusion=selected_out)

    walk(root)
    if set(selection.include_files) != included:
        raise ValueError("include_files contains missing files or excluded cache/output paths")
    return AgentSourceInventory(root, tuple(entries), tuple(excluded))


def _open_source(root: Path, relative: str) -> int:
    """Open through non-symlink directory descriptors to reject parent replacement races."""
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        parts = relative.split("/")
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
    finally:
        os.close(descriptor)


def capture_agent_source(inventory: AgentSourceInventory, output: Path) -> str:
    """Capture into an invocation-owned archive, failing on detectable selected-file changes."""
    stream_size = 10240
    with output.open("xb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            for entry in inventory.entries:
                header = tarfile.TarInfo(entry.path)
                header.mode = entry.mode
                header.type = tarfile.DIRTYPE if entry.directory else tarfile.REGTYPE
                header.size = entry.size
                stream_size += len(header.tobuf(format=tarfile.PAX_FORMAT)) + ((entry.size + 511) // 512) * 512
                if stream_size > MAX_STREAM_BYTES:
                    raise ValueError("Agent archive expanded stream exceeds limit")
                if entry.directory:
                    if _fingerprint((inventory.root / entry.path).lstat()) != entry.fingerprint:
                        raise ValueError(f"Agent source changed during capture: {entry.path}")
                    archive.addfile(header)
                    continue
                with os.fdopen(_open_source(inventory.root, entry.path), "rb") as source:
                    if _fingerprint(os.fstat(source.fileno())) != entry.fingerprint:
                        raise ValueError(f"Agent source changed during capture: {entry.path}")
                    archive.addfile(header, source)
                    if source.read(1) or _fingerprint(os.fstat(source.fileno())) != entry.fingerprint:
                        raise ValueError(f"Agent source changed during capture: {entry.path}")
    if output.stat().st_size > MAX_STREAM_BYTES:
        raise ValueError("Agent archive compressed stream exceeds limit")
    with output.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


class _AgentTarInfo(tarfile.TarInfo):
    def _proc_member(self, archive):
        archive.physical_count = getattr(archive, "physical_count", 0) + 1
        if archive.physical_count > MAX_ENTRIES * 2:
            raise ValueError("Too many agent archive records")
        if self.type == tarfile.XHDTYPE:
            if self.size > MAX_EXTENSION_BYTES or getattr(archive, "in_extension", False):
                raise ValueError("Excessive agent archive extension")
            archive.in_extension = True
            try:
                return super()._proc_member(archive)  # ty: ignore[unresolved-attribute]
            finally:
                archive.in_extension = False
        if self.type not in {tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE}:
            raise ValueError("Unsupported agent archive entry type")
        return super()._proc_member(archive)  # ty: ignore[unresolved-attribute]

    def _proc_gnusparse_00(self, next, raw_headers):
        raise ValueError("Unsupported sparse archive")

    def _proc_gnusparse_01(self, next, pax_headers):
        raise ValueError("Unsupported sparse archive")

    def _proc_gnusparse_10(self, next, pax_headers, archive):
        raise ValueError("Unsupported sparse archive")


class _BoundedReader:
    def __init__(self, stream: BinaryIO):
        self.stream = stream
        self.size = 0

    def read(self, size: int) -> bytes:
        if size < 0:
            raise ValueError("Unbounded agent archive read")
        value = self.stream.read(min(size, MAX_STREAM_BYTES - self.size + 1))
        self.size += len(value)
        if self.size > MAX_STREAM_BYTES:
            raise ValueError("Agent archive expanded stream exceeds limit")
        return value


def extract_agent_source(source: Path, destination: Path) -> None:
    """Extract verified bytes into a fresh private directory, restoring only normalized modes."""
    seen: set[str] = set()
    directories: set[str] = set()
    total = 0
    with source.open("rb") as raw, gzip.GzipFile(fileobj=raw, mode="rb") as zipped:
        bounded = _BoundedReader(cast(BinaryIO, zipped))
        with tarfile.open(fileobj=cast(BinaryIO, bounded), mode="r|", tarinfo=_AgentTarInfo) as archive:
            for entry in archive:
                name = entry.name.rstrip("/") if entry.isdir() else entry.name
                validate_archive_path(name)
                if name.casefold() in seen or len(seen) >= MAX_ENTRIES:
                    raise ValueError("Duplicate or excessive agent archive entries")
                if entry.pax_headers.keys() - {"path"} or entry.sparse is not None:
                    raise ValueError("Unsupported agent archive metadata")
                seen.add(name.casefold())
                parts = name.split("/")
                if any("/".join(parts[:i]) not in directories for i in range(1, len(parts))):
                    raise ValueError("Agent archive has missing or case-colliding parent directories")
                path = destination / name
                if entry.isdir():
                    if entry.size or entry.mode != 0o755:
                        raise ValueError("Invalid agent archive directory")
                    path.mkdir(mode=0o755)
                    path.chmod(0o755)
                    directories.add(name)
                else:
                    total += entry.size
                    if (
                        entry.mode not in {0o644, 0o755}
                        or entry.size < 0
                        or entry.size > MAX_FILE_BYTES
                        or total > MAX_PAYLOAD_BYTES
                    ):
                        raise ValueError("Agent archive mode or payload exceeds limits")
                    body = archive.extractfile(entry)
                    if body is None:
                        raise ValueError("Missing agent archive body")
                    with body, path.open("xb") as output:
                        count = 0
                        while chunk := body.read(CHUNK_BYTES):
                            count += len(chunk)
                            if count > entry.size:
                                raise ValueError("Agent archive body exceeds declared size")
                            output.write(chunk)
                    if count != entry.size:
                        raise ValueError("Truncated agent archive body")
                    path.chmod(entry.mode)
            while chunk := archive.fileobj.read(CHUNK_BYTES):
                if any(chunk):
                    raise ValueError("Unexpected trailing agent archive content")


def publish_agent_source(
    root: Path,
    *,
    files_client: FilesClient,
    fileset_ref: str,
    options: AgentSourceOptions | None = None,
    jobs_dir: Path | None = None,
) -> HarborAgentSource:
    """Retain one unique publication; uncertain upload responses require verified readback."""
    path = f"{uuid4()}/agent.tar.gz"
    candidate = HarborAgentSource(fileset_ref=f"{fileset_ref}#{path}", sha256="0" * 64)
    workspace, name, _ = parse_fileset_ref(candidate.fileset_ref, workspace_fallback=None)
    inventory = inspect_agent_source(root, options=options, jobs_dir=jobs_dir)
    logger.info(
        "Agent source: %d entries, %d bytes, %d exclusions",
        len(inventory.entries),
        inventory.total_bytes,
        len(inventory.excluded),
    )
    with private_directory() as owned:
        archive = owned / "agent.tar.gz"
        digest = capture_agent_source(inventory, archive)
        source = HarborAgentSource(fileset_ref=candidate.fileset_ref, sha256=digest)
        files_client.create_fileset(workspace=workspace, body=CreateFilesetRequest(name=name), exist_ok=True).data()
        logger.info("Uploading agent source to %s", source.fileset_ref)
        with archive.open("rb") as stream:
            try:
                files_client.with_headers({"Content-Length": str(archive.stat().st_size)}).upload_file(
                    workspace=workspace, name=name, path=path, content=iter(lambda: stream.read(CHUNK_BYTES), b"")
                ).data()
            except NemoTransportError:
                pass
        with (owned / "readback").open("xb") as output:
            download_verified(files_client, source.fileset_ref, output, limit=MAX_STREAM_BYTES, expected_digest=digest)
        logger.info("Verified agent source %s (sha256=%s)", source.fileset_ref, source.sha256)
        return source


@contextmanager
def prepared_agent_source(
    source: HarborAgentSource,
    *,
    parent: Path,
    files_client: FilesClient | None = None,
    async_sdk: AsyncNemoClient | None = None,
) -> Iterator[Path]:
    """Each attempt owns its directory; keep this context open through runtime teardown."""
    parent.mkdir(parents=True, exist_ok=True)
    with private_directory(parent) as owned:
        archive = owned / "agent.tar.gz"
        with archive.open("xb") as output:
            if async_sdk is not None:

                async def download(client: AsyncNemoClient) -> None:
                    await download_verified_async(
                        AsyncFilesClient.from_client(client),
                        source.fileset_ref,
                        output,
                        limit=MAX_STREAM_BYTES,
                        expected_digest=source.sha256,
                    )

                run_with_isolated_async_client(async_sdk, download)
            elif files_client is not None:
                download_verified(
                    files_client, source.fileset_ref, output, limit=MAX_STREAM_BYTES, expected_digest=source.sha256
                )
            else:
                raise ValueError("Agent source requires an authenticated Files client")
        destination = owned / "agent"
        destination.mkdir(mode=0o700)
        extract_agent_source(archive, destination)
        try:
            yield destination
        except Exception as exc:
            exc.add_note(f"Executing verified Harbor agent source {source.fileset_ref} (sha256={source.sha256})")
            raise
