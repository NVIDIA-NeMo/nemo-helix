# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import shutil
import tempfile
from abc import abstractmethod
from asyncio import CancelledError, create_task, wait
from dataclasses import dataclass, field
from os import walk
from pathlib import Path
from typing import NoReturn, Protocol

import anyio
from anyio.to_thread import run_sync
from filesets import AsyncFilesetFileSystem, FilesetFileSystem, build_fileset_ref, parse_fileset_ref
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.jobs.schemas import FileStorageType

logger = logging.getLogger(__name__)


@dataclass
class TmpDirPath:
    """
    Simple wrapper of a path of interest (`path`) and a containing folder (`tmp_dir`).
    We provide an explicit cleanup function rather than a context manager to play
    nicely with FastAPI background tasks that can run after the `path` has been returned.
    """

    path: Path
    tmp_dir: Path

    def cleanup_tmp_dir(self):
        shutil.rmtree(self.tmp_dir)


def _filter_files_by_patterns(
    relative_paths: list[str],
    ignore_patterns: list[str] | str | None = None,
) -> list[str]:
    """
    Filter a list of relative file paths by ignore patterns.

    Uses pathlib.Path.match() for glob pattern matching.

    Args:
        relative_paths: List of relative path strings
        ignore_patterns: Glob patterns for files to ignore. Supports:
            - "*.ext" - match files by extension (at any depth)
            - "name" - match files by name (at any depth)
            - "dir/" - match all files inside directories named "dir"

    Returns:
        Filtered list of relative path strings (files not matching any pattern)
    """
    if ignore_patterns is None:
        return relative_paths

    if isinstance(ignore_patterns, str):
        ignore_patterns = [ignore_patterns]

    def matches(rel_path: str) -> bool:
        path = Path(rel_path)
        for pattern in ignore_patterns:
            if pattern.endswith("/"):
                # Directory pattern: match files under directories with this name
                dir_name = pattern.rstrip("/")
                if dir_name in path.parts[:-1]:
                    return True
            elif path.match(pattern):
                return True
        return False

    return [p for p in relative_paths if not matches(p)]


async def _list_local_files(local_path: Path) -> list[str]:
    """List all files in a local directory recursively."""
    files: list[str] = []
    async_path = anyio.Path(local_path)
    async for file_path in async_path.rglob("*"):
        if await file_path.is_file():
            files.append(str(Path(file_path).relative_to(local_path)))
    return files


def _list_local_files_sync(local_path: Path) -> list[str]:
    """List all files in a local directory recursively."""
    return [str(file_path.relative_to(local_path)) for file_path in local_path.rglob("*") if file_path.is_file()]


def _list_selected_files(local_path: Path, ignore_patterns: list[str] | str | None = None) -> list[str]:
    """List files for staging from selected directories, propagating traversal errors."""

    def raise_error(error: OSError) -> NoReturn:
        raise error

    patterns = [ignore_patterns] if isinstance(ignore_patterns, str) else ignore_patterns or []
    ignored_directories = {pattern.rstrip("/") for pattern in patterns if pattern.endswith("/")}
    files: list[str] = []
    for directory, directories, names in walk(local_path, onerror=raise_error, followlinks=False):
        if ignored_directories:
            directories[:] = [name for name in directories if name not in ignored_directories]
        for name in names:
            path = Path(directory) / name
            if path.is_file():
                files.append(str(path.relative_to(local_path)))
    return files


def _stage_local_files(
    local_path: Path, staging_root: Path, ignore_patterns: list[str] | str | None
) -> tuple[Path, list[str]]:
    """Acquire a complete upload copy before sending any directory contents.

    Runtime-owned files can disappear between traversal and opening them (for
    example, a harness finishing a temporary clone). Restart acquisition from a
    fresh listing in that case; never omit one failed file from an existing
    manifest. Persistent churn and all other filesystem errors still fail the
    upload. The caller owns ``staging_root`` and keeps it until upload finishes.
    Uploads stage beside ``local_path`` so the copy uses the job's own volume
    rather than container temporary storage; a full volume is an upload
    failure, not a reason to omit files.

    This stabilizes the upload inputs, not concurrent in-place writes: producers
    must still finish writing their result files before publishing them.
    """
    for attempt in range(3):
        snapshot = staging_root / "files"
        snapshot.mkdir()
        try:
            if not local_path.is_dir():
                raise FileNotFoundError(f"Result directory no longer exists: {local_path}")
            files = _filter_files_by_patterns(
                _list_selected_files(local_path, ignore_patterns=ignore_patterns), ignore_patterns
            )
            if not local_path.is_dir():
                raise FileNotFoundError(f"Result directory no longer exists: {local_path}")
            for rel_path in files:
                destination = snapshot / rel_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(local_path / rel_path, destination)
        except FileNotFoundError:
            shutil.rmtree(snapshot)
            if attempt == 2 or not local_path.is_dir():
                raise
            logger.warning("Result directory changed during staging; retrying %s", local_path)
        else:
            return snapshot, files
    raise AssertionError("Unreachable: staging either returns or raises")


async def _stage_local_files_async(
    local_path: Path, staging_root: Path, ignore_patterns: list[str] | str | None
) -> tuple[Path, list[str]]:
    """Keep the acquisition worker alive until it releases its temporary directory.

    Native asyncio cancellation can interrupt ``run_sync`` before its worker
    stops. Drain the owned task through repeated cancellation before the caller
    cleans up staging, consuming any later worker error without masking the
    first cancellation.
    """
    acquisition = create_task(run_sync(_stage_local_files, local_path, staging_root, ignore_patterns))
    cancellation: CancelledError | None = None
    while not acquisition.done():
        try:
            # Do not forward cancellation or eagerly report the worker's error:
            # shield() can report it before we restore cancellation on Python 3.14.
            await wait((acquisition,))
        except CancelledError as error:
            cancellation = cancellation or error
    if cancellation is not None:
        if not acquisition.cancelled():
            acquisition.exception()
        raise cancellation
    return acquisition.result()


class FileManager(Protocol):
    """
    Protocol for a generic file provider. Both the async and sync versions must be implemented,
    but it's up to the implementer to decide how to best deduplicate logic (ex. write it sync, use async wrappers).
    """

    @abstractmethod
    def validate_storage(self) -> None: ...

    @abstractmethod
    def upload(
        self, local_path: Path, remote_path: str, ignore_patterns: list[str] | str | None = None, *, stage: bool = False
    ) -> str: ...

    @abstractmethod
    def download_from_url(self, url: str, local_dir: str | Path | None = None) -> TmpDirPath: ...

    @abstractmethod
    def url(self) -> str: ...

    @abstractmethod
    def storage_type(self) -> FileStorageType: ...


class AsyncFileManager(Protocol):
    @abstractmethod
    async def validate_storage(self) -> None: ...

    @abstractmethod
    async def upload(
        self, local_path: Path, remote_path: str, ignore_patterns: list[str] | str | None = None, *, stage: bool = False
    ) -> str: ...

    @abstractmethod
    async def download_from_url(self, url: str, local_dir: str | Path | None = None) -> TmpDirPath: ...

    @abstractmethod
    def url(self) -> str: ...

    @abstractmethod
    def storage_type(self) -> FileStorageType: ...


class FileStorageDoesNotExist(Exception): ...


@dataclass
class BaseFilesetFileManager:
    """Base class for Fileset-backed file managers."""

    workspace: str
    fileset_name: str

    def url(self, remote_path: str | None = None) -> str:
        """Return fileset reference for the given path."""
        if remote_path:
            return build_fileset_ref(remote_path, workspace=self.workspace, fileset=self.fileset_name)
        return f"{self.workspace}/{self.fileset_name}"

    def storage_type(self) -> FileStorageType:
        return FileStorageType.FILESET

    def _fileset_path(self, remote_path: str | None = None) -> str:
        """Return internal path format for FilesetFileSystem (without fileset:// prefix)."""
        if remote_path:
            return build_fileset_ref(remote_path, workspace=self.workspace, fileset=self.fileset_name)
        return f"{self.workspace}/{self.fileset_name}"


@dataclass
class FilesetFileManager(BaseFilesetFileManager):
    """Synchronous FileManager implementation for Filesets."""

    filesystem: FilesetFileSystem
    ensure_fileset_exists: bool = True

    _fs: FilesetFileSystem = field(init=False)

    def __post_init__(self) -> None:
        self._fs = self.filesystem

    def validate_storage(self) -> None:
        """Check if fileset exists, create if ensure_fileset_exists=True."""
        try:
            self._fs.info(self._fileset_path())
        except FileNotFoundError:
            if self.ensure_fileset_exists:
                logger.info(f"Creating new fileset: [{self.fileset_name}] in workspace [{self.workspace}]")
                self._fs._client.create_fileset(
                    body=CreateFilesetRequest(name=self.fileset_name), workspace=self.workspace
                )
            else:
                raise FileStorageDoesNotExist(
                    f"Fileset [{self.fileset_name}] in workspace [{self.workspace}] does not exist."
                )

    def upload(
        self, local_path: Path, remote_path: str, ignore_patterns: list[str] | str | None = None, *, stage: bool = False
    ) -> str:
        """Upload a file or directory to the fileset.

        With ``stage``, copy a directory's selected files first and upload that copy (see ``_stage_local_files``).
        """
        full_remote_path = self._fileset_path(remote_path)
        if local_path.is_dir() and not stage:
            all_files = _list_local_files_sync(local_path)
            files_to_upload = _filter_files_by_patterns(all_files, ignore_patterns)
            for rel_path in files_to_upload:
                self._fs.put_file(str(local_path / rel_path), f"{full_remote_path}/{rel_path}")
        elif local_path.is_dir():
            with tempfile.TemporaryDirectory(prefix="nemo-result-upload-", dir=local_path.parent) as staging_dir:
                snapshot, files_to_upload = _stage_local_files(local_path, Path(staging_dir), ignore_patterns)
                for rel_path in files_to_upload:
                    try:
                        self._fs.put_file(str(snapshot / rel_path), f"{full_remote_path}/{rel_path}")
                    except Exception as error:
                        error.add_note(f"Failed to upload {local_path / rel_path} to {full_remote_path}/{rel_path}")
                        raise
        else:
            self._fs.put_file(str(local_path), full_remote_path)
        return self.url(remote_path)

    def download_from_url(self, url: str, local_dir: str | Path | None = None) -> TmpDirPath:
        """Download from fileset:// URL."""
        workspace, fileset, remote_path = parse_fileset_ref(url, workspace_fallback=self.workspace)
        if workspace != self.workspace or fileset != self.fileset_name:
            raise ValueError(f"URL [{url}] is not a valid fileset:// URL for this manager.")

        if local_dir is None:
            local_dir = tempfile.mkdtemp()
        if isinstance(local_dir, str):
            local_dir = Path(local_dir)

        full_remote_path = self._fileset_path(remote_path)
        if not remote_path or remote_path.endswith("/"):
            self._fs.get(full_remote_path, str(local_dir), recursive=True)
            return TmpDirPath(path=local_dir, tmp_dir=local_dir)

        info = self._fs.info(full_remote_path)

        if info["type"] == "directory":
            self._fs.get(full_remote_path, str(local_dir), recursive=True)
            return TmpDirPath(path=local_dir / remote_path.split("/")[-1], tmp_dir=local_dir)

        local_file_path = local_dir / Path(remote_path).name
        self._fs.get_file(full_remote_path, str(local_file_path))
        return TmpDirPath(path=local_file_path, tmp_dir=local_dir)


@dataclass
class AsyncFilesetFileManager(BaseFilesetFileManager):
    """Asynchronous FileManager implementation for Filesets."""

    filesystem: AsyncFilesetFileSystem
    ensure_fileset_exists: bool = True

    _fs: AsyncFilesetFileSystem = field(init=False)

    def __post_init__(self) -> None:
        self._fs = self.filesystem

    async def _validate_storage(self) -> None:
        """Check if fileset exists, create if ensure_fileset_exists=True."""
        try:
            await self._fs._info(self._fileset_path())
        except FileNotFoundError:
            if self.ensure_fileset_exists:
                logger.info(f"Creating new fileset: [{self.fileset_name}] in workspace [{self.workspace}]")
                await self._fs._client.create_fileset(
                    body=CreateFilesetRequest(name=self.fileset_name), workspace=self.workspace
                )
            else:
                raise FileStorageDoesNotExist(
                    f"Fileset [{self.fileset_name}] in workspace [{self.workspace}] does not exist."
                )

    async def _upload(
        self, local_path: Path, remote_path: str, ignore_patterns: list[str] | str | None = None, *, stage: bool = False
    ) -> str:
        """Upload a file or directory to the fileset.

        With ``stage``, copy a directory's selected files first and upload that copy (see ``_stage_local_files``).
        """
        full_remote_path = self._fileset_path(remote_path)
        if local_path.is_dir() and not stage:
            all_files = await _list_local_files(local_path)
            files_to_upload = _filter_files_by_patterns(all_files, ignore_patterns)
            for rel_path in files_to_upload:
                await self._fs._put_file(str(local_path / rel_path), f"{full_remote_path}/{rel_path}")
        elif local_path.is_dir():
            with tempfile.TemporaryDirectory(prefix="nemo-result-upload-", dir=local_path.parent) as staging_dir:
                snapshot, files_to_upload = await _stage_local_files_async(
                    local_path, Path(staging_dir), ignore_patterns
                )
                for rel_path in files_to_upload:
                    try:
                        await self._fs._put_file(str(snapshot / rel_path), f"{full_remote_path}/{rel_path}")
                    except Exception as error:
                        error.add_note(f"Failed to upload {local_path / rel_path} to {full_remote_path}/{rel_path}")
                        raise
        else:
            await self._fs._put_file(str(local_path), full_remote_path)
        return self.url(remote_path)

    async def _download_from_url(self, url: str, local_dir: str | Path | None = None) -> TmpDirPath:
        """Download from fileset:// URL."""
        workspace, fileset, remote_path = parse_fileset_ref(url, workspace_fallback=self.workspace)
        if workspace != self.workspace or fileset != self.fileset_name:
            raise ValueError(f"URL [{url}] is not a valid fileset:// URL for this manager.")

        if local_dir is None:
            local_dir = tempfile.mkdtemp()
        if isinstance(local_dir, str):
            local_dir = Path(local_dir)

        full_remote_path = self._fileset_path(remote_path)
        if not remote_path or remote_path.endswith("/"):
            await self._fs._get(full_remote_path, str(local_dir), recursive=True)
            return TmpDirPath(path=local_dir, tmp_dir=local_dir)

        info = await self._fs._info(full_remote_path)

        if info["type"] == "directory":
            await self._fs._get(full_remote_path, str(local_dir), recursive=True)
            return TmpDirPath(path=local_dir / remote_path.split("/")[-1], tmp_dir=local_dir)
        else:
            local_file_path = local_dir / Path(remote_path).name
            await self._fs._get_file(full_remote_path, str(local_file_path))
            return TmpDirPath(path=local_file_path, tmp_dir=local_dir)

    async def validate_storage(self) -> None:
        await self._validate_storage()

    async def upload(
        self, local_path: Path, remote_path: str, ignore_patterns: list[str] | str | None = None, *, stage: bool = False
    ) -> str:
        return await self._upload(local_path, remote_path, ignore_patterns, stage=stage)

    async def download_from_url(self, url: str, local_dir: str | Path | None = None) -> TmpDirPath:
        return await self._download_from_url(url, local_dir)
