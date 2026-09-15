# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""High-level fileset transfers on the typed Files client.

``upload``, ``download``, ``list_files`` and ``delete`` (and their async twins)
drive :class:`~filesets.filesystem.filesystem.FilesetFileSystem` from a
:class:`~nemo_platform_plugin.files.client.FilesClient`. They resolve the
``[workspace/]fileset#path`` reference forms, expand glob patterns, create
filesets on demand, and report progress through fsspec callbacks. The CLI and
the SDK ``FilesResource`` both build on these functions.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import PurePath

from fsspec.callbacks import DEFAULT_CALLBACK, Callback
from fsspec.core import has_magic
from nemo_platform_plugin.files.client import AsyncFilesClient, FilesClient
from nemo_platform_plugin.files.types import (
    CacheStatus,
    CreateFilesetRequest,
    FilesetFileOutput,
    FilesetOutput,
    ListFilesQueryParams,
)

from filesets.filesystem.filesystem import (
    AsyncFilesetFileSystem,
    FilesetFileSystem,
    build_fileset_ref,
    parse_fileset_path,
)

FILESET_REQUIRED_MESSAGE = "Fileset must be specified either as a parameter or in the remote_path."
FILESET_REQUIRED_WITHOUT_AUTO_CREATE_MESSAGE = (
    "Fileset must be specified either as a parameter or in the remote_path when fileset_auto_create is False."
)


@dataclass
class ListFilesResponse:
    """Response from listing files in a fileset.

    Attributes:
        data: List of files in the fileset.

    Properties:
        cache_status: Aggregate cache status of all files.
            - "caching" if any file is actively being cached
            - "not_cached" if any file is not cached (and none are caching)
            - "cached" if all files are fully cached
            - "not_cacheable" if all files cannot be cached
            - None if no cache information is available
    """

    data: list[FilesetFileOutput]

    @property
    def cache_status(self) -> CacheStatus | None:
        """Get aggregate cache status of all files.

        Returns the most relevant status based on priority:
        - "caching" if any file is actively being cached
        - "not_cached" if any file is not cached (and none are caching)
        - "cached" if all files are fully cached
        - "not_cacheable" if all files cannot be cached
        - None if no cache information is available
        """
        if not self.data:
            return None

        statuses = [f.cache_status for f in self.data if f.cache_status is not None]
        if not statuses:
            return None

        # Priority: caching > not_cached > cached > not_cacheable
        if "caching" in statuses:
            return CacheStatus.CACHING
        if "not_cached" in statuses:
            return CacheStatus.NOT_CACHED
        if all(s == "cached" for s in statuses):
            return CacheStatus.CACHED
        if all(s == "not_cacheable" for s in statuses):
            return CacheStatus.NOT_CACHEABLE

        # Mixed cached/not_cacheable - return cached since some files are cached
        return CacheStatus.CACHED


def generate_fileset_name() -> str:
    """Generate a unique fileset name using UUID."""
    return f"fileset-{uuid.uuid4().hex[:8]}"


def matches_glob(filepath: str, pattern: str) -> bool:
    """Match filepath against a glob pattern using pathlib.

    Simple patterns (no /) only match top-level files.
    Path patterns (with /) match the full relative path from the right.

    Examples:
        matches_glob("train.json", "*.json") -> True
        matches_glob("subdir/nested.json", "*.json") -> False (nested file)
        matches_glob("subdir/nested.json", "subdir/*.json") -> True
        matches_glob("subdir/nested.json", "*/*.json") -> True

    Args:
        filepath: The file path to check (relative path within fileset).
        pattern: Glob pattern to match against.

    Returns:
        True if the filepath matches the pattern.
    """
    if "/" not in pattern:
        # Simple pattern - only matches top-level files
        return "/" not in filepath and PurePath(filepath).match(pattern)
    # Path pattern - match from the right
    return PurePath(filepath).match(pattern)


def _resolve_target(
    remote_path: str,
    *,
    fileset: str | None,
    workspace: str | None,
    client_workspace: str | None,
) -> tuple[str, str | None, str]:
    """Return ``(workspace, fileset, path)`` for a remote path that may embed a fileset ref."""
    ws, path_fileset, path = parse_fileset_path(remote_path, workspace_fallback=workspace or client_workspace)
    return ws, fileset or path_fileset, path


def _resolve_upload_fileset(fileset: str | None, *, fileset_auto_create: bool) -> str:
    if fileset is not None:
        return fileset
    if fileset_auto_create:
        return generate_fileset_name()
    raise ValueError(FILESET_REQUIRED_WITHOUT_AUTO_CREATE_MESSAGE)


def list_query_params(path: str, *, include_cache_status: bool = False) -> ListFilesQueryParams | None:
    """Query params ``list_files`` sends for *path*: a prefix goes to the server, a glob is filtered client-side."""
    query_params: ListFilesQueryParams = {}
    if not has_magic(path) and path:
        query_params["path"] = path
    if include_cache_status:
        query_params["include_cache_status"] = True
    return query_params or None


def _filter_listed(files: list[FilesetFileOutput], path: str) -> ListFilesResponse:
    if has_magic(path):
        files = [f for f in files if matches_glob(f.path, path)]
    return ListFilesResponse(data=files)


def _pairs_for(paths: list[str], *, workspace: str, fileset: str, local_path: str) -> tuple[list[str], list[str]]:
    """Build parallel remote/local path lists that preserve directory structure."""
    rpaths = [build_fileset_ref(p, workspace=workspace, fileset=fileset) for p in paths]
    lpaths = [str(PurePath(local_path) / p) for p in paths]
    return rpaths, lpaths


def _async_transfer_kwargs(callback: Callback | None, max_workers: int | None) -> dict:
    kwargs: dict = {"batch_size": max_workers}
    if callback is not None:
        kwargs["callback"] = callback
    return kwargs


# ---------------------------------------------------------------------------
# Sync API
# ---------------------------------------------------------------------------


def _fs(client: FilesClient, filesystem: FilesetFileSystem | None) -> FilesetFileSystem:
    return filesystem if filesystem is not None else FilesetFileSystem(client=client)


def list_files(
    client: FilesClient,
    *,
    remote_path: str = "",
    fileset: str | None = None,
    workspace: str | None = None,
    include_cache_status: bool = False,
) -> ListFilesResponse:
    """List all files in a fileset path (recursive), with optional glob pattern support.

    Args:
        client: Typed Files client.
        remote_path: Path within the fileset to list. Can be a full path
            (e.g., "workspace/fileset#data/" or "fileset#data/") if fileset is not provided,
            or a relative path (e.g., "data/") if fileset is provided.
            Supports glob patterns (*, ?, []) for filtering files.
            Defaults to "" (root of fileset).
        fileset: Fileset name. If not provided, inferred from remote_path.
        workspace: Workspace name. If not provided, inferred from remote_path
            or uses the client's default workspace.
        include_cache_status: Check and return cache status for each file.
            When False (default), external storage files return None for cache_status.

    Returns:
        ListFilesResponse with data (list of FilesetFileOutput) and cache_status property.
    """
    ws, fileset, path = _resolve_target(
        remote_path, fileset=fileset, workspace=workspace, client_workspace=client.workspace
    )
    if fileset is None:
        raise ValueError(FILESET_REQUIRED_MESSAGE)

    response = client.list_files(
        workspace=ws,
        name=fileset,
        query_params=list_query_params(path, include_cache_status=include_cache_status),
    ).data()
    return _filter_listed(list(response.data), path)


def download(
    client: FilesClient,
    *,
    remote_path: str | list[str] = "",
    local_path: str,
    fileset: str | None = None,
    workspace: str | None = None,
    callback: Callback | None = None,
    max_workers: int | None = None,
    filesystem: FilesetFileSystem | None = None,
) -> None:
    """Download files from a fileset to a local path.

    Args:
        client: Typed Files client.
        remote_path: Path(s) within the fileset to download. Can be:
            - A single path (str): Full path (e.g., "workspace/fileset#data/"),
              relative path (e.g., "data/"), or glob pattern (e.g., "*.json").
            - A list of paths (list[str]): Multiple specific file paths to download.
              When using a list, fileset and workspace must be provided explicitly.
            Defaults to "" (root of fileset).
        local_path: Local destination path (directory).
        fileset: Fileset name. If not provided, inferred from remote_path (str only).
        workspace: Workspace name. If not provided, inferred from remote_path
            or uses the client's default workspace.
        callback: Optional progress callback (e.g., RichProgressCallback).
        max_workers: Maximum number of concurrent file transfers.
        filesystem: Filesystem to transfer through; defaults to one built on *client*.
    """
    fs = _fs(client, filesystem)
    callback = callback or DEFAULT_CALLBACK

    if isinstance(remote_path, list):
        if not remote_path:
            return
        ws = workspace or client.workspace
        if fileset is None:
            raise ValueError("fileset must be provided when remote_path is a list.")
        if ws is None:
            raise ValueError("workspace must be provided when remote_path is a list.")
        rpaths, lpaths = _pairs_for(remote_path, workspace=ws, fileset=fileset, local_path=local_path)
        fs.get(rpath=rpaths, lpath=lpaths, callback=callback)
        return

    ws, fileset, path = _resolve_target(
        remote_path, fileset=fileset, workspace=workspace, client_workspace=client.workspace
    )
    if fileset is None:
        raise ValueError(FILESET_REQUIRED_MESSAGE)

    if has_magic(path):
        matching = list_files(client, remote_path=path, fileset=fileset, workspace=ws)
        if not matching.data:
            return
        rpaths, lpaths = _pairs_for(
            [f.path for f in matching.data], workspace=ws, fileset=fileset, local_path=local_path
        )
        fs.get(rpath=rpaths, lpath=lpaths, callback=callback)
        return

    fs.get(
        rpath=build_fileset_ref(path, workspace=ws, fileset=fileset),
        lpath=local_path,
        recursive=True,
        callback=callback,
    )


def upload(
    client: FilesClient,
    *,
    local_path: str,
    remote_path: str = "",
    fileset: str | None = None,
    workspace: str | None = None,
    callback: Callback | None = None,
    max_workers: int | None = None,
    fileset_auto_create: bool = False,
    filesystem: FilesetFileSystem | None = None,
) -> FilesetOutput:
    """Upload files from a local path to a fileset.

    Args:
        client: Typed Files client.
        local_path: Local source path (file or directory). A trailing slash on a
            directory uploads its contents rather than the directory itself.
        remote_path: Path within the fileset to upload to. Can be a full path
            (e.g., "workspace/fileset#data/" or "fileset#data/") if fileset is not provided,
            or a relative path (e.g., "data/") if fileset is provided.
            Defaults to "" (root of fileset).
        fileset: Fileset name. If not provided, inferred from remote_path.
        workspace: Workspace name. If not provided, inferred from remote_path
            or uses the client's default workspace.
        callback: Optional progress callback (e.g., RichProgressCallback).
        max_workers: Maximum number of concurrent file transfers.
        fileset_auto_create: If True, create the fileset if it doesn't exist.
            When no fileset is specified (neither as param nor in remote_path),
            a unique name is generated (e.g., "fileset-a1b2c3d4").
        filesystem: Filesystem to transfer through; defaults to one built on *client*.

    Returns:
        FilesetOutput: The fileset that was uploaded to. Check ``fileset.name`` to see
            the generated name when using fileset_auto_create without specifying
            a fileset.
    """
    ws, fileset, path = _resolve_target(
        remote_path, fileset=fileset, workspace=workspace, client_workspace=client.workspace
    )
    fileset = _resolve_upload_fileset(fileset, fileset_auto_create=fileset_auto_create)

    fileset_ref = build_fileset_ref(path, workspace=ws, fileset=fileset)
    if fileset_auto_create:
        client.create_fileset(workspace=ws, body=CreateFilesetRequest(name=fileset), exist_ok=True)

    _fs(client, filesystem).put(
        lpath=local_path, rpath=fileset_ref, recursive=True, callback=callback or DEFAULT_CALLBACK
    )

    return client.get_fileset(name=fileset, workspace=ws).data()


def delete(
    client: FilesClient,
    *,
    remote_path: str,
    fileset: str | None = None,
    workspace: str | None = None,
    filesystem: FilesetFileSystem | None = None,
) -> None:
    """Delete a file from a fileset.

    Args:
        client: Typed Files client.
        remote_path: Path of the file to delete. Can be a full path
            (e.g., "workspace/fileset#data/file.txt") if fileset is not provided,
            or a relative path (e.g., "data/file.txt") if fileset is provided.
        fileset: Fileset name. If not provided, inferred from remote_path.
        workspace: Workspace name. If not provided, inferred from remote_path
            or uses the client's default workspace.
        filesystem: Filesystem to delete through; defaults to one built on *client*.
    """
    ws, fileset, path = _resolve_target(
        remote_path, fileset=fileset, workspace=workspace, client_workspace=client.workspace
    )
    if fileset is None:
        raise ValueError(FILESET_REQUIRED_MESSAGE)

    _fs(client, filesystem).rm(build_fileset_ref(path, workspace=ws, fileset=fileset))


# ---------------------------------------------------------------------------
# Async API
# ---------------------------------------------------------------------------


def _async_fs(client: AsyncFilesClient, filesystem: AsyncFilesetFileSystem | None) -> AsyncFilesetFileSystem:
    return filesystem if filesystem is not None else AsyncFilesetFileSystem(client=client)


async def async_list_files(
    client: AsyncFilesClient,
    *,
    remote_path: str = "",
    fileset: str | None = None,
    workspace: str | None = None,
    include_cache_status: bool = False,
) -> ListFilesResponse:
    """Async twin of :func:`list_files`."""
    ws, fileset, path = _resolve_target(
        remote_path, fileset=fileset, workspace=workspace, client_workspace=client.workspace
    )
    if fileset is None:
        raise ValueError(FILESET_REQUIRED_MESSAGE)

    response = await client.list_files(
        workspace=ws,
        name=fileset,
        query_params=list_query_params(path, include_cache_status=include_cache_status),
    )
    return _filter_listed(list(response.data().data), path)


async def async_download(
    client: AsyncFilesClient,
    *,
    remote_path: str | list[str] = "",
    local_path: str,
    fileset: str | None = None,
    workspace: str | None = None,
    callback: Callback | None = None,
    max_workers: int | None = None,
    filesystem: AsyncFilesetFileSystem | None = None,
) -> None:
    """Async twin of :func:`download`."""
    fs = _async_fs(client, filesystem)
    cb = callback or DEFAULT_CALLBACK

    if isinstance(remote_path, list):
        if not remote_path:
            return
        ws = workspace or client.workspace
        if fileset is None:
            raise ValueError("fileset must be provided when remote_path is a list.")
        if ws is None:
            raise ValueError("workspace must be provided when remote_path is a list.")
        rpaths, lpaths = _pairs_for(remote_path, workspace=ws, fileset=fileset, local_path=local_path)
        await fs._get(rpaths, lpaths, batch_size=max_workers, callback=cb)
        return

    ws, fileset, path = _resolve_target(
        remote_path, fileset=fileset, workspace=workspace, client_workspace=client.workspace
    )
    if fileset is None:
        raise ValueError(FILESET_REQUIRED_MESSAGE)

    if has_magic(path):
        matching = await async_list_files(client, remote_path=path, fileset=fileset, workspace=ws)
        if not matching.data:
            return
        rpaths, lpaths = _pairs_for(
            [f.path for f in matching.data], workspace=ws, fileset=fileset, local_path=local_path
        )
        await fs._get(rpaths, lpaths, batch_size=max_workers, callback=cb)
        return

    await fs._get(
        build_fileset_ref(path, workspace=ws, fileset=fileset),
        local_path,
        recursive=True,
        batch_size=max_workers,
        callback=cb,
    )


async def async_upload(
    client: AsyncFilesClient,
    *,
    local_path: str,
    remote_path: str = "",
    fileset: str | None = None,
    workspace: str | None = None,
    callback: Callback | None = None,
    max_workers: int | None = None,
    fileset_auto_create: bool = False,
    filesystem: AsyncFilesetFileSystem | None = None,
) -> FilesetOutput:
    """Async twin of :func:`upload`."""
    ws, fileset, path = _resolve_target(
        remote_path, fileset=fileset, workspace=workspace, client_workspace=client.workspace
    )
    fileset = _resolve_upload_fileset(fileset, fileset_auto_create=fileset_auto_create)

    fileset_ref = build_fileset_ref(path, workspace=ws, fileset=fileset)
    if fileset_auto_create:
        await client.create_fileset(workspace=ws, body=CreateFilesetRequest(name=fileset), exist_ok=True)

    await _async_fs(client, filesystem)._put(
        lpath=local_path, rpath=fileset_ref, recursive=True, **_async_transfer_kwargs(callback, max_workers)
    )

    return (await client.get_fileset(name=fileset, workspace=ws)).data()


async def async_delete(
    client: AsyncFilesClient,
    *,
    remote_path: str,
    fileset: str | None = None,
    workspace: str | None = None,
    filesystem: AsyncFilesetFileSystem | None = None,
) -> None:
    """Async twin of :func:`delete`."""
    ws, fileset, path = _resolve_target(
        remote_path, fileset=fileset, workspace=workspace, client_workspace=client.workspace
    )
    if fileset is None:
        raise ValueError(FILESET_REQUIRED_MESSAGE)

    await _async_fs(client, filesystem)._rm(build_fileset_ref(path, workspace=ws, fileset=fileset))
