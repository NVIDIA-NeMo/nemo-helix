# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FilesetFileSystem - fsspec filesystem for NeMo Platform fileset storage."""

from __future__ import annotations

import inspect
import os
from collections.abc import AsyncIterator, Coroutine, Iterator, Sequence
from datetime import datetime, timezone
from glob import has_magic
from pathlib import Path
from typing import Any, Literal, TypedDict, TypeVar, overload

import anyio
import httpx
from anyio import to_thread
from fsspec.asyn import AbstractAsyncStreamedFile, AsyncFileSystem, _get_batch_size
from fsspec.callbacks import DEFAULT_CALLBACK, Callback
from fsspec.implementations.local import LocalFileSystem, make_path_posix, trailing_sep
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem
from fsspec.utils import other_paths
from nemo_platform_plugin.files.client import AsyncFilesClient, FilesClient
from nemo_platform_plugin.files.types import FilesetFileOutput, ListFilesQueryParams

T = TypeVar("T")


async def run_coros_in_chunks(
    coros: Sequence[Coroutine[Any, Any, T]],
    batch_size: int | None = None,
    callback: Callback = DEFAULT_CALLBACK,
    timeout: float | None = None,
    return_exceptions: bool = False,
    nofiles: bool = False,
) -> list[T | None | BaseException]:
    """Run coroutines with bounded concurrency using AnyIO task groups.

    This intentionally differs from fsspec's chunked wave-based scheduling.
    We admit all work up front and use a CapacityLimiter so a new task can start
    as soon as any slot frees up, which keeps bulk downloads/uploads saturated.

    Running tasks are cancelled by the task group; coroutine objects that were
    never entered are explicitly closed in the outer finally block.
    """

    if batch_size is None:
        batch_size = _get_batch_size(nofiles=nofiles)
    if batch_size == -1 or not isinstance(batch_size, int):
        batch_size = len(coros) if coros else 1

    results: list[T | None | BaseException] = [None] * len(coros)
    exceptions: list[BaseException] = []

    limiter = anyio.CapacityLimiter(batch_size)

    async def run_one(coro: Coroutine[Any, Any, T], idx: int) -> None:
        entered = False
        try:
            async with limiter:
                entered = True
                if timeout is None:
                    results[idx] = await coro
                else:
                    with anyio.fail_after(timeout):
                        results[idx] = await coro
        except Exception as e:
            if return_exceptions:
                results[idx] = e
            else:
                exceptions.append(e)
                raise
        finally:
            if entered:
                callback.relative_update(1)

    try:
        async with anyio.create_task_group() as tg:
            for i, coro in enumerate(coros):
                tg.start_soon(run_one, coro, i)
    except ExceptionGroup:
        # Re-raise the first actual failure for compatibility with callers that
        # expect a plain exception instead of an ExceptionGroup.
        if not return_exceptions and exceptions:
            raise exceptions[0] from None
        raise
    finally:
        # Any coroutine objects that were never entered by a task must be explicitly
        # closed, otherwise Python warns that they were never awaited.
        for coro in coros:
            if inspect.getcoroutinestate(coro) == inspect.CORO_CREATED:
                coro.close()

    return results


class FileInfo(TypedDict):
    """File or directory info returned by fsspec methods."""

    name: str
    size: int
    type: Literal["file", "directory"]


class FilesetPathError(ValueError):
    """Error raised when handling a fileset path."""


def parse_fileset_ref(ref: str, *, workspace_fallback: str | None) -> tuple[str, str, str]:
    """Parse fileset reference -> (workspace, fileset, file_path).

    Extracts components from a fileset reference. Workspace must be provided
    either in the ref or via workspace_fallback.

    Args:
        ref: The fileset reference to parse.
        workspace_fallback: Workspace to use if not specified in the ref.

    Returns:
        Tuple of (workspace, fileset, file_path).

    Raises:
        FilesetPathError: If workspace is not in the ref and no fallback provided.

    Supported formats:
        - URL format: fileset://workspace/fileset[#path]
        - Hash format: [workspace/]fileset#path
        - Path format: [workspace/]fileset (1-2 segments, no path)
        - Legacy format: workspace/fileset/path (3+ segments, for backwards compatibility)

    The `#` separator distinguishes fileset name from file path.
    If `#` is omitted and there are 3+ path segments, assumes legacy format.

    Examples:
        parse_fileset_ref("default/my-fileset#data/file.txt", workspace_fallback=None)
            -> ("default", "my-fileset", "data/file.txt")
        parse_fileset_ref("my-fileset#file.txt", workspace_fallback="default")
            -> ("default", "my-fileset", "file.txt")
        parse_fileset_ref("default/my-fileset", workspace_fallback=None)
            -> ("default", "my-fileset", "")
        parse_fileset_ref("default/my-fileset/data/file.txt", workspace_fallback=None)
            -> ("default", "my-fileset", "data/file.txt")  # legacy format
    """
    ref = ref.removeprefix("fileset://").lstrip("/")
    if not ref:
        raise FilesetPathError("Path must include fileset name")

    # If # is present, use hash format parsing
    if "#" in ref:
        fileset_part, file_path = ref.split("#", 1)
        file_path = file_path.lstrip("/")

        # Parse workspace/fileset or just fileset
        if "/" in fileset_part:
            workspace, fileset = fileset_part.rsplit("/", 1)
        else:
            workspace = ""
            fileset = fileset_part
    else:
        # No # - could be path format (1-2 segments) or legacy format (3+ segments)
        parts = ref.split("/")

        if len(parts) == 1:
            # Just fileset name
            workspace = ""
            fileset = parts[0]
            file_path = ""
        elif len(parts) == 2:
            # workspace/fileset (root)
            workspace, fileset = parts
            file_path = ""
        else:
            # Legacy format: workspace/fileset/path/...
            workspace = parts[0]
            fileset = parts[1]
            file_path = "/".join(parts[2:])

    if not fileset:
        raise FilesetPathError("Fileset name is required")

    # Apply fallback if workspace wasn't in the ref
    if not workspace:
        if workspace_fallback:
            workspace = workspace_fallback
        else:
            raise FilesetPathError(
                f"Workspace required - provide in ref (e.g., 'workspace/{fileset}') or pass workspace_fallback"
            )

    return workspace, fileset, file_path


def parse_fileset_path(
    path: str,
    *,
    workspace_fallback: str | None = None,
) -> tuple[str, str | None, str]:
    """Parse a path that may or may not include a fileset reference.

    This is designed for SDK methods where the path parameter could be:
    - A full fileset ref: "workspace/fileset#path" or "fileset://..."
    - A pure file path: "data/file.txt" (when fileset/workspace are separate params)

    Args:
        path: The path to parse. May be a full fileset ref or just a file path.
        workspace_fallback: Workspace to use when workspace is not in the path.

    Returns:
        Tuple of (workspace, fileset, file_path).

    Raises:
        FilesetPathError: If workspace cannot be determined (not in path and no fallback).

    Examples:
        # Full fileset refs - delegates to parse_fileset_ref
        parse_fileset_path("workspace/fileset#data/file.txt")
            -> ("workspace", "fileset", "data/file.txt")
        parse_fileset_path("fileset://ws/fs#file.txt")
            -> ("ws", "fs", "file.txt")

        # Pure file paths - uses workspace_fallback
        parse_fileset_path("data/file.txt", workspace_fallback="default")
            -> ("default", None, "data/file.txt")
        parse_fileset_path("file.txt", workspace_fallback="default")
            -> ("default", None, "file.txt")
    """
    # If it has a # or fileset:// prefix, it's a fileset ref
    if "#" in path or path.startswith("fileset://"):
        return parse_fileset_ref(path, workspace_fallback=workspace_fallback)

    # Otherwise, treat the entire input as a file path
    if not workspace_fallback:
        raise FilesetPathError("workspace required when parsing a file path without fileset reference")
    return workspace_fallback, None, path


def build_fileset_ref(
    ref: str,
    *,
    workspace: str | None = None,
    fileset: str | None = None,
) -> str:
    """Build a fileset reference in canonical format: workspace/fileset[#file_path].

    Can normalize an existing ref or construct a new one from a file path.

    Args:
        ref: The fileset reference to normalize, or a file path if fileset is provided.
        workspace: Workspace to use if not specified in ref.
        fileset: Fileset to use if ref is a pure file path (no # separator).

    Returns:
        Ref in format "workspace/fileset#file_path" or "workspace/fileset".

    Raises:
        FilesetPathError: If workspace or fileset cannot be determined.

    Examples:
        # Normalizing existing refs
        build_fileset_ref("ws/fs#dir/file.txt")
            -> "ws/fs#dir/file.txt"
        build_fileset_ref("ws/fs/dir/file.txt")  # legacy format
            -> "ws/fs#dir/file.txt"
        build_fileset_ref("my-fileset#file.txt", workspace="default")
            -> "default/my-fileset#file.txt"

        # Constructing refs from file paths
        build_fileset_ref("data/file.txt", workspace="ws", fileset="fs")
            -> "ws/fs#data/file.txt"
        build_fileset_ref("", workspace="ws", fileset="fs")
            -> "ws/fs"
    """
    # If fileset provided and ref doesn't contain #, treat ref as a file path
    if fileset is not None and "#" not in ref:
        file_path = ref.lstrip("/")
        ws = workspace
        fs = fileset
        if not ws:
            raise FilesetPathError(f"workspace required when constructing ref from file path '{ref}'")
    else:
        # Parse as fileset ref
        ws, fs, file_path = parse_fileset_ref(ref, workspace_fallback=workspace)

    if file_path:
        return f"{ws}/{fs}#{file_path}"
    return f"{ws}/{fs}"


class FilesetFileSystem(AbstractFileSystem):
    """
    Synchronous fsspec filesystem for NeMo Platform fileset storage.

    URL format: fileset://[workspace/]fileset_name[#path]
    """

    protocol = "fileset"
    _client: FilesClient

    @classmethod
    def register_fsspec(cls) -> None:
        """Register the fileset protocol with fsspec.

        After calling this, you can use fsspec.filesystem("fileset", client=client).
        """
        from fsspec import register_implementation

        register_implementation(cls.protocol, cls, clobber=True)

    default_batch_size = 4
    blocksize = 16 * 1024 * 1024
    _fallback_timestamp = datetime.fromtimestamp(0, tz=timezone.utc)

    def __init__(
        self,
        *,
        client: FilesClient,
        batch_size: int | None = None,
        blocksize: int | None = None,
        **kwargs,
    ) -> None:
        if batch_size is None:
            batch_size = self.default_batch_size

        if blocksize is None:
            blocksize = self.blocksize

        super().__init__(**kwargs)
        self._client = client
        self.batch_size = batch_size
        self.blocksize = blocksize

    @property
    def _workspace(self) -> str | None:
        return self._client.workspace

    def to_fileset_files(self, results: dict[str, Any]) -> list[FilesetFileOutput]:
        """Convert fsspec find results to FilesetFileOutput objects."""
        files = []
        for name, info in results.items():
            if info.get("type") == "directory":
                continue
            workspace, fileset, file_path = parse_fileset_ref(name, workspace_fallback=None)
            files.append(
                FilesetFileOutput(
                    file_ref=f"{workspace}/{fileset}#{file_path}",
                    file_url=f"/apis/files/v2/workspaces/{workspace}/filesets/{fileset}/-/{file_path}",
                    path=file_path,
                    size=info.get("size", 0),
                )
            )
        return files

    def invalidate_cache(self, path: str | None = None) -> None:
        """Discard cached directory information."""
        if path is None:
            self.dircache.clear()
        else:
            self.dircache.pop(path.rstrip("/"), None)
        super().invalidate_cache(path)

    def created(self, path: str) -> datetime:
        self.info(path)
        return self._fallback_timestamp

    def modified(self, path: str) -> datetime:
        self.info(path)
        return self._fallback_timestamp

    def _populate_dircache_from_response(
        self,
        response,
        workspace: str,
        fileset: str,
        prefix: str,
    ) -> dict[str, list[FileInfo]]:
        """Parse recursive API response and populate dircache for all directory levels."""
        base_path = f"{workspace}/{fileset}"
        root = f"{base_path}#{prefix}" if prefix else base_path

        dir_contents: dict[str, list[FileInfo]] = {root: []}
        seen_subdirs: dict[str, set[str]] = {root: set()}

        for file_info in response.data:
            file_path = file_info.path.lstrip("/")
            full_path = f"{base_path}#{file_path}"

            parent = self._parent(full_path)
            if parent not in dir_contents:
                dir_contents[parent] = []
                seen_subdirs[parent] = set()
            dir_contents[parent].append({"name": full_path, "size": file_info.size, "type": "file"})

            current = parent
            while current != root and len(current) > len(root):
                parent_of_current = self._parent(current)
                if parent_of_current not in dir_contents:
                    dir_contents[parent_of_current] = []
                    seen_subdirs[parent_of_current] = set()
                subdir_name = current.split("#", 1)[1].rsplit("/", 1)[-1]
                if subdir_name not in seen_subdirs[parent_of_current]:
                    seen_subdirs[parent_of_current].add(subdir_name)
                    dir_contents[parent_of_current].append({"name": current, "size": 0, "type": "directory"})
                current = parent_of_current

        if self.dircache.use_listings_cache:
            for path, contents in dir_contents.items():
                self.dircache[path] = contents

        return dir_contents

    def info(self, path: str, **kwargs) -> FileInfo:
        """Get file info, using dircache when available."""
        _, _, file_path = parse_fileset_ref(path, workspace_fallback=self._workspace)
        path_key = build_fileset_ref(path)
        parent_path = self._parent(path_key)

        if path_key in self.dircache:
            return {"name": path_key, "size": 0, "type": "directory"}

        if parent_path != path_key and parent_path in self.dircache:
            for entry in self.dircache[parent_path]:
                if entry["name"].rstrip("/") == path_key:
                    return entry
            raise FileNotFoundError(path)

        if not file_path:
            try:
                self.ls(path_key, detail=True)
                return {"name": path_key, "size": 0, "type": "directory"}
            except Exception as e:
                raise FileNotFoundError(path) from e

        try:
            self.ls(parent_path, detail=True)
        except Exception as e:
            raise FileNotFoundError(path) from e

        if parent_path in self.dircache:
            for entry in self.dircache[parent_path]:
                if entry["name"].rstrip("/") == path_key:
                    return entry

        raise FileNotFoundError(path)

    def cat_file(self, path: str, start: int | None = None, end: int | None = None, **kwargs) -> bytes:
        """Fetch file content with optional byte range."""
        workspace, fileset, file_path = parse_fileset_ref(path, workspace_fallback=self._workspace)
        if not file_path:
            raise IsADirectoryError(path)

        client = self._client
        if start is not None or end is not None:
            client = client.with_headers({"Range": f"bytes={start or 0}-{(end - 1) if end else ''}"})

        response = client.download_file(workspace=workspace, name=fileset, path=file_path)
        return response.read()

    @classmethod
    def _parent(cls, path: str) -> str:
        """Get the parent directory path, handling the # separator correctly."""
        workspace, fileset, file_path = parse_fileset_ref(path.rstrip("/"), workspace_fallback=None)

        fileset_root = f"{workspace}/{fileset}"
        if not file_path:
            return fileset_root
        if "/" in file_path:
            return f"{fileset_root}#{file_path.rsplit('/', 1)[0]}"
        return fileset_root

    def ls(self, path: str, detail: bool = True, refresh: bool = False, **kwargs) -> list[FileInfo] | list[str]:
        """List files in a fileset or directory."""
        workspace, fileset, prefix = parse_fileset_ref(path, workspace_fallback=self._workspace)
        prefix = prefix.rstrip("/")
        path_key = build_fileset_ref(prefix, workspace=workspace, fileset=fileset)

        if self.dircache.use_listings_cache and not refresh:
            try:
                out = self.dircache[path_key]
                return out if detail else [f["name"] for f in out]
            except KeyError:
                pass

        query_params: ListFilesQueryParams | None = {"path": prefix} if prefix else None
        response = self._client.list_files(
            workspace=workspace,
            name=fileset,
            query_params=query_params,
        ).data()
        dir_contents = self._populate_dircache_from_response(response, workspace, fileset, prefix)

        result = dir_contents.get(path_key, [])
        return result if detail else [f["name"] for f in result]

    def rm_file(self, path: str, **kwargs) -> None:
        """Delete a single file."""
        workspace, fileset, file_path = parse_fileset_ref(path, workspace_fallback=self._workspace)
        if not file_path:
            raise ValueError("Cannot delete fileset root via rm")
        self._client.delete_file(workspace=workspace, name=fileset, path=file_path)
        self.invalidate_cache(self._parent(build_fileset_ref(path)))

    def pipe_file(self, path: str, value: bytes, mode: str = "overwrite", **kwargs) -> None:
        """Write bytes to a file."""
        workspace, fileset, file_path = parse_fileset_ref(path, workspace_fallback=self._workspace)
        if not file_path:
            raise ValueError("File path required for upload")
        self._client.upload_file(workspace=workspace, name=fileset, path=file_path, content=value)
        self.invalidate_cache(self._parent(build_fileset_ref(path)))

    def pipe_stream(
        self,
        path: str,
        stream: Iterator[bytes],
        content_length: int | None = None,
    ) -> None:
        """Write a sync byte stream to a file."""
        if hasattr(stream, "__anext__"):
            raise TypeError("FilesetFileSystem.pipe_stream requires a sync iterator")

        workspace, fileset, file_path = parse_fileset_ref(path, workspace_fallback=self._workspace)
        if not file_path:
            raise ValueError("File path required for upload")

        client = self._client
        if content_length is not None:
            client = client.with_headers({"Content-Length": str(content_length)})

        client.upload_file(workspace=workspace, name=fileset, path=file_path, content=stream)
        self.invalidate_cache(self._parent(build_fileset_ref(path)))

    def put(
        self,
        lpath,
        rpath,
        recursive=False,
        callback=DEFAULT_CALLBACK,
        maxdepth=None,
        **kwargs,
    ) -> None:
        """Copy local file(s) into the fileset."""
        kwargs.pop("batch_size", None)
        if isinstance(lpath, list) and isinstance(rpath, list):
            rpaths = rpath
            lpaths = lpath
        else:
            source_is_str = isinstance(lpath, str)
            if source_is_str:
                lpath = make_path_posix(lpath)
            fs = LocalFileSystem()
            lpaths = fs.expand_path(lpath, recursive=recursive, maxdepth=maxdepth)
            if source_is_str and (not recursive or maxdepth is not None):
                lpaths = [path for path in lpaths if not (trailing_sep(path) or fs.isdir(path))]
                if not lpaths:
                    return

            source_is_file = len(lpaths) == 1
            dest_is_dir = isinstance(rpath, str) and (trailing_sep(rpath) or self.isdir(rpath))

            rpath = self._strip_protocol(rpath)
            exists = source_is_str and (
                (has_magic(lpath) and source_is_file)
                or (not has_magic(lpath) and dest_is_dir and not trailing_sep(lpath))
            )
            rpaths = other_paths(
                lpaths,
                rpath,
                exists=exists,
                flatten=not source_is_str,
            )

        file_pairs = [(local, remote) for local, remote in zip(lpaths, rpaths) if not os.path.isdir(local)]

        callback.set_size(len(file_pairs))
        for local, remote in file_pairs:
            with callback.branched(local, remote) as child:
                self.put_file(local, remote, callback=child, **kwargs)
            callback.relative_update(1)

    def put_file(
        self,
        lpath: str,
        rpath: str,
        callback: Callback = DEFAULT_CALLBACK,
        mode: str = "overwrite",
        **kwargs,
    ) -> None:
        """Upload a local file to a fileset."""
        workspace, fileset, file_path = parse_fileset_ref(rpath, workspace_fallback=self._workspace)
        if not file_path:
            raise ValueError("File path required for upload")

        file_size = os.path.getsize(lpath)
        callback.set_size(file_size)

        def stream_file() -> Iterator[bytes]:
            with open(lpath, "rb") as f:
                while chunk := f.read(self.blocksize):
                    callback.relative_update(len(chunk))
                    yield chunk

        self._client.with_headers({"Content-Length": str(file_size)}).upload_file(
            workspace=workspace,
            name=fileset,
            path=file_path,
            content=stream_file(),
        )
        self.invalidate_cache(self._parent(build_fileset_ref(rpath)))

    @overload
    def find(
        self, path: str, maxdepth: int | None = None, withdirs: bool = False, detail: Literal[False] = ..., **kwargs
    ) -> list[str]: ...

    @overload
    def find(
        self, path: str, maxdepth: int | None = None, withdirs: bool = False, detail: Literal[True] = ..., **kwargs
    ) -> dict[str, FileInfo]: ...

    def find(
        self, path: str, maxdepth: int | None = None, withdirs: bool = False, detail: bool = False, **kwargs
    ) -> dict[str, FileInfo] | list[str]:
        """Find all files under path using a single recursive listing."""
        workspace, fileset, prefix = parse_fileset_ref(path, workspace_fallback=self._workspace)
        prefix = prefix.rstrip("/")
        query_params: ListFilesQueryParams | None = {"path": prefix} if prefix else None
        response = self._client.list_files(
            workspace=workspace,
            name=fileset,
            query_params=query_params,
        ).data()

        self._populate_dircache_from_response(response, workspace, fileset, prefix)

        out: dict[str, FileInfo] = {}
        seen_dirs: set[str] = set()

        if withdirs:
            root_path = build_fileset_ref(path, workspace=self._workspace)
            out[root_path] = {"name": root_path, "size": 0, "type": "directory"}

        for file_info in response.data:
            file_path = file_info.path.lstrip("/")
            full_path = f"{workspace}/{fileset}#{file_path}"
            out[full_path] = {"name": full_path, "size": file_info.size, "type": "file"}

            if withdirs:
                parts = file_path.split("/")
                for i in range(1, len(parts)):
                    dir_path = "/".join(parts[:i])
                    full_dir_path = f"{workspace}/{fileset}#{dir_path}"
                    if full_dir_path not in seen_dirs:
                        seen_dirs.add(full_dir_path)
                        out[full_dir_path] = {
                            "name": full_dir_path,
                            "size": 0,
                            "type": "directory",
                        }

        names = sorted(out)
        if detail:
            return {name: out[name] for name in names}
        return names

    def get_file(self, rpath: str, lpath: str, callback: Callback = DEFAULT_CALLBACK, outfile=None, **kwargs) -> None:
        """Download a file to local path."""
        workspace, fileset, file_path = parse_fileset_ref(rpath, workspace_fallback=self._workspace)

        if not file_path:
            return

        response = self._client.download_file(
            workspace=workspace,
            name=fileset,
            path=file_path,
        )

        with response.stream() as chunks:
            content_length = response.http_response.headers.get("content-length")
            if content_length:
                callback.set_size(int(content_length))
            close_file = outfile is None
            if outfile is None:
                Path(lpath).parent.mkdir(parents=True, exist_ok=True)
                f = open(lpath, "wb")
            else:
                f = outfile
            try:
                try:
                    for chunk in chunks:
                        f.write(chunk)
                        callback.relative_update(len(chunk))
                except httpx.StreamConsumed:
                    content = response.http_response.content
                    f.write(content)
                    callback.relative_update(len(content))
            finally:
                if close_file:
                    f.close()

    def get(
        self,
        rpath: str | list[str],
        lpath: str | list[str],
        recursive: bool = True,
        callback: Callback = DEFAULT_CALLBACK,
        maxdepth: int | None = None,
        **kwargs,
    ) -> None:
        """Download files using a single find call for efficiency."""
        kwargs.pop("batch_size", None)
        if isinstance(rpath, list) and isinstance(lpath, list):
            if not rpath:
                return
            callback.set_size(len(rpath))
            for remote, local in zip(rpath, lpath, strict=True):
                with callback.branched(remote, local) as child:
                    self.get_file(remote, local, callback=child, **kwargs)
                callback.relative_update(1)
            return

        if not isinstance(rpath, str) or not isinstance(lpath, str):
            raise TypeError("rpath and lpath must both be strings or both be lists")

        source_files = self.find(rpath, maxdepth=maxdepth, withdirs=False)
        if not source_files:
            return

        rpath_normalized = build_fileset_ref(rpath, workspace=self._workspace).rstrip("/")
        lpath_stripped = lpath.rstrip("/")
        source_is_file = len(source_files) == 1 and self._strip_protocol(source_files[0]) == rpath_normalized

        if source_is_file:
            source = source_files[0]
            dest_is_dir = lpath.endswith("/") or os.path.isdir(lpath)
            if dest_is_dir:
                _, _, source_file_path = parse_fileset_ref(source, workspace_fallback=None)
                filename = source_file_path.rsplit("/", 1)[-1]
                dest = f"{lpath_stripped}/{filename}"
            else:
                dest = lpath_stripped

            callback.set_size(1)
            with callback.branched(source, dest) as child:
                self.get_file(source, dest, callback=child, **kwargs)
            callback.relative_update(1)
            return

        _, _, file_path = parse_fileset_ref(rpath, workspace_fallback=self._workspace)
        copy_contents_directly = rpath.endswith("/") or not file_path
        source_name = file_path.rsplit("/", 1)[-1] if file_path else ""

        file_pairs = []
        for source in source_files:
            source_stripped = self._strip_protocol(source)
            relative = source_stripped[len(rpath_normalized) :].lstrip("#/")

            if copy_contents_directly:
                dest = f"{lpath_stripped}/{relative}"
            else:
                dest = f"{lpath_stripped}/{source_name}/{relative}"

            file_pairs.append((source, dest))

        callback.set_size(len(file_pairs))
        for source, dest in file_pairs:
            with callback.branched(source, dest) as child:
                self.get_file(source, dest, callback=child, **kwargs)
            callback.relative_update(1)

    def _open(
        self,
        path: str,
        mode: str = "rb",
        block_size: int | None = None,
        autocommit: bool = True,
        cache_options: dict | None = None,
        **kwargs,
    ) -> FilesetFile:
        """Open a file for reading or writing."""
        return FilesetFile(
            self,
            path,
            mode=mode,
            block_size=block_size or self.blocksize,
            autocommit=autocommit,
            cache_options=cache_options,
            **kwargs,
        )


class AsyncFilesetFileSystem(AsyncFileSystem):
    """
    Asynchronous fsspec filesystem for NeMo Platform fileset storage.

    URL format: fileset://[workspace/]fileset_name[#path]

    The optional `#` separator distinguishes the fileset name from the file path.
    If omitted, assumes root of fileset. Workspace is optional - if omitted,
    uses the client's default workspace.

    Examples:
        >>> from nemo_platform_plugin.files.client import AsyncFilesClient
        >>> client = AsyncFilesClient(base_url="http://localhost:8000", workspace="default")
        >>> fs = AsyncFilesetFileSystem(client=client)
        >>> await fs._ls("my-fileset")  # root of fileset, workspace from client default
        >>> await fs._ls("my-fileset#data/")  # specific path within fileset
        >>> await fs._ls("default/my-fileset#data/")  # explicit workspace
    """

    protocol = "fileset"
    _client: AsyncFilesClient

    # Default concurrency for file transfers
    default_batch_size = 4

    # Default block/chunk size for uploads/downloads
    blocksize = 16 * 1024 * 1024  # 16MB

    # The Files API does not currently expose file timestamps. Return a stable
    # fallback after confirming the path exists so fsspec consumers like DuckDB
    # do not fail on metadata lookups.
    _fallback_timestamp = datetime.fromtimestamp(0, tz=timezone.utc)

    def __init__(
        self,
        *,
        client: AsyncFilesClient,
        batch_size: int | None = None,
        blocksize: int | None = None,
        **kwargs,
    ) -> None:
        if batch_size is None:
            batch_size = self.default_batch_size

        if blocksize is None:
            blocksize = self.blocksize

        super().__init__(asynchronous=True, batch_size=batch_size, blocksize=blocksize, **kwargs)
        self._client = client

    @property
    def _workspace(self) -> str | None:
        return self._client.workspace

    def to_fileset_files(self, results: dict[str, Any]) -> list[FilesetFileOutput]:
        """Convert fsspec find results to FilesetFileOutput objects.

        Args:
            results: Dict from find(detail=True) mapping paths to file info.

        Returns:
            List of FilesetFileOutput objects with path, size, and file_ref.
        """
        files = []
        for name, info in results.items():
            if info.get("type") == "directory":
                continue
            workspace, fileset, file_path = parse_fileset_ref(name, workspace_fallback=None)
            files.append(
                FilesetFileOutput(
                    file_ref=f"{workspace}/{fileset}#{file_path}",
                    file_url=f"/apis/files/v2/workspaces/{workspace}/filesets/{fileset}/-/{file_path}",
                    path=file_path,
                    size=info.get("size", 0),
                )
            )
        return files

    def invalidate_cache(self, path: str | None = None) -> None:
        """Discard cached directory information."""
        if path is None:
            self.dircache.clear()
        else:
            self.dircache.pop(path.rstrip("/"), None)
        super().invalidate_cache(path)

    def created(self, path: str) -> datetime:
        self.info(path)
        return self._fallback_timestamp

    def modified(self, path: str) -> datetime:
        self.info(path)
        return self._fallback_timestamp

    def _populate_dircache_from_response(
        self,
        response,
        workspace: str,
        fileset: str,
        prefix: str,
    ) -> dict[str, list[FileInfo]]:
        """Parse recursive API response and populate dircache for all directory levels.

        The API returns a flat list of all files. We build directory listings for
        each level, adding both files and intermediate directory entries.

        Path format: workspace/fileset#file_path (using # to separate fileset from path)
        """
        base_path = f"{workspace}/{fileset}"
        # Root is the fileset root or a subdirectory within it
        root = f"{base_path}#{prefix}" if prefix else base_path

        dir_contents: dict[str, list[FileInfo]] = {root: []}
        seen_subdirs: dict[str, set[str]] = {root: set()}

        for file_info in response.data:
            file_path = file_info.path.lstrip("/")
            full_path = f"{base_path}#{file_path}"

            # Add file to its parent directory
            parent = self._parent(full_path)
            if parent not in dir_contents:
                dir_contents[parent] = []
                seen_subdirs[parent] = set()
            dir_contents[parent].append({"name": full_path, "size": file_info.size, "type": "file"})

            # Add intermediate directories (walk from parent up to root)
            current = parent
            while current != root and len(current) > len(root):
                parent_of_current = self._parent(current)
                if parent_of_current not in dir_contents:
                    dir_contents[parent_of_current] = []
                    seen_subdirs[parent_of_current] = set()
                # Note: current always has # here because the loop exits when current == root
                subdir_name = current.split("#", 1)[1].rsplit("/", 1)[-1]
                if subdir_name not in seen_subdirs[parent_of_current]:
                    seen_subdirs[parent_of_current].add(subdir_name)
                    dir_contents[parent_of_current].append({"name": current, "size": 0, "type": "directory"})
                current = parent_of_current

        if self.dircache.use_listings_cache:
            for path, contents in dir_contents.items():
                self.dircache[path] = contents

        return dir_contents

    async def _info(self, path: str, **kwargs) -> FileInfo:
        """Get file info, using dircache when available.

        Checks dircache first to avoid redundant API calls. For cache misses,
        uses _ls which populates the cache for all directory levels.
        """
        _, _, file_path = parse_fileset_ref(path, workspace_fallback=self._workspace)
        path_key = build_fileset_ref(path)
        parent_path = self._parent(path_key)

        # Check if this path is a cached directory (we've listed it before)
        if path_key in self.dircache:
            return {"name": path_key, "size": 0, "type": "directory"}

        # Check if this path exists in its parent's cached listing
        # Skip for fileset root (root is its own parent)
        if parent_path != path_key and parent_path in self.dircache:
            for entry in self.dircache[parent_path]:
                if entry["name"].rstrip("/") == path_key:
                    return entry
            # Parent was cached but this path wasn't in it
            raise FileNotFoundError(path)

        # Cache miss - fetch via _ls which populates cache
        if not file_path:
            # Fileset root - call _ls to populate cache and verify existence
            try:
                await self._ls(path_key, detail=True)
                return {"name": path_key, "size": 0, "type": "directory"}
            except Exception as e:
                raise FileNotFoundError(path) from e

        # File path - list parent directory (populates cache for siblings too)
        try:
            await self._ls(parent_path, detail=True)
        except Exception as e:
            raise FileNotFoundError(path) from e

        # Now check cache
        if parent_path in self.dircache:
            for entry in self.dircache[parent_path]:
                if entry["name"].rstrip("/") == path_key:
                    return entry

        raise FileNotFoundError(path)

    async def _cat_file(self, path: str, start: int | None = None, end: int | None = None, **kwargs) -> bytes:
        """Fetch file content with optional byte range."""
        workspace, fileset, file_path = parse_fileset_ref(path, workspace_fallback=self._workspace)
        if not file_path:
            raise IsADirectoryError(path)

        client = self._client
        if start is not None or end is not None:
            client = client.with_headers({"Range": f"bytes={start or 0}-{(end - 1) if end else ''}"})

        response = await client.download_file(workspace=workspace, name=fileset, path=file_path)
        return await response.read()

    @classmethod
    def _parent(cls, path: str) -> str:
        """Get the parent directory path, handling the # separator correctly.

        Override of fsspec's _parent to handle our path format where # separates
        the fileset name from the file path (e.g., workspace/fileset#dir/file.txt).
        The fileset root is its own parent (like / in Unix).
        """
        workspace, fileset, file_path = parse_fileset_ref(path.rstrip("/"), workspace_fallback=None)

        fileset_root = f"{workspace}/{fileset}"
        if not file_path:
            return fileset_root  # Root is its own parent
        if "/" in file_path:
            return f"{fileset_root}#{file_path.rsplit('/', 1)[0]}"
        return fileset_root

    async def _ls(self, path: str, detail: bool = True, refresh: bool = False, **kwargs) -> list[FileInfo] | list[str]:
        """List files in a fileset or directory.

        Uses dircache to avoid redundant API calls. The cache is populated for
        ALL directory levels found in the API response, so subsequent _ls calls
        for nested paths will hit the cache.

        Args:
            path: Path to list
            detail: If True, return list of dicts. If False, return list of paths.
            refresh: If True, bypass cache and fetch fresh listing.
        """
        workspace, fileset, prefix = parse_fileset_ref(path, workspace_fallback=self._workspace)
        prefix = prefix.rstrip("/")
        path_key = build_fileset_ref(prefix, workspace=workspace, fileset=fileset)

        # Check cache first (unless refresh requested)
        if self.dircache.use_listings_cache and not refresh:
            try:
                out = self.dircache[path_key]
                return out if detail else [f["name"] for f in out]
            except KeyError:
                pass

        # Fetch from backend and populate cache for all directory levels
        query_params: ListFilesQueryParams | None = {"path": prefix} if prefix else None
        response = await self._client.list_files(
            workspace=workspace,
            name=fileset,
            query_params=query_params,
        )
        response = response.data()
        dir_contents = self._populate_dircache_from_response(response, workspace, fileset, prefix)

        # Return the listing for the requested path
        result = dir_contents.get(path_key, [])
        return result if detail else [f["name"] for f in result]

    async def _rm_file(self, path: str, **kwargs) -> None:
        """Delete a single file."""
        workspace, fileset, file_path = parse_fileset_ref(path, workspace_fallback=self._workspace)
        if not file_path:
            raise ValueError("Cannot delete fileset root via rm")
        await self._client.delete_file(workspace=workspace, name=fileset, path=file_path)
        # Invalidate parent directory's cache since file info is stored there
        self.invalidate_cache(self._parent(build_fileset_ref(path)))

    async def _pipe_file(self, path: str, value: bytes, mode: str = "overwrite", **kwargs) -> None:
        """Write bytes to a file."""
        workspace, fileset, file_path = parse_fileset_ref(path, workspace_fallback=self._workspace)
        if not file_path:
            raise ValueError("File path required for upload")
        await self._client.upload_file(workspace=workspace, name=fileset, path=file_path, content=value)
        # Invalidate parent directory's cache since file info is stored there
        self.invalidate_cache(self._parent(build_fileset_ref(path)))

    async def _pipe_stream(
        self,
        path: str,
        stream: AsyncIterator[bytes] | Iterator[bytes],
        content_length: int | None = None,
    ) -> None:
        """Write a stream to a file.

        Uses httpx streaming upload to avoid buffering the entire content in memory.
        If content_length is not provided, uses chunked transfer encoding.

        Accepts both sync and async iterators. Sync iterators are wrapped as async
        because httpx's AsyncClient requires async iterators for streaming content.

        Args:
            path: Fileset path in format workspace/fileset#file_path
            stream: Sync or async iterator yielding byte chunks
            content_length: Optional content length for Content-Length header.
                If not provided, uses chunked transfer encoding.
        """
        workspace, fileset, file_path = parse_fileset_ref(path, workspace_fallback=self._workspace)
        if not file_path:
            raise ValueError("File path required for upload")

        # httpx AsyncClient requires async iterators for streaming content.
        if not hasattr(stream, "__anext__"):
            stream = to_async_iterator(stream)

        client = self._client
        if content_length is not None:
            client = client.with_headers({"Content-Length": str(content_length)})

        await client.upload_file(workspace=workspace, name=fileset, path=file_path, content=stream)

        # Invalidate parent directory's cache since file info is stored there
        self.invalidate_cache(self._parent(build_fileset_ref(path)))

    async def pipe_stream(
        self,
        path: str,
        stream: AsyncIterator[bytes] | Iterator[bytes],
        content_length: int | None = None,
    ) -> None:
        """Write a byte stream to a file."""
        await self._pipe_stream(path, stream, content_length)

    async def _put(
        self,
        lpath,
        rpath,
        recursive=False,
        callback=DEFAULT_CALLBACK,
        batch_size=None,
        maxdepth=None,
        **kwargs,
    ):
        """Copy local file(s) into the fileset."""
        if isinstance(lpath, list) and isinstance(rpath, list):
            rpaths = rpath
            lpaths = lpath
        else:
            source_is_str = isinstance(lpath, str)
            if source_is_str:
                lpath = make_path_posix(lpath)
            fs = LocalFileSystem()
            lpaths = fs.expand_path(lpath, recursive=recursive, maxdepth=maxdepth)
            if source_is_str and (not recursive or maxdepth is not None):
                lpaths = [path for path in lpaths if not (trailing_sep(path) or fs.isdir(path))]
                if not lpaths:
                    return

            source_is_file = len(lpaths) == 1
            dest_is_dir = isinstance(rpath, str) and (trailing_sep(rpath) or await self._isdir(rpath))

            rpath = self._strip_protocol(rpath)
            exists = source_is_str and (
                (has_magic(lpath) and source_is_file)
                or (not has_magic(lpath) and dest_is_dir and not trailing_sep(lpath))
            )
            rpaths = other_paths(
                lpaths,
                rpath,
                exists=exists,
                flatten=not source_is_str,
            )

        is_dir = {path: os.path.isdir(path) for path in lpaths}
        rdirs = [remote for local, remote in zip(lpaths, rpaths) if is_dir[local]]
        file_pairs = [(local, remote) for local, remote in zip(lpaths, rpaths) if not is_dir[local]]

        async with anyio.create_task_group() as tg:
            for directory in rdirs:
                tg.start_soon(self._makedirs, directory, True)

        callback.set_size(len(file_pairs))
        put_file = callback.branch_coro(self._put_file)
        await run_coros_in_chunks(
            [put_file(local, remote, **kwargs) for local, remote in file_pairs],
            batch_size=batch_size or self.batch_size,
            callback=callback,
        )

    async def _put_file(
        self,
        lpath: str,
        rpath: str,
        mode: str = "overwrite",
        callback: Callback = DEFAULT_CALLBACK,
        **kwargs,
    ) -> None:
        """Upload a local file to a fileset.

        Uses streaming upload to avoid buffering the entire file in memory.
        Supports per-chunk progress via callback.relative_update(chunk_size).
        """
        workspace, fileset, file_path = parse_fileset_ref(rpath, workspace_fallback=self._workspace)
        if not file_path:
            raise ValueError("File path required for upload")

        # Get file size for callback and Content-Length header
        file_size = (await anyio.Path(lpath).stat()).st_size
        callback.set_size(file_size)

        # Create async generator that streams file content with progress
        async def stream_file() -> AsyncIterator[bytes]:
            async with await anyio.open_file(lpath, "rb") as f:
                while chunk := await f.read(self.blocksize):
                    callback.relative_update(len(chunk))
                    yield chunk

        await self._client.with_headers({"Content-Length": str(file_size)}).upload_file(
            workspace=workspace,
            name=fileset,
            path=file_path,
            content=stream_file(),
        )
        # Invalidate parent directory's cache since file info is stored there
        self.invalidate_cache(self._parent(build_fileset_ref(rpath)))

    @overload
    async def _find(
        self, path: str, maxdepth: int | None = None, withdirs: bool = False, detail: Literal[False] = ..., **kwargs
    ) -> list[str]: ...

    @overload
    async def _find(
        self, path: str, maxdepth: int | None = None, withdirs: bool = False, detail: Literal[True] = ..., **kwargs
    ) -> dict[str, FileInfo]: ...

    async def _find(
        self, path: str, maxdepth: int | None = None, withdirs: bool = False, detail: bool = False, **kwargs
    ) -> dict[str, FileInfo] | list[str]:
        """Find all files under path using a single recursive listing.

        Also populates the dircache so subsequent _ls calls benefit.
        """
        workspace, fileset, prefix = parse_fileset_ref(path, workspace_fallback=self._workspace)
        prefix = prefix.rstrip("/")
        query_params: ListFilesQueryParams | None = {"path": prefix} if prefix else None
        response = await self._client.list_files(
            workspace=workspace,
            name=fileset,
            query_params=query_params,
        )
        response = response.data()

        # Populate dircache for all directory levels (benefits subsequent _ls calls)
        self._populate_dircache_from_response(response, workspace, fileset, prefix)

        # Build the flat output dict
        out: dict[str, FileInfo] = {}
        seen_dirs: set[str] = set()

        # Add root path if withdirs requested
        if withdirs:
            root_path = build_fileset_ref(path, workspace=self._workspace)
            out[root_path] = {"name": root_path, "size": 0, "type": "directory"}

        for file_info in response.data:
            file_path = file_info.path.lstrip("/")
            full_path = f"{workspace}/{fileset}#{file_path}"
            out[full_path] = {"name": full_path, "size": file_info.size, "type": "file"}

            # Add parent directories if withdirs requested
            if withdirs:
                parts = file_path.split("/")
                for i in range(1, len(parts)):
                    dir_path = "/".join(parts[:i])
                    full_dir_path = f"{workspace}/{fileset}#{dir_path}"
                    if full_dir_path not in seen_dirs:
                        seen_dirs.add(full_dir_path)
                        out[full_dir_path] = {
                            "name": full_dir_path,
                            "size": 0,
                            "type": "directory",
                        }

        names = sorted(out)
        if detail:
            return {name: out[name] for name in names}
        return names

    async def _get_file(self, rpath: str, lpath: str, callback: Callback = DEFAULT_CALLBACK, **kwargs) -> None:
        """Download a file to local path.

        Uses streaming response to avoid buffering the entire response in memory.
        Supports per-chunk progress via callback.relative_update(chunk_size).
        """
        workspace, fileset, file_path = parse_fileset_ref(rpath, workspace_fallback=self._workspace)

        if not file_path:
            return

        response = await self._client.download_file(
            workspace=workspace,
            name=fileset,
            path=file_path,
        )

        async with response.stream() as chunks:
            content_length = response.http_response.headers.get("content-length")
            if content_length:
                callback.set_size(int(content_length))
            await anyio.Path(lpath).parent.mkdir(parents=True, exist_ok=True)
            async with await anyio.open_file(lpath, "wb") as f:
                async for chunk in chunks:
                    await f.write(chunk)
                    callback.relative_update(len(chunk))

    async def _get(
        self,
        rpath: str | list[str],
        lpath: str | list[str],
        recursive: bool = True,
        callback: Callback = DEFAULT_CALLBACK,
        maxdepth: int | None = None,
        batch_size: int | None = None,
        **kwargs,
    ) -> None:
        """Download files using a single _find call for efficiency.

        Uses run_coros_in_chunks which provides proper task cancellation
        for the direct list-download path.

        When rpath and lpath are both lists, downloads each (remote, local) pair
        directly without path expansion. This is useful for downloading a specific
        set of files (e.g., from glob expansion in the SDK layer).
        """
        # Handle list inputs (pre-expanded paths from SDK layer)
        if isinstance(rpath, list) and isinstance(lpath, list):
            if not rpath:
                return
            callback.set_size(len(rpath))
            get_file_with_callback = callback.branch_coro(self._get_file)
            await run_coros_in_chunks(
                [get_file_with_callback(remote, local, **kwargs) for remote, local in zip(rpath, lpath, strict=True)],
                batch_size=batch_size or self.batch_size,
                callback=callback,
            )
            return

        if not isinstance(rpath, str) or not isinstance(lpath, str):
            raise TypeError("rpath and lpath must both be strings or both be lists")

        source_files = await self._find(rpath, maxdepth=maxdepth, withdirs=False)
        if not source_files:
            return

        # Normalize rpath to new format for comparison (since _find returns new format paths)
        rpath_normalized = build_fileset_ref(rpath, workspace=self._workspace).rstrip("/")
        lpath_stripped = lpath.rstrip("/")
        source_is_file = len(source_files) == 1 and self._strip_protocol(source_files[0]) == rpath_normalized

        # Single file download
        if source_is_file:
            source = source_files[0]
            # For single file: check if dest is a directory (trailing slash or existing dir)
            dest_is_dir = lpath.endswith("/") or await anyio.Path(lpath).is_dir()
            if dest_is_dir:
                # lpath="foo/" or existing dir - put file inside directory with original name
                # Extract filename from the file path portion (after #)
                _, _, source_file_path = parse_fileset_ref(source, workspace_fallback=None)
                filename = source_file_path.rsplit("/", 1)[-1]
                dest = f"{lpath_stripped}/{filename}"
            else:
                # lpath="foo" - save as this exact path
                dest = lpath_stripped

            callback.set_size(1)
            get_file_with_callback = callback.branch_coro(self._get_file)
            await get_file_with_callback(source, dest, **kwargs)
            return

        # Directory download - multiple files
        # Trailing slash on SOURCE controls whether to preserve source dir name:
        # - "source#subdir/" -> copy contents directly into dest
        # - "source#subdir"  -> create dest/subdir/ and copy contents there
        #
        # SPECIAL CASE: Fileset root (workspace/fileset with no file path) always
        # copies contents directly, matching HuggingFace Hub behavior. Users who want
        # to preserve the fileset name can include it in local_path.
        _, _, file_path = parse_fileset_ref(rpath, workspace_fallback=self._workspace)
        copy_contents_directly = rpath.endswith("/") or not file_path

        # Extract directory name from the file path portion (e.g., "subdir" from "a/b/subdir")
        source_name = file_path.rsplit("/", 1)[-1] if file_path else ""

        file_pairs = []
        for source in source_files:
            source_stripped = self._strip_protocol(source)
            # Strip rpath_normalized and any separator (# for root, / for subdirs)
            relative = source_stripped[len(rpath_normalized) :].lstrip("#/")

            if copy_contents_directly:
                # Source has trailing slash OR is fileset root: copy contents directly
                dest = f"{lpath_stripped}/{relative}"
            else:
                # Source has no trailing slash: preserve source directory name
                dest = f"{lpath_stripped}/{source_name}/{relative}"

            file_pairs.append((source, dest))

        callback.set_size(len(file_pairs))
        get_file_with_callback = callback.branch_coro(self._get_file)
        await run_coros_in_chunks(
            [get_file_with_callback(src, dst, **kwargs) for src, dst in file_pairs],
            batch_size=batch_size or self.batch_size,
            callback=callback,
        )

    def _open(
        self,
        path: str,
        mode: str = "rb",
        block_size: int | None = None,
        autocommit: bool = True,
        cache_options: dict | None = None,
        **kwargs,
    ) -> FilesetFile:
        """Open a file for reading or writing (sync)."""
        return FilesetFile(
            self,
            path,
            mode=mode,
            block_size=block_size or self.blocksize,
            autocommit=autocommit,
            cache_options=cache_options,
            **kwargs,
        )

    async def open_async(
        self,
        path: str,
        mode: str = "rb",
        block_size: int | None = None,
        **kwargs,
    ) -> AsyncFilesetFile:
        """Open a file for reading or writing (async)."""
        if "b" not in mode:
            raise ValueError("Only binary mode is supported for async open")
        return AsyncFilesetFile(self, path, mode=mode, block_size=block_size or self.blocksize, **kwargs)


class FilesetFile(AbstractBufferedFile):
    """Buffered file for sync reads and writes."""

    def _upload_chunk(self, final: bool = False) -> bool:
        """Upload buffer contents on final flush."""
        if final:
            self.fs.pipe_file(self.path, self.buffer.getvalue())
        return True


class AsyncFilesetFile(AbstractAsyncStreamedFile):
    """Async streamed file for async reads and writes."""

    async def _fetch_range(self, start: int, end: int) -> bytes:
        return await self.fs._cat_file(self.path, start=start, end=end)

    async def _upload_chunk(self, final: bool = False) -> bool:
        """Upload buffer contents on final flush."""
        if final:
            await self.fs._pipe_file(self.path, self.buffer.getvalue())
        return True


async def to_async_iterator(it: Iterator[bytes]) -> AsyncIterator[bytes]:
    """Convert a sync iterator to an async iterator without blocking the event loop.

    Runs each next() call in a thread pool so slow/large iterators don't block.
    """
    sentinel = object()
    while True:
        chunk = await to_thread.run_sync(next, it, sentinel)
        if chunk is sentinel:
            return
        yield chunk
