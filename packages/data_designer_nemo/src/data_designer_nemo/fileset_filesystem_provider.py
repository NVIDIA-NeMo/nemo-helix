# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from os import PathLike
from pathlib import PurePosixPath
from typing import Self

from data_designer.engine.resources.seed_reader import (
    SeedReaderConfigError,
    SeedReaderError,
    SeedReaderFileSystemContext,
)
from data_designer_nemo.filesystem import make_filesystem
from filesets import FilesetFileSystem, FilesetPathError, build_fileset_ref, parse_fileset_ref
from fsspec.implementations.dirfs import DirFileSystem
from nemo_platform import NeMoPlatform


class _FilesetDirFileSystem(DirFileSystem):
    """DirFileSystem that handles FilesetFileSystem's '#' path separator.

    FilesetFileSystem returns paths using '#' to separate the fileset name from
    the file path (e.g. "ws/fs#data.parquet"). Standard DirFileSystem._relpath
    builds its strip-prefix with '/' (e.g. "ws/fs/"), so the startswith check
    fails for fileset-root paths. For subdirectory roots (e.g. "ws/fs#subdir"),
    files use '/' after '#' and the standard logic already works; the '#' branch
    below is a no-op in that case.

    All methods besides _relpath are inherited from DirFileSystem unchanged, so
    this remains a complete AbstractFileSystem implementation.
    """

    def _relpath(self, path: str | list) -> str | list:
        if isinstance(path, list):
            return [self._relpath(p) for p in path]
        if not self.path:
            return path
        if path == self.path:
            return ""
        for sep in ("#", "/"):
            prefix = self.path + sep
            if path.startswith(prefix):
                return path[len(prefix) :]
        raise AssertionError(f"Path {path!r} does not start with root {self.path!r}")


class _FilesetMetadataRootPath(PurePosixPath):
    """Metadata root path that preserves canonical fileset refs.

    Data Designer builds row metadata with ``context.root_path / relative_path``.
    For fileset-root seeds, a plain ``PurePosixPath("ws/fs")`` would produce
    ``ws/fs/data.parquet``. Fileset refs use ``#`` between the fileset and file
    path, so this metadata-only path adapter makes that first join render as
    ``ws/fs#data.parquet`` while later joins stay slash-separated.
    """

    def __truediv__(self, key: str | PathLike[str]) -> Self:
        if not key:
            return self
        separator = "/" if "#" in str(self) else "#"
        return type(self)(f"{self}{separator}{key}")


class FilesetFileSystemProvider:
    """Filesystem provider that roots directory-style seed readers in a fileset."""

    def __init__(
        self,
        sdk: NeMoPlatform,
        *,
        workspace: str,
        validated_roots: set[str] | None = None,
    ) -> None:
        self._sdk = sdk
        self._filesystem: FilesetFileSystem | None = None
        self._workspace = workspace
        self._validated_roots = set(validated_roots or ())

    def create_context(self, *, runtime_path: str) -> SeedReaderFileSystemContext:
        workspace, fileset, fragment = self._parse(runtime_path)
        root = build_fileset_ref(fragment, workspace=workspace, fileset=fileset)
        fs = self._get_filesystem()
        rooted_fs = _FilesetDirFileSystem(path=root, fs=fs)
        root_path = PurePosixPath(root) if fragment else _FilesetMetadataRootPath(root)
        return SeedReaderFileSystemContext(fs=rooted_fs, root_path=root_path)

    def ensure_root_exists(self, *, runtime_path: str) -> None:
        workspace, fileset, fragment = self._parse(runtime_path)
        root = build_fileset_ref(fragment, workspace=workspace, fileset=fileset)
        if root in self._validated_roots:
            return

        fs = self._get_filesystem()
        if fs.exists(root):
            self._validated_roots.add(root)
            return

        fileset_root = build_fileset_ref("", workspace=workspace, fileset=fileset)
        fully_qualified_fileset_name = f"{workspace}/{fileset}"
        if not fs.exists(fileset_root):
            raise SeedReaderConfigError(f"🛑 Fileset {fully_qualified_fileset_name!r} not found.")
        raise SeedReaderConfigError(f"🛑 Path {fragment!r} not found in fileset {fully_qualified_fileset_name!r}.")

    def _get_filesystem(self) -> FilesetFileSystem:
        if self._filesystem is not None:
            return self._filesystem

        filesystem = make_filesystem(self._sdk)
        self._filesystem = filesystem
        return filesystem

    def _parse(self, runtime_path: str) -> tuple[str, str, str]:
        try:
            return parse_fileset_ref(runtime_path, workspace_fallback=self._workspace)
        except FilesetPathError as error:
            raise SeedReaderError(f"🛑 Invalid fileset seed source path {runtime_path!r}: {error}") from error
