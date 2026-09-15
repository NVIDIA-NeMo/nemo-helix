# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The file-source seam.

The profiler core reads dataset files only through a :class:`FileSource`, so it never touches a
storage API directly. :class:`LocalFileSource` covers a directory on disk; a ranged-read source over
the Files storage API is a later drop-in behind the same two methods.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol


@dataclass(frozen=True)
class FileEntry:
    """A file's identity — everything the profiler needs before it reads the contents."""

    path: str  # POSIX-style path relative to the source root
    size_bytes: int


class FileSource(Protocol):
    """Read-only access to a set of dataset files."""

    def list_files(self) -> list[FileEntry]:
        """Every file in the source, in a stable order."""
        ...

    def open(self, path: str) -> BinaryIO:
        """A binary, seekable stream for one file (``path`` as returned by :meth:`list_files`)."""
        ...


class LocalFileSource:
    """A directory of dataset files on the local filesystem."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        if not self._root.is_dir():
            raise NotADirectoryError(f"{self._root} is not a directory")

    def list_files(self) -> list[FileEntry]:
        # Symlinks are skipped rather than followed. `is_file()` resolves them and `open` follows
        # them, so a link planted inside the root reads a file outside it, and its column names reach
        # the profile. `rglob` already declines to descend into a symlinked directory, so this covers
        # the rest. A root that legitimately contains links is not a case the profiler needs to serve.
        entries = [
            FileEntry(path=path.relative_to(self._root).as_posix(), size_bytes=path.stat().st_size)
            for path in sorted(self._root.rglob("*"))
            if path.is_file() and not path.is_symlink()
        ]
        return entries

    def open(self, path: str) -> BinaryIO:
        """One file under the root, opened without following a link out of it.

        `list_files` already declines to *list* a symlink, and enforcing it only there left the rule
        holding for exactly as long as every caller kept coming through `list_files`. It is enforced
        here too because this is the method that reads:

        * ``self._root / path`` discards the root entirely when ``path`` is absolute, so
          ``open("/etc/hostname")`` read that file with no traversal sequence needed;
        * ``..`` walked out of the root the ordinary way;
        * and an entry listed as a regular file and replaced with a symlink before it was read was
          followed, which no check made before the open can catch.

        Nothing in the profiler reaches any of these today -- every caller passes a `FileEntry.path`
        built by `relative_to(self._root)`, which can hold neither. The Files-backed source this
        seam exists for changes that: the root becomes trusted and its *contents* become whatever an
        uploader put there, which is where a symlink planted in a tarball is the ordinary attack.
        """
        target = self._root / path
        if not target.resolve().is_relative_to(self._root.resolve()):
            raise ValueError(f"{path!r} resolves outside the source root")
        # `O_NOFOLLOW` rather than a check before the open: containment is verified against the path
        # as it was a moment ago, and the entry can be swapped underneath that. This is the one race
        # a caller cannot close for itself. It guards the final component only -- a symlinked
        # *directory* on the way to the file is still followed, which the containment check above
        # catches for the path as it stood but cannot catch for a swap after it.
        fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            return os.fdopen(fd, "rb")
        except BaseException:
            # `os.open` succeeded and `os.fdopen` did not, so nothing owns this descriptor yet and
            # the caller has no handle to close. The profiler opens a file per shard, so leaking one
            # per failure exhausts the process rather than the request.
            os.close(fd)
            raise
