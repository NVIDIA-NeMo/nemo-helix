# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One build context, fetched as the in-cluster build fetches it, then packed for a remote build service.

The tar holds regular files and directories only, with ``Dockerfile`` at its root: no link, device or FIFO, so
nothing outside the context reaches the service. The same tree always packs to the same bytes.
"""

from __future__ import annotations

import gzip
import os
import shutil
import stat
import tarfile
from pathlib import Path, PurePosixPath

from nemo_builder_plugin.run.fetch import MAX_UNPACKED_BYTES, MAX_UNPACKED_ENTRIES, fetch_source
from nemo_builder_plugin.steps import ContextSource, WorkLayout
from nemo_helix_plugin.files.client import FilesClient

MAX_PACKED_ENTRIES = MAX_UNPACKED_ENTRIES
MAX_PACKED_BYTES = MAX_UNPACKED_BYTES
DOCKERFILE = "Dockerfile"


class ContextRefused(Exception):
    """A context this step won't send: one with a link or a special file, no root Dockerfile, or too large."""


def _context_dir(layout: WorkLayout, source: ContextSource) -> Path:
    """The context, refused if any directory on the way down from the fetched tree is a link or isn't there."""
    if source.archive is not None:
        path = Path(layout.archive(source.fileset, source.archive))
    else:
        path = Path(layout.fileset(source.fileset))
    relative = PurePosixPath(source.context_path or ".")
    if relative.is_absolute() or ".." in relative.parts:
        raise ContextRefused(f"context_path {source.context_path!r} must be relative, without '..'")
    for part in relative.parts:
        path = path / part
        if not stat.S_ISDIR(_mode(path)):
            raise ContextRefused(f"context_path {source.context_path!r} is not a directory in the source")
    return path


def _mode(path: Path) -> int:
    try:
        return path.lstat().st_mode
    except FileNotFoundError:
        return 0


def _entries(context: Path) -> list[tuple[str, bool]]:
    """Every entry under ``context`` as ``(name, is_dir)``, sorted; refuses links, special files and the caps."""
    entries: list[tuple[str, bool]] = []
    total = 0
    for directory, dirnames, filenames in os.walk(context, followlinks=False):
        for name in dirnames + filenames:
            path = Path(directory, name)
            relative = path.relative_to(context).as_posix()
            found = path.lstat()
            mode = found.st_mode
            if stat.S_ISREG(mode):
                total += found.st_size
            elif not stat.S_ISDIR(mode):
                kind = "a link" if stat.S_ISLNK(mode) else "not a regular file or directory"
                raise ContextRefused(f"{relative} is {kind}")
            entries.append((relative, stat.S_ISDIR(mode)))
            if len(entries) > MAX_PACKED_ENTRIES:
                raise ContextRefused(f"the context holds more than {MAX_PACKED_ENTRIES} entries")
            if total > MAX_PACKED_BYTES:
                raise ContextRefused(f"the context holds more than {MAX_PACKED_BYTES} bytes")
    if (DOCKERFILE, False) not in entries:
        raise ContextRefused(f"the context has no regular file {DOCKERFILE} at its root")
    return sorted(entries)


def pack_context(files: FilesClient, source: ContextSource, *, workspace: str, scratch: Path) -> Path:
    """Fetch ``source`` under ``scratch`` and pack it to ``scratch/context.tar.gz``, which is returned. What was
    fetched is removed once it is packed, so only the tar stays on disk."""
    work = scratch / "work"
    try:
        return _pack(files, source, workspace=workspace, work=work, packed=scratch / "context.tar.gz")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _pack(files: FilesClient, source: ContextSource, *, workspace: str, work: Path, packed: Path) -> Path:
    layout = WorkLayout(PurePosixPath(work))
    fetch_source(files, layout, source, workspace=workspace)
    context = _context_dir(layout, source)
    with packed.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for name, is_dir in _entries(context):
                info = tarfile.TarInfo(name)
                info.uid = info.gid = info.mtime = 0
                if is_dir:
                    info.type, info.mode = tarfile.DIRTYPE, 0o755
                    tar.addfile(info)
                    continue
                # A fresh regular member, read without following a link: a hard link packs as a copy.
                with open(os.open(context / name, os.O_RDONLY | os.O_NOFOLLOW), "rb") as file:
                    found = os.fstat(file.fileno())
                    if not stat.S_ISREG(found.st_mode):
                        raise ContextRefused(f"{name} changed while it was packed")
                    info.size = found.st_size
                    info.mode = 0o755 if found.st_mode & 0o111 else 0o644
                    tar.addfile(info, file)
    return packed
