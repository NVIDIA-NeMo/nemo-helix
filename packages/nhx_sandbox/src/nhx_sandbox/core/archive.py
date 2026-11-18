# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gzipped tar archives of directories, the format ``SandboxBackend.upload_dir`` and ``download_dir`` use.

A tar keeps what a file-by-file copy loses: permission bits, symlinks and empty directories. A script
that loses its execute bit on upload fails only later, when something runs it.

Extraction on the caller's side uses tarfile's ``data`` filter. An archive downloaded from a sandbox
was produced by untrusted code, so it may contain absolute paths, ``..`` members or links pointing
outside the target, and none of those may land outside it.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path


def pack_dir(source: Path) -> bytes:
    """Return the contents of ``source`` as a gzipped tar, with members relative to ``source``."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(source.rglob("*")):
            archive.add(path, arcname=path.relative_to(source).as_posix(), recursive=False)
    return buffer.getvalue()


def unpack_into(archive: bytes, target: Path) -> None:
    """Extract a gzipped tar into ``target``, refusing members that would land outside it.

    Raises:
        tarfile.TarError: The archive is malformed, or has a member the ``data`` filter refuses, such
            as an absolute path, a device file or a link out of ``target``.
    """
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        tar.extractall(target, filter="data")
