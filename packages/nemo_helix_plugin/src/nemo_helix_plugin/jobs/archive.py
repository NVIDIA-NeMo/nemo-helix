# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe tar extraction shared by plugin job-resource SDKs.

Plugins fetch artifact tarballs from the platform jobs service and unpack
them locally. The bytes arrive over HTTP so we validate every member up
front and reject anything that would escape ``output_path`` or create a
link / special file.
"""

from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Callable


def safe_extract_tar(
    tar: tarfile.TarFile,
    output_path: Path,
    error_cls: Callable[[str], BaseException] = ValueError,
) -> None:
    """Validate every member, then extract into ``output_path``.

    Two-pass so a rejected archive never leaves partial files behind. Per-member
    ``tar.extract()`` calls satisfy CodeQL's tar-slip taint tracking which does
    not follow a separate validation loop into a bulk ``extractall``.

    Args:
        tar: open :class:`tarfile.TarFile` to read from.
        output_path: destination directory; created if missing.
        error_cls: exception factory used when a member is rejected. Plugins
            pass their domain-specific error so callers see a consistent type.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    base_path = output_path.resolve()
    members = tar.getmembers()
    for member in members:
        target_path = (output_path / member.name).resolve()
        if target_path != base_path and base_path not in target_path.parents:
            raise error_cls(f"Refusing to extract unsafe tar member: {member.name}")
        if member.issym() or member.islnk():
            raise error_cls(f"Refusing to extract tar link member: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise error_cls(f"Refusing to extract special tar member: {member.name}")
    for member in members:
        tar.extract(member, path=output_path)
