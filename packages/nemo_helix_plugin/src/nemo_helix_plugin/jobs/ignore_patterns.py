# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared result-path matching for local copies and hosted uploads."""

from fnmatch import fnmatchcase
from pathlib import Path


def matches_ignore_patterns(path: Path, patterns: list[str], *, is_directory: bool = False) -> bool:
    """Match a path relative to the saved root, treating trailing-slash patterns as directories.

    File patterns use ``Path.match`` and never prune directories. A single folder glob
    matches at any depth; multi-folder patterns start at the saved root.
    """
    directories = path.parts if is_directory else path.parts[:-1]
    for pattern in patterns:
        if pattern.endswith("/"):
            parts = Path(pattern.rstrip("/")).parts
            if len(parts) == 1:
                if any(fnmatchcase(directory, parts[0]) for directory in directories):
                    return True
            elif parts and len(directories) >= len(parts) and all(map(fnmatchcase, directories, parts)):
                return True
        elif not is_directory and path.match(pattern):
            return True
    return False
