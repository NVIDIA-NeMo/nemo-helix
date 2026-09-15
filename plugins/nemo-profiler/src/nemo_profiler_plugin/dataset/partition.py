# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Partition grouping.

A partition is a group of files profiled as a unit. This stage groups by top-level directory; a
later stage refines partitions whose files turn out to disagree on column schema.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from nemo_profiler_plugin.dataset.splits import is_split_directory
from nemo_profiler_plugin.source import FileEntry


def _top_dir(path: str) -> str:
    """The partition a file belongs to: its first path segment, or ``""`` for a root-level file.

    Empty works as a name because no directory can be called it, so root-level files never collide
    with a directory named ``default``.

    Split directories (``train/``, ``test/``) are excluded. Grouping on them would put one dataset's
    train and test in separate partitions, each with its own schema and classification, which is
    what :mod:`splits` models instead.
    """
    parts = PurePosixPath(path).parts
    if len(parts) <= 1 or is_split_directory(parts[0]):
        return ""
    return parts[0]


def group_partitions(entries: list[FileEntry]) -> list[tuple[str, list[FileEntry]]]:
    """Group files into (name, files) partitions by top-level directory, sorted by name.

    The name is the shared path prefix, not a label derived from it: a lone group under ``data/`` is
    named ``"data"``, not ``"default"``, which would discard the only thing identifying it.

    Files under a split directory group under ``""`` with root-level files, since those are one
    dataset's splits rather than separate partitions.
    """
    by_dir: dict[str, list[FileEntry]] = {}
    for entry in entries:
        by_dir.setdefault(_top_dir(entry.path), []).append(entry)

    if len(by_dir) == 1:
        # One group is one partition, keeping whatever directory it came from.
        return [(next(iter(by_dir)), list(entries))]

    return sorted(by_dir.items())
