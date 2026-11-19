# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared result-path matching for local copies and hosted uploads, using Git ignore syntax."""

from io import BytesIO
from pathlib import PurePosixPath

from dulwich.ignore import IgnoreFilter, read_ignore_patterns


class _IgnoreRules:
    """Git ignore rules for paths relative to a saved root; a pruned directory excludes its contents."""

    def __init__(self, patterns: list[str]) -> None:
        # Each item is one .gitignore line; the reader drops comments, blank lines, and trailing spaces.
        self._filter = IgnoreFilter(read_ignore_patterns(BytesIO("\n".join(patterns).encode())))
        self._pruned: dict[PurePosixPath, bool] = {}

    def excludes(self, path: PurePosixPath, *, is_directory: bool = False) -> bool:
        """Return whether Git would leave ``path`` out of a work-tree walk."""
        if is_directory:
            return self._is_pruned(path)
        return self._is_pruned(path.parent) or bool(self._filter.is_ignored(path.as_posix()))

    def _is_pruned(self, directory: PurePosixPath) -> bool:
        if directory == PurePosixPath("."):
            return False
        if directory not in self._pruned:
            # ``is_ignored("a/")`` is true for ``a/*``, yet Git still enters ``a/``; only pruning skips it.
            # ``may_prune_directory`` checks every prefix, so no parent recursion is needed here.
            self._pruned[directory] = self._filter.may_prune_directory(f"{directory.as_posix()}/")
        return self._pruned[directory]


def compile_ignore_rules(patterns: list[str] | str | None) -> _IgnoreRules | None:
    """Validate Git ignore patterns and build their matcher; return ``None`` when nothing is excluded."""
    if not patterns:
        return None
    patterns = [patterns] if isinstance(patterns, str) else patterns
    if any("\n" in pattern or "\r" in pattern for pattern in patterns):
        raise ValueError("Ignore patterns must be single lines.")
    return _IgnoreRules(patterns)
