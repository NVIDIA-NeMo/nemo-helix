# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a :class:`~nemo_helix_plugin.refs.LocalDir` to a usable :class:`Path`.

The context-manager shape mirrors :func:`fileset_path` so callers can
``with cls(...) as root:`` without caring which arm they took.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path

from nemo_helix_plugin.refs import LocalDir


class UsageSourceError(FileNotFoundError):
    """Raised when a local-path input does not exist or is not readable."""


@contextlib.contextmanager
def local_path(ref: LocalDir) -> Iterator[Path]:
    """Yield the resolved local path for *ref*.

    No staging or cleanup happens — the caller's working dir is used as-is.
    Existence and readability are checked here so the caller can rely on
    the yielded path being usable.
    """
    p = Path(str(ref)).expanduser().resolve()
    if not p.exists():
        raise UsageSourceError(f"local path does not exist: {p}")
    yield p
