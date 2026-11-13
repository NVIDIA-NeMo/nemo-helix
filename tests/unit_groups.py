# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Split test files across CI runners by a stable hash of their path.

Shards skip other shards' files in ``pytest_ignore_collect``, so each runner also skips
importing them. Collection alone is several minutes of the full unit suite.
"""

import hashlib
from fnmatch import fnmatch
from pathlib import Path

import pytest

_SHARD = pytest.StashKey[tuple[int, int]]()


def shard_of(relative_path: str, total: int) -> int:
    """Return the 1-based shard that owns a test file."""
    return int(hashlib.sha256(relative_path.encode()).hexdigest(), 16) % total + 1


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--test-shard", default=None, metavar="I/N", help="Run only the test files in shard I of N.")


def pytest_configure(config: pytest.Config) -> None:
    value = config.getoption("test_shard")
    if value is None:
        return
    index, _, total = value.partition("/")
    if not (index.isdigit() and total.isdigit() and 1 <= int(index) <= int(total)):
        raise pytest.UsageError(f"--test-shard must be I/N with 1 <= I <= N, got {value!r}")
    config.stash[_SHARD] = (int(index), int(total))


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    shard = config.stash.get(_SHARD, None)
    if shard is None or not any(fnmatch(collection_path.name, p) for p in config.getini("python_files")):
        return None
    index, total = shard
    return True if shard_of(collection_path.relative_to(config.rootpath).as_posix(), total) != index else None
