# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Path-based unit test groups, balanced using CI JUnit durations.

Groups skip other groups' files in ``pytest_ignore_collect``, so each CI runner also skips
importing them; collection alone is minutes of the full unit suite. Default/local runs
still execute the whole suite.
"""

from fnmatch import fnmatch
from pathlib import Path

import pytest

UNIT_GROUPS = ("core", "plugins", "misc")


def unit_group(path: Path) -> str:
    """Assign a test path relative to the repo root."""
    if path.parts[:2] == ("services", "core"):
        return "core"
    if path.parts[:1] == ("plugins",):
        return "plugins"
    return "misc"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--unit-group",
        choices=UNIT_GROUPS,
        default=None,
        help="Run only the test files in one duration-balanced unit group.",
    )


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    group = config.getoption("unit_group")
    if group is None or not any(fnmatch(collection_path.name, p) for p in config.getini("python_files")):
        return None
    return True if unit_group(collection_path.relative_to(config.rootpath)) != group else None
