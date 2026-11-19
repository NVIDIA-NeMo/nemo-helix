# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package-based integration groups, balanced using CI JUnit durations.

Select after normal collection/marker filtering so pytest.ini remains the source
of discovery paths. Default/local integration runs still execute the whole suite.

Under ``-m integration`` this plugin also skips importing test files that cannot hold
integration tests (or belong to the other group), so each xdist worker no longer imports
the ~35k unit tests only to deselect them.
"""

from collections.abc import Generator
from fnmatch import fnmatch
from pathlib import Path

import pytest


def integration_group(path: Path) -> str:
    """Assign a test path; secrets joins platform tests to balance the groups."""
    if path.parts[:1] == ("services",) and path.parts[:3] != ("services", "core", "secrets"):
        return "services"
    return "platform"


def may_hold_integration_tests(path: Path) -> bool:
    """Integration tests are auto-marked under an ``integration/`` dir or name the marker in source."""
    return "integration" in path.parts[:-1] or "integration" in path.read_text(errors="ignore")


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    if config.getoption("markexpr") != "integration" or not any(
        fnmatch(collection_path.name, pattern) for pattern in config.getini("python_files")
    ):
        return None
    group = config.getoption("integration_group")
    if group is not None and integration_group(collection_path.relative_to(config.rootpath)) != group:
        return True
    return None if may_hold_integration_tests(collection_path) else True


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--integration-group",
        choices=("services", "platform"),
        default=None,
        help="Run one duration-balanced integration group (requires -m integration).",
    )


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> Generator[None, None, None]:
    # Run last, after auto-marking, -m deselection, and service-pool expansion.
    yield
    group = config.getoption("integration_group")
    if group is None:
        return
    if any(item.get_closest_marker("integration") is None for item in items):
        raise pytest.UsageError("--integration-group requires integration-only collection (use -m integration)")
    selected = []
    deselected = []
    for item in items:
        target = selected if integration_group(item.path.relative_to(config.rootpath)) == group else deselected
        target.append(item)
    config.hook.pytest_deselected(items=deselected)
    items[:] = selected
