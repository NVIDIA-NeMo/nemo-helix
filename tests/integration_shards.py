# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package-based integration shards, balanced using CI JUnit durations.

Select after normal collection/marker filtering so pytest.ini remains the source
of discovery paths. Default/local integration runs still execute the whole suite.
"""

from collections.abc import Generator
from pathlib import Path

import pytest


def integration_shard(path: Path) -> str:
    """Assign a repository-relative test path to exactly one shard.

    November 2026 CI timings: services without secrets ~2,150 aggregate seconds;
    packages/plugins/root tests plus secrets ~2,280 seconds. These are summed
    test durations, not wall time; keep packages intact when rebalancing.
    """
    if path.parts[:1] == ("services",) and path.parts[:3] != ("services", "core", "secrets"):
        return "services"
    return "platform"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--integration-shard",
        choices=("services", "platform"),
        default=None,
        help="Run one duration-balanced integration shard (requires -m integration).",
    )


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> Generator[None, None, None]:
    # Run last, after auto-marking, -m deselection, and service-pool expansion.
    yield
    shard = config.getoption("integration_shard")
    if shard is None:
        return
    if any(item.get_closest_marker("integration") is None for item in items):
        raise pytest.UsageError("--integration-shard requires integration-only collection (use -m integration)")
    selected = []
    deselected = []
    for item in items:
        target = selected if integration_shard(item.path.relative_to(config.rootpath)) == shard else deselected
        target.append(item)
    config.hook.pytest_deselected(items=deselected)
    items[:] = selected
