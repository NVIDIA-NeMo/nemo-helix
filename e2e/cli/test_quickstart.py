# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for quickstart CLI commands."""

from __future__ import annotations

import pytest
from nhx.testing import NemoRun, assert_exit_0, run_nemo_local

pytestmark = [pytest.mark.timeout(300)]


@pytest.mark.platform("docker")
def test_quickstart_lifecycle() -> None:
    """Quickstart local Docker operations: status and logs.

    - status / logs use run_nemo_local (local Docker operations, no cluster URL needed).
    - logs are fetched only when a standard quickstart container is running; skipped otherwise
      (e.g. when pytest manages its own Docker backend with a different container name).
    """
    result = run_nemo_local("quickstart", "status")
    assert_exit_0(result, "quickstart status failed")
    out = result.stdout + result.stderr
    assert "running" in out.lower() or "status" in out.lower()

    is_running = "running: yes" in out.lower()

    if is_running:
        result = run_nemo_local("quickstart", "logs", "--tail", "5")
        assert_exit_0(result, "quickstart logs failed")


def test_cluster_info(nemo_run: NemoRun) -> None:
    """Cluster-info verifies the E2E cluster is reachable via the API."""
    result = nemo_run("cluster-info")
    assert_exit_0(result, "cluster-info failed")
    out = result.stdout + result.stderr
    assert "url" in out.lower() or "status" in out.lower() or "healthy" in out.lower()
