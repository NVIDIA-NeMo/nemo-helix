# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for workspaces CLI commands."""

from __future__ import annotations

import json
import uuid

import pytest
from nhx.testing import NemoRun, assert_exit_0

pytestmark = [pytest.mark.timeout(600)]


def test_workspaces_lifecycle(workspace: str, nemo_run: NemoRun) -> None:
    """Full workspaces lifecycle via CLI: create, list, get, update, delete."""
    name = f"e2e-cli-ws-{uuid.uuid4().hex[:8]}"

    result = nemo_run("workspaces", "create", name)
    assert_exit_0(result, "workspaces create failed")
    assert json.loads(result.stdout).get("name") == name

    result = nemo_run("workspaces", "list")
    assert_exit_0(result, "workspaces list failed")
    data = json.loads(result.stdout)
    assert "data" in data
    names = [item.get("name") for item in data["data"] if isinstance(item, dict)]
    assert name in names
    assert workspace in names

    result = nemo_run("workspaces", "get", name)
    assert_exit_0(result, "workspaces get failed")
    assert json.loads(result.stdout).get("name") == name

    result = nemo_run("workspaces", "update", name, "--description", "updated description")
    assert_exit_0(result, "workspaces update failed")
    assert json.loads(result.stdout).get("description") == "updated description"

    result = nemo_run("workspaces", "delete", name)
    assert_exit_0(result, "workspaces delete failed")

    # Workspace deletion is async — skip verify-absent since the workspace may still
    # appear briefly in list while the background cleanup controller processes it.
