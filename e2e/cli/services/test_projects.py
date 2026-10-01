# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for projects CLI commands."""

from __future__ import annotations

import json
import uuid

import pytest
from nhx.testing import NemoRun, assert_exit_0

pytestmark = [pytest.mark.timeout(600)]


def test_projects_lifecycle(workspace: str, nemo_run: NemoRun) -> None:
    """Full projects lifecycle via CLI: create, list, get, update, delete."""
    name = f"e2e-cli-proj-{uuid.uuid4().hex[:8]}"

    result = nemo_run("projects", "create", name, workspace=workspace)
    assert_exit_0(result, "projects create failed")
    assert json.loads(result.stdout).get("name") == name

    result = nemo_run("projects", "list", workspace=workspace)
    assert_exit_0(result, "projects list failed")
    assert "data" in json.loads(result.stdout)
    assert any(p["name"] == name for p in json.loads(result.stdout).get("data", []))

    result = nemo_run("projects", "get", name, workspace=workspace)
    assert_exit_0(result, "projects get failed")
    assert json.loads(result.stdout).get("name") == name

    result = nemo_run(
        "projects",
        "update",
        name,
        "--description",
        "updated description",
        workspace=workspace,
    )
    assert_exit_0(result, "projects update failed")
    assert json.loads(result.stdout).get("description") == "updated description"

    result = nemo_run("projects", "delete", name, workspace=workspace)
    assert_exit_0(result, "projects delete failed")

    result = nemo_run("projects", "list", workspace=workspace)
    assert_exit_0(result, "projects list failed")
    assert all(p["name"] != name for p in json.loads(result.stdout).get("data", []))
