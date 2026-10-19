# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for members CLI commands."""

from __future__ import annotations

import json
import uuid

import pytest
from nhx.testing import NemoRun, assert_exit_0

pytestmark = [pytest.mark.timeout(600)]


def test_members_lifecycle(workspace: str, nemo_run: NemoRun) -> None:
    """Full members lifecycle via CLI: create, list, update, delete."""
    principal = f"e2e-cli-user-{uuid.uuid4().hex[:8]}@example.com"

    result = nemo_run(
        "workspaces",
        "members",
        "create",
        "--principal",
        principal,
        "--roles",
        "Viewer",
        workspace=workspace,
    )
    assert_exit_0(result, "members create failed")
    assert json.loads(result.stdout).get("principal") == principal

    result = nemo_run("workspaces", "members", "list", workspace=workspace)
    assert_exit_0(result, "members list failed")
    data = json.loads(result.stdout)
    assert "data" in data
    assert any(m["principal"] == principal for m in data["data"])

    result = nemo_run(
        "workspaces",
        "members",
        "update",
        principal,
        "--roles",
        "Editor",
        workspace=workspace,
    )
    assert_exit_0(result, "members update failed")
    assert "Editor" in json.loads(result.stdout).get("roles", [])

    result = nemo_run("workspaces", "members", "delete", principal, workspace=workspace)
    assert_exit_0(result, "members delete failed")

    result = nemo_run("workspaces", "members", "list", workspace=workspace)
    assert_exit_0(result, "members list failed")
    assert all(m["principal"] != principal for m in json.loads(result.stdout).get("data", []))
