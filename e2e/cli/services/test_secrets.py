# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for secrets CLI commands."""

from __future__ import annotations

import json
import uuid

import pytest
from nhx.testing import NemoRun, assert_exit_0

pytestmark = [pytest.mark.timeout(600)]


def test_secrets_lifecycle(workspace: str, nemo_run: NemoRun) -> None:
    """Full secrets lifecycle via CLI: create, list, get, update, delete."""
    name = f"e2e-cli-secret-{uuid.uuid4().hex[:8]}"

    result = nemo_run(
        "secrets",
        "create",
        name,
        "--value",
        "initial-value",
        workspace=workspace,
    )
    assert_exit_0(result, "secrets create failed")
    assert json.loads(result.stdout).get("name") == name

    result = nemo_run("secrets", "list", workspace=workspace)
    assert_exit_0(result, "secrets list failed")
    assert "data" in json.loads(result.stdout)
    assert any(s["name"] == name for s in json.loads(result.stdout).get("data", []))

    result = nemo_run("secrets", "get", name, workspace=workspace)
    assert_exit_0(result, "secrets get failed")
    assert json.loads(result.stdout).get("name") == name

    result = nemo_run(
        "secrets",
        "update",
        name,
        "--description",
        "updated description",
        workspace=workspace,
    )
    assert_exit_0(result, "secrets update failed")
    assert json.loads(result.stdout).get("description") == "updated description"

    result = nemo_run("secrets", "delete", name, workspace=workspace)
    assert_exit_0(result, "secrets delete failed")

    result = nemo_run("secrets", "list", workspace=workspace)
    assert_exit_0(result, "secrets list failed")
    assert all(s["name"] != name for s in json.loads(result.stdout).get("data", []))
