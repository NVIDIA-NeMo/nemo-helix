# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for guardrail CLI commands."""

from __future__ import annotations

import json
import uuid

import pytest
from nhx.testing import NemoRun, assert_exit_0

pytestmark = [pytest.mark.timeout(600)]


def test_guardrail_config_lifecycle(workspace: str, nemo_run: NemoRun) -> None:
    """Full guardrail config lifecycle via CLI: create, list, get, update, delete."""
    name = f"e2e-cli-guard-{uuid.uuid4().hex[:8]}"

    result = nemo_run("guardrail", "configs", "create", name, workspace=workspace)
    assert_exit_0(result, "guardrail configs create failed")
    assert json.loads(result.stdout).get("name") == name

    result = nemo_run("guardrail", "configs", "list", workspace=workspace)
    assert_exit_0(result, "guardrail configs list failed")
    data = json.loads(result.stdout)
    assert "data" in data
    assert any(c["name"] == name for c in data["data"])

    result = nemo_run("guardrail", "configs", "get", name, workspace=workspace)
    assert_exit_0(result, "guardrail configs get failed")
    assert json.loads(result.stdout).get("name") == name

    result = nemo_run(
        "guardrail",
        "configs",
        "update",
        name,
        "--description",
        "updated description",
        workspace=workspace,
    )
    assert_exit_0(result, "guardrail configs update failed")
    assert json.loads(result.stdout).get("description") == "updated description"

    result = nemo_run("guardrail", "configs", "delete", name, workspace=workspace)
    assert_exit_0(result, "guardrail configs delete failed")

    result = nemo_run("guardrail", "configs", "list", workspace=workspace)
    assert_exit_0(result, "guardrail configs list failed")
    assert all(c["name"] != name for c in json.loads(result.stdout).get("data", []))
