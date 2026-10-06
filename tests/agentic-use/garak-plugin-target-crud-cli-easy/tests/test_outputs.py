# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Verify that Garak Plugin target CRUD operations were performed correctly.

Checks:
- harbor-scan-target was deleted (should not exist)
- harbor-scan-target-final exists with correct model and type
- Agent trajectory shows all intermediate operations were executed
"""

import os

import pytest
from nemo_helix_plugin.garak_plugin.client import GarakPluginClient
from trace_reader import get_session

WORKSPACE = "default"


@pytest.fixture
def client() -> GarakPluginClient:
    nhx_base_url = os.environ.get("NHX_BASE_URL", "http://localhost:8080")
    return GarakPluginClient(base_url=nhx_base_url, workspace=WORKSPACE)


def test_original_target_was_deleted(client: GarakPluginClient) -> None:
    """Verify that harbor-scan-target was successfully deleted."""
    targets = client.list_scan_targets(workspace=WORKSPACE)
    target_names = [t.name for t in targets.items()]

    assert "harbor-scan-target" not in target_names, (
        f"Target 'harbor-scan-target' should have been deleted but still exists! Found targets: {target_names}"
    )


def test_final_target_exists(client: GarakPluginClient) -> None:
    """Verify that harbor-scan-target-final was created with correct config."""
    target = client.get_scan_target(workspace=WORKSPACE, name="harbor-scan-target-final").data()

    assert target is not None, "Target 'harbor-scan-target-final' was not found!"
    assert target.name == "harbor-scan-target-final", (
        f"Expected target name 'harbor-scan-target-final', got '{target.name}'"
    )
    assert target.model == "final-model-endpoint", f"Expected model 'final-model-endpoint', got '{target.model}'"
    assert target.type == "openai", f"Expected type 'openai', got '{target.type}'"


def test_agent_performed_all_crud_operations() -> None:
    """
    Verify the agent executed all intermediate CRUD operations via CLI.

    Why this test exists:
    The tests above only verify final state (harbor-scan-target deleted,
    harbor-scan-target-final exists). But the task requires the agent to
    perform a full CRUD lifecycle on harbor-scan-target: create it, list
    targets, get it by name, update its description, then delete it.

    Since the garak_plugin service does hard deletes with no scan trail, we
    cannot verify these intermediate operations via the API. Instead, we
    read the Claude Code session transcript to confirm the agent actually
    executed all the required CLI commands.
    """
    session = get_session()
    commands = session.get_bash_commands()

    # Helper to check if any single command contains all specified patterns
    def has_command(*patterns: str) -> bool:
        return any(all(p in cmd for p in patterns) for cmd in commands)

    # 1. Created harbor-scan-target
    assert has_command("garak-plugin", "targets", "create", "harbor-scan-target"), (
        f"Agent did not create 'harbor-scan-target'. Commands: {commands}"
    )

    # 2. Listed scan targets
    assert has_command("garak-plugin", "targets", "list"), f"Agent did not list scan targets. Commands: {commands}"

    # 3. Got harbor-scan-target by name
    assert has_command("garak-plugin", "targets", "get", "harbor-scan-target"), (
        f"Agent did not get 'harbor-scan-target' by name. Commands: {commands}"
    )

    # 4. Updated harbor-scan-target description
    assert has_command("garak-plugin", "targets", "update", "harbor-scan-target"), (
        f"Agent did not update 'harbor-scan-target'. Commands: {commands}"
    )

    # 5. Deleted harbor-scan-target
    assert has_command("garak-plugin", "targets", "delete", "harbor-scan-target"), (
        f"Agent did not delete 'harbor-scan-target'. Commands: {commands}"
    )

    # 6. Created harbor-scan-target-final
    assert has_command("garak-plugin", "targets", "create", "harbor-scan-target-final"), (
        f"Agent did not create 'harbor-scan-target-final'. Commands: {commands}"
    )

    print(f"Test passed: Agent performed all CRUD operations. Total commands: {len(commands)}")
