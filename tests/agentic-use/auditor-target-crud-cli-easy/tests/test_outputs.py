# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Verify that Auditor target CRUD operations were performed correctly.

Checks:
- harbor-audit-target was deleted (should not exist)
- harbor-audit-target-final exists with correct model and type
- Agent trajectory shows all intermediate operations were executed
"""

import os

import pytest
from nemo_helix_plugin.auditor.client import AuditorClient
from trace_reader import get_session

WORKSPACE = "default"


@pytest.fixture
def client() -> AuditorClient:
    nhx_base_url = os.environ.get("NHX_BASE_URL", "http://localhost:8080")
    return AuditorClient(base_url=nhx_base_url, workspace=WORKSPACE)


def test_original_target_was_deleted(client: AuditorClient) -> None:
    """Verify that harbor-audit-target was successfully deleted."""
    targets = client.list_audit_targets(workspace=WORKSPACE)
    target_names = [t.name for t in targets.items()]

    assert "harbor-audit-target" not in target_names, (
        f"Target 'harbor-audit-target' should have been deleted but still exists! Found targets: {target_names}"
    )


def test_final_target_exists(client: AuditorClient) -> None:
    """Verify that harbor-audit-target-final was created with correct config."""
    target = client.get_audit_target(workspace=WORKSPACE, name="harbor-audit-target-final").data()

    assert target is not None, "Target 'harbor-audit-target-final' was not found!"
    assert target.name == "harbor-audit-target-final", (
        f"Expected target name 'harbor-audit-target-final', got '{target.name}'"
    )
    assert target.model == "final-model-endpoint", f"Expected model 'final-model-endpoint', got '{target.model}'"
    assert target.type == "openai", f"Expected type 'openai', got '{target.type}'"


def test_agent_performed_all_crud_operations() -> None:
    """
    Verify the agent executed all intermediate CRUD operations via CLI.

    Why this test exists:
    The tests above only verify final state (harbor-audit-target deleted,
    harbor-audit-target-final exists). But the task requires the agent to
    perform a full CRUD lifecycle on harbor-audit-target: create it, list
    targets, get it by name, update its description, then delete it.

    Since the auditor service does hard deletes with no audit trail, we
    cannot verify these intermediate operations via the API. Instead, we
    read the Claude Code session transcript to confirm the agent actually
    executed all the required CLI commands.
    """
    session = get_session()
    commands = session.get_bash_commands()

    # Helper to check if any single command contains all specified patterns
    def has_command(*patterns: str) -> bool:
        return any(all(p in cmd for p in patterns) for cmd in commands)

    # 1. Created harbor-audit-target
    assert has_command("auditor", "targets", "create", "harbor-audit-target"), (
        f"Agent did not create 'harbor-audit-target'. Commands: {commands}"
    )

    # 2. Listed audit targets
    assert has_command("auditor", "targets", "list"), f"Agent did not list audit targets. Commands: {commands}"

    # 3. Got harbor-audit-target by name
    assert has_command("auditor", "targets", "get", "harbor-audit-target"), (
        f"Agent did not get 'harbor-audit-target' by name. Commands: {commands}"
    )

    # 4. Updated harbor-audit-target description
    assert has_command("auditor", "targets", "update", "harbor-audit-target"), (
        f"Agent did not update 'harbor-audit-target'. Commands: {commands}"
    )

    # 5. Deleted harbor-audit-target
    assert has_command("auditor", "targets", "delete", "harbor-audit-target"), (
        f"Agent did not delete 'harbor-audit-target'. Commands: {commands}"
    )

    # 6. Created harbor-audit-target-final
    assert has_command("auditor", "targets", "create", "harbor-audit-target-final"), (
        f"Agent did not create 'harbor-audit-target-final'. Commands: {commands}"
    )

    print(f"Test passed: Agent performed all CRUD operations. Total commands: {len(commands)}")
