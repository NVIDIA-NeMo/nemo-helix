# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Verify that auditor config CRUD operations were performed correctly.

Checks:
- harbor-test-config was deleted (should not exist)
- harbor-final-config exists with correct description and probe spec
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


def test_harbor_test_config_deleted(client: AuditorClient) -> None:
    """Test that harbor-test-config was deleted after CRUD operations."""
    configs = client.list_audit_configs(workspace=WORKSPACE)
    config_names = [c.name for c in configs.items()]
    assert "harbor-test-config" not in config_names, (
        f"Config 'harbor-test-config' should have been deleted but still exists! Found: {config_names}"
    )


def test_harbor_final_config_exists(client: AuditorClient) -> None:
    """Test that harbor-final-config was created and has correct metadata."""
    response = client.get_audit_config(workspace=WORKSPACE, name="harbor-final-config").data()
    assert response.name == "harbor-final-config", f"Expected config name 'harbor-final-config', got '{response.name}'"
    assert response.description == "Final config for verification", (
        f"Expected description 'Final config for verification', got '{response.description}'"
    )


def test_harbor_final_config_probe_spec(client: AuditorClient) -> None:
    """Test that harbor-final-config uses the dan.DanInTheWild probe."""
    response = client.get_audit_config(workspace=WORKSPACE, name="harbor-final-config").data()
    probe_spec = response.plugins["probe_spec"]
    assert "dan.DanInTheWild" in probe_spec, f"Expected probe_spec to contain 'dan.DanInTheWild', got '{probe_spec}'"


def test_agent_performed_crud_operations() -> None:
    """
    Verify the agent executed the expected CRUD operations via trajectory analysis.

    This is needed because the API does hard deletes with no audit trail,
    so we cannot verify intermediate operations via the API alone.
    """
    session = get_session()
    commands = session.get_bash_commands()

    def has_command(*patterns: str) -> bool:
        return any(all(p in cmd for p in patterns) for cmd in commands)

    # Agent should have listed global configs
    assert has_command("auditor", "configs", "list"), f"Agent did not list audit configs. Commands: {commands}"

    # Agent should have created harbor-test-config
    assert has_command("auditor", "configs", "create", "harbor-test-config"), (
        f"Agent did not create 'harbor-test-config'. Commands: {commands}"
    )

    # Agent should have retrieved the config
    assert has_command("auditor", "configs", "get", "harbor-test-config"), (
        f"Agent did not retrieve 'harbor-test-config'. Commands: {commands}"
    )

    # Agent should have updated the config
    assert has_command("auditor", "configs", "update", "harbor-test-config"), (
        f"Agent did not update 'harbor-test-config'. Commands: {commands}"
    )

    # Agent should have deleted harbor-test-config
    assert has_command("auditor", "configs", "delete", "harbor-test-config"), (
        f"Agent did not delete 'harbor-test-config'. Commands: {commands}"
    )

    # Agent should have created harbor-final-config
    assert has_command("auditor", "configs", "create", "harbor-final-config"), (
        f"Agent did not create 'harbor-final-config'. Commands: {commands}"
    )

    print(f"Test passed: Agent performed all CRUD operations. Total commands: {len(commands)}")
