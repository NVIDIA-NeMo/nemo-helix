# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that the agent created an audit target and invoked a default audit via CLI.

The inference provider is pre-configured by the Dockerfile setup script.
Tests focus on the auditor workflow: target creation, config creation, audit invocation.

Tests:
- Audit target exists and references a model through the pre-configured provider
- Audit config exists
- Agent trajectory shows config discovery and audit invocation
"""

import base64
import json
import os
import re

import pytest
from nemo_helix_plugin.auditor.client import AuditorClient
from trace_reader import get_session

WORKSPACE = "default"
TARGET_NAME = "audit-target"
CONFIG_NAME = "default"
TARGET_REF = f"{WORKSPACE}/{TARGET_NAME}"
CONFIG_REF = f"{WORKSPACE}/{CONFIG_NAME}"


def _make_unsigned_jwt() -> str:
    """Create an unsigned JWT (alg=none) for local quickstart auth."""
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode()).rstrip(b"=").decode()
    payload = (
        base64.urlsafe_b64encode(
            json.dumps({"sub": "verifier@harbor.local", "email": "verifier@harbor.local"}).encode()
        )
        .rstrip(b"=")
        .decode()
    )
    return f"{header}.{payload}."


@pytest.fixture
def client() -> AuditorClient:
    nhx_base_url = os.environ.get("NHX_BASE_URL", "http://localhost:8080")
    return AuditorClient(base_url=nhx_base_url, workspace=WORKSPACE, auth=_make_unsigned_jwt())


# --- Audit target checks ---


def test_audit_target_exists(client: AuditorClient) -> None:
    """Verify the audit target was created."""
    targets = client.list_audit_targets(workspace=WORKSPACE)
    target_names = [t.name for t in targets.items()]
    assert TARGET_NAME in target_names, f"Target '{TARGET_NAME}' not found. Found: {target_names}"


def test_audit_target_model(client: AuditorClient) -> None:
    """Verify the audit target references the correct model."""
    target = client.get_audit_target(workspace=WORKSPACE, name=TARGET_NAME).data()
    assert target.model is not None and len(target.model) > 0, f"Target '{TARGET_NAME}' has no model configured"
    print(f"Target model: {target.model}, type: {target.type}")


# --- Audit config checks ---


def test_audit_config_exists(client: AuditorClient) -> None:
    """Verify the audit config was created."""
    config = client.get_audit_config(workspace=WORKSPACE, name=CONFIG_NAME).data()
    assert config.name == CONFIG_NAME, f"Expected config '{CONFIG_NAME}', got '{config.name}'"


def test_audit_config_has_probe_spec(client: AuditorClient) -> None:
    """Verify the audit config has probes configured."""
    config = client.get_audit_config(workspace=WORKSPACE, name=CONFIG_NAME).data()
    assert config.plugins["probe_spec"], f"Config '{CONFIG_NAME}' has no probe_spec configured"
    print(f"Config probe_spec: {config.plugins['probe_spec']}")


# --- Agent trajectory checks ---


def _has_token(cmd: str, token: str) -> bool:
    """Check if token appears as a command-ish token, not as an arbitrary substring."""
    return bool(re.search(r"(?:^|[\s/\-_.])(" + re.escape(token) + r")(?:[\s/\-_.]|$)", cmd))


def _has_token_sequence(cmd: str, *tokens: str) -> bool:
    words = cmd.split()
    return any(tuple(words[i : i + len(tokens)]) == tokens for i in range(len(words) - len(tokens) + 1))


def test_agent_invoked_audit() -> None:
    """Verify the agent invoked the audit with the config and target."""
    session = get_session()
    commands = session.get_bash_commands()

    has_audit_run = any(
        _has_token_sequence(cmd, "auditor", "audit", "run") and CONFIG_REF in cmd and TARGET_REF in cmd
        for cmd in commands
    )

    assert has_audit_run, (
        f"Agent never invoked auditor audit run with '{CONFIG_REF}' and '{TARGET_REF}'. Commands: {commands}"
    )


def test_agent_created_target() -> None:
    """Verify the agent created the audit target via CLI."""
    session = get_session()
    commands = session.get_bash_commands()

    has_target_create = any(
        _has_token_sequence(cmd, "auditor", "targets", "create") and _has_token(cmd, TARGET_NAME) for cmd in commands
    )

    assert has_target_create, f"Agent did not create '{TARGET_NAME}'. Commands: {commands}"


def test_agent_created_config() -> None:
    """Verify the agent created or interacted with an audit config."""
    session = get_session()
    commands = session.get_bash_commands()

    has_config_create = any(
        (
            _has_token_sequence(cmd, "auditor", "configs", "create")
            or _has_token_sequence(cmd, "auditor", "config", "create")
        )
        for cmd in commands
    )

    has_config_list = any(
        (
            _has_token_sequence(cmd, "auditor", "configs", "list")
            or _has_token_sequence(cmd, "auditor", "config", "list")
        )
        for cmd in commands
    )

    has_config_get = any(
        (
            _has_token_sequence(cmd, "auditor", "configs", "get")
            or _has_token_sequence(cmd, "auditor", "config", "get")
            or (
                (_has_token_sequence(cmd, "auditor", "configs") or _has_token_sequence(cmd, "auditor", "config"))
                and CONFIG_REF in cmd
            )
        )
        for cmd in commands
    )

    assert has_config_create or has_config_list or has_config_get, (
        "Agent never created or discovered an audit config. Expected the agent to create or find the default config."
    )
