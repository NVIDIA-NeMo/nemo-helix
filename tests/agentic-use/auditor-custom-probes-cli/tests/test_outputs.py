# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that the agent created a custom audit config with selected probes,
created an audit target, and invoked an audit with the custom config.

The inference provider is pre-configured by the Dockerfile setup script.
Tests focus on the custom probes workflow: config creation with specific probes,
target creation, and audit invocation with the custom config.

Tests:
- Audit target exists and references a model through the pre-configured provider
- Custom audit config exists with the three specified probes
- Agent trajectory shows config verification and audit invocation
"""

import base64
import json
import os

import pytest
from nemo_helix import NeMoHelix
from trace_reader import get_session

WORKSPACE = "default"
TARGET_NAME = "custom-audit-target"
CONFIG_NAME = "custom-probes-config"
EXPECTED_PROBES = ["dan.DanInTheWild", "dan.AutoDANCached", "dan.Ablation_Dan_11_0"]


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
def client() -> NeMoHelix:
    nhx_base_url = os.environ.get("NHX_BASE_URL", "http://localhost:8080")
    return NeMoHelix(base_url=nhx_base_url, workspace=WORKSPACE, access_token=_make_unsigned_jwt())


# --- Audit target checks ---


def test_audit_target_exists(client: NeMoHelix) -> None:
    """Verify the audit target was created."""
    targets = client.auditor.targets.list(workspace=WORKSPACE)
    target_names = [t["name"] for t in targets["data"]]
    assert TARGET_NAME in target_names, f"Target '{TARGET_NAME}' not found. Found: {target_names}"


def test_audit_target_model(client: NeMoHelix) -> None:
    """Verify the audit target references a model."""
    target = client.auditor.targets.get(workspace=WORKSPACE, name=TARGET_NAME)
    assert isinstance(target.model, str) and len(target.model) > 0, f"Target '{TARGET_NAME}' has no model configured"
    print(f"Target model: {target.model}, type: {target.type}")


# --- Custom audit config checks ---


def test_custom_config_exists(client: NeMoHelix) -> None:
    """Verify the custom audit config was created."""
    configs = client.auditor.configs.list(workspace=WORKSPACE)
    config_names = [c["name"] for c in configs["data"]]
    assert CONFIG_NAME in config_names, f"Config '{CONFIG_NAME}' not found. Found: {config_names}"


def test_custom_config_description(client: NeMoHelix) -> None:
    """Verify the custom config has the correct description."""
    config = client.auditor.configs.get(workspace=WORKSPACE, name=CONFIG_NAME)
    assert config.description == "Custom config with selected probes", (
        f"Expected description 'Custom config with selected probes', got '{config.description}'"
    )


def _get_probes_from_config(config: object) -> list[str]:
    """Extract probe names from a config, checking both probes dict and probe_spec string."""
    plugins = config.plugins

    # Check plugins.probes dict first (keys are probe names)
    probes_dict = plugins.probes if hasattr(plugins, "probes") else {}
    if isinstance(probes_dict, dict) and probes_dict:
        return list(probes_dict.keys())

    # Fall back to probe_spec (comma-separated string like "dan.DanInTheWild,dan.AutoDANCached")
    probe_spec = plugins.probe_spec if hasattr(plugins, "probe_spec") else ""
    if isinstance(probe_spec, str) and probe_spec and probe_spec != "all":
        return [p.strip() for p in probe_spec.split(",")]

    return []


def test_custom_config_has_selected_probes(client: NeMoHelix) -> None:
    """Verify the custom config uses the three specified probes."""
    config = client.auditor.configs.get(workspace=WORKSPACE, name=CONFIG_NAME)
    # Probes may be in plugins.probes dict (keys) or in probe_spec (comma-separated string)
    probe_names = _get_probes_from_config(config)

    for probe in EXPECTED_PROBES:
        assert probe in probe_names, f"Expected probe '{probe}' in config probes, got: {probe_names}"

    print(f"Config probes: {probe_names}")


# --- Audit invocation checks ---


def test_agent_invoked_audit_with_custom_config() -> None:
    """Verify the agent invoked the audit with the custom config and target."""
    session = get_session()
    commands = session.get_bash_commands()

    has_audit_run = any(
        _has_token(cmd, "auditor")
        and _has_token(cmd, "audit")
        and _has_token(cmd, "run")
        and CONFIG_NAME in cmd
        and TARGET_NAME in cmd
        for cmd in commands
    )

    assert has_audit_run, (
        f"Agent never invoked auditor audit run with '{CONFIG_NAME}' and '{TARGET_NAME}'. Commands: {commands}"
    )


# --- Agent trajectory checks ---


def _has_token(cmd: str, token: str) -> bool:
    """Check if token appears as a whole word in cmd (not as a substring of another word)."""
    import re

    return bool(re.search(r"(?:^|[\s/\-_.])(" + re.escape(token) + r")(?:[\s/\-_.]|$)", cmd))


def test_agent_verified_config() -> None:
    """Verify the agent retrieved the config to confirm probe selection."""
    session = get_session()
    commands = session.get_bash_commands()

    has_config_get = any(
        _has_token(cmd, "auditor") and _has_token(cmd, "configs") and _has_token(cmd, "get") for cmd in commands
    )

    has_config_list = any(
        _has_token(cmd, "auditor") and _has_token(cmd, "configs") and _has_token(cmd, "list") for cmd in commands
    )

    assert has_config_get or has_config_list, (
        "Agent never verified the custom config. Expected the agent to retrieve or list configs after creation."
    )


def test_agent_created_custom_config() -> None:
    """Verify the agent created the custom config via CLI."""
    session = get_session()
    commands = session.get_bash_commands()

    def has_command(*patterns: str) -> bool:
        return any(all(p in cmd for p in patterns) for cmd in commands)

    # Agent may create config with inline args or via --input-file with JSON
    # (when using --input-file, CONFIG_NAME is in a separate cat/echo command that writes the JSON file)
    has_inline_create = has_command("auditor", "configs", "create", CONFIG_NAME)
    has_file_create = any(
        all(p in cmd for p in ["auditor", "configs", "create"]) and CONFIG_NAME in cmd for cmd in commands
    )

    assert has_inline_create or has_file_create, f"Agent did not create '{CONFIG_NAME}'. Commands: {commands}"
