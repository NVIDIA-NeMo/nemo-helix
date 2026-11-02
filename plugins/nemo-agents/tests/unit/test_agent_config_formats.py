# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest
from nemo_agents_plugin import agent_config_formats
from nemo_agents_plugin.agent_config_formats import (
    InvalidAgentConfigError,
    UnsupportedAgentConfigFormatError,
    resolve_agent_config_for_deployment,
    validate_agent_config,
)
from nemo_agents_plugin.entities import NAT_WORKFLOW_CONFIG_FORMAT, NEMO_AGENTS_SPEC_CONFIG_FORMAT


def _nemo_agents_config() -> dict[str, Any]:
    return {
        "config_format": NEMO_AGENTS_SPEC_CONFIG_FORMAT,
        "name": "test-agent",
        "default_harness": "hermes",
        "harnesses": {"hermes": {"kind": "hermes"}},
    }


def test_nat_config_validation_preserves_legacy_payload() -> None:
    config = {"workflow": {"_type": "chat_completion"}}

    validated = validate_agent_config(NAT_WORKFLOW_CONFIG_FORMAT, config)

    assert validated is config


def test_nemo_agents_config_validation_normalizes_payload() -> None:
    validated = validate_agent_config(NEMO_AGENTS_SPEC_CONFIG_FORMAT, _nemo_agents_config())

    assert validated["config_format"] == NEMO_AGENTS_SPEC_CONFIG_FORMAT
    assert validated["environment"]["provider"] == "local"


def test_nemo_agents_config_validation_rejects_invalid_payload() -> None:
    config = _nemo_agents_config()
    config["default_harness"] = "missing"

    with pytest.raises(InvalidAgentConfigError, match="Invalid agent config"):
        validate_agent_config(NEMO_AGENTS_SPEC_CONFIG_FORMAT, config)


@pytest.mark.usefixtures("installed_adapter_manifests")
def test_nemo_agents_config_validation_leaves_adapter_ids_to_deploy_time() -> None:
    """Registration can't know which image will run the agent, so an adapter only that image installs is accepted."""
    config = _nemo_agents_config()
    config["harnesses"]["acme"] = {"kind": "acme.custom.native"}

    validated = validate_agent_config(NEMO_AGENTS_SPEC_CONFIG_FORMAT, config)

    assert validated["harnesses"]["acme"]["kind"] == "acme.custom.native"


def test_unknown_config_format_is_rejected() -> None:
    with pytest.raises(UnsupportedAgentConfigFormatError, match="Unsupported config_format 'custom-v2'"):
        validate_agent_config("custom-v2", {})


@pytest.mark.parametrize("config_format", [NAT_WORKFLOW_CONFIG_FORMAT, NEMO_AGENTS_SPEC_CONFIG_FORMAT])
def test_empty_config_can_register_but_not_deploy(config_format: str) -> None:
    assert validate_agent_config(config_format, {}) == {}

    with pytest.raises(InvalidAgentConfigError, match="Agent 'default/test-agent' has no configuration"):
        resolve_agent_config_for_deployment(config_format, {}, workspace="default", agent_name="test-agent")


def test_nat_deployment_resolution_applies_legacy_injections(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Any]] = []

    def inject_gateway(config: dict[str, Any], workspace: str) -> dict[str, Any]:
        calls.append(("gateway", workspace))
        return {**config, "gateway": True}

    def inject_model(config: dict[str, Any]) -> dict[str, Any]:
        calls.append(("model", None))
        return {**config, "model": True}

    def inject_trace(config: dict[str, Any], *, workspace: str, agent_name: str) -> None:
        calls.append(("trace", (workspace, agent_name)))
        config["trace"] = True

    monkeypatch.setattr(agent_config_formats, "inject_gateway_url", inject_gateway)
    monkeypatch.setattr(agent_config_formats, "inject_default_model", inject_model)
    monkeypatch.setattr(agent_config_formats, "inject_nemo_trace_fields", inject_trace)

    resolved = resolve_agent_config_for_deployment(
        NAT_WORKFLOW_CONFIG_FORMAT,
        {"workflow": {}},
        workspace="test-workspace",
        agent_name="test-agent",
    )

    assert resolved == {"workflow": {}, "gateway": True, "model": True, "trace": True}
    assert calls == [
        ("gateway", "test-workspace"),
        ("model", None),
        ("trace", ("test-workspace", "test-agent")),
    ]


def test_nemo_agents_deployment_resolution_injects_gateway_and_normalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def inject_gateway(config: dict[str, Any], workspace: str) -> dict[str, Any]:
        calls.append(workspace)
        return config

    monkeypatch.setattr(agent_config_formats, "inject_fabric_gateway_url", inject_gateway)

    resolved = resolve_agent_config_for_deployment(
        NEMO_AGENTS_SPEC_CONFIG_FORMAT,
        _nemo_agents_config(),
        workspace="test-workspace",
        agent_name="test-agent",
    )

    assert resolved["config_format"] == NEMO_AGENTS_SPEC_CONFIG_FORMAT
    assert resolved["environment"]["provider"] == "local"
    assert "workflow" not in resolved
    assert calls == ["test-workspace"]


def test_nemo_agents_deployment_resolution_stamps_registered_agent_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_config_formats,
        "inject_fabric_gateway_url",
        lambda config, workspace: config,
    )

    resolved = resolve_agent_config_for_deployment(
        NEMO_AGENTS_SPEC_CONFIG_FORMAT,
        _nemo_agents_config(),
        workspace="test-workspace",
        agent_name="test-agent-hzwy9s",
    )

    # The spec's own ``name`` stays untouched; only telemetry is retagged so
    # per-agent trace queries match the Platform-registered name.
    assert resolved["name"] == "test-agent"
    assert resolved["telemetry"]["agent_name"] == "test-agent-hzwy9s"


def test_nemo_agents_deployment_resolution_keeps_explicit_telemetry_agent_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_config_formats,
        "inject_fabric_gateway_url",
        lambda config, workspace: config,
    )
    config = _nemo_agents_config()
    config["telemetry"] = {"enabled": True, "agent_name": "author-chosen-name"}

    resolved = resolve_agent_config_for_deployment(
        NEMO_AGENTS_SPEC_CONFIG_FORMAT,
        config,
        workspace="test-workspace",
        agent_name="test-agent-hzwy9s",
    )

    assert resolved["telemetry"]["agent_name"] == "author-chosen-name"
