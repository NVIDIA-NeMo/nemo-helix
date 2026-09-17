# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the agent resolver's pure logic and end-to-end manifest build.

The orchestrator is exercised with a fake SDK (plain dicts, matching what
``client.agents.get`` / ``client.agents.deployments.list`` return), so no live platform or
agent-hardener install is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from nemo_agent_hardener_plugin.agent_resolver import (
    AgentResolutionError,
    derive_secret_names,
    detect_custom_components,
    gateway_backend,
    inject_gateway_url,
    parse_agent_ref,
    resolve_agent_to_manifest,
    strip_platform_telemetry,
)

# `list` is shadowed by the fake's own `list` method inside the class body, so the parameter type
# is aliased at module level.
DeploymentRows = list[dict[str, Any]]


class _FakeDeployments:
    def __init__(self, deployments: DeploymentRows) -> None:
        self._deployments = deployments

    def list(self, workspace: str = "default") -> dict:
        # Mirror the real SDK shape: a paginated dict, not a bare list.
        return {"data": self._deployments, "pagination": {"total_results": len(self._deployments)}}


class _FakeAgents:
    def __init__(self, agent: dict | None, deployments: list[dict]) -> None:
        self._agent = agent
        self.deployments = _FakeDeployments(deployments)

    def get(self, name: str, workspace: str = "default") -> dict:
        if self._agent is None:
            raise KeyError(name)
        return self._agent


class _FakeSDK:
    def __init__(self, agent: dict | None, deployments: list[dict] | None = None) -> None:
        self.agents = _FakeAgents(agent, deployments or [])


# --------------------------------------------------------------------------- parse_agent_ref
def test_parse_agent_ref_plain_name_uses_default_workspace():
    assert parse_agent_ref("calculator", "default") == ("default", "calculator")


def test_parse_agent_ref_qualified():
    assert parse_agent_ref("team-a/calculator", "default") == ("team-a", "calculator")


def test_parse_agent_ref_rejects_url():
    with pytest.raises(AgentResolutionError):
        parse_agent_ref("http://localhost:9001", "default")


# --------------------------------------------------------------------------- inject_gateway_url
def test_inject_gateway_url_sets_base_url_for_openai_and_nim():
    config = {"llms": {"a": {"_type": "openai"}, "b": {"_type": "nim"}, "c": {"_type": "custom"}}}
    out = inject_gateway_url(config, "ws1", "http://host:8080/")
    expected = "http://host:8080/apis/inference-gateway/v2/workspaces/ws1/openai/-/v1"
    assert out["llms"]["a"]["base_url"] == expected
    assert out["llms"]["b"]["base_url"] == expected
    assert "base_url" not in out["llms"]["c"]  # non-IGW type untouched
    assert config["llms"]["a"] == {"_type": "openai"}  # original not mutated


def test_inject_gateway_url_preserves_explicit_base_url():
    config = {"llms": {"a": {"_type": "openai", "base_url": "http://explicit"}}}
    out = inject_gateway_url(config, "ws1", "http://host:8080")
    assert out["llms"]["a"]["base_url"] == "http://explicit"


# --------------------------------------------------------------------------- gateway_backend
def test_gateway_backend_declared_for_local_gateway():
    assert gateway_backend("http://localhost:8080") == {"name": "nemo-inference-gateway", "ports": [8080]}
    assert gateway_backend("http://127.0.0.1:9000") == {"name": "nemo-inference-gateway", "ports": [9000]}


def test_gateway_backend_none_for_remote_gateway():
    assert gateway_backend("https://gateway.example.com") is None


# --------------------------------------------------------------------------- detect_custom_components
def test_detect_custom_components_flags_dotted_and_colon_types():
    config = {
        "functions": {"f1": {"_type": "my_pkg.tools:search"}, "f2": {"_type": "current_datetime"}},
        "workflow": {"_type": "react_agent"},
    }
    assert detect_custom_components(config) == ["my_pkg.tools:search"]


def test_detect_custom_components_empty_for_config_only_agent():
    config = {"functions": {"f": {"_type": "current_datetime"}}, "workflow": {"_type": "react_agent"}}
    assert detect_custom_components(config) == []


# --------------------------------------------------------------------------- derive_secret_names
def test_derive_secret_names_finds_env_refs_and_falls_back():
    config = {"functions": {"gh": {"_type": "github", "github_token": "${GITHUB_TOKEN}"}}}
    assert "GITHUB_TOKEN" in derive_secret_names(config)


def test_derive_secret_names_default_when_none_found():
    assert derive_secret_names({"workflow": {"_type": "react_agent"}}) == ["INFERENCE_API_KEY"]


# --------------------------------------------------------------------------- resolve_agent_to_manifest
def test_resolve_config_only_agent_builds_manifest_and_scaffolds(tmp_path):
    agent = {"config": {"llms": {"main": {"_type": "openai"}}, "workflow": {"_type": "react_agent"}}}
    sdk = _FakeSDK(
        agent,
        deployments=[{"agent": "calc", "status": "running", "port": 9123}],
    )
    resolved = resolve_agent_to_manifest(
        "calc",
        sdk=sdk,
        base_url="http://host:8080",
        default_workspace="default",
        manifest_dir=tmp_path,
    )
    # Port taken from the running deployment.
    assert resolved.port == 9123
    # Manifest shape matches agent-hardener's AgentSpec keys.
    agent_block = resolved.manifest["agent"]
    assert agent_block["name"] == "calc"
    assert agent_block["workflow"] == "workflow.yaml"
    assert agent_block["port"] == 9123
    # Workflow materialized with IGW base_url injected.
    written = yaml.safe_load(resolved.workflow_path.read_text())
    assert (
        written["llms"]["main"]["base_url"]
        == "http://host:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
    )
    # Scaffold project created (config-only path).
    assert (resolved.project_dir / "pyproject.toml").exists()


def test_resolve_declares_gateway_backend_for_local_platform(tmp_path):
    agent = {"config": {"llms": {"main": {"_type": "openai"}}, "workflow": {"_type": "react_agent"}}}
    sdk = _FakeSDK(agent, deployments=[{"agent": "calc", "status": "running", "port": 9123}])
    resolved = resolve_agent_to_manifest(
        "calc", sdk=sdk, base_url="http://localhost:8080", default_workspace="default", manifest_dir=tmp_path
    )
    assert resolved.manifest["backends"] == [{"name": "nemo-inference-gateway", "ports": [8080]}]


def test_resolve_no_backend_for_remote_platform(tmp_path):
    agent = {"config": {"llms": {"main": {"_type": "openai"}}, "workflow": {"_type": "react_agent"}}}
    sdk = _FakeSDK(agent, deployments=[{"agent": "calc", "status": "running", "port": 9123}])
    resolved = resolve_agent_to_manifest(
        "calc", sdk=sdk, base_url="https://gw.example.com", default_workspace="default", manifest_dir=tmp_path
    )
    assert resolved.manifest["backends"] == []


def test_resolve_defaults_port_when_no_running_deployment(tmp_path):
    agent = {"config": {"workflow": {"_type": "react_agent"}}}
    sdk = _FakeSDK(agent, deployments=[])
    resolved = resolve_agent_to_manifest(
        "calc", sdk=sdk, base_url="http://h:8080", default_workspace="default", manifest_dir=tmp_path
    )
    assert resolved.port == 8000
    assert any("no running deployment" in w for w in resolved.warnings)


def test_resolve_forwards_egress_and_overrides(tmp_path):
    agent = {"config": {"llms": {"main": {"_type": "openai"}}, "workflow": {"_type": "react_agent"}}}
    sdk = _FakeSDK(agent, deployments=[{"agent": "calc", "status": "running", "port": 9123}])
    resolved = resolve_agent_to_manifest(
        "calc",
        sdk=sdk,
        base_url="http://host:8080",
        default_workspace="default",
        manifest_dir=tmp_path,
        egress=["en.wikipedia.org", "raw.githubusercontent.com"],
        port=7000,
        secrets=["MY_KEY"],
    )
    agent_block = resolved.manifest["agent"]
    # Egress is allow-listed on the manifest; port/secrets overrides win over derivation.
    assert agent_block["egress"] == ["en.wikipedia.org", "raw.githubusercontent.com"]
    assert agent_block["port"] == 7000
    assert agent_block["secrets"] == ["MY_KEY"]
    assert resolved.port == 7000
    assert resolved.secrets == ["MY_KEY"]


def test_resolve_omits_egress_key_when_none(tmp_path):
    agent = {"config": {"workflow": {"_type": "react_agent"}}}
    sdk = _FakeSDK(agent, deployments=[{"agent": "calc", "status": "running", "port": 9123}])
    resolved = resolve_agent_to_manifest(
        "calc", sdk=sdk, base_url="http://h:8080", default_workspace="default", manifest_dir=tmp_path
    )
    # No egress supplied → no egress key, and port/secrets fall back to derivation.
    assert "egress" not in resolved.manifest["agent"]
    assert resolved.port == 9123


def test_resolve_custom_code_requires_project_dir(tmp_path):
    agent = {"config": {"functions": {"f": {"_type": "my_pkg:tool"}}, "workflow": {"_type": "react_agent"}}}
    sdk = _FakeSDK(agent, deployments=[])
    with pytest.raises(AgentResolutionError, match="custom components"):
        resolve_agent_to_manifest(
            "calc", sdk=sdk, base_url="http://h:8080", default_workspace="default", manifest_dir=tmp_path
        )


def test_resolve_missing_agent_raises(tmp_path):
    sdk = _FakeSDK(agent=None)
    with pytest.raises(AgentResolutionError, match="not found"):
        resolve_agent_to_manifest(
            "ghost", sdk=sdk, base_url="http://h:8080", default_workspace="default", manifest_dir=tmp_path
        )


# --------------------------------------------------------------------------- platform telemetry
REPO_ROOT = Path(__file__).resolve().parents[4]
REACT_AGENT = REPO_ROOT / "plugins/nemo-agents/examples/react-agent/react-agent.yml"


def test_strip_platform_telemetry_drops_the_block_and_empty_general():
    config = {"workflow": {"_type": "react_agent"}, "general": {"telemetry": {"tracing": {}}}}
    assert strip_platform_telemetry(config) == {"workflow": {"_type": "react_agent"}}
    assert "general" in config  # input untouched


def test_strip_platform_telemetry_keeps_other_general_keys():
    config = {"general": {"telemetry": {"tracing": {}}, "cache": {"enabled": True}}}
    assert strip_platform_telemetry(config) == {"general": {"cache": {"enabled": True}}}


def test_strip_platform_telemetry_is_a_noop_without_telemetry():
    config = {"workflow": {"_type": "react_agent"}}
    assert strip_platform_telemetry(config) == config


def test_scaffolded_victim_workflow_has_no_platform_telemetry(tmp_path):
    """A deployed agent carries `nemo_files` tracing, which the scaffolded victim cannot resolve.

    `nemo_files` is registered by nemo-agents-plugin (a `nat.plugins` entry point), but
    `scaffold_project` pins the victim to `nvidia-nat[langchain]` only — so leaving it in makes NAT
    exit on config validation and the run dies at a health-check timeout. Uses the real shipped
    example so the fixture cannot drift from what users actually register.
    """
    stored = yaml.safe_load(REACT_AGENT.read_text(encoding="utf-8"))
    assert stored["general"]["telemetry"]["tracing"]["nemo_trace"]["_type"] == "nemo_files", (
        "example no longer carries platform telemetry — this regression test needs a new fixture"
    )

    resolved = resolve_agent_to_manifest(
        "react-agent",
        sdk=_FakeSDK({"config": stored}, deployments=[]),
        base_url="http://localhost:8080",
        default_workspace="default",
        manifest_dir=tmp_path,
    )

    written = yaml.safe_load(resolved.workflow_path.read_text(encoding="utf-8"))
    assert "nemo_files" not in resolved.workflow_path.read_text(encoding="utf-8")
    assert "telemetry" not in written.get("general", {})
    # the parts the sandbox *can* serve survive
    assert written["workflow"]["_type"] == "react_agent"
    assert set(written["functions"]) == {"wiki", "clock"}


def test_project_source_keeps_telemetry(tmp_path):
    """A user-supplied project installs its own dependencies, so its telemetry is its own business."""
    stored = yaml.safe_load(REACT_AGENT.read_text(encoding="utf-8"))
    project = tmp_path / "my-project"
    project.mkdir()

    resolved = resolve_agent_to_manifest(
        "react-agent",
        sdk=_FakeSDK({"config": stored}, deployments=[]),
        base_url="http://localhost:8080",
        default_workspace="default",
        manifest_dir=tmp_path,
        project_dir=str(project),
    )

    written = yaml.safe_load(resolved.workflow_path.read_text(encoding="utf-8"))
    assert written["general"]["telemetry"]["tracing"]["nemo_trace"]["_type"] == "nemo_files"
