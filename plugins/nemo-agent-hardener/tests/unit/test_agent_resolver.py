# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the agent resolver's pure logic and end-to-end manifest build.

The orchestrator is exercised with a fake SDK (plain dicts, matching what
``client.agents.get`` / ``client.agents.deployments.list`` return), so no live platform or
agent-hardener install is needed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from nemo_agent_hardener_plugin.agent_resolver import (
    AgentResolutionError,
    build_manifest_dict,
    check_relay_artifacts_dir,
    derive_agent_env,
    derive_egress,
    derive_secret_names,
    detect_custom_components,
    gateway_backend,
    inject_gateway_url,
    materialize_agent_package,
    parse_agent_ref,
    require_guardable_harness,
    resolve_agent_to_manifest,
    shipped_dockerfile,
)

# `list` is shadowed by the fake's own `list` method inside the class body, so the parameter type
# is aliased at module level.
DeploymentRows = list[dict[str, Any]]


def _agent(**overrides: Any) -> dict:
    """A registered nemo-agents-spec-v1 agent, the only shape this plugin resolves."""
    config: dict[str, Any] = {
        "config_format": "nemo-agents-spec-v1",
        "name": "calc",
        "default_harness": "deepagents",
        "harnesses": {"deepagents": {"kind": "deepagents"}},
        "models": {"default": {"provider": "nvidia", "model": "m", "api_key_env": "NVIDIA_API_KEY"}},
        "telemetry": {"enabled": True, "provider": "relay", "output_dir": "./artifacts/relay"},
    }
    config.update(overrides)
    return {"config": config, "config_format": "nemo-agents-spec-v1"}


@pytest.fixture(autouse=True)
def _allow_dev_contract_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the Fabric packager render against a source checkout.

    It refuses a dev/local version by default because the image pins that exact version and no
    index serves it — correct for a real build, but these tests only render the Dockerfile.
    """
    monkeypatch.setenv("NEMO_AGENTS_ALLOW_UNPUBLISHED_CONTRACT_VERSION", "1")


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
def test_inject_gateway_url_binds_the_agents_models():
    """The victim needs no raw model key: models resolve through the Inference Gateway."""
    config = {"models": {"default": {"provider": "nvidia", "model": "m"}}}
    out = inject_gateway_url(config, "ws1", "http://host:8080/")
    assert "/apis/inference-gateway/v2/workspaces/ws1/" in out["models"]["default"]["base_url"]
    assert config["models"]["default"] == {"provider": "nvidia", "model": "m"}  # original not mutated


# --------------------------------------------------------------------------- gateway_backend
def test_gateway_backend_declared_for_local_gateway():
    assert gateway_backend("http://localhost:8080") == {"name": "nemo-inference-gateway", "ports": [8080]}
    assert gateway_backend("http://127.0.0.1:9000") == {"name": "nemo-inference-gateway", "ports": [9000]}


def test_gateway_backend_none_for_remote_gateway():
    assert gateway_backend("https://gateway.example.com") is None


# --------------------------------------------------------------------------- detect_custom_components
def test_detect_custom_components_flags_local_paths():
    """Skills live beside the config; an image carrying only the config cannot find them."""
    assert detect_custom_components({"skills": {"paths": ["./skills/triage"]}}) == ["./skills/triage"]


def test_detect_custom_components_empty_for_a_self_contained_agent():
    assert detect_custom_components({"skills": {"paths": []}}) == []
    assert detect_custom_components({}) == []


# --------------------------------------------------------------------------- derive_secret_names
def test_derive_secret_names_reads_declarations_not_values():
    """The spec names its credentials, so no secret is ever copied into the manifest."""
    config = {
        "models": {"default": {"api_key_env": "NVIDIA_API_KEY"}},
        "mcp": {"servers": {"gh": {"env": {"GITHUB_TOKEN": "${GITHUB_TOKEN}"}}}},
    }
    assert derive_secret_names(config) == ["GITHUB_TOKEN", "NVIDIA_API_KEY"]


def test_derive_secret_names_default_when_none_declared():
    assert derive_secret_names({}) == ["INFERENCE_API_KEY"]


def test_telemetry_dir_complaint_names_only_a_path_agent_hardener_will_not_read():
    """The check exists to catch a victim writing telemetry somewhere the run never looks."""
    sandbox_dir = "/home/sandbox/.agent-hardener/relay"
    assert check_relay_artifacts_dir({"telemetry": {"output_dir": sandbox_dir}}) is None
    assert check_relay_artifacts_dir({}) is None

    complaint = check_relay_artifacts_dir({"telemetry": {"output_dir": "./artifacts/relay"}})
    assert complaint is not None
    # The message has to name both paths, or it says "wrong" without saying what to write instead.
    assert "./artifacts/relay" in complaint
    assert sandbox_dir in complaint


# --------------------------------------------------------------------------- resolve_agent_to_manifest
def test_resolve_builds_a_runnable_agent_package(tmp_path):
    agent = _agent()
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
    # Agent Hardener's contract is "a directory with a runnable agent in it", so the manifest names an
    # image and a launch command rather than a config for Agent Hardener to interpret.
    agent_block = resolved.manifest["agent"]
    assert agent_block["name"] == "calc"
    assert agent_block["dockerfile"] == "Dockerfile"
    assert agent_block["port"] == 9123
    assert "nemo_agents_plugin.fabric.server" in agent_block["start_command"]
    # Absolute interpreter and config path: `openshell sandbox exec` drops the image's ENV, so
    # neither PATH nor AGENT_CONFIG_PATH is visible to the launch command.
    assert agent_block["start_command"].startswith("/workspace/.venv/bin/python")
    assert "/workspace/agent.yaml" in agent_block["start_command"]

    written = yaml.safe_load(resolved.agent_config_path.read_text(encoding="utf-8"))
    assert "/apis/inference-gateway/v2/workspaces/default/" in written["models"]["default"]["base_url"]

    # The image comes from the platform's own Fabric pipeline, with the sandbox profile applied.
    dockerfile = (resolved.project_dir / "Dockerfile").read_text(encoding="utf-8")
    assert "nemo_agents_plugin.fabric.server" in dockerfile
    assert "sandbox" in dockerfile  # the openshell profile's non-root user


def test_resolve_declares_gateway_backend_for_local_platform(tmp_path):
    agent = _agent()
    sdk = _FakeSDK(agent, deployments=[{"agent": "calc", "status": "running", "port": 9123}])
    resolved = resolve_agent_to_manifest(
        "calc", sdk=sdk, base_url="http://localhost:8080", default_workspace="default", manifest_dir=tmp_path
    )
    assert resolved.manifest["backends"] == [{"name": "nemo-inference-gateway", "ports": [8080]}]


def test_resolve_no_backend_for_remote_platform(tmp_path):
    agent = _agent()
    sdk = _FakeSDK(agent, deployments=[{"agent": "calc", "status": "running", "port": 9123}])
    resolved = resolve_agent_to_manifest(
        "calc", sdk=sdk, base_url="https://gw.example.com", default_workspace="default", manifest_dir=tmp_path
    )
    assert resolved.manifest["backends"] == []


def test_resolve_defaults_port_when_no_running_deployment(tmp_path):
    agent = _agent()
    sdk = _FakeSDK(agent, deployments=[])
    resolved = resolve_agent_to_manifest(
        "calc", sdk=sdk, base_url="http://h:8080", default_workspace="default", manifest_dir=tmp_path
    )
    assert resolved.port == 8000
    assert any("no running deployment" in w for w in resolved.warnings)


def test_resolve_forwards_egress_and_overrides(tmp_path):
    agent = _agent()
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
    agent = _agent()
    sdk = _FakeSDK(agent, deployments=[{"agent": "calc", "status": "running", "port": 9123}])
    resolved = resolve_agent_to_manifest(
        "calc", sdk=sdk, base_url="http://h:8080", default_workspace="default", manifest_dir=tmp_path
    )
    # No egress supplied → no egress key, and port/secrets fall back to derivation.
    assert "egress" not in resolved.manifest["agent"]
    assert resolved.port == 9123


def test_resolve_requires_a_project_dir_for_local_artifacts(tmp_path):
    """Skills live beside the config; packaging only the config yields an agent missing them."""
    agent = _agent(skills={"paths": ["./skills/triage"]})
    sdk = _FakeSDK(agent, deployments=[])
    with pytest.raises(AgentResolutionError, match="local paths"):
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


def test_the_victims_relay_telemetry_is_preserved(tmp_path):
    """The opposite of the NAT behaviour, and load-bearing.

    NAT configs carried a ``nemo_files`` exporter the sandboxed victim could not resolve, so it was
    stripped. A Relay victim's telemetry is what Agent Hardener reads to see which tools an attack
    reached — strip it and every run reports an uninstrumented victim.
    """
    resolved = resolve_agent_to_manifest(
        "calc",
        sdk=_FakeSDK(_agent()),
        base_url="http://host:8080",
        default_workspace="default",
        manifest_dir=tmp_path,
    )
    written = yaml.safe_load(resolved.agent_config_path.read_text(encoding="utf-8"))
    assert written["telemetry"]["provider"] == "relay"
    # The agent's telemetry.output_dir is a path *inside the container*, and the manifest's
    # relay_artifacts names a directory on the *host* for the fetched copy. Forwarding one to the
    # other made the host write /home/sandbox, which is autofs on macOS. So the manifest carries no
    # relay_artifacts and agent-hardener's own host default applies.
    assert "relay_artifacts" not in resolved.manifest["agent"]


def _fake_ethos(monkeypatch: pytest.MonkeyPatch, contents: dict[str, str] | None) -> list[str]:
    """Stand in for the fileset download, recording the ref asked for."""
    asked: list[str] = []

    def fake_download(_sdk: object, ref: str, dest: Path) -> Path:
        asked.append(ref)
        if contents is None:
            raise FileNotFoundError(ref)
        for name, text in contents.items():
            (dest / name).write_text(text, encoding="utf-8")
        return dest

    monkeypatch.setattr("nemo_agent_hardener_plugin.filesets.download_fileset", fake_download)
    return asked


def test_shipped_dockerfile_is_read_from_the_agents_ethos_fileset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Registration uploads the whole agent directory, so an author's Dockerfile is already stored.

    Reading it back is what lets an agent pick its own nemo-relay and build from a source checkout —
    the rendered Dockerfile pins the packaging machine's nemo-platform version, which no index serves
    when the platform is installed from git.
    """
    asked = _fake_ethos(monkeypatch, {"agent.yaml": "x", "Dockerfile": "FROM python:3.12-slim\n"})

    found = shipped_dockerfile(SimpleNamespace(), "ledger", "default")

    assert found == "FROM python:3.12-slim\n"
    assert asked == ["default/ledger-ethos"]


def test_an_agent_without_a_dockerfile_falls_back_to_rendering(monkeypatch: pytest.MonkeyPatch) -> None:
    """Most agents ship only agent.yaml, and that must stay the ordinary path rather than an error."""
    _fake_ethos(monkeypatch, {"agent.yaml": "x"})
    assert shipped_dockerfile(SimpleNamespace(), "ledger", "default") is None


def test_an_unreadable_ethos_fileset_falls_back_to_rendering(monkeypatch: pytest.MonkeyPatch) -> None:
    """An agent registered before ethos filesets existed has none; that is not an error."""
    _fake_ethos(monkeypatch, None)
    assert shipped_dockerfile(SimpleNamespace(), "ledger", "default") is None


def test_materialize_prefers_the_shipped_dockerfile(tmp_path: Path) -> None:
    """The author's file is written verbatim — no re-render, so none of the pins are substituted."""
    materialize_agent_package({"name": "x"}, tmp_path, dockerfile_override="FROM scratch\n")

    assert (tmp_path / "Dockerfile").read_text(encoding="utf-8") == "FROM scratch\n"


def test_gateway_harnesses_are_rejected_at_init() -> None:
    """Claude and Codex run Relay as a compiled gateway that cannot load the guardrail plugin.

    Rejected here because this is the last point where the harness is known — build_manifest_dict
    emits a plain BYO victim, after which the agent is indistinguishable from a hand-built image.
    """
    for harness in ("claude", "codex"):
        with pytest.raises(AgentResolutionError, match="cannot load Agent Hardener's guardrail plugin"):
            require_guardable_harness({"default_harness": harness}, "ws/agent")


def test_guardable_harnesses_pass_through() -> None:
    assert require_guardable_harness({"default_harness": "deepagents"}, "ws/a") == "deepagents"
    assert require_guardable_harness({"default_harness": "hermes"}, "ws/a") == "hermes"
    assert require_guardable_harness({}, "ws/a") is None  # a BYO-shaped config names no harness


def test_the_manifest_carries_the_harness() -> None:
    """Agent Hardener needs it to stage Hermes' extra wiring — not to choose a victim kind."""
    manifest = build_manifest_dict(agent_name="a", project_dir=".", port=8000, secrets=[], harness="hermes")

    assert manifest["agent"]["harness"] == "hermes"


def test_declared_env_reaches_the_manifest() -> None:
    """Egress discovery reads agent_env; without it the agent gets a backend URL it cannot reach."""
    config = {"environment": {"env": {"BACKEND_URL": "https://ledger.internal", "KEY": "${SECRET}"}}}

    assert derive_agent_env(config) == {"BACKEND_URL": "https://ledger.internal"}


def test_declared_env_is_optional() -> None:
    assert derive_agent_env({}) == {}
    assert derive_agent_env({"environment": {"workspace": "./workspace"}}) == {}


def test_network_mcp_servers_and_model_endpoints_are_allow_listed() -> None:
    """Discovery scans code; a config-only agent's hosts are declarations, so nothing finds them.

    Without this the agent is admitted, started, and then refused the connection by its own policy.
    """
    config = {
        "mcp": {
            "servers": {
                "remote": {"transport": "streamable-http", "url": "https://ledger-tools.internal/mcp"},
                "local": {"transport": "stdio", "url": "/workspace/.venv/bin/python", "args": ["s.py"]},
            }
        },
        "models": {"default": {"base_url": "https://inference-api.nvidia.com/v1"}},
    }

    assert derive_egress(config) == [
        "inference-api.nvidia.com:443",
        "ledger-tools.internal:443",
    ]


def test_a_stdio_server_contributes_no_egress() -> None:
    """Its url is a local interpreter path; a filesystem path in a network policy is nonsense."""
    config = {"mcp": {"servers": {"local": {"transport": "stdio", "url": "/usr/bin/python"}}}}

    assert derive_egress(config) == []
