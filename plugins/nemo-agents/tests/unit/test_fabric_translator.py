# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Platform agent config to FabricConfig translation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from nemo_agents_plugin.agent_config import AgentConfig, load_agent_config
from nemo_agents_plugin.fabric.gateway_credentials import PLATFORM_IGW_API_KEY_ENV, PLATFORM_IGW_API_KEY_PLACEHOLDER
from nemo_agents_plugin.fabric.translator import FabricTranslationError, translate_agent_config
from nemo_agents_plugin.telemetry.intake_export import configure_intake_atif_export
from nemo_fabric.models import RelayHttpStorageConfig


def _example_yaml_config() -> dict[str, Any]:
    return {
        "config_format": "nemo-agents-spec-v1",
        "name": "example-agent",
        "description": "Example Agent",
        "default_harness": "hermes",
        "harnesses": {
            "hermes": {
                "kind": "hermes",
                "model": {
                    "provider": "nvidia",
                    "model": "nvidia/nemotron-3-nano-30b-a3b",
                    "api_key_env": "NVIDIA_API_KEY",
                    "base_url": "https://integrate.api.nvidia.com/v1",
                    "temperature": 0.0,
                },
                "settings": {"max_tokens": 512},
            },
            "codex": {
                "kind": "codex",
                "settings": {
                    "sandbox": "workspace-write",
                },
            },
        },
        "models": {
            "default": {
                "provider": "openai",
                "model": "openai/gpt-5.4",
                "base_url": "http://platform:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1",
            },
        },
        "environment": {
            "workspace": "./workspace",
            "artifacts": "./artifacts",
        },
        "instructions": {
            "system": {
                "content": "You are a concise assistant.",
            },
        },
        "telemetry": {
            "enabled": False,
            "provider": "relay",
            "output_dir": "./artifacts/relay",
            "project": "example-agent",
            "atif": {
                "enabled": True,
                "filename_template": "trajectory-{session_id}.atif.json",
            },
            "atof": {
                "enabled": True,
                "filename": "events.atof.jsonl",
                "mode": "overwrite",
            },
        },
    }


class TestTranslateAgentConfig:
    def test_repository_example_uses_current_codex_and_isolated_hermes_adapters(self) -> None:
        example_path = Path(__file__).parents[2] / "examples/nemo-agent-config/agent.yaml"
        config = load_agent_config(example_path)

        codex_config = translate_agent_config(config, harness_name="codex")
        claude_config = translate_agent_config(config, harness_name="claude")
        deepagents_config = translate_agent_config(config, harness_name="deepagents")
        hermes_config = translate_agent_config(config, harness_name="hermes")

        codex_harness = codex_config.harness
        claude_harness = claude_config.harness
        deepagents_harness = deepagents_config.harness
        hermes_harness = hermes_config.harness
        assert codex_harness is not None
        assert claude_harness is not None
        assert deepagents_harness is not None
        assert hermes_harness is not None
        assert codex_harness.adapter_id == "nvidia.fabric.codex"
        assert "skip_git_repo_check" not in codex_harness.settings
        assert claude_harness.adapter_id == "nvidia.fabric.claude"
        assert claude_harness.settings["permission_mode"] == "dontAsk"
        assert deepagents_harness.adapter_id == "nvidia.fabric.langchain.deepagents"
        assert deepagents_harness.settings["deepagents"] == {}
        assert hermes_harness.adapter_id == "nvidia.fabric.hermes"
        assert hermes_harness.settings["max_tokens"] == 512

    def test_translates_default_harness(self) -> None:
        config = AgentConfig.model_validate(_example_yaml_config())

        fabric_config = translate_agent_config(config)

        harness = fabric_config.harness
        instructions = fabric_config.instructions
        environment = fabric_config.environment
        assert harness is not None
        assert instructions is not None
        system_instruction = instructions.system
        assert system_instruction is not None
        assert environment is not None
        assert fabric_config.metadata.name == "example-agent"
        assert fabric_config.metadata.description == "Example Agent"
        assert harness.adapter_id == "nvidia.fabric.hermes"
        assert harness.resolution == "preinstalled"
        assert harness.settings["max_tokens"] == 512
        assert system_instruction.content == "You are a concise assistant."
        assert system_instruction.mode == "replace"
        assert fabric_config.models["default"].provider == "nvidia"
        assert fabric_config.models["default"].model == "nvidia/nemotron-3-nano-30b-a3b"
        assert fabric_config.models["default"].base_url == "https://integrate.api.nvidia.com/v1"
        assert environment.provider == "local"
        assert environment.workspace == "./workspace"
        assert environment.artifacts == "./artifacts"
        assert fabric_config.relay is None

    def test_selected_harness_uses_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NEMO_BASE_URL", raising=False)
        monkeypatch.delenv("NMP_BASE_URL", raising=False)
        monkeypatch.delenv("NMP_WORKSPACE", raising=False)
        config = AgentConfig.model_validate(_example_yaml_config())

        fabric_config = translate_agent_config(config, harness_name="codex")

        harness = fabric_config.harness
        environment = fabric_config.environment
        assert harness is not None
        assert environment is not None
        assert harness.adapter_id == "nvidia.fabric.codex"
        assert harness.settings["sandbox"] == "workspace-write"
        assert fabric_config.models["default"].provider == "openai"
        assert fabric_config.models["default"].model == "openai/gpt-5.4"
        assert (
            fabric_config.models["default"].base_url
            == "http://platform:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
        )
        assert fabric_config.models["default"].api_key_env == PLATFORM_IGW_API_KEY_ENV
        assert environment.env == {PLATFORM_IGW_API_KEY_ENV: PLATFORM_IGW_API_KEY_PLACEHOLDER}
        assert config.models["default"].api_key_env is None

    def test_forwards_platform_runtime_environment_to_child_tools(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEMO_BASE_URL", "http://command-platform:8080")
        monkeypatch.setenv("NMP_BASE_URL", "http://shared-platform:8080")
        monkeypatch.setenv("NMP_WORKSPACE", "request-workspace")
        config = AgentConfig.model_validate(_example_yaml_config())

        fabric_config = translate_agent_config(config, harness_name="codex")

        environment = fabric_config.environment
        assert environment is not None
        assert environment.env == {
            "NEMO_BASE_URL": "http://command-platform:8080",
            "NMP_BASE_URL": "http://shared-platform:8080",
            "NMP_WORKSPACE": "request-workspace",
            PLATFORM_IGW_API_KEY_ENV: PLATFORM_IGW_API_KEY_PLACEHOLDER,
        }

    def test_promotes_legacy_model_settings_base_url(self) -> None:
        payload = _example_yaml_config()
        payload["default_harness"] = "codex"
        payload["models"]["default"]["base_url"] = None
        payload["models"]["default"]["settings"] = {
            "base_url": "http://legacy:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
        }
        config = AgentConfig.model_validate(payload)

        fabric_config = translate_agent_config(config)

        assert (
            fabric_config.models["default"].base_url
            == "http://legacy:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
        )
        assert "base_url" not in fabric_config.models["default"].settings

    def test_translates_shared_capability_sections(self) -> None:
        payload = copy.deepcopy(_example_yaml_config())
        payload["skills"] = {"paths": ["skills/review"]}
        payload["mcp"] = {
            "servers": {
                "repo": {
                    "transport": "stdio",
                    "url": "repo-mcp --root .",
                    "exposure": "fabric_managed",
                    "allowed_tools": ["read_file", "search_files"],
                    "blocked_tools": ["write_file"],
                },
                "disabled": {
                    "transport": "streamable-http",
                    "url": "https://mcp.example.com",
                    "allowed_tools": [],
                    "custom_headers": {"Authorization": "Bearer ${MCP_ACCESS_TOKEN}"},
                },
            }
        }
        payload["tools"] = {"blocked": ["shell", "browser"]}
        config = AgentConfig.model_validate(payload)

        fabric_config = translate_agent_config(config)

        skills = fabric_config.skills
        mcp = fabric_config.mcp
        tools = fabric_config.tools
        assert skills is not None
        assert mcp is not None
        assert tools is not None
        assert skills.paths == ["skills/review"]
        assert mcp.servers["repo"].transport == "stdio"
        assert mcp.servers["repo"].url == "repo-mcp --root ."
        assert mcp.servers["repo"].exposure == "fabric_managed"
        assert mcp.servers["repo"].allowed_tools == ["read_file", "search_files"]
        assert mcp.servers["repo"].blocked_tools == ["write_file"]
        assert mcp.servers["disabled"].allowed_tools == []
        assert mcp.servers["disabled"].custom_headers == {"Authorization": "Bearer ${MCP_ACCESS_TOKEN}"}
        assert tools.blocked == ["shell", "browser"]

    def test_top_level_prompts_rejected_until_shared_prompt_contract_exists(self) -> None:
        payload = copy.deepcopy(_example_yaml_config())
        payload["prompts"] = {"system": "prompts/system.md"}
        config = AgentConfig.model_validate(payload)

        with pytest.raises(FabricTranslationError, match="Top-level prompts are not translated yet"):
            translate_agent_config(config)

    def test_environment_spec_env_forwarded_platform_values_win(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Simulates a config after an EnvironmentSpec merge: environment.env holds
        # the spec's plaintext vars. Platform-injected runtime values win on key
        # collision.
        monkeypatch.setenv("NMP_WORKSPACE", "runtime-ws")
        monkeypatch.delenv("NEMO_BASE_URL", raising=False)
        monkeypatch.delenv("NMP_BASE_URL", raising=False)
        payload = copy.deepcopy(_example_yaml_config())
        payload["environment"]["env"] = {"CUSTOM": "from-spec", "NMP_WORKSPACE": "spec-should-lose"}
        config = AgentConfig.model_validate(payload)

        fabric_config = translate_agent_config(config, harness_name="codex")

        environment = fabric_config.environment
        assert environment is not None
        assert environment.env["CUSTOM"] == "from-spec"
        assert environment.env["NMP_WORKSPACE"] == "runtime-ws"

    def test_environment_mirror_fields_forwarded(self) -> None:
        payload = copy.deepcopy(_example_yaml_config())
        payload["environment"].update(
            {
                "control_location": "in_env_control",
                "ownership": "fabric_owned",
                "connection": {"url": "http://sandbox"},
                "metadata": {"team": "platform"},
            }
        )
        config = AgentConfig.model_validate(payload)

        fabric_config = translate_agent_config(config, harness_name="codex")

        environment = fabric_config.environment
        assert environment is not None
        assert environment.control_location == "in_env_control"
        assert environment.ownership == "fabric_owned"
        assert environment.connection == {"url": "http://sandbox"}
        assert environment.metadata == {"team": "platform"}

    def test_runtime_constraints_forwarded(self) -> None:
        payload = copy.deepcopy(_example_yaml_config())
        payload["runtime"] = {"max_turns": 20, "timeout_seconds": 120.5}
        config = AgentConfig.model_validate(payload)

        fabric_config = translate_agent_config(config)

        assert fabric_config.runtime.max_turns == 20
        assert fabric_config.runtime.timeout_seconds == 120.5

    @pytest.mark.parametrize(
        ("kind", "adapter_id"),
        [
            ("claude", "nvidia.fabric.claude"),
            ("codex", "nvidia.fabric.codex"),
            ("deepagents", "nvidia.fabric.langchain.deepagents"),
            ("hermes", "nvidia.fabric.hermes"),
        ],
    )
    def test_supported_harness_kinds_translate_to_adapter_ids(
        self,
        kind: str,
        adapter_id: str,
    ) -> None:
        payload = _example_yaml_config()
        payload["default_harness"] = "selected"
        payload["harnesses"] = {"selected": {"kind": kind}}
        config = AgentConfig.model_validate(payload)

        fabric_config = translate_agent_config(config)

        harness = fabric_config.harness
        assert harness is not None
        assert harness.adapter_id == adapter_id

    def test_every_configured_model_is_forwarded(self) -> None:
        """FabricConfig.models is a keyed map; secondary keys must survive translation."""
        payload = _example_yaml_config()
        payload["models"]["fast"] = {"provider": "openai", "model": "openai/gpt-5.4-mini"}
        payload["models"]["judge"] = {"provider": "openai", "model": "openai/gpt-5.4-judge"}
        config = AgentConfig.model_validate(payload)

        fabric_config = translate_agent_config(config)

        assert set(fabric_config.models) == {"default", "fast", "judge"}
        assert fabric_config.models["fast"].model == "openai/gpt-5.4-mini"
        assert fabric_config.models["judge"].model == "openai/gpt-5.4-judge"

    def test_selected_harness_model_is_published_as_default(self) -> None:
        """harness.model outranks models.default, and lands on the "default" key."""
        payload = _example_yaml_config()
        payload["models"]["fast"] = {"provider": "openai", "model": "openai/gpt-5.4-mini"}
        config = AgentConfig.model_validate(payload)

        fabric_config = translate_agent_config(config)

        # The hermes harness declares its own model; models.default is shadowed.
        assert fabric_config.models["default"].model == "nvidia/nemotron-3-nano-30b-a3b"
        assert fabric_config.models["fast"].model == "openai/gpt-5.4-mini"

    def test_models_default_is_published_when_the_harness_declares_no_model(self) -> None:
        payload = _example_yaml_config()
        payload["default_harness"] = "codex"
        payload["models"]["fast"] = {"provider": "openai", "model": "openai/gpt-5.4-mini"}
        config = AgentConfig.model_validate(payload)

        fabric_config = translate_agent_config(config)

        assert fabric_config.models["default"].model == "openai/gpt-5.4"
        assert fabric_config.models["fast"].model == "openai/gpt-5.4-mini"

    def test_gateway_credential_bound_on_every_routed_model(self) -> None:
        """A secondary IGW-routed model needs its api_key_env too, not just the primary."""
        payload = _example_yaml_config()
        payload["default_harness"] = "codex"
        payload["models"]["fast"] = {
            "provider": "openai",
            "model": "openai/gpt-5.4-mini",
            "base_url": "http://platform:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1",
        }
        config = AgentConfig.model_validate(payload)

        fabric_config = translate_agent_config(config)

        assert fabric_config.models["fast"].api_key_env == PLATFORM_IGW_API_KEY_ENV
        assert fabric_config.environment is not None
        assert fabric_config.environment.env[PLATFORM_IGW_API_KEY_ENV] == PLATFORM_IGW_API_KEY_PLACEHOLDER

    def test_non_routed_secondary_model_is_left_uncredentialed(self) -> None:
        payload = _example_yaml_config()
        payload["default_harness"] = "codex"
        payload["models"]["fast"] = {
            "provider": "nvidia",
            "model": "nvidia/nemotron-3-nano-30b-a3b",
            "base_url": "https://integrate.api.nvidia.com/v1",
        }
        config = AgentConfig.model_validate(payload)

        fabric_config = translate_agent_config(config)

        assert fabric_config.models["fast"].api_key_env is None

    def test_harness_only_model_still_translates_without_a_models_map(self) -> None:
        payload = _example_yaml_config()
        payload["models"] = {}
        config = AgentConfig.model_validate(payload)

        fabric_config = translate_agent_config(config)

        assert set(fabric_config.models) == {"default"}
        assert fabric_config.models["default"].model == "nvidia/nemotron-3-nano-30b-a3b"

    def test_unknown_selected_harness_rejected(self) -> None:
        config = AgentConfig.model_validate(_example_yaml_config())

        with pytest.raises(FabricTranslationError, match="Unknown configured harness 'claude'"):
            translate_agent_config(config, harness_name="claude")

    def test_unsupported_harness_kind_rejected(self) -> None:
        payload = _example_yaml_config()
        payload["harnesses"]["custom"] = {"kind": "custom"}
        payload["default_harness"] = "custom"
        config = AgentConfig.model_validate(payload)

        with pytest.raises(FabricTranslationError, match="Unsupported harness kind 'custom'"):
            translate_agent_config(config)

    @pytest.mark.parametrize(
        "adapter_id",
        [
            "nvidia.fabric.insights-analyst",
            "nvidia.fabric.some.future.adapter",
        ],
    )
    def test_fully_qualified_adapter_id_passes_through(self, adapter_id: str) -> None:
        """Plugin-owned adapters ship their own descriptor, so no short-name entry is needed."""
        payload = _example_yaml_config()
        payload["default_harness"] = "selected"
        payload["harnesses"] = {"selected": {"kind": adapter_id}}
        config = AgentConfig.model_validate(payload)

        fabric_config = translate_agent_config(config)

        assert fabric_config.harness is not None
        assert fabric_config.harness.adapter_id == adapter_id

    def test_adapter_id_passthrough_requires_the_nvidia_fabric_prefix(self) -> None:
        payload = _example_yaml_config()
        payload["default_harness"] = "selected"
        payload["harnesses"] = {"selected": {"kind": "acme.fabric.custom"}}
        config = AgentConfig.model_validate(payload)

        with pytest.raises(FabricTranslationError, match="Unsupported harness kind 'acme.fabric.custom'"):
            translate_agent_config(config)

    def test_missing_model_rejected(self) -> None:
        payload = _example_yaml_config()
        payload["models"] = {}
        payload["default_harness"] = "codex"
        config = AgentConfig.model_validate(payload)

        with pytest.raises(FabricTranslationError, match="no models.default is configured"):
            translate_agent_config(config)

    def test_relay_telemetry_uses_latest_fabric_shape(self) -> None:
        payload = copy.deepcopy(_example_yaml_config())
        payload["telemetry"]["enabled"] = True
        config = AgentConfig.model_validate(payload)

        fabric_config = translate_agent_config(config)

        telemetry = fabric_config.telemetry
        assert telemetry is not None
        relay_provider = telemetry.providers["relay"]
        assert not isinstance(relay_provider, dict)
        relay = fabric_config.relay
        assert relay is not None
        observability = relay.observability
        assert observability is not None
        assert relay_provider.config is None
        assert relay.project == "example-agent"
        assert relay.output_dir == "./artifacts/relay"
        assert observability.model_dump(exclude_none=True) == {
            "version": 3,
            "enable_full_payloads": False,
            "atif": {
                "enabled": True,
                "filename_template": "trajectory-{session_id}.atif.json",
                "output_directory": "./artifacts/relay",
                "agent_name": "example-agent",
                "model_name": "nvidia/nemotron-3-nano-30b-a3b",
            },
            "atof": {
                "enabled": True,
                "sinks": [
                    {
                        "type": "file",
                        "output_directory": "./artifacts/relay",
                        "filename": "events.atof.jsonl",
                        "mode": "overwrite",
                    }
                ],
            },
        }

    def test_relay_telemetry_prefers_platform_registered_agent_name(self) -> None:
        payload = copy.deepcopy(_example_yaml_config())
        payload["telemetry"]["enabled"] = True
        payload["telemetry"]["agent_name"] = "example-agent-hzwy9s"
        payload["telemetry"]["opentelemetry"] = {
            "enabled": True,
            "endpoints": [{"type": "full", "endpoint": "http://otel-collector:4317"}],
        }
        config = AgentConfig.model_validate(payload)

        fabric_config = translate_agent_config(config)

        relay = fabric_config.relay
        assert relay is not None
        observability = relay.observability
        assert observability is not None
        assert observability.model_dump(exclude_none=True)["atif"]["agent_name"] == "example-agent-hzwy9s"
        opentelemetry = observability.opentelemetry
        assert opentelemetry is not None
        assert opentelemetry.endpoints[0].service_name == "example-agent-hzwy9s"

    def test_relay_telemetry_keeps_explicit_atif_agent_name(self) -> None:
        payload = copy.deepcopy(_example_yaml_config())
        payload["telemetry"]["enabled"] = True
        payload["telemetry"]["agent_name"] = "example-agent-hzwy9s"
        payload["telemetry"]["atif"]["agent_name"] = "author-chosen-name"
        config = AgentConfig.model_validate(payload)

        fabric_config = translate_agent_config(config)

        relay = fabric_config.relay
        assert relay is not None
        observability = relay.observability
        assert observability is not None
        assert observability.model_dump(exclude_none=True)["atif"]["agent_name"] == "author-chosen-name"

    def test_relay_atof_endpoint_sinks_translate_to_stream_sinks(self) -> None:
        payload = copy.deepcopy(_example_yaml_config())
        payload["telemetry"]["enabled"] = True
        payload["telemetry"]["atof"] = {
            "enabled": True,
            "endpoints": [
                {
                    "type": "file",
                    "endpoint": "http://localhost:4318/v1/events",
                    "timeout_millis": 3000,
                }
            ],
        }
        config = AgentConfig.model_validate(payload)

        fabric_config = translate_agent_config(config)

        relay = fabric_config.relay
        assert relay is not None
        observability = relay.observability
        assert observability is not None
        assert observability.model_dump(exclude_none=True)["atof"]["sinks"] == [
            {
                "type": "file",
                "output_directory": "./artifacts/relay",
                "mode": "append",
            },
            {
                "type": "stream",
                "url": "http://localhost:4318/v1/events",
                "transport": "http_post",
                "timeout_millis": 3000,
                "field_name_policy": "preserve",
            },
        ]

    def test_relay_opentelemetry_export_translates_to_fabric(self) -> None:
        payload = copy.deepcopy(_example_yaml_config())
        payload["telemetry"]["enabled"] = True
        payload["telemetry"]["opentelemetry"] = {
            "enabled": True,
            "endpoints": [
                {
                    "type": "full",
                    "endpoint": "http://otel-collector:4317",
                    "transport": "grpc",
                    "header_env": {"Authorization": "OTEL_AUTH_HEADER"},
                    "resource_attributes": {"deployment.environment": "test"},
                },
                {
                    "type": "gen_ai",
                    "endpoint": "http://shared-collector:4318/v1/traces",
                    "service_name": "shared-agent-service",
                },
            ],
        }
        config = AgentConfig.model_validate(payload)

        fabric_config = translate_agent_config(config)

        relay = fabric_config.relay
        assert relay is not None
        observability = relay.observability
        assert observability is not None
        opentelemetry = observability.opentelemetry
        assert opentelemetry is not None
        assert opentelemetry.enabled is True
        assert len(opentelemetry.endpoints) == 2
        endpoint = opentelemetry.endpoints[0]
        assert endpoint.type == "full"
        assert endpoint.endpoint == "http://otel-collector:4317"
        assert endpoint.transport == "grpc"
        assert endpoint.header_env == {"Authorization": "OTEL_AUTH_HEADER"}
        assert endpoint.resource_attributes == {"deployment.environment": "test"}
        assert endpoint.service_name == "example-agent"
        assert opentelemetry.endpoints[1].service_name == "shared-agent-service"


def test_auto_wired_intake_telemetry_translates_to_a_relay_http_storage() -> None:
    """The wired dict has to be a shape Fabric accepts, not merely a plausible one.

    The unit tests around ``configure_intake_atif_export`` assert its output
    structurally; this puts that output through the real translator so a field
    Relay does not recognise fails here rather than inside a running job.
    """
    payload = _example_yaml_config()
    # The shared fixture opts out; this is about a config that does not.
    payload.pop("telemetry")
    configure_intake_atif_export(
        payload,
        workspace="team-a",
        base_url="http://nemo-platform-api:8080",
        header_env={"X-NMP-Principal-Id": "NMP_AGENT_TELEMETRY_HEADER_X_NMP_PRINCIPAL_ID"},
    )

    fabric_config = translate_agent_config(AgentConfig.model_validate(payload))

    relay = fabric_config.relay
    assert relay is not None
    observability = relay.observability
    assert observability is not None
    atif = observability.atif
    assert atif is not None
    storage = atif.storage[0]
    assert isinstance(storage, RelayHttpStorageConfig)
    assert storage.endpoint == "http://nemo-platform-api:8080/apis/intake/v2/workspaces/team-a/ingest/atif"
    assert storage.header_env == {"X-NMP-Principal-Id": "NMP_AGENT_TELEMETRY_HEADER_X_NMP_PRINCIPAL_ID"}
    assert not storage.headers, "credentials must reach Relay through the environment, not the config"
    # Relay identifies the trajectory by these; both come from the agent config.
    assert atif.agent_name == payload["name"]
    assert atif.model_name
