# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Platform-owned agent.yaml config models."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from nemo_agents_plugin.agent_config import (
    AgentConfig,
    AgentConfigLoadError,
    load_agent_config,
    load_agent_config_from_dir,
)
from pydantic import ValidationError


def _example_yaml_config() -> dict:
    return {
        "config_format": "nemo-agents-spec-v1",
        "name": "test-agent",
        "description": "Test agent config",
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
                "settings": {
                    "max_tokens": 512,
                    "reasoning_config": {"effort": "none"},
                },
            },
            "codex": {
                "kind": "codex",
                "settings": {
                    "sandbox": "workspace-write",
                    "reasoning_effort": "high",
                },
            },
        },
        "models": {
            "default": {
                "provider": "openai",
                "model": "openai/gpt-5.4",
            },
        },
        "prompts": {
            "system": "prompts/system.md",
        },
        "instructions": {
            "system": {
                "content": "You are a concise smoke test assistant.",
            },
        },
        "skills": None,
        "environment": {
            "workspace": "./workspace",
            "artifacts": "./artifacts",
        },
        "telemetry": {
            "enabled": False,
            "provider": "relay",
            "output_dir": "./artifacts/relay",
            "project": "test-agent",
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


class TestAgentConfig:
    def test_example_yaml_config_validates(self) -> None:
        config = AgentConfig.model_validate(_example_yaml_config())

        assert config.config_format == "nemo-agents-spec-v1"
        assert config.name == "test-agent"
        assert config.default_harness == "hermes"
        assert config.harnesses["hermes"].model is not None
        assert config.harnesses["hermes"].model.provider == "nvidia"
        assert config.harnesses["hermes"].model.base_url == "https://integrate.api.nvidia.com/v1"
        assert config.harnesses["hermes"].settings["max_tokens"] == 512
        assert config.harnesses["codex"].settings["sandbox"] == "workspace-write"
        assert config.models["default"].model == "openai/gpt-5.4"
        assert config.instructions is not None
        assert config.instructions.system is not None
        assert config.instructions.system.content == "You are a concise smoke test assistant."
        assert config.instructions.system.mode == "replace"
        assert config.skills is None
        assert config.telemetry.atif == {
            "enabled": True,
            "filename_template": "trajectory-{session_id}.atif.json",
        }

    def test_defaults_fill_optional_sections(self) -> None:
        config = AgentConfig.model_validate(
            {
                "config_format": "nemo-agents-spec-v1",
                "name": "minimal-agent",
                "default_harness": "codex",
                "harnesses": {"codex": {"kind": "codex"}},
            }
        )

        assert config.description == ""
        assert config.models == {}
        assert config.prompts == {}
        assert config.instructions is None
        assert config.skills is None
        assert config.mcp is None
        assert config.tools is None
        assert config.runtime.timeout_seconds is None
        assert config.runtime.max_turns is None
        assert config.environment.provider == "local"
        assert config.environment.workspace == "./workspace"
        assert config.environment.artifacts == "./artifacts"
        # Tri-state: unset, so the backend may wire an export for the context.
        assert config.telemetry.enabled is None

    def test_shared_capability_sections_validate(self) -> None:
        payload = _example_yaml_config()
        payload["skills"] = {"paths": ["skills/review"]}
        payload["mcp"] = {
            "servers": {
                "repo": {
                    "transport": "stdio",
                    "url": "repo-mcp --root .",
                    "allowed_tools": ["read_file", "search_files"],
                    "blocked_tools": ["write_file"],
                },
                "private-api": {
                    "transport": "streamable-http",
                    "url": "https://mcp.example.com",
                    "custom_headers": {"Authorization": "Bearer ${MCP_ACCESS_TOKEN}"},
                },
            }
        }
        payload["tools"] = {"blocked": ["shell", "browser"]}

        config = AgentConfig.model_validate(payload)

        assert config.skills is not None
        assert config.skills.paths == ["skills/review"]
        assert config.mcp is not None
        assert config.mcp.servers["repo"].transport == "stdio"
        assert config.mcp.servers["repo"].exposure == "harness_native"
        assert config.mcp.servers["repo"].allowed_tools == ["read_file", "search_files"]
        assert config.mcp.servers["repo"].blocked_tools == ["write_file"]
        assert config.mcp.servers["private-api"].custom_headers == {"Authorization": "Bearer ${MCP_ACCESS_TOKEN}"}
        assert config.tools is not None
        assert config.tools.blocked == ["shell", "browser"]

    def test_mcp_server_preserves_explicit_empty_tool_allowlist(self) -> None:
        payload = _example_yaml_config()
        payload["mcp"] = {
            "servers": {
                "repo": {
                    "transport": "stdio",
                    "url": "repo-mcp --root .",
                    "allowed_tools": [],
                }
            }
        }

        config = AgentConfig.model_validate(payload)

        assert config.mcp is not None
        assert config.mcp.servers["repo"].allowed_tools == []
        assert config.mcp.servers["repo"].blocked_tools == []

    def test_opentelemetry_export_config_validates(self) -> None:
        payload = _example_yaml_config()
        payload["telemetry"]["opentelemetry"] = {
            "enabled": True,
            "endpoints": [
                {
                    "type": "full",
                    "endpoint": "http://otel-collector:4317",
                    "transport": "grpc",
                }
            ],
        }

        config = AgentConfig.model_validate(payload)

        assert config.telemetry.opentelemetry == payload["telemetry"]["opentelemetry"]

    def test_runtime_constraints_validate(self) -> None:
        payload = _example_yaml_config()
        payload["runtime"] = {"max_turns": 20, "timeout_seconds": 120.5}

        config = AgentConfig.model_validate(payload)

        assert config.runtime.max_turns == 20
        assert config.runtime.timeout_seconds == 120.5

    @pytest.mark.parametrize(("field", "value"), [("max_turns", 0), ("timeout_seconds", 0)])
    def test_runtime_constraints_must_be_positive(self, field: str, value: int) -> None:
        payload = _example_yaml_config()
        payload["runtime"] = {field: value}

        with pytest.raises(ValidationError, match="Input should be greater than 0"):
            AgentConfig.model_validate(payload)

    def test_default_harness_must_reference_configured_harness(self) -> None:
        with pytest.raises(ValidationError, match="default_harness must reference one of harnesses: codex"):
            AgentConfig.model_validate(
                {
                    "config_format": "nemo-agents-spec-v1",
                    "name": "bad-agent",
                    "default_harness": "hermes",
                    "harnesses": {"codex": {"kind": "codex"}},
                }
            )

    def test_unknown_top_level_fields_rejected(self) -> None:
        payload = _example_yaml_config()
        payload["unexpected"] = "value"

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            AgentConfig.model_validate(payload)

    def test_config_format_must_match_platform_spec_version(self) -> None:
        payload = _example_yaml_config()
        payload["config_format"] = "nat-workflow-v1"

        with pytest.raises(ValidationError, match="Input should be 'nemo-agents-spec-v1'"):
            AgentConfig.model_validate(payload)

    def test_unknown_nested_fields_rejected_outside_settings(self) -> None:
        payload = _example_yaml_config()
        payload["harnesses"]["codex"]["unknown"] = "value"

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            AgentConfig.model_validate(payload)

    def test_blank_instruction_content_rejected(self) -> None:
        payload = _example_yaml_config()
        payload["instructions"]["system"]["content"] = "   "

        with pytest.raises(ValidationError, match="String should match pattern"):
            AgentConfig.model_validate(payload)


def _write_agent_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


class TestLoadAgentConfig:
    def test_load_agent_config_reads_yaml_file(self, tmp_path: Path) -> None:
        config_path = tmp_path / "custom-agent.yaml"
        _write_agent_yaml(config_path, _example_yaml_config())

        config = load_agent_config(config_path)

        assert config.name == "test-agent"
        assert config.default_harness == "hermes"

    def test_load_agent_config_from_dir_uses_canonical_filename(self, tmp_path: Path) -> None:
        config_path = tmp_path / "agent.yaml"
        _write_agent_yaml(config_path, _example_yaml_config())

        config = load_agent_config_from_dir(tmp_path)

        assert config.name == "test-agent"

    def test_missing_file_reports_load_error(self, tmp_path: Path) -> None:
        with pytest.raises(AgentConfigLoadError, match="Unable to read agent config"):
            load_agent_config(tmp_path / "missing.yaml")

    def test_invalid_yaml_reports_load_error(self, tmp_path: Path) -> None:
        config_path = tmp_path / "agent.yaml"
        config_path.write_text("name: [", encoding="utf-8")

        with pytest.raises(AgentConfigLoadError, match="YAML parse error"):
            load_agent_config(config_path)

    def test_non_mapping_yaml_reports_load_error(self, tmp_path: Path) -> None:
        config_path = tmp_path / "agent.yaml"
        config_path.write_text("- not-a-mapping\n", encoding="utf-8")

        with pytest.raises(AgentConfigLoadError, match="root must be a YAML mapping"):
            load_agent_config(config_path)

    def test_validation_error_reports_load_error(self, tmp_path: Path) -> None:
        config_path = tmp_path / "agent.yaml"
        payload = _example_yaml_config()
        del payload["default_harness"]
        _write_agent_yaml(config_path, payload)

        with pytest.raises(AgentConfigLoadError, match="Invalid agent config"):
            load_agent_config(config_path)
