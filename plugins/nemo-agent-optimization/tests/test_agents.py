# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from nemo_agent_optimization_plugin.agents import resolve_agent_config
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.run_dependencies import LocalRunError


def test_resolve_agent_config_returns_none_without_agent() -> None:
    assert resolve_agent_config(None, workspace="default", sdk=None) is None


def test_resolve_agent_config_loads_a_local_agent_yaml(tmp_path: Path) -> None:
    agent_path = tmp_path / "agent.yaml"
    agent_path.write_text(
        yaml.safe_dump(
            {
                "config_format": "nemo-agents-spec-v1",
                "name": "calculator-agent",
                "default_harness": "deepagents",
                "harnesses": {"deepagents": {"kind": "deepagents", "settings": {"deepagents": {}}}},
                "models": {"default": {"provider": "nvidia", "model": "calculator-model"}},
                "instructions": {"system": {"content": "Return only the answer."}},
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_agent_config(str(agent_path), workspace="default", sdk=None)

    assert resolved is not None
    assert resolved["schema_version"] == "fabric.agent/v1alpha1"
    assert resolved["metadata"]["name"] == "calculator-agent"
    assert resolved["instructions"]["system"]["content"] == "Return only the answer."


def test_resolve_agent_config_resolves_a_platform_agent() -> None:
    platform_agent = {
        "config_format": "nemo-agents-spec-v1",
        "name": "react-agent",
        "default_harness": "hermes",
        "harnesses": {
            "hermes": {
                "kind": "hermes",
                "model": {
                    "provider": "openai",
                    "model": "demo-model",
                    "base_url": "http://localhost:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1",
                    "api_key_env": "NEMO_AGENTS_IGW_API_KEY",
                },
                "settings": {"max_tokens": 256, "reasoning_config": {"effort": "none"}},
            }
        },
        "instructions": {"system": {"content": "Be brief."}},
        "environment": {"provider": "local", "workspace": "./workspace", "artifacts": "./artifacts"},
        "models": {
            "judge": {
                "provider": "openai",
                "model": "demo-model",
                "base_url": "http://localhost:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1",
                "api_key_env": "NEMO_AGENTS_IGW_API_KEY",
            }
        },
    }

    class _StubAgents:
        def get(self, *, name: str, workspace: str) -> dict[str, Any]:
            assert name == "react-agent"
            assert workspace == "default"
            return {"config": platform_agent}

    class _StubSDK:
        agents = _StubAgents()

    resolved = resolve_agent_config("react-agent", workspace="default", sdk=cast(NeMoPlatform, _StubSDK()))

    assert resolved is not None
    assert resolved["schema_version"] == "fabric.agent/v1alpha1"
    assert resolved["harness"]["adapter_id"] == "nvidia.fabric.hermes"
    assert resolved["models"]["default"]["model"] == "demo-model"
    assert resolved["models"]["judge"]["model"] == "demo-model"


def test_resolve_agent_config_requires_sdk_for_platform_ref() -> None:
    with pytest.raises(LocalRunError, match="requires a platform SDK"):
        resolve_agent_config("react-agent", workspace="default", sdk=None)


def test_resolve_agent_config_rejects_endpoint_uri() -> None:
    with pytest.raises(LocalRunError, match="Endpoint URL"):
        resolve_agent_config("http://example.com/agent", workspace="default", sdk=None)
