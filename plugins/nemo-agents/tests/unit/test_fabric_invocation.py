# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Fabric one-shot invocation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from nemo_agents_plugin.agent_config import AgentConfig
from nemo_agents_plugin.fabric.invocation import (
    AgentConfigInvocationRequest,
    FabricDirectories,
    invoke_agent_config_once,
    invoke_agent_config_request_once,
)
from nemo_agents_plugin.fabric.runtime import FabricRuntimeResult


def _agent_config() -> dict[str, Any]:
    return {
        "config_format": "nemo-agents-spec-v1",
        "name": "fabric-agent",
        "default_harness": "hermes",
        "harnesses": {
            "hermes": {
                "kind": "hermes",
            },
        },
        "models": {
            "default": {
                "provider": "openai",
                "model": "openai/gpt-5.4",
            },
        },
    }


@pytest.mark.asyncio
async def test_invoke_agent_config_request_once_translates_and_runs_one_input(tmp_path: Path) -> None:
    captured: list[Any] = []

    async def _run_fabric_agent_once(request: Any) -> FabricRuntimeResult:
        captured.append(request)
        return FabricRuntimeResult(status="succeeded", response=f"response:{request.input}")

    agent_config = AgentConfig.model_validate(_agent_config())
    with patch("nemo_agents_plugin.fabric.invocation.run_fabric_agent_once", _run_fabric_agent_once):
        result = await invoke_agent_config_request_once(
            AgentConfigInvocationRequest(
                agent_config=agent_config,
                input="one",
                base_dir=tmp_path,
                request_id="job-1",
                caller_context={"source": "job"},
                timeout_seconds=12,
            )
        )

    assert result.response == "response:one"
    request = captured[0]
    assert request.input == "one"
    assert request.base_dir == tmp_path
    assert request.request_id == "job-1"
    assert request.caller_context == {"source": "job"}
    assert request.timeout_seconds == 12
    assert request.fabric_config.metadata.name == "fabric-agent"


@pytest.mark.asyncio
async def test_one_shot_deepagents_skills_are_staged_inside_the_workspace(tmp_path: Path) -> None:
    skill = tmp_path / "skills" / "eval-author"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Eval Author\n")
    config = _agent_config()
    config["default_harness"] = "deepagents"
    config["harnesses"] = {"deepagents": {"kind": "deepagents"}}
    config["environment"] = {"workspace": "./workspace", "artifacts": "./artifacts"}
    config["skills"] = {"paths": ["skills/eval-author"]}
    captured: list[Any] = []

    async def _run_fabric_agent_once(request: Any) -> FabricRuntimeResult:
        captured.append(request)
        return FabricRuntimeResult(status="succeeded")

    agent_config = AgentConfig.model_validate(config)
    with patch("nemo_agents_plugin.fabric.invocation.run_fabric_agent_once", _run_fabric_agent_once):
        await invoke_agent_config_request_once(
            AgentConfigInvocationRequest(agent_config=agent_config, input="one", base_dir=tmp_path)
        )

    staged = tmp_path / "workspace" / ".nemo-deepagents" / "skills" / "eval-author" / "SKILL.md"
    assert staged.read_text() == "# Eval Author\n"
    assert [str(path) for path in captured[0].fabric_config.skills.paths] == ["/.nemo-deepagents/skills"]


@pytest.mark.asyncio
async def test_invoke_agent_config_once_translates_and_runs_each_input(tmp_path: Path) -> None:
    captured: list[Any] = []

    async def _run_fabric_agent_once(request: Any) -> FabricRuntimeResult:
        captured.append(request)
        return FabricRuntimeResult(status="succeeded", response=f"response:{request.input}")

    agent_config = AgentConfig.model_validate(_agent_config())
    with patch("nemo_agents_plugin.fabric.invocation.run_fabric_agent_once", _run_fabric_agent_once):
        results = await invoke_agent_config_once(agent_config, ["one", "two"], base_dir=tmp_path)

    assert [result.response for result in results] == ["response:one", "response:two"]
    assert [request.input for request in captured] == ["one", "two"]
    assert all(request.base_dir == tmp_path for request in captured)
    assert captured[0].fabric_config.metadata.name == "fabric-agent"


@pytest.mark.asyncio
async def test_invoke_agent_config_once_creates_local_workspace_dir(tmp_path: Path) -> None:
    config = _agent_config()
    config["environment"] = {"workspace": "./workspace"}

    async def _run_fabric_agent_once(request: Any) -> FabricRuntimeResult:
        assert (tmp_path / "workspace").is_dir()
        return FabricRuntimeResult(status="succeeded")

    agent_config = AgentConfig.model_validate(config)
    with patch("nemo_agents_plugin.fabric.invocation.run_fabric_agent_once", _run_fabric_agent_once):
        await invoke_agent_config_once(agent_config, ["one"], base_dir=tmp_path)


@pytest.mark.asyncio
async def test_invoke_agent_config_request_once_creates_local_artifacts_dir(tmp_path: Path) -> None:
    config = _agent_config()
    config["environment"] = {"workspace": "./workspace", "artifacts": "./artifacts"}

    async def _run_fabric_agent_once(request: Any) -> FabricRuntimeResult:
        assert (tmp_path / "workspace").is_dir()
        assert (tmp_path / "artifacts").is_dir()
        return FabricRuntimeResult(status="succeeded")

    agent_config = AgentConfig.model_validate(config)
    with patch("nemo_agents_plugin.fabric.invocation.run_fabric_agent_once", _run_fabric_agent_once):
        await invoke_agent_config_request_once(
            AgentConfigInvocationRequest(agent_config=agent_config, input="one", base_dir=tmp_path)
        )


@pytest.mark.asyncio
async def test_invoke_agent_config_once_rejects_absolute_local_workspace(tmp_path: Path) -> None:
    config = _agent_config()
    config["environment"] = {"workspace": str(tmp_path.parent / "outside")}
    agent_config = AgentConfig.model_validate(config)

    expected_msg = "Local workspace path must be relative"

    with pytest.raises(ValueError, match=expected_msg):
        FabricDirectories.create(agent_config, tmp_path)

    with pytest.raises(ValueError, match=expected_msg):
        await invoke_agent_config_once(agent_config, ["one"], base_dir=tmp_path)


@pytest.mark.asyncio
async def test_invoke_agent_config_once_rejects_workspace_traversal(tmp_path: Path) -> None:
    config = _agent_config()
    config["environment"] = {"workspace": "../../outside"}
    agent_config = AgentConfig.model_validate(config)

    expected_msg = "Local workspace path must remain within"

    with pytest.raises(ValueError, match=expected_msg):
        FabricDirectories.create(agent_config, tmp_path)

    with pytest.raises(ValueError, match=expected_msg):
        await invoke_agent_config_once(agent_config, ["one"], base_dir=tmp_path)
