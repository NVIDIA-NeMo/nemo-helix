# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from nemo_agents_plugin.fabric.translator import translate_agent_config
from prompt_master_plugin.config import PromptMasterConfig
from prompt_master_plugin.runner import (
    PromptMasterExecutionError,
    build_optimizer_agent,
    extract_optimized_prompt,
    optimize_prompt,
)


def _config() -> PromptMasterConfig:
    return PromptMasterConfig.model_validate(
        {
            "model": {
                "provider": "openai",
                "model": "gpt-5.6",
                "api_key_env": "OPENAI_API_KEY",
                "temperature": 0.0,
            },
            "prompt_override": "You are a custom one-shot Prompt Master runner.",
            "timeout_seconds": 45,
        }
    )


def _agent_config() -> dict[str, Any]:
    return {
        "schema_version": "fabric.agent/v1alpha1",
        "metadata": {"name": "calculator-agent"},
        "harness": {"adapter_id": "nvidia.fabric.langchain.deepagents"},
        "models": {"default": {"provider": "nvidia", "model": "calculator-model"}},
        "instructions": {
            "system": {"content": "You are a concise calculator agent. Solve arithmetic and return only the answer."}
        },
    }


def test_builds_a_deepagents_fabric_agent_with_the_bundled_skill() -> None:
    agent = build_optimizer_agent(_config())

    assert agent.default_harness == "deepagents"
    assert agent.harnesses["deepagents"].kind == "deepagents"
    assert agent.models["default"].model == "gpt-5.6"
    assert agent.models["default"].api_key_env == "OPENAI_API_KEY"
    assert agent.skills is not None
    assert len(agent.skills.paths) == 1
    # The path must be the skills *library*, not the skill: harnesses enumerate the directory's
    # children and take each one holding a SKILL.md, so naming the skill itself loads nothing —
    # silently, with no error.  Asserting the library both ways pins the level, not just the name.
    library_path = Path(agent.skills.paths[0])
    assert (library_path / "prompt-master" / "SKILL.md").is_file()
    assert not (library_path / "SKILL.md").exists()
    assert agent.environment.workspace == "workspace"
    assert agent.environment.artifacts == "artifacts"
    assert agent.instructions is not None
    assert agent.instructions.system is not None
    assert agent.instructions.system.content == "You are a custom one-shot Prompt Master runner."


def test_optimizer_agent_translates_to_a_fabric_config() -> None:
    fabric_config = translate_agent_config(build_optimizer_agent(_config()))

    assert fabric_config.harness is not None
    assert fabric_config.harness.adapter_id == "nvidia.fabric.langchain.deepagents"
    assert fabric_config.models["default"].model == "gpt-5.6"
    assert fabric_config.skills is not None
    assert len(fabric_config.skills.paths) == 1


def test_executes_the_skill_through_fabric_and_returns_the_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_invoke(request: Any) -> Any:
        captured["request"] = request
        return SimpleNamespace(
            status="succeeded",
            response=(
                "```\n"
                "You are a coding assistant. Diagnose the smallest root cause, apply a scoped fix, "
                "and verify it with focused tests.\n"
                "```\n"
                "🎯 Target: Fabric agent, 💡 Added scope and verification criteria."
            ),
            error=None,
        )

    monkeypatch.setattr(
        "prompt_master_plugin.runner.invoke_agent_config_request_once",
        fake_invoke,
    )

    optimized = asyncio.run(optimize_prompt(_config(), agent_config=_agent_config(), base_dir=tmp_path))

    request = captured["request"]
    assert request.timeout_seconds == 45
    assert request.base_dir == tmp_path
    assert "Use the prompt-master skill" in request.input
    assert "<existing_prompt>You are a concise calculator agent." in request.input
    assert "nvidia.fabric.langchain.deepagents" in request.input
    assert "calculator-model" in request.input
    assert optimized.startswith("You are a coding assistant.")
    assert optimized.endswith("focused tests.")


def test_rejects_a_failed_fabric_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_invoke(_request: Any) -> Any:
        return SimpleNamespace(status="failed", response=None, error="provider unavailable")

    monkeypatch.setattr(
        "prompt_master_plugin.runner.invoke_agent_config_request_once",
        fake_invoke,
    )

    with pytest.raises(PromptMasterExecutionError, match="provider unavailable"):
        asyncio.run(optimize_prompt(_config(), agent_config=_agent_config(), base_dir=tmp_path))


def test_wraps_fabric_execution_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_agents_plugin.fabric.runtime import FabricRuntimeExecutionError

    async def fake_invoke(_request: Any) -> Any:
        raise FabricRuntimeExecutionError("adapter could not start")

    monkeypatch.setattr(
        "prompt_master_plugin.runner.invoke_agent_config_request_once",
        fake_invoke,
    )

    with pytest.raises(PromptMasterExecutionError, match="adapter could not start"):
        asyncio.run(optimize_prompt(_config(), agent_config=_agent_config(), base_dir=tmp_path))


def test_rejects_an_agent_without_system_instructions(tmp_path: Path) -> None:
    agent_config = _agent_config()
    agent_config.pop("instructions")

    with pytest.raises(PromptMasterExecutionError, match="instructions.system.content"):
        asyncio.run(optimize_prompt(_config(), agent_config=agent_config, base_dir=tmp_path))


def test_extracts_the_first_copyable_prompt_block() -> None:
    response = """Strategy note first.

```markdown
Role: You are a precise assistant.
Task: Answer only from supplied context.
```

🎯 Target: Fabric agent, 💡 Tightened grounding.
"""

    assert extract_optimized_prompt(response) == (
        "Role: You are a precise assistant.\nTask: Answer only from supplied context."
    )


def test_rejects_a_response_without_a_prompt_block() -> None:
    with pytest.raises(PromptMasterExecutionError, match="copyable prompt block"):
        extract_optimized_prompt("No fenced block was returned.")
