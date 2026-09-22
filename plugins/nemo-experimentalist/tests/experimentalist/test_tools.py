# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import cast

import pytest
from nemo_experimentalist_plugin.experimentalist.components.coder import CodeEditBuilder, CodeEditBuilderConfig
from nemo_experimentalist_plugin.experimentalist.components.holdout_utils import BLOCKED_MESSAGE
from nemo_experimentalist_plugin.experimentalist.components.tools import GuardedShellTools
from nemo_helix_plugin.nooa_model_client import ConfiguredModelClients, ConfiguredModelRefs, activate_model_clients
from nooa.agentdoc import pformat
from nooa.errors import GenerationError
from nooa.tools import ShellResult
from nooa.unifiedllm import CompletionClient, FakeLLMClient, LLMResponse, ToolCall


def _exec_response(code: str) -> LLMResponse:
    """A scripted LLM turn that drives CodeAct's ``execute_python`` tool with ``code``."""
    return LLMResponse(
        raw_response=None,
        content="",
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": ""},
        tool_calls=[ToolCall(id="call_exec", name="execute_python", arguments=json.dumps({"code": code}))],
    )


async def test_guarded_shell_tools_runs_allowed_commands(tmp_path):
    shell = GuardedShellTools(cwd=tmp_path)
    try:
        result = await shell.run("python -", stdin="print('allowed')")
    finally:
        await shell.close()

    assert result.stdout.strip() == "allowed"
    assert result.returncode == 0
    assert result.success


async def test_guarded_shell_tools_returns_failure_for_blocked_paths(tmp_path):
    shell = GuardedShellTools(cwd=tmp_path)
    try:
        result = await shell.run("cat dataset/validation/secret")
    finally:
        await shell.close()

    assert isinstance(result, ShellResult)
    assert result.stdout == ""
    assert result.stderr == BLOCKED_MESSAGE
    assert result.returncode == 1
    assert not result.success


def test_coder_hides_skill_registry_that_can_replace_guarded_shell(tmp_path):
    nooa_skill = Path(__file__).resolve().parents[2] / "framework-skills" / "nooa"
    coder = CodeEditBuilder(workspace=tmp_path, framework_skills_dirs=[nooa_skill])

    rendered = pformat(coder)

    assert "skills=SkillRegistry" not in rendered
    assert "shell=ShellTools" in rendered
    assert "nooa=Nooa" in rendered


def test_coder_uses_default_model_for_architecture_docs(tmp_path: Path) -> None:
    default = cast(CompletionClient, FakeLLMClient())
    fast = cast(CompletionClient, FakeLLMClient())
    clients = ConfiguredModelClients(
        default=default,
        fast=fast,
        refs=ConfiguredModelRefs(default="default/quality", fast="default/fast"),
    )

    with activate_model_clients(clients):
        coder = CodeEditBuilder(workspace=tmp_path)

    assert coder._architecture_model is default


async def test_architecture_doc_stops_at_the_configured_iteration_limit(tmp_path: Path) -> None:
    """The configured limit, not the one fixed on the @strategy decorator, must bound the run."""
    assert CodeEditBuilderConfig().max_architecture_doc_iterations == 100, "the documented default"

    # Each scripted turn runs a no-op cell instead of returning a result, so the only way
    # out is exhausting the limit. A turn past the limit would report a larger count.
    never_finishes = FakeLLMClient(scripted_responses=[_exec_response("x = 1") for _ in range(4)])
    builder = CodeEditBuilder(workspace=tmp_path, config=CodeEditBuilderConfig(max_architecture_doc_iterations=3))
    builder._architecture_model = never_finishes

    with pytest.raises(GenerationError, match=r"after 3 iterations \(max_iterations=3\)"):
        await builder.create_architecture_doc(tmp_path)


async def test_coder_lists_agent_mutation_models_from_catalog(tmp_path: Path) -> None:
    catalog = tmp_path / "models.yaml"
    catalog.write_text(
        """\
default_endpoint: https://models.example/v1
default_api_key_env: EXAMPLE_API_KEY
id_field: model_id
models:
  - model_id: provider/quality
    provider: Example
    range: quality
    notes: Quality model.
  - model_id: provider/fast
    provider: Example
    range: fast
    notes: Fast model.
""",
        encoding="utf-8",
    )
    coder = CodeEditBuilder(workspace=tmp_path, config=CodeEditBuilderConfig(model_catalog_path=catalog))

    assert await coder.list_available_models() == ["provider/quality", "provider/fast"]
