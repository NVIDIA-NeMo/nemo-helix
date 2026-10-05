# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from nemo_agents_plugin.agent_config import AgentConfig
from nemo_agents_plugin.fabric.gateway_credentials import PLATFORM_IGW_API_KEY_ENV, PLATFORM_IGW_API_KEY_PLACEHOLDER
from nemo_agents_plugin.fabric.runtime import FabricRuntimeExecutionError
from nemo_agents_plugin.fabric.translator import translate_agent_config
from prompt_master_plugin import runner as runner_module
from prompt_master_plugin.runner import (
    PromptMasterExecutionError,
    build_optimization_input,
    build_optimizer_agent,
    extract_optimized_prompt,
    optimize_prompt,
    run_prompt_master,
    stage_skills,
)

PLATFORM_URL = "http://platform:8080"
GATEWAY_PATH = "/apis/inference-gateway/v2/workspaces/team/openai/-/v1"
GATEWAY_URL = f"{PLATFORM_URL}{GATEWAY_PATH}"
PROXY_URL = "http://127.0.0.1:4321"
RESPONSE = "```\nnew prompt\n```\n🎯 Target: Fabric agent."
#: The skills library as agent.yaml names it, seen from the Fabric workspace root.
SKILLS_SOURCE = "/.agents/skills"

OVERRIDES: dict[str, Any] = {
    "models": {"default": {"model": "gpt-5.6", "temperature": 0.0}},
    "runtime": {"timeout_seconds": 45},
}


def _optimizer(overrides: dict[str, Any] | None = None) -> AgentConfig:
    return build_optimizer_agent(overrides, workspace="team", platform_base_url=PLATFORM_URL)


def _agent_config() -> dict[str, Any]:
    return {
        "config_format": "nemo-agents-spec-v1",
        "name": "calculator-agent",
        "default_harness": "hermes",
        "harnesses": {"hermes": {"kind": "hermes", "model": {"provider": "openai", "model": "harness-model"}}},
        "instructions": {"system": {"content": "You are a concise calculator agent."}},
    }


def _fake_invoke(monkeypatch: pytest.MonkeyPatch, result: Any) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def fake(request: Any) -> Any:
        captured["request"] = request
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(runner_module, "invoke_agent_config_request_once", fake)
    return captured


def _optimize(tmp_path: Path, **kwargs: Any) -> Any:
    return asyncio.run(
        optimize_prompt(_optimizer(OVERRIDES), agent_config=_agent_config(), base_dir=tmp_path, **kwargs)
    )


def test_builds_bundled_optimizer_agent() -> None:
    agent = _optimizer()

    assert agent.default_harness == "deepagents"
    assert agent.models["default"].provider == "nvidia"
    assert agent.models["default"].base_url == GATEWAY_URL
    assert agent.models["default"].api_key_env is None
    assert agent.instructions and agent.instructions.system
    assert "prompt-master skill" in agent.instructions.system.content
    # The harness sees the skills *library* (directory of skill dirs) at a virtual path under the
    # Fabric workspace, where stage_skills() puts it -- never a host path, which the
    # workspace-rooted Deep Agents filesystem cannot see.
    assert agent.skills and agent.skills.paths == [SKILLS_SOURCE]
    assert agent.environment.workspace == "workspace"


def test_merges_overrides_into_bundled_agent() -> None:
    agent = _optimizer({**OVERRIDES, "instructions": {"system": {"content": "custom"}}})

    assert agent.models["default"].model == "gpt-5.6"
    assert agent.models["default"].provider == "nvidia"
    assert agent.runtime.timeout_seconds == 45
    assert agent.instructions and agent.instructions.system
    assert agent.instructions.system.content == "custom"


def test_reads_platform_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEMO_BASE_URL", raising=False)
    monkeypatch.setenv("NHX_BASE_URL", "http://from-env:9000/")

    agent = build_optimizer_agent(None, workspace="default")

    assert (
        agent.models["default"].base_url
        == "http://from-env:9000/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
    )


def test_translates_with_gateway_credential() -> None:
    fabric = translate_agent_config(_optimizer())

    assert fabric.harness and fabric.harness.adapter_id == "nvidia.fabric.langchain.deepagents"
    assert fabric.models["default"].api_key_env == PLATFORM_IGW_API_KEY_ENV
    assert fabric.environment and fabric.environment.env[PLATFORM_IGW_API_KEY_ENV] == PLATFORM_IGW_API_KEY_PLACEHOLDER


def test_builds_optimization_input() -> None:
    task = build_optimization_input(_agent_config())

    assert "using the hermes harness and harness-model model" in task
    assert "<existing_prompt>You are a concise calculator agent.</existing_prompt>" in task


def test_prefers_default_model() -> None:
    config = _agent_config() | {"models": {"default": {"provider": "nvidia", "model": "top-model"}}}

    assert "top-model model" in build_optimization_input(config)


def test_rejects_missing_system_prompt() -> None:
    config = _agent_config()
    del config["instructions"]

    with pytest.raises(PromptMasterExecutionError, match="instructions.system.content"):
        build_optimization_input(config)


def test_runs_through_fabric(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _fake_invoke(monkeypatch, SimpleNamespace(status="succeeded", response=RESPONSE, error=None))

    outcome = _optimize(tmp_path)

    request = captured["request"]
    assert request.timeout_seconds == 45
    assert request.base_dir == tmp_path
    assert "Use the prompt-master skill" in request.input
    assert request.agent_config.models["default"].base_url == GATEWAY_URL
    assert outcome.optimized_prompt == "new prompt"
    assert outcome.response == RESPONSE


def test_routes_through_auth_proxy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _fake_invoke(monkeypatch, SimpleNamespace(status="succeeded", response=RESPONSE, error=None))

    _optimize(tmp_path, proxy_origin=PROXY_URL)

    assert captured["request"].agent_config.models["default"].base_url == f"{PROXY_URL}{GATEWAY_PATH}"


def test_run_prompt_master_opens_proxy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _fake_invoke(monkeypatch, SimpleNamespace(status="succeeded", response=RESPONSE, error=None))
    monkeypatch.setattr(runner_module, "platform_auth_proxy", lambda: contextlib.nullcontext(PROXY_URL))

    outcome = run_prompt_master(_optimizer(), _agent_config(), tmp_path)

    assert outcome.optimized_prompt == "new prompt"
    assert captured["request"].agent_config.models["default"].base_url == f"{PROXY_URL}{GATEWAY_PATH}"


@pytest.mark.parametrize(
    ("result", "match"),
    [
        (SimpleNamespace(status="failed", response=None, error="provider unavailable"), "provider unavailable"),
        (SimpleNamespace(status="succeeded", response="", error=None), "no response"),
        (FabricRuntimeExecutionError("adapter could not start"), "adapter could not start"),
    ],
)
def test_rejects_failed_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, result: Any, match: str) -> None:
    _fake_invoke(monkeypatch, result)

    with pytest.raises(PromptMasterExecutionError, match=match):
        _optimize(tmp_path)


def test_falls_back_to_the_first_fenced_block_without_a_target_line() -> None:
    response = "Note.\n\n```markdown\nRole: assistant.\nTask: answer.\n```\n\nDone."

    assert extract_optimized_prompt(response) == "Role: assistant.\nTask: answer."


def test_rejects_missing_prompt_block() -> None:
    with pytest.raises(PromptMasterExecutionError, match="copyable prompt block"):
        extract_optimized_prompt("No fenced block.")


def test_missing_prompt_block_error_quotes_the_response() -> None:
    # The job log only ever sees the exception message, so the model's actual reply must ride
    # along or the next parse failure is undiagnosable (as the first one was).
    with pytest.raises(PromptMasterExecutionError, match="Here is an improved prompt without a fence"):
        extract_optimized_prompt("Here is an improved prompt without a fence.\nRole: assistant.")


def test_missing_prompt_block_error_truncates_a_long_response() -> None:
    response = "word " * 1000

    with pytest.raises(PromptMasterExecutionError) as excinfo:
        extract_optimized_prompt(response)

    assert len(str(excinfo.value)) < 600


def test_rejects_missing_bundled_skill_before_invoking_fabric(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A skills library without prompt-master/SKILL.md loads nothing, silently, and the optimizer
    # then answers without Prompt Master's output format.  Fail before paying for that model call.
    captured = _fake_invoke(monkeypatch, SimpleNamespace(status="succeeded", response=RESPONSE, error=None))
    monkeypatch.setattr(runner_module, "SKILLS_DIR", tmp_path / "no-skills-here")

    with pytest.raises(PromptMasterExecutionError, match=r"prompt-master/SKILL\.md"):
        _optimize(tmp_path)

    assert "request" not in captured


def test_stages_the_skill_library_into_the_workspace(tmp_path: Path) -> None:
    staged = stage_skills(tmp_path)

    assert staged == tmp_path / "workspace" / ".agents" / "skills"
    assert (staged / "prompt-master" / "SKILL.md").is_file()
    assert (staged / "prompt-master" / "references" / "templates.md").is_file()


def test_restaging_replaces_a_previous_staging(tmp_path: Path) -> None:
    stale = stage_skills(tmp_path) / "stale-skill" / "SKILL.md"
    stale.parent.mkdir()
    stale.write_text("---\nname: stale-skill\ndescription: left over\n---\n", encoding="utf-8")

    staged = stage_skills(tmp_path)

    assert not stale.exists()
    assert (staged / "prompt-master" / "SKILL.md").is_file()


def test_stages_skills_before_invoking_fabric(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _fake_invoke(monkeypatch, SimpleNamespace(status="succeeded", response=RESPONSE, error=None))

    _optimize(tmp_path)

    assert (tmp_path / "workspace" / ".agents" / "skills" / "prompt-master" / "SKILL.md").is_file()
    assert captured["request"].agent_config.skills.paths == [SKILLS_SOURCE]


def test_deep_agents_discovers_the_staged_skill(tmp_path: Path) -> None:
    # Pin the contract against the harness itself: the Deep Agents adapter roots its filesystem
    # at the workspace in virtual mode, and its skills middleware must find prompt-master there.
    # (A host path, which the runner used to pass, resolves to nothing under that root.)
    backends = pytest.importorskip("deepagents.backends")
    skills_middleware = pytest.importorskip("deepagents.middleware.skills")
    stage_skills(tmp_path)
    backend = backends.FilesystemBackend(root_dir=str(tmp_path / "workspace"), virtual_mode=True)

    skills, error = skills_middleware._list_skills_with_errors(backend, SKILLS_SOURCE)

    assert error is None
    assert [skill["name"] for skill in skills] == ["prompt-master"]
    assert skills[0]["path"] == f"{SKILLS_SOURCE}/prompt-master/SKILL.md"


def test_splits_the_prompt_from_the_target_line() -> None:
    # Prompt Master's documented shape, exactly as the 120B optimizer returned it: no fence.
    response = (
        "You are a calculator agent. Solve arithmetic and numeric comparison requests. "
        "Return only the numerical answer. 🎯 Target: NeMo Fabric agent with deepagents harness "
        "and nvidia-nemotron-3-5-lightning-30b-a3b model,💡 Removed redundancy."
    )

    assert extract_optimized_prompt(response) == (
        "You are a calculator agent. Solve arithmetic and numeric comparison requests. "
        "Return only the numerical answer."
    )


def test_unwraps_a_fenced_prompt_before_the_target_line() -> None:
    response = "```markdown\nRole: assistant.\nTask: answer.\n```\n\n🎯 Target: Cursor,💡 Tightened."

    assert extract_optimized_prompt(response) == "Role: assistant.\nTask: answer."


def test_keeps_fences_inside_an_unwrapped_prompt() -> None:
    # A fence inside the prompt is the prompt's own example, not the delimiter.
    prompt = 'Reply in JSON, for example:\n```json\n{"answer": 4}\n```\nNo prose.'

    assert extract_optimized_prompt(prompt + "\n\n🎯 Target: GPT,💡 Locked the format.") == prompt


@pytest.mark.parametrize(
    "tail",
    [
        "\n\n🎯 **Target:** Cursor,💡 note",
        "\n🎯Target: Cursor",
        "\n\n---\n🎯 Target: Cursor",
        " 🎯\ufe0f Target: Cursor",
    ],
)
def test_tolerates_target_line_variants(tail: str) -> None:
    assert extract_optimized_prompt("The prompt." + tail) == "The prompt."


def test_rejects_a_reply_that_is_only_a_target_line() -> None:
    with pytest.raises(PromptMasterExecutionError, match="empty prompt"):
        extract_optimized_prompt("🎯 Target: Cursor,💡 Nothing to show.")
