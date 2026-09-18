# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Studio local assistant bridge."""

import asyncio
import json
import logging
import re
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from nmp.common.entities.client import EntityConflictError, EntityNotFoundError
from nmp.common.service.dependencies import get_entity_client
from nmp.studio import assistant, assistant_artifacts, assistant_skills, studio_links
from nmp.studio.config import StudioConfig
from nmp.studio.entities import AssistantConversation, AssistantMessage, LegacyAssistantConversation
from nmp.studio.service import StudioService


class FakeEntityStore:
    """Small async EntityClient fake for Assistant route tests."""

    def __init__(self) -> None:
        self.entities: dict[tuple[str, str], AssistantConversation] = {}

    async def create(self, entity: AssistantConversation) -> AssistantConversation:
        now = datetime.now(UTC)
        entity._created_at = now
        entity._updated_at = now
        self.entities[(entity.workspace, entity.name)] = entity
        return entity

    async def get(
        self,
        entity_type: type[AssistantConversation],
        name: str,
        *,
        workspace: str | None = None,
    ) -> AssistantConversation:
        del entity_type
        try:
            return self.entities[(workspace or "default", name)]
        except KeyError as exc:
            raise EntityNotFoundError(name) from exc

    async def list(
        self,
        entity_type: type[AssistantConversation],
        *,
        workspace: str = "default",
        filter_obj: dict[str, Any] | None = None,
        **_: Any,
    ) -> SimpleNamespace:
        del entity_type
        owner_id = filter_obj.get("owner_id") if filter_obj else None
        data = [
            entity
            for (entity_workspace, _), entity in self.entities.items()
            if entity_workspace == workspace and (owner_id is None or entity.owner_id == owner_id)
        ]
        data.sort(key=lambda entity: entity.updated_at or datetime.min.replace(tzinfo=UTC), reverse=True)
        return SimpleNamespace(data=data)

    async def update(self, entity: AssistantConversation) -> AssistantConversation:
        entity._updated_at = datetime.now(UTC)
        self.entities[(entity.workspace, entity.name)] = entity
        return entity

    async def delete(
        self,
        entity_type: type[AssistantConversation],
        name: str,
        *,
        workspace: str | None = None,
        expected_db_version: int | None = None,
    ) -> None:
        del entity_type, expected_db_version
        try:
            del self.entities[(workspace or "default", name)]
        except KeyError as exc:
            raise EntityNotFoundError(name) from exc


@pytest.fixture(autouse=True)
def reset_assistant_state():
    """Reset module-level bridge state between tests."""
    assistant._initialized_sessions.clear()
    assistant._session_streams.clear()
    assistant._pending_permissions.clear()
    assistant._pending_agent_inputs.clear()
    assistant._session_workspace_cache.clear()
    yield
    assistant._initialized_sessions.clear()
    assistant._session_streams.clear()
    assistant._pending_permissions.clear()
    assistant._pending_agent_inputs.clear()
    assistant._session_workspace_cache.clear()


@pytest.fixture
def entity_store() -> FakeEntityStore:
    return FakeEntityStore()


@pytest.fixture
def service_client(entity_store: FakeEntityStore) -> TestClient:
    service = StudioService()
    service.app.dependency_overrides[get_entity_client] = lambda: entity_store
    return TestClient(service.app)


def service_client_with_feature_flags(
    monkeypatch: pytest.MonkeyPatch, feature_flags: dict[str, bool | str]
) -> TestClient:
    monkeypatch.setattr(
        "nmp.studio.config.Configuration.get_global_settings_from_env",
        lambda: {"studio": {"feature_flags": feature_flags}},
    )
    return TestClient(StudioService().with_config(StudioConfig()).app)


def test_history_preserves_distinct_answers_under_an_agent_header():
    artifacts = assistant_artifacts.ChatArtifactsResponse()
    question_labels_by_tool_use_id: dict[str, dict[str, str]] = {}
    input_selection_tools_by_tool_use_id: dict[str, assistant_artifacts.InputSelectionTool] = {}
    questions = [
        {
            "header": "Agent",
            "question": "Which framework should the agent use?",
            "options": [{"label": "LangGraph"}, {"label": "NAT"}],
        },
        {
            "header": "Agent",
            "question": "How should the agent handle retries?",
            "options": [{"label": "Retry once"}, {"label": "Fail immediately"}],
        },
    ]

    assistant_artifacts.record_tool_artifacts(
        artifacts,
        "AskUserQuestion",
        {"questions": questions},
        "toolu_questions",
        question_labels_by_tool_use_id,
        input_selection_tools_by_tool_use_id,
    )
    assistant_artifacts.record_user_tool_result_artifacts(
        artifacts,
        [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_questions",
                "content": (
                    "Your questions have been answered: "
                    '"Which framework should the agent use?"="NAT", '
                    '"How should the agent handle retries?"="Retry once".'
                ),
            }
        ],
        question_labels_by_tool_use_id,
        input_selection_tools_by_tool_use_id,
    )

    assert [selection.model_dump() for selection in artifacts.selections] == [
        {"label": "Which framework should the agent use", "value": "NAT"},
        {"label": "How should the agent handle retries", "value": "Retry once"},
    ]
    assert artifacts.agent is None


def test_history_records_studio_picker_results_as_selections():
    artifacts = assistant_artifacts.ChatArtifactsResponse()
    question_labels_by_tool_use_id: dict[str, dict[str, str]] = {}
    input_selection_tools_by_tool_use_id: dict[str, assistant_artifacts.InputSelectionTool] = {}
    picker_calls = [
        ("toolu_agent", "mcp__nemo_studio__select_agent", {}, {"agent": "calculator-agent"}),
        (
            "toolu_model",
            "mcp__nemo_studio__select_model",
            {"output_key": "selected_model"},
            {"selected_model": "nvidia/llama-3.3-nemotron-super-49b-v1"},
        ),
        (
            "toolu_dataset",
            "mcp__nemo_studio__select_dataset_file",
            {},
            {"dataset_fileset": "evaluation-data", "dataset_path": "inputs/test.jsonl"},
        ),
        (
            "toolu_eval",
            "mcp__nemo_studio__select_eval_config",
            {},
            {"eval_config_fileset": "agent-evals", "eval_config": "configs/default.yml"},
        ),
    ]

    for tool_use_id, tool_name, tool_input, _result in picker_calls:
        assistant_artifacts.record_tool_artifacts(
            artifacts,
            tool_name,
            tool_input,
            tool_use_id,
            question_labels_by_tool_use_id,
            input_selection_tools_by_tool_use_id,
        )

    assistant_artifacts.record_user_tool_result_artifacts(
        artifacts,
        [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"status": "submitted", **result}),
                    }
                ],
            }
            for tool_use_id, _tool_name, _tool_input, result in picker_calls
        ],
        question_labels_by_tool_use_id,
        input_selection_tools_by_tool_use_id,
    )

    assert [selection.model_dump() for selection in artifacts.selections] == [
        {"label": "Agent", "value": "calculator-agent"},
        {"label": "Model", "value": "nvidia/llama-3.3-nemotron-super-49b-v1"},
        {"label": "Dataset", "value": "evaluation-data/inputs/test.jsonl"},
        {"label": "Eval config", "value": "agent-evals/configs/default.yml"},
    ]
    assert artifacts.agent == "calculator-agent"
    assert artifacts.model == "nvidia/llama-3.3-nemotron-super-49b-v1"
    assert artifacts.model_source == "selection"


def supported_destinations_from_description(description: str) -> set[str]:
    _, _, values = description.partition("Supported values: ")
    return set(values.removesuffix(".").split(", "))


def _inference_source_dir(root: Path) -> Path:
    source_dir = root / "packages" / "nemo_platform_ext" / "skills" / "inference"
    source_dir.mkdir(parents=True)
    return source_dir


def _inference_skill(source_dir: Path) -> assistant_skills.Skill:
    return assistant_skills.Skill(
        name="inference",
        description="Use NeMo Platform inference.",
        version="0.1",
        content="# Inference",
        raw="# Inference",
        source_dir=source_dir,
        source_plugin="platform",
        source_dist="nemo-platform-ext",
    )


def _expected_inference_skill_response(*, installed: bool) -> dict[str, Any]:
    return {
        "name": "inference",
        "claude_name": "nemo-inference",
        "description": "Use NeMo Platform inference.",
        "source": "nemo-platform",
        "source_path": "packages/nemo_platform_ext/skills/inference",
        "install_path": ".claude/skills/nemo-inference/SKILL.md",
        "installed": installed,
    }


def test_vendored_load_skills_from_root_loads_selected_root_without_registry_private_helper(tmp_path: Path):
    source_dir = _inference_source_dir(tmp_path)
    (source_dir / "SKILL.md").write_text(
        "---\nname: inference\ndescription: Use NeMo Platform inference.\nversion: 2\n---\n# Inference\n",
        encoding="utf-8",
    )

    loaded = assistant_skills.load_skills_from_root(
        tmp_path / "packages" / "nemo_platform_ext" / "skills",
        source_plugin="platform",
        source_dist="nemo-platform-ext",
    )

    assert list(loaded) == ["inference"]
    assert loaded["inference"].description == "Use NeMo Platform inference."
    assert loaded["inference"].version == "2"
    assert loaded["inference"].source_plugin == "platform"
    assert loaded["inference"].source_dist == "nemo-platform-ext"


def test_create_session_returns_uuid(service_client: TestClient):
    response = service_client.post("/v2/assistant/sessions")

    assert response.status_code == 200
    uuid.UUID(response.json()["session_id"])


def test_create_session_persists_workspace_and_owner(
    service_client: TestClient,
    entity_store: FakeEntityStore,
):
    response = service_client.post(
        "/v2/assistant/sessions?workspace=team-a",
        headers={"X-NMP-Principal-Id": "alice@example.com"},
    )

    session_id = response.json()["session_id"]
    persisted = entity_store.entities[("team-a", f"assistant-{session_id}")]
    assert persisted.owner_id == "alice@example.com"
    assert persisted.messages == []


def test_recent_conversation_messages_caps_model_context_without_mutating_history(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(assistant, "MAX_MODEL_CONTEXT_TURNS_PER_SESSION", 2)
    conversation = [
        AssistantMessage(role=role, content=f"{role}-{turn}") for turn in range(3) for role in ("user", "assistant")
    ]

    recent = assistant._recent_conversation_messages(conversation)

    assert [message.model_dump() for message in recent] == [
        {"role": "user", "content": "user-1"},
        {"role": "assistant", "content": "assistant-1"},
        {"role": "user", "content": "user-2"},
        {"role": "assistant", "content": "assistant-2"},
    ]
    assert len(conversation) == 6


def test_recent_conversation_messages_removes_legacy_studio_context():
    contextual_prompt = assistant._build_claude_prompt(
        "create a guardrail",
        "default",
        "https://studio.test",
        "/workspaces/default/guardrails",
    )
    conversation = [AssistantMessage(role="user", content=contextual_prompt)]

    recent = assistant._recent_conversation_messages(conversation)

    assert recent[0].content == "create a guardrail"
    assert conversation[0].content == contextual_prompt


def test_recent_conversation_messages_preserves_user_prompt_with_studio_context_markers():
    user_prompt = "\n".join(
        [
            assistant.STUDIO_CONTEXT_START,
            "Explain how these Studio wrapper markers work.",
            assistant.STUDIO_CONTEXT_END,
            "",
            assistant.STUDIO_CONTEXT_USER_REQUEST_PREFIX,
            "This text is part of my example.",
        ]
    )
    conversation = [AssistantMessage(role="user", content=user_prompt)]

    recent = assistant._recent_conversation_messages(conversation)

    assert recent[0].content == user_prompt


def test_list_history_sessions_includes_persisted_conversation(
    service_client: TestClient,
    entity_store: FakeEntityStore,
):
    session_id = str(uuid.uuid4())
    conversation = AssistantConversation(
        name=f"assistant-{session_id}",
        workspace="default",
        session_id=session_id,
        owner_id="local-user",
        messages=[
            AssistantMessage(role="user", content="Help me build an agent"),
            AssistantMessage(role="assistant", content="What should it do?"),
        ],
    )
    conversation._created_at = datetime.fromtimestamp(40, UTC)
    conversation._updated_at = datetime.fromtimestamp(42, UTC)
    entity_store.entities[("default", conversation.name)] = conversation

    response = service_client.get("/v2/assistant/history/sessions")

    assert response.status_code == 200
    assert response.json() == [
        {
            "session_id": session_id,
            "mtime": 42,
            "title": "Help me build an agent",
            "first_prompt": "Help me build an agent",
            "message_count": 1,
            "token_count": 0,
            "tool_call_count": 0,
            "tool_calls": [],
            "chat_artifacts": {
                "agent": None,
                "model": None,
                "model_source": None,
                "assistant_model": None,
                "workspace": None,
                "selections": [],
                "files": [],
                "links": [],
                "jobs": [],
                "tools": [],
            },
        }
    ]


def test_legacy_conversation_remains_listable_readable_and_deletable(
    service_client: TestClient,
    entity_store: FakeEntityStore,
):
    session_id = str(uuid.uuid4())
    conversation = LegacyAssistantConversation.model_validate(
        {
            "name": f"copilot-{session_id}",
            "workspace": "default",
            "session_id": session_id,
            "owner_id": "local-user",
            "messages": [
                {"role": "user", "content": "Legacy prompt"},
                {"role": "assistant", "content": "Legacy answer"},
            ],
            "chat_artifacts": {
                "model_source": "copilot",
                "copilot_model": "nvidia/legacy-model",
            },
        }
    )
    conversation._created_at = datetime.fromtimestamp(40, UTC)
    conversation._updated_at = datetime.fromtimestamp(42, UTC)
    entity_store.entities[("default", conversation.name)] = conversation

    list_response = service_client.get("/v2/assistant/history/sessions")
    history_response = service_client.get(f"/v2/assistant/history/sessions/{session_id}")
    delete_response = service_client.delete(f"/v2/assistant/history/sessions/{session_id}")

    assert list_response.status_code == 200
    assert list_response.json()[0]["chat_artifacts"]["assistant_model"] == "nvidia/legacy-model"
    assert history_response.status_code == 200
    assert history_response.json()["items"] == [
        {"kind": "user", "text": "Legacy prompt"},
        {"kind": "assistant", "parts": [{"type": "text", "text": "Legacy answer"}]},
    ]
    assert delete_response.status_code == 204
    assert ("default", conversation.name) not in entity_store.entities


def test_history_is_scoped_to_workspace_and_owner(
    service_client: TestClient,
    entity_store: FakeEntityStore,
):
    alice_id = service_client.post(
        "/v2/assistant/sessions?workspace=team-a",
        headers={"X-NMP-Principal-Id": "alice@example.com"},
    ).json()["session_id"]
    bob_id = service_client.post(
        "/v2/assistant/sessions?workspace=team-a",
        headers={"X-NMP-Principal-Id": "bob@example.com"},
    ).json()["session_id"]
    entity_store.entities[("team-a", f"assistant-{alice_id}")].messages = [
        AssistantMessage(role="user", content="Alice's private prompt"),
        AssistantMessage(role="assistant", content="Alice's answer"),
    ]
    entity_store.entities[("team-a", f"assistant-{bob_id}")].messages = [
        AssistantMessage(role="user", content="Bob's private prompt"),
        AssistantMessage(role="assistant", content="Bob's answer"),
    ]

    response = service_client.get(
        "/v2/assistant/history/sessions?workspace=team-a",
        headers={"X-NMP-Principal-Id": "alice@example.com"},
    )

    assert response.status_code == 200
    assert [session["session_id"] for session in response.json()] == [alice_id]
    forbidden = service_client.get(
        f"/v2/assistant/history/sessions/{bob_id}?workspace=team-a",
        headers={"X-NMP-Principal-Id": "alice@example.com"},
    )
    assert forbidden.status_code == 404


def test_delete_history_enforces_owner_and_removes_conversation(
    service_client: TestClient,
    entity_store: FakeEntityStore,
):
    session_id = service_client.post(
        "/v2/assistant/sessions?workspace=team-a",
        headers={"X-NMP-Principal-Id": "alice@example.com"},
    ).json()["session_id"]

    forbidden = service_client.delete(
        f"/v2/assistant/history/sessions/{session_id}?workspace=team-a",
        headers={"X-NMP-Principal-Id": "bob@example.com"},
    )
    assert forbidden.status_code == 404
    assert ("team-a", f"assistant-{session_id}") in entity_store.entities

    deleted = service_client.delete(
        f"/v2/assistant/history/sessions/{session_id}?workspace=team-a",
        headers={"X-NMP-Principal-Id": "alice@example.com"},
    )
    assert deleted.status_code == 204
    assert ("team-a", f"assistant-{session_id}") not in entity_store.entities


def test_delete_history_rejects_active_session(
    service_client: TestClient,
    entity_store: FakeEntityStore,
):
    session_id = service_client.post("/v2/assistant/sessions").json()["session_id"]
    assistant._session_streams[session_id] = asyncio.Queue()

    response = service_client.delete(f"/v2/assistant/history/sessions/{session_id}")

    assert response.status_code == 409
    assert ("default", f"assistant-{session_id}") in entity_store.entities


def test_assistant_turn_is_persisted_and_reused_as_context(
    service_client: TestClient,
    entity_store: FakeEntityStore,
    monkeypatch: pytest.MonkeyPatch,
):
    session_id = service_client.post("/v2/assistant/sessions").json()["session_id"]
    invocations: list[list[dict[str, str]]] = []

    async def fake_invoke(
        agent_url: str,
        headers: dict[str, str],
        messages: list[dict[str, str]],
        studio_session_id: str,
    ) -> tuple[str, str]:
        del agent_url, headers
        assert studio_session_id == session_id
        invocations.append(messages)
        return f"answer-{len(invocations)}", "nvidia/assistant-model"

    monkeypatch.setattr(assistant, "_invoke_assistant", fake_invoke)

    first = service_client.post(
        f"/v2/assistant/sessions/{session_id}/messages",
        json={"message": "first question", "workspace": "default"},
    )
    second = service_client.post(
        f"/v2/assistant/sessions/{session_id}/messages",
        json={"message": "second question", "workspace": "default"},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert "event: done" in first.text
    assert "event: done" in second.text
    persisted = entity_store.entities[("default", f"assistant-{session_id}")]
    assert [message.model_dump() for message in persisted.messages] == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "answer-1"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "answer-2"},
    ]
    assert invocations[1][:2] == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "answer-1"},
    ]
    history = service_client.get(f"/v2/assistant/history/sessions/{session_id}")
    assert [item["kind"] for item in history.json()["items"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_build_claude_argv_uses_new_session_then_resume_flag():
    session_id = str(uuid.uuid4())

    argv = assistant._build_claude_argv(session_id, "hello", "http://test/mcp", "Studio context")
    assert argv[:3] == ["claude", "-p", "hello"]
    assert "--output-format" in argv
    assert "stream-json" in argv
    mcp_config = json.loads(argv[argv.index("--mcp-config") + 1])
    assert mcp_config["mcpServers"][assistant.CLAUDE_MCP_SERVER_NAME] == {
        "type": "http",
        "url": "http://test/mcp",
        "timeout": assistant.CLAUDE_MCP_TOOL_TIMEOUT_MS,
    }
    assert "--allowedTools" in argv
    allowed_tools = argv[argv.index("--allowedTools") + 1].split(",")
    assert f"mcp__{assistant.CLAUDE_MCP_SERVER_NAME}__select_agent" in allowed_tools
    assert f"mcp__{assistant.CLAUDE_MCP_SERVER_NAME}__select_eval_config" in allowed_tools
    assert f"mcp__{assistant.CLAUDE_MCP_SERVER_NAME}__select_dataset_file" in allowed_tools
    assert f"mcp__{assistant.CLAUDE_MCP_SERVER_NAME}__select_model" in allowed_tools
    assert f"mcp__{assistant.CLAUDE_MCP_SERVER_NAME}__job_progress" in allowed_tools
    assert f"mcp__{assistant.CLAUDE_MCP_SERVER_NAME}__studio_link" in allowed_tools
    assert "--disallowedTools" not in argv
    assert "--append-system-prompt" in argv
    assert argv[argv.index("--append-system-prompt") + 1] == assistant.STUDIO_ASSISTANT_CONTEXT
    assert "--permission-prompt-tool" in argv
    assert f"mcp__{assistant.CLAUDE_MCP_SERVER_NAME}__approval_prompt" in argv
    assert "--append-system-prompt" in argv
    assert "Studio context" in argv
    assert "--session-id" in argv
    assert session_id in argv

    assistant._initialized_sessions.add(session_id)
    resumed_argv = assistant._build_claude_argv(session_id, "again", "http://test/mcp", "Studio context")
    assert "-r" in resumed_argv
    assert "--session-id" not in resumed_argv
    assert "--append-system-prompt" in resumed_argv


def test_history_list_excludes_disk_only_sessions_but_direct_get_remains_available(
    service_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workdir = tmp_path / "repo"
    projects_dir = tmp_path / "claude-projects"
    project_dir = projects_dir / str(workdir).replace("/", "-")
    project_dir.mkdir(parents=True)
    session_id = str(uuid.uuid4())
    history = project_dir / f"{session_id}.jsonl"
    first_prompt = assistant._build_claude_prompt(
        "first prompt",
        "default",
        "https://studio.test/studio",
        "/workspaces/default/dashboard/assistant",
    )
    history.write_text(
        "\n".join(
            [
                json.dumps({"type": "user", "message": {"content": first_prompt}}),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "id": "msg_1",
                            "model": "claude-sonnet-4-5",
                            "content": [
                                {"type": "thinking", "thinking": "checking"},
                                {"type": "text", "text": "done"},
                                {
                                    "type": "tool_use",
                                    "id": "toolu_1",
                                    "name": "Bash",
                                    "input": {"command": "pwd"},
                                },
                            ],
                            "usage": {
                                "input_tokens": 10,
                                "cache_creation_input_tokens": 2,
                                "cache_read_input_tokens": 3,
                                "output_tokens": 4,
                            },
                        },
                        "requestId": "req_1",
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "id": "msg_2",
                            "model": "claude-sonnet-4-6",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "toolu_write",
                                    "name": "Write",
                                    "input": {"file_path": "agents/beach-finder.yml", "content": "name: beach-finder"},
                                },
                                {
                                    "type": "tool_use",
                                    "id": "toolu_link",
                                    "name": "mcp__nemo_studio__studio_link",
                                    "input": {"destination": "agents", "label": "Agents"},
                                },
                                {
                                    "type": "tool_use",
                                    "id": "toolu_job",
                                    "name": "mcp__nemo_studio__job_progress",
                                    "input": {
                                        "job_name": "agent-eval-1",
                                        "job_type": "agent_evaluation",
                                        "source": "evaluator",
                                    },
                                },
                                {
                                    "type": "tool_use",
                                    "id": "toolu_question",
                                    "name": "AskUserQuestion",
                                    "input": {
                                        "questions": [
                                            {
                                                "question": "Which agent should be used?",
                                                "header": "Agent",
                                                "options": [{"label": "beach-finder"}],
                                            }
                                        ]
                                    },
                                },
                            ],
                        },
                        "requestId": "req_2",
                    }
                ),
                json.dumps(
                    {
                        "type": "user",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_question",
                                    "content": (
                                        'Your question has been answered: "Which agent should be used?"='
                                        '"beach-finder". You can now continue with this answer in mind.'
                                    ),
                                }
                            ]
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "id": "msg_3",
                            "model": "claude-sonnet-4-6",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "\n".join(
                                        [
                                            "Draft Spec: `cat-identifier`",
                                            "Name: `cat-identifier`",
                                            "",
                                            "Model",
                                            "`cloud, nvidia/llama-3.3-nemotron-super-49b-v1` - default, good reasoning",
                                            "",
                                            "Framework",
                                            "langgraph-nat",
                                        ]
                                    ),
                                }
                            ],
                        },
                        "requestId": "req_3",
                    }
                ),
                json.dumps(
                    {
                        "type": "user",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_1",
                                    "content": "done",
                                }
                            ]
                        },
                        "toolUseResult": {"totalTokens": 11},
                    }
                ),
                json.dumps({"type": "user", "isSidechain": True, "message": {"content": "ignored"}}),
                "not-json",
            ]
        )
    )

    monkeypatch.setattr(assistant, "SERVER_CWD", workdir)
    monkeypatch.setattr(assistant, "CLAUDE_PROJECTS_DIR", projects_dir)

    list_response = service_client.get("/v2/assistant/history/sessions")

    assert list_response.status_code == 200
    assert list_response.json() == []

    history_response = service_client.get(f"/v2/assistant/history/sessions/{session_id}")

    assert history_response.status_code == 200
    assert history_response.json() == {
        "session_id": session_id,
        "items": [
            {"kind": "user", "text": "first prompt"},
            {
                "kind": "assistant",
                "parts": [
                    {"type": "thinking", "thinking": "checking"},
                    {"type": "text", "text": "done"},
                    {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "pwd"}},
                ],
            },
            {
                "kind": "assistant",
                "parts": [
                    {
                        "type": "tool_use",
                        "id": "toolu_write",
                        "name": "Write",
                        "input": {"file_path": "agents/beach-finder.yml", "content": "name: beach-finder"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_link",
                        "name": "mcp__nemo_studio__studio_link",
                        "input": {"destination": "agents", "label": "Agents"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_job",
                        "name": "mcp__nemo_studio__job_progress",
                        "input": {
                            "job_name": "agent-eval-1",
                            "job_type": "agent_evaluation",
                            "source": "evaluator",
                        },
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_question",
                        "name": "AskUserQuestion",
                        "input": {
                            "questions": [
                                {
                                    "question": "Which agent should be used?",
                                    "header": "Agent",
                                    "options": [{"label": "beach-finder"}],
                                }
                            ]
                        },
                    },
                ],
            },
            {"kind": "user", "text": "Which agent should be used?\nbeach-finder"},
            {
                "kind": "assistant",
                "parts": [
                    {
                        "type": "text",
                        "text": "\n".join(
                            [
                                "Draft Spec: `cat-identifier`",
                                "Name: `cat-identifier`",
                                "",
                                "Model",
                                "`cloud, nvidia/llama-3.3-nemotron-super-49b-v1` - default, good reasoning",
                                "",
                                "Framework",
                                "langgraph-nat",
                            ]
                        ),
                    }
                ],
            },
        ],
        "chat_artifacts": {
            "agent": "cat-identifier",
            "model": "cloud, nvidia/llama-3.3-nemotron-super-49b-v1",
            "model_source": "spec",
            "assistant_model": "claude-sonnet-4-6",
            "workspace": "default",
            "selections": [{"label": "Agent", "value": "beach-finder"}],
            "files": [{"action": "Wrote", "path": "agents/beach-finder.yml"}],
            "links": [{"label": "Agents", "destination": "agents", "href": "/workspaces/default/agents"}],
            "jobs": [
                {
                    "name": "agent-eval-1",
                    "job_type": "agent_evaluation",
                    "source": "evaluator",
                    "href": None,
                }
            ],
            "tools": [
                "Bash",
                "Write",
                "mcp__nemo_studio__studio_link",
                "mcp__nemo_studio__job_progress",
                "AskUserQuestion",
            ],
        },
    }
    assert session_id in assistant._initialized_sessions


def test_list_claude_skills_returns_claude_install_metadata(
    service_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source_dir = _inference_source_dir(tmp_path)
    installed_skill = tmp_path / ".claude" / "skills" / "nemo-inference" / "SKILL.md"
    installed_skill.parent.mkdir(parents=True)
    installed_skill.write_text("# Installed")
    skill = _inference_skill(source_dir)

    monkeypatch.setattr(assistant, "SERVER_CWD", tmp_path)
    monkeypatch.setattr(assistant_skills, "load_skills", lambda: {"inference": skill})

    response = service_client.get("/v2/assistant/skills")

    assert response.status_code == 200
    assert response.json() == [_expected_inference_skill_response(installed=True)]


@pytest.mark.parametrize(
    ("tool_name", "tool_input", "result", "expected"),
    [
        (
            "mcp__nemo_studio__select_agent",
            {},
            '{"status":"submitted","agent":"beach-finder"}',
            "Selected agent: beach-finder",
        ),
        (
            "mcp__nemo_studio__select_model",
            {"display_label": "Fallback model", "output_key": "fallback_model"},
            '{"status":"submitted","fallback_model":"nemotron"}',
            "Fallback model: nemotron",
        ),
        (
            "mcp__nemo_studio__select_dataset_file",
            {},
            '{"status":"submitted","dataset_fileset":"eval-data","dataset_path":"input.jsonl"}',
            "Selected dataset: eval-data/input.jsonl",
        ),
        (
            "mcp__nemo_studio__select_eval_config",
            {},
            '{"status":"submitted","needs_eval_config":true}',
            "I don't have an evaluation config yet",
        ),
    ],
)
def test_history_interaction_text_restores_studio_picker_submissions(
    tool_name: str,
    tool_input: dict[str, Any],
    result: str,
    expected: str,
):
    assert (
        assistant._history_interaction_text(
            assistant.HistoryToolUse(name=tool_name, input=tool_input),
            result,
        )
        == expected
    )


def test_load_claude_skills_falls_back_on_duplicate_skill_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    skill = _inference_skill(_inference_source_dir(tmp_path))
    fallback_called = False

    def fallback() -> dict[str, assistant_skills.Skill]:
        nonlocal fallback_called
        fallback_called = True
        return {"inference": skill}

    monkeypatch.setattr(
        assistant_skills,
        "load_skills",
        lambda: (_ for _ in ()).throw(assistant_skills.DuplicateSkillError("vendored drift")),
    )
    monkeypatch.setattr(assistant_skills, "_load_skills_from_preferred_entry_points", fallback)

    with caplog.at_level(logging.WARNING):
        loaded = assistant_skills._load_claude_skills()

    assert loaded == {"inference": skill}
    assert fallback_called
    assert "vendored drift" in caplog.text


def test_list_claude_skills_returns_500_when_fallback_also_fails(
    service_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        assistant_skills,
        "load_skills",
        lambda: (_ for _ in ()).throw(assistant_skills.DuplicateSkillError("registry drift")),
    )
    monkeypatch.setattr(
        assistant_skills,
        "_load_skills_from_preferred_entry_points",
        lambda: (_ for _ in ()).throw(assistant_skills.DuplicateSkillError("fallback drift")),
    )

    response = service_client.get("/v2/assistant/skills")

    assert response.status_code == 500
    assert response.json()["detail"] == "fallback drift"


def test_invalid_session_id_returns_400(service_client: TestClient):
    response = service_client.get("/v2/assistant/history/sessions/not-a-uuid")

    assert response.status_code == 400
    assert response.json()["detail"] == "session_id must be a UUID"


async def test_stream_claude_hides_startup_oserror(monkeypatch: pytest.MonkeyPatch):
    session_id = str(uuid.uuid4())

    async def fail_start(*args: Any, **kwargs: Any):
        raise OSError("secret local path")

    monkeypatch.setattr(assistant.shutil, "which", lambda name: "/usr/bin/claude")
    monkeypatch.setattr(assistant.asyncio, "create_subprocess_exec", fail_start)

    chunks = [chunk async for chunk in assistant._stream_claude(session_id, "hello", "http://test/mcp")]

    assert chunks == ['event: error\ndata: {"exit_code": null, "stderr": "Failed to start Claude Code process"}\n\n']
    assert "secret local path" not in chunks[0]
    assert session_id not in assistant._session_streams


def test_mcp_initialize_and_tools_list(service_client: TestClient):
    session_id = str(uuid.uuid4())

    initialize_response = service_client.post(
        f"/v2/assistant/mcp/{session_id}",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        },
    )
    tools_response = service_client.post(
        f"/v2/assistant/mcp/{session_id}",
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    )

    assert initialize_response.status_code == 200
    assert initialize_response.json()["result"]["serverInfo"]["name"] == "nemo-studio-permissions"
    assert tools_response.status_code == 200
    tools = tools_response.json()["result"]["tools"]
    assert tools[0]["name"] == "approval_prompt"
    assert {tool["name"] for tool in tools} == {
        "approval_prompt",
        "select_agent",
        "select_eval_config",
        "select_dataset_file",
        "select_model",
        "job_progress",
        "studio_link",
    }
    studio_link_tool = next(tool for tool in tools if tool["name"] == "studio_link")
    assert "Default to using this for Studio-related responses" in studio_link_tool["description"]
    assert "After creating an agent, use destination='agent_chat'" in studio_link_tool["description"]
    assert "chat with or try a model" not in studio_link_tool["description"]
    destination_description = studio_link_tool["inputSchema"]["properties"]["destination"]["description"]
    supported_destinations = supported_destinations_from_description(destination_description)
    assert "base_models" in supported_destinations
    assert "evaluation_results" in supported_destinations
    assert "model_chat" not in supported_destinations
    assert "customizations" not in supported_destinations
    assert "settings" in supported_destinations


def test_mcp_tools_list_includes_feature_flag_enabled_destinations(monkeypatch: pytest.MonkeyPatch):
    service_client = service_client_with_feature_flags(
        monkeypatch,
        {
            "customizer_enabled": "preview",
            "model_compare_enabled": True,
        },
    )
    session_id = str(uuid.uuid4())

    tools_response = service_client.post(
        f"/v2/assistant/mcp/{session_id}",
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    )

    assert tools_response.status_code == 200
    studio_link_tool = next(tool for tool in tools_response.json()["result"]["tools"] if tool["name"] == "studio_link")
    destination_description = studio_link_tool["inputSchema"]["properties"]["destination"]["description"]
    supported_destinations = supported_destinations_from_description(destination_description)
    assert "customizations" in supported_destinations
    assert "model_chat" in supported_destinations
    assert "chat with or try a model" in studio_link_tool["description"]


@pytest.mark.parametrize("feature_flag", ["assistant_studio_enabled", "copilot_studio_enabled"])
def test_assistant_destinations_accept_current_and_legacy_feature_flags(feature_flag: str):
    enabled_destinations = studio_links.enabled_destinations({feature_flag: True})

    assert "assistant" in enabled_destinations
    assert "dashboard" in enabled_destinations


def test_build_studio_system_prompt_preserves_empty_enabled_destinations():
    prompt = assistant._build_studio_system_prompt(
        "default",
        "https://studio.test",
        "/workspaces/default/dashboard/assistant",
        {},
    )

    destinations_line = next(
        line for line in prompt.splitlines() if line.startswith("Enabled Studio link destinations")
    )
    assert destinations_line == "Enabled Studio link destinations for this Studio instance: ."


def test_build_studio_system_prompt_includes_message_summary_contract():
    prompt = assistant._build_studio_system_prompt(
        "default",
        "https://studio.test",
        "/workspaces/default/dashboard/assistant",
        {},
    )

    assert "Conditional message-summary behavior:" in prompt
    assert assistant.STUDIO_MESSAGE_SUMMARY_START in prompt
    assert assistant.STUDIO_MESSAGE_SUMMARY_END in prompt
    assert "title: <meaningful 3-7 word title" in prompt
    assert "worked_for: <elapsed time if you know it, otherwise unknown>" in prompt
    assert "summary: <concise Markdown" in prompt
    assert "details_label: worked for <same elapsed time or unknown>" in prompt
    assert "behind a 'worked for <time>' accordion" in prompt
    assert "Never end a message with only a plain-text question" in prompt
    assert "call the matching select_* tool before completing the message" in prompt
    assert "you MUST call AskUserQuestion to render a selectable options picker" in prompt
    assert "Only ask a concise plain-text question for genuinely open-ended" in prompt
    assert "A timeout, disconnect, or other interactive-tool error is not permission to continue" in prompt
    assert "summary's final sentence MUST state the exact unresolved selection or action" in prompt
    assert "Never show only the investigation result" in prompt
    assert "use a numbered or bulleted list" in prompt
    assert "repeat those links at the bottom of the summary" in prompt
    assert "Put repeated links on separate lines without a heading" in prompt
    assert "A summary block is required when you called one or more tools" in prompt
    assert "For a short informational answer" in prompt
    assert "omit the summary block entirely" in prompt
    assert "Do not emit the summary markers with a shortened duplicate" in prompt
    assert "ask it normally without a summary block" in prompt
    assert "Do not omit the summary block because the message is short." not in prompt


def test_build_assistant_system_prompt_uses_assistant_identity():
    prompt = assistant._build_assistant_system_prompt(
        str(uuid.uuid4()),
        "default",
        "https://studio.test",
        "/workspaces/default/dashboard/assistant",
        {},
    )

    assert "Your identity in this interface is NeMo Assistant." in prompt
    assert "Claude" not in prompt


def test_history_summary_reads_model_generated_session_title(tmp_path: Path):
    history = tmp_path / "session.jsonl"
    history.write_text(
        "\n".join(
            [
                json.dumps({"type": "user", "message": {"content": "A long initial request"}}),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": "\n".join(
                                        [
                                            assistant.STUDIO_MESSAGE_SUMMARY_START,
                                            "title: Create Spam Detector Agent",
                                            "worked_for: unknown",
                                            "summary: Created the requested agent.",
                                            "details_label: worked for unknown",
                                            assistant.STUDIO_MESSAGE_SUMMARY_END,
                                        ]
                                    ),
                                }
                            ]
                        },
                    }
                ),
            ]
        )
    )

    summary = assistant._summarize_history_session(history)

    assert summary.title == "Create Spam Detector Agent"


def test_studio_link_destinations_cover_registered_workspace_routes():
    repo_root = Path(__file__).resolve().parents[4]
    routes_index = (repo_root / "web/packages/studio/src/routes/index.tsx").read_text()
    registered_route_keys = set(re.findall(r"ROUTES\.workspace\.([A-Za-z0-9_]+)", routes_index))
    route_destination_map = {
        "agentDetail": "agent",
        "agentEvaluationDetail": "agent_evaluation",
        "agentEvaluationsList": "agent_evaluations",
        "agentOptimizations": "agent_optimizations",
        "agentsList": "agents",
        "baseModels": "base_models",
        "baseModelsModel": "base_model",
        "claudeCodeChat": "assistant",
        "customizationJobDetails": "customization",
        "customizationJobList": "customizations",
        "dashboard": "dashboard",
        "dataDesignerJobDetails": "data_designer_job",
        "dataDesignerJobList": "data_designer",
        "dataDesignerJobNew": "data_designer_new",
        "deployments": "deployments",
        "deploymentsDeployment": "deployment",
        "evaluation": "evaluation",
        "evaluationBenchmarkDetails": "evaluation_benchmark",
        "evaluationBenchmarks": "evaluation_benchmarks",
        "evaluationMetricDetails": "evaluation_metric",
        "evaluationMetricNew": "evaluation_metric_new",
        "evaluationMetrics": "evaluation_metrics",
        "evaluationMetricsRun": "evaluation_run",
        "evaluationResultDetails": "evaluation_result",
        "evaluationResults": "evaluation_results",
        "experiment": "experiment",
        "experimentDetail": "experiment_detail",
        "experimentGroupDetail": "experiment_group",
        "filesetDetail": "fileset",
        "filesetDetails": "fileset_panel",
        "filesetFile": "fileset_file",
        "filesetNew": "fileset_new",
        "filesets": "filesets",
        "guardrails": "guardrails",
        "index": "workspace",
        "inferenceProviders": "inference_providers",
        "intake": "intake",
        "intakeSession": "intake_session",
        "intakeSpans": "intake_spans",
        "intakeTraces": "intake_traces",
        "jobDetail": "job",
        "jobs": "jobs",
        "members": "members",
        "modelCompare": "model_chat",
        "newCustomizationJob": "customization_new",
        "plugin": "plugin",
        "promptTuningForm": "prompt_tuning",
        "safeSynthesizer": "safe_synthesizer",
        "safeSynthesizerJob": "safe_synthesizer_job",
        "safeSynthesizerJobReport": "safe_synthesizer_report",
        "safeSynthesizerNew": "safe_synthesizer_new",
        "secrets": "secrets",
        "settings": "settings",
    }

    assert registered_route_keys - set(route_destination_map) == set()
    assert {
        route_key: destination
        for route_key, destination in route_destination_map.items()
        if route_key in registered_route_keys and destination not in studio_links.STUDIO_LINK_DESTINATIONS
    } == {}


def test_build_studio_link_result_returns_virtual_model_chat_link():
    result = studio_links.build_studio_link_result(
        "default",
        None,
        {"destination": "virtual_model_chat", "name": "demo guarded"},
        enabled_destinations={"virtual_model_chat": studio_links.STUDIO_LINK_DESTINATIONS["virtual_model_chat"]},
    )

    assert result == {
        "workspace": "default",
        "destination": "virtual_model_chat",
        "path": "/workspaces/default/virtual-models?virtualModel=demo%20guarded&tab=chat",
        "url": None,
        "markdown": (
            "[Chat with VirtualModel demo guarded]"
            "(/workspaces/default/virtual-models?virtualModel=demo%20guarded&tab=chat)"
        ),
    }


def test_mcp_studio_link_returns_agents_page_markdown(service_client: TestClient):
    session_id = str(uuid.uuid4())

    response = service_client.post(
        f"/v2/assistant/mcp/{session_id}?workspace=default",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "studio_link",
                "arguments": {"destination": "agents"},
            },
        },
    )

    assert response.status_code == 200
    result_text = response.json()["result"]["content"][0]["text"]
    assert json.loads(result_text) == {
        "workspace": "default",
        "destination": "agents",
        "path": "/workspaces/default/agents",
        "url": None,
        "markdown": "[Agents](/workspaces/default/agents)",
    }


def test_mcp_studio_link_returns_custom_models_full_url(monkeypatch: pytest.MonkeyPatch):
    service_client = service_client_with_feature_flags(monkeypatch, {"customizer_enabled": True})
    session_id = str(uuid.uuid4())

    response = service_client.post(
        f"/v2/assistant/mcp/{session_id}?workspace=default&studio_base_url=https%3A%2F%2Fstudio.test%2Fstudio",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "studio_link",
                "arguments": {"destination": "custom_models"},
            },
        },
    )

    assert response.status_code == 200
    result_text = response.json()["result"]["content"][0]["text"]
    assert json.loads(result_text) == {
        "workspace": "default",
        "destination": "customizations",
        "path": "/workspaces/default/customizations",
        "url": "https://studio.test/studio/workspaces/default/customizations",
        "markdown": "[Custom Models](https://studio.test/studio/workspaces/default/customizations)",
    }


def test_mcp_studio_link_returns_base_models_markdown(service_client: TestClient):
    session_id = str(uuid.uuid4())

    response = service_client.post(
        f"/v2/assistant/mcp/{session_id}?workspace=default",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "studio_link",
                "arguments": {"destination": "available_base_models"},
            },
        },
    )

    assert response.status_code == 200
    result_text = response.json()["result"]["content"][0]["text"]
    assert json.loads(result_text) == {
        "workspace": "default",
        "destination": "base_models",
        "path": "/workspaces/default/base-models",
        "url": None,
        "markdown": "[Base Models](/workspaces/default/base-models)",
    }


def test_mcp_studio_link_returns_jobs_page_markdown(service_client: TestClient):
    session_id = str(uuid.uuid4())

    response = service_client.post(
        f"/v2/assistant/mcp/{session_id}?workspace=default",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "studio_link",
                "arguments": {"destination": "jobs", "label": "Open Jobs"},
            },
        },
    )

    assert response.status_code == 200
    result_text = response.json()["result"]["content"][0]["text"]
    assert json.loads(result_text) == {
        "workspace": "default",
        "destination": "jobs",
        "path": "/workspaces/default/jobs",
        "url": None,
        "markdown": "[Open Jobs](/workspaces/default/jobs)",
    }


def test_mcp_studio_link_encodes_detail_route_parts(service_client: TestClient):
    session_id = str(uuid.uuid4())

    response = service_client.post(
        f"/v2/assistant/mcp/{session_id}?workspace=default%20workspace",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "studio_link",
                "arguments": {
                    "destination": "agent",
                    "name": "triage agent",
                    "label": "Open agent",
                },
            },
        },
    )

    assert response.status_code == 200
    result_text = response.json()["result"]["content"][0]["text"]
    assert json.loads(result_text) == {
        "workspace": "default workspace",
        "destination": "agent",
        "path": "/workspaces/default%20workspace/agents/triage%20agent",
        "url": None,
        "markdown": "[Open agent](/workspaces/default%20workspace/agents/triage%20agent)",
    }


def test_mcp_studio_link_returns_agent_deployment_detail_markdown(service_client: TestClient):
    session_id = str(uuid.uuid4())

    response = service_client.post(
        f"/v2/assistant/mcp/{session_id}?workspace=default",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "studio_link",
                "arguments": {
                    "destination": "agent_deployment",
                    "name": "spanish-translator",
                },
            },
        },
    )

    assert response.status_code == 200
    result_text = response.json()["result"]["content"][0]["text"]
    assert json.loads(result_text) == {
        "workspace": "default",
        "destination": "agent_deployment",
        "path": "/workspaces/default/agents/spanish-translator",
        "url": None,
        "markdown": "[Agent deployment spanish-translator](/workspaces/default/agents/spanish-translator)",
    }


def test_mcp_studio_link_returns_agent_chat_markdown(service_client: TestClient):
    session_id = str(uuid.uuid4())

    response = service_client.post(
        f"/v2/assistant/mcp/{session_id}?workspace=default",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "studio_link",
                "arguments": {
                    "destination": "agent_chat",
                    "name": "spanish-translator",
                },
            },
        },
    )

    assert response.status_code == 200
    result_text = response.json()["result"]["content"][0]["text"]
    assert json.loads(result_text) == {
        "workspace": "default",
        "destination": "agent_chat",
        "path": "/workspaces/default/agents/spanish-translator?tab=chat-playground",
        "url": None,
        "markdown": "[Chat with agent spanish-translator](/workspaces/default/agents/spanish-translator?tab=chat-playground)",
    }


def test_mcp_studio_link_returns_model_chat_markdown(monkeypatch: pytest.MonkeyPatch):
    service_client = service_client_with_feature_flags(monkeypatch, {"model_compare_enabled": True})
    session_id = str(uuid.uuid4())

    response = service_client.post(
        f"/v2/assistant/mcp/{session_id}?workspace=default",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "studio_link",
                "arguments": {"destination": "model_chat"},
            },
        },
    )

    assert response.status_code == 200
    result_text = response.json()["result"]["content"][0]["text"]
    assert json.loads(result_text) == {
        "workspace": "default",
        "destination": "model_chat",
        "path": "/workspaces/default/model-compare",
        "url": None,
        "markdown": "[Chat with models](/workspaces/default/model-compare)",
    }


def test_mcp_studio_link_rejects_disabled_feature_flag_destination(service_client: TestClient):
    session_id = str(uuid.uuid4())

    response = service_client.post(
        f"/v2/assistant/mcp/{session_id}?workspace=default",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "studio_link",
                "arguments": {"destination": "model_chat"},
            },
        },
    )

    assert response.status_code == 200
    result_text = response.json()["result"]["content"][0]["text"]
    result = json.loads(result_text)
    assert result["error"] == "Studio destination is disabled by feature flag: model_chat"
    assert "model_chat" not in result["available_destinations"]
    assert "base_models" in result["available_destinations"]


def test_build_studio_link_result_preserves_empty_enabled_destinations():
    result = studio_links.build_studio_link_result(
        "default",
        None,
        {"destination": "agents"},
        {},
    )

    assert result == {
        "error": "Studio destination is disabled by feature flag: agents",
        "available_destinations": [],
    }


def test_mcp_studio_link_returns_fileset_file_markdown(service_client: TestClient):
    session_id = str(uuid.uuid4())

    response = service_client.post(
        f"/v2/assistant/mcp/{session_id}?workspace=default",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "studio_link",
                "arguments": {
                    "destination": "fileset_file",
                    "fileset_name": "training data",
                    "file_path": "nested/examples.jsonl",
                },
            },
        },
    )

    assert response.status_code == 200
    result_text = response.json()["result"]["content"][0]["text"]
    assert json.loads(result_text) == {
        "workspace": "default",
        "destination": "fileset_file",
        "path": "/workspaces/default/filesets/training%20data/file/nested%2Fexamples.jsonl",
        "url": None,
        "markdown": "[File nested/examples.jsonl](/workspaces/default/filesets/training%20data/file/nested%2Fexamples.jsonl)",
    }


def test_mcp_studio_link_returns_intake_span_markdown(monkeypatch: pytest.MonkeyPatch):
    service_client = service_client_with_feature_flags(monkeypatch, {"intake_enabled": True})
    session_id = str(uuid.uuid4())

    response = service_client.post(
        f"/v2/assistant/mcp/{session_id}?workspace=default",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "studio_link",
                "arguments": {
                    "destination": "intake_span",
                    "session_id": "session 00",
                    "trace_id": "trace 01",
                    "span_id": "span 02",
                },
            },
        },
    )

    assert response.status_code == 200
    result_text = response.json()["result"]["content"][0]["text"]
    assert json.loads(result_text) == {
        "workspace": "default",
        "destination": "intake_span",
        "path": "/workspaces/default/intake/sessions/session%2000?traceId=trace%2001&spanId=span%2002",
        "url": None,
        "markdown": "[Span span 02](/workspaces/default/intake/sessions/session%2000?traceId=trace%2001&spanId=span%2002)",
    }


@pytest.mark.parametrize(
    ("destination", "arguments", "expected_path"),
    [
        (
            "intake_session",
            {"session_id": "session 00"},
            "/workspaces/default/intake/sessions/session%2000",
        ),
        (
            "intake_trace",
            {"session_id": "session 00", "trace_id": "trace 01"},
            "/workspaces/default/intake/sessions/session%2000?traceId=trace%2001",
        ),
    ],
)
def test_build_studio_link_result_returns_canonical_intake_session_paths(
    destination: str,
    arguments: dict[str, str],
    expected_path: str,
):
    result = studio_links.build_studio_link_result(
        "default",
        None,
        {"destination": destination, **arguments},
    )

    assert result["path"] == expected_path


def test_build_studio_link_result_requires_session_for_intake_trace():
    result = studio_links.build_studio_link_result(
        "default",
        None,
        {"destination": "intake_trace", "trace_id": "trace 01"},
    )

    assert result == {"error": "session_id is required for Studio destination: intake_trace"}


def test_mcp_studio_link_returns_started_evaluation_result_markdown(service_client: TestClient):
    session_id = str(uuid.uuid4())

    response = service_client.post(
        f"/v2/assistant/mcp/{session_id}?workspace=default",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "studio_link",
                "arguments": {
                    "destination": "evaluation_result",
                    "job_name": "eval run 01",
                },
            },
        },
    )

    assert response.status_code == 200
    result_text = response.json()["result"]["content"][0]["text"]
    assert json.loads(result_text) == {
        "workspace": "default",
        "destination": "evaluation_result",
        "path": "/workspaces/default/evaluation/results/eval%20run%2001",
        "url": None,
        "markdown": "[Evaluation result eval run 01](/workspaces/default/evaluation/results/eval%20run%2001)",
    }


def test_mcp_rejects_malformed_json(service_client: TestClient):
    session_id = str(uuid.uuid4())

    response = service_client.post(
        f"/v2/assistant/mcp/{session_id}",
        content="{",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid JSON body"


def test_mcp_rejects_non_object_json(service_client: TestClient):
    session_id = str(uuid.uuid4())

    response = service_client.post(f"/v2/assistant/mcp/{session_id}", json=[])

    assert response.status_code == 400
    assert response.json()["detail"] == "JSON body must be an object"


def test_mcp_rejects_non_object_params(service_client: TestClient):
    session_id = str(uuid.uuid4())

    response = service_client.post(
        f"/v2/assistant/mcp/{session_id}",
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": []},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "JSON-RPC params must be an object"


def test_mcp_tools_call_denies_without_active_stream(service_client: TestClient):
    session_id = str(uuid.uuid4())

    response = service_client.post(
        f"/v2/assistant/mcp/{session_id}",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "approval_prompt",
                "arguments": {"tool_name": "Bash", "input": {"command": "pwd"}},
            },
        },
    )

    assert response.status_code == 200
    result_text = response.json()["result"]["content"][0]["text"]
    assert json.loads(result_text) == {
        "behavior": "deny",
        "message": "no active Studio assistant session",
    }


def test_mcp_tools_call_job_progress_returns_rendered(service_client: TestClient):
    session_id = str(uuid.uuid4())

    response = service_client.post(
        f"/v2/assistant/mcp/{session_id}",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "job_progress",
                "arguments": {
                    "job_name": "eval-job-1",
                },
            },
        },
    )

    assert response.status_code == 200
    result_text = response.json()["result"]["content"][0]["text"]
    assert json.loads(result_text) == {"status": "rendered"}


def test_mcp_tools_call_select_agent_denies_without_active_stream(service_client: TestClient):
    session_id = str(uuid.uuid4())

    response = service_client.post(
        f"/v2/assistant/mcp/{session_id}",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "select_agent",
                "arguments": {"title": "Select an agent"},
            },
        },
    )

    assert response.status_code == 200
    result_text = response.json()["result"]["content"][0]["text"]
    assert json.loads(result_text) == {
        "status": "error",
        "message": "no active Studio assistant session",
    }


def test_mcp_tools_call_select_dataset_file_denies_without_active_stream(service_client: TestClient):
    session_id = str(uuid.uuid4())

    response = service_client.post(
        f"/v2/assistant/mcp/{session_id}",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "select_dataset_file",
                "arguments": {"title": "Select a dataset"},
            },
        },
    )

    assert response.status_code == 200
    result_text = response.json()["result"]["content"][0]["text"]
    assert json.loads(result_text) == {
        "status": "error",
        "message": "no active Studio assistant session",
    }


def test_mcp_tools_call_select_model_denies_without_active_stream(service_client: TestClient):
    session_id = str(uuid.uuid4())

    response = service_client.post(
        f"/v2/assistant/mcp/{session_id}",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "select_model",
                "arguments": {"title": "Select a model"},
            },
        },
    )

    assert response.status_code == 200
    result_text = response.json()["result"]["content"][0]["text"]
    assert json.loads(result_text) == {
        "status": "error",
        "message": "no active Studio assistant session",
    }


async def test_resolve_permission_rejects_cross_session_request():
    owner_session_id = str(uuid.uuid4())
    other_session_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    future = asyncio.get_running_loop().create_future()
    assistant._pending_permissions[request_id] = (owner_session_id, future)

    with pytest.raises(HTTPException) as exc_info:
        await assistant.resolve_permission(
            other_session_id,
            request_id,
            assistant.PermissionDecision(approved=True),
        )

    assert exc_info.value.status_code == 404
    assert not future.done()


async def test_resolve_permission_sets_result_for_owning_session():
    session_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    future = asyncio.get_running_loop().create_future()
    assistant._pending_permissions[request_id] = (session_id, future)

    response = await assistant.resolve_permission(
        session_id,
        request_id,
        assistant.PermissionDecision(approved=True),
    )

    assert response == {"ok": True}
    assert future.result() == {"approved": True, "reason": None, "updated_input": None}


async def test_resolve_agent_input_sets_result_for_owning_session():
    session_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    future = asyncio.get_running_loop().create_future()
    assistant._pending_agent_inputs[request_id] = (session_id, future)

    response = await assistant.resolve_agent_input(
        session_id,
        request_id,
        assistant.AgentInputDecision(value={"agent": "react-agent"}),
    )

    assert response == {"ok": True}
    assert future.result() == {"skipped": False, "value": {"agent": "react-agent"}}


async def test_request_agent_input_rejects_reserved_response_keys():
    session_id = str(uuid.uuid4())
    assistant._session_streams[session_id] = asyncio.Queue()

    request_task = asyncio.create_task(assistant._request_agent_input(session_id, "agent", {}))
    _, payload = await assistant._session_streams[session_id].get()
    request_id = json.loads(payload)["request_id"]

    await assistant.resolve_agent_input(
        session_id,
        request_id,
        assistant.AgentInputDecision(value={"agent": "react-agent", "status": "submitted"}),
    )

    assert await request_task == {
        "status": "error",
        "message": "input value included reserved keys: status",
    }


async def test_permission_request_waits_until_user_resolves_it():
    session_id = str(uuid.uuid4())
    assistant._session_streams[session_id] = asyncio.Queue()

    request_task = asyncio.create_task(
        assistant._request_permission(
            session_id,
            {"tool_name": "AskUserQuestion", "input": {"question": "Continue?"}},
        )
    )
    _, payload = await assistant._session_streams[session_id].get()
    request_id = json.loads(payload)["request_id"]

    await asyncio.sleep(0)
    assert not request_task.done()

    await assistant.resolve_permission(
        session_id,
        request_id,
        assistant.PermissionDecision(approved=True),
    )

    assert await request_task == {"behavior": "allow", "updatedInput": {"question": "Continue?"}}


async def test_agent_input_request_cleans_up_when_wait_is_cancelled():
    session_id = str(uuid.uuid4())
    assistant._session_streams[session_id] = asyncio.Queue()

    request_task = asyncio.create_task(assistant._request_agent_input(session_id, "agent", {}))
    _, payload = await assistant._session_streams[session_id].get()
    request_id = json.loads(payload)["request_id"]

    assert request_id in assistant._pending_agent_inputs
    request_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request_task
    assert request_id not in assistant._pending_agent_inputs


async def test_blocking_mcp_tool_response_streams_keepalives_until_user_responds():
    session_id = str(uuid.uuid4())
    assistant._session_streams[session_id] = asyncio.Queue()
    result = asyncio.get_running_loop().create_future()

    response = await assistant._blocking_mcp_tool_response(session_id, 7, result)

    assert response.media_type == "text/event-stream"
    assert response.headers["cache-control"] == "no-cache, no-transform"
    assert response.headers["x-accel-buffering"] == "no"

    assert isinstance(response, StreamingResponse)
    iterator = cast(AsyncIterator[str], response.body_iterator)
    assert await anext(iterator) == ": keepalive\n\n"

    result.set_result({"status": "answered", "response": "A detailed answer"})
    final_event = await anext(iterator)
    assert final_event.startswith("event: message\ndata: ")
    payload = json.loads(final_event.removeprefix("event: message\ndata: ").removesuffix("\n\n"))
    assert payload == {
        "jsonrpc": "2.0",
        "id": 7,
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({"status": "answered", "response": "A detailed answer"}),
                }
            ]
        },
    }

    with pytest.raises(StopAsyncIteration):
        await anext(iterator)


def test_platform_route_stream_uses_deployed_assistant(monkeypatch: pytest.MonkeyPatch):
    service = StudioService()
    app = FastAPI()
    app.include_router(service.app.router, prefix="/apis/studio")
    service.configure_app(app)
    session_id = str(uuid.uuid4())
    entity_store = FakeEntityStore()
    conversation = AssistantConversation(
        name=f"assistant-{session_id}",
        workspace="default",
        session_id=session_id,
        owner_id="local-user",
    )
    entity_store.entities[("default", conversation.name)] = conversation
    app.dependency_overrides[get_entity_client] = lambda: entity_store
    client = TestClient(app)
    captured: dict[str, Any] = {}

    async def fake_stream(
        session_id: str,
        message: str,
        agent_url: str,
        headers: dict[str, str],
        studio_system_prompt: str,
        conversation: AssistantConversation,
        entity_store: FakeEntityStore,
    ):
        del conversation, entity_store
        captured.update(
            {
                "session_id": session_id,
                "message": message,
                "agent_url": agent_url,
                "headers": headers,
                "studio_system_prompt": studio_system_prompt,
            }
        )
        yield assistant._sse(json.dumps({"type": "system", "subtype": "init"}))
        yield assistant._sse("", event="done")

    monkeypatch.setattr(assistant, "_stream_assistant", fake_stream)

    response = client.post(
        f"/apis/studio/v2/assistant/sessions/{session_id}/messages",
        json={
            "message": "hello",
            "workspace": "default",
            "studio_base_url": "https://studio.test/studio",
            "studio_pathname": "/workspaces/default/dashboard/assistant",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-store"
    assert "event: done" in response.text
    assert captured["session_id"] == session_id
    assert captured["agent_url"] == (
        "http://127.0.0.1:8080/apis/agents/v2/workspaces/default/agents/nemo-studio-assistant/-/v1/chat/completions"
    )
    assert captured["headers"] == {}
    assert captured["message"] == "hello"
    assert "Current Studio workspace: default" in captured["studio_system_prompt"]
    assert "Studio UI base URL: https://studio.test/studio" in captured["studio_system_prompt"]
    assert "Current Studio route path: /workspaces/default/dashboard/assistant" in captured["studio_system_prompt"]
    assert "you MUST call select_agent" in captured["studio_system_prompt"]
    assert "you MUST call select_model" in captured["studio_system_prompt"]
    # The options-picker directive is present and rewritten to the deployed agent's tool name.
    assert "you MUST call ask_user_question to render a selectable options picker" in captured["studio_system_prompt"]
    assert "AskUserQuestion" not in captured["studio_system_prompt"]
    assert "Prefer NeMo Studio MCP tools and Studio views over CLI commands" in captured["studio_system_prompt"]
    assert "Do not tell the user to run nemo CLI commands" in captured["studio_system_prompt"]
    assert "when a Studio view, Studio link, or Studio progress card is available" in captured["studio_system_prompt"]
    assert "Default to trying to include a Studio link in Studio-related responses" in captured["studio_system_prompt"]
    assert "link to the closest list page for the current workspace" in captured["studio_system_prompt"]
    assert "Base Models or available base models use destination='base_models'" in captured["studio_system_prompt"]
    assert "never use customizations for Base Models" in captured["studio_system_prompt"]
    assert "Enabled Studio link destinations for this Studio instance" in captured["studio_system_prompt"]
    assert "Only call studio_link with one of the enabled destinations above" in captured["studio_system_prompt"]
    assert "Do not invent Studio route paths manually" in captured["studio_system_prompt"]
    assert "/workspaces/{workspace}/evaluation/..." in captured["studio_system_prompt"]
    assert "never nest evaluation links under /dashboard/evaluations/" in captured["studio_system_prompt"]
    assert "destination='evaluation_results'" in captured["studio_system_prompt"]
    assert "/workspaces/{workspace}/evaluation/results" in captured["studio_system_prompt"]
    assert "The model_chat destination is not enabled in this Studio instance" in captured["studio_system_prompt"]
    assert "Direct Studio link requests are mandatory tool-use requests" in captured["studio_system_prompt"]
    assert "Never answer a Studio link request by saying you cannot generate URLs" in captured["studio_system_prompt"]
    assert (
        "After any successful Studio action, you must include a Studio link in the response"
        in captured["studio_system_prompt"]
    )
    assert "Before your final response" in captured["studio_system_prompt"]
    assert "studio_link" in captured["studio_system_prompt"]
    assert "Required job-progress behavior:" in captured["studio_system_prompt"]
    assert "you MUST call job_progress" in captured["studio_system_prompt"]
    assert f"studio_session_id='{session_id}'" in captured["studio_system_prompt"]
    assert "The approval context for nemo_api is injected by Studio" in captured["studio_system_prompt"]
    assert "mutating calls will automatically pause for the user's approval" in captured["studio_system_prompt"]
    assert (
        "For a newly created agent, use studio_link with destination='agent_chat'" in captured["studio_system_prompt"]
    )
    assert "destination='agent_chat'" in captured["studio_system_prompt"]


def test_platform_route_stream_infers_studio_url_from_browser_headers(monkeypatch: pytest.MonkeyPatch):
    service = StudioService()
    app = FastAPI()
    app.include_router(service.app.router, prefix="/apis/studio")
    service.configure_app(app)
    session_id = str(uuid.uuid4())
    entity_store = FakeEntityStore()
    conversation = AssistantConversation(
        name=f"assistant-{session_id}",
        workspace="default",
        session_id=session_id,
        owner_id="local-user",
    )
    entity_store.entities[("default", conversation.name)] = conversation
    app.dependency_overrides[get_entity_client] = lambda: entity_store
    client = TestClient(app)
    captured: dict[str, Any] = {}

    async def fake_stream(
        session_id: str,
        message: str,
        agent_url: str,
        headers: dict[str, str],
        studio_system_prompt: str,
        conversation: AssistantConversation,
        entity_store: FakeEntityStore,
    ):
        del conversation, entity_store
        captured.update(
            {
                "session_id": session_id,
                "message": message,
                "agent_url": agent_url,
                "headers": headers,
                "studio_system_prompt": studio_system_prompt,
            }
        )
        yield assistant._sse(json.dumps({"type": "system", "subtype": "init"}))
        yield assistant._sse("", event="done")

    monkeypatch.setattr(assistant, "_stream_assistant", fake_stream)

    response = client.post(
        f"/apis/studio/v2/assistant/sessions/{session_id}/messages",
        json={
            "message": "can you give me a link to it?",
            "workspace": "default",
        },
        headers={
            "host": "attacker.example",
            "origin": "https://studio.example.com",
            "referer": "https://studio.example.com/workspaces/default/dashboard/assistant",
        },
    )

    assert response.status_code == 200
    assert captured["agent_url"] == (
        "http://127.0.0.1:8080/apis/agents/v2/workspaces/default/agents/nemo-studio-assistant/-/v1/chat/completions"
    )
    assert captured["message"] == "can you give me a link to it?"
    assert "Studio UI base URL: https://studio.example.com" in captured["studio_system_prompt"]
    assert "Current Studio route path: /workspaces/default/dashboard/assistant" in captured["studio_system_prompt"]


def test_assistant_url_uses_only_configured_base_url(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NMP_BASE_URL", "https://platform.example.com")

    assert assistant._studio_assistant_url("default") == (
        "https://platform.example.com/apis/agents/v2/workspaces/default/"
        "agents/nemo-studio-assistant/-/v1/chat/completions"
    )


def test_assistant_request_headers_strip_credentials_for_http():
    request = Request(
        {
            "type": "http",
            "headers": [
                (b"authorization", b"Bearer secret-token"),
                (b"cookie", b"session=secret-cookie"),
            ],
        }
    )

    assert assistant._assistant_request_headers(request, "http://127.0.0.1:8080/agent") == {}


def test_assistant_request_headers_forward_credentials_for_https():
    request = Request(
        {
            "type": "http",
            "headers": [
                (b"authorization", b"Bearer secret-token"),
                (b"cookie", b"session=secret-cookie"),
            ],
        }
    )

    assert assistant._assistant_request_headers(request, "https://platform.test/agent") == {
        "authorization": "Bearer secret-token",
        "cookie": "session=secret-cookie",
    }


def test_validated_workspace_uses_default_and_rejects_path_injection():
    assert assistant._validated_workspace_or_default(None) == "default"
    assert assistant._validated_workspace_or_default(" default ") == "default"

    with pytest.raises(HTTPException, match="workspace must match the expected entity-name pattern"):
        assistant._validated_workspace_or_default("../internal?target=metadata")


def test_platform_route_rejects_workspace_path_injection(service_client: TestClient):
    session_id = str(uuid.uuid4())

    response = service_client.post(
        f"/v2/assistant/sessions/{session_id}/messages",
        json={"message": "hello", "workspace": "../internal?target=metadata"},
        headers={"host": "attacker.example"},
    )

    assert response.status_code == 422


_WORKSPACES_LIST_URL = "http://127.0.0.1:8080/apis/entities/v2/workspaces"


@pytest.mark.asyncio
async def test_authorized_workspace_default_skips_lookup():
    # No HTTP mock configured: the default fallback must not make a network call.
    assert await assistant._authorized_workspace("default", {}, "sess") == "default"


@pytest.mark.asyncio
async def test_authorized_workspace_returns_name_from_entity_store(respx_mock):
    respx_mock.get(_WORKSPACES_LIST_URL).mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"name": "team-a"}, {"name": "my-ws"}], "pagination": {"total_pages": 1}},
        )
    )

    # The returned value is the platform's own copy of the name (not client input).
    assert await assistant._authorized_workspace("my-ws", {}, "sess") == "my-ws"


@pytest.mark.asyncio
async def test_authorized_workspace_paginates_until_match(respx_mock):
    respx_mock.get(_WORKSPACES_LIST_URL).mock(
        side_effect=[
            httpx.Response(200, json={"data": [{"name": "a"}], "pagination": {"total_pages": 2}}),
            httpx.Response(200, json={"data": [{"name": "target"}], "pagination": {"total_pages": 2}}),
        ]
    )

    assert await assistant._authorized_workspace("target", {}, "sess") == "target"


@pytest.mark.asyncio
async def test_authorized_workspace_caches_per_session(respx_mock):
    route = respx_mock.get(_WORKSPACES_LIST_URL).mock(
        return_value=httpx.Response(200, json={"data": [{"name": "my-ws"}], "pagination": {"total_pages": 1}})
    )

    first = await assistant._authorized_workspace("my-ws", {}, "sess")
    second = await assistant._authorized_workspace("my-ws", {}, "sess")

    assert first == second == "my-ws"
    # The second resolution is served from the per-session cache, not the network.
    assert route.call_count == 1
    # A different session does not share the cache.
    await assistant._authorized_workspace("my-ws", {}, "other-sess")
    assert route.call_count == 2


@pytest.mark.asyncio
async def test_authorized_workspace_cache_is_not_shared_across_callers(respx_mock):
    """A cached authorization decision must never be reused for a different caller."""
    route = respx_mock.get(_WORKSPACES_LIST_URL).mock(
        return_value=httpx.Response(200, json={"data": [{"name": "my-ws"}], "pagination": {"total_pages": 1}})
    )
    session_id = "shared-session"
    caller_a = {"authorization": "Bearer token-a"}
    caller_b = {"authorization": "Bearer token-b"}

    await assistant._authorized_workspace("my-ws", caller_a, session_id)
    assert route.call_count == 1
    # Same session id, different credential: must re-verify against the Entity Store.
    await assistant._authorized_workspace("my-ws", caller_b, session_id)
    assert route.call_count == 2
    # Each caller still gets its own cache hit on repeat.
    await assistant._authorized_workspace("my-ws", caller_a, session_id)
    assert route.call_count == 2
    # The raw credential is never retained in the cache key material.
    assert "token-a" not in str(assistant._session_workspace_cache)


@pytest.mark.asyncio
async def test_authorized_workspace_unauthorized_caller_is_rejected_on_cached_session(respx_mock):
    """An unauthorized caller cannot ride a session that already resolved the workspace."""
    session_id = "shared-session"
    respx_mock.get(_WORKSPACES_LIST_URL).mock(
        side_effect=[
            httpx.Response(200, json={"data": [{"name": "my-ws"}], "pagination": {"total_pages": 1}}),
            # The second caller is not a member of that workspace.
            httpx.Response(200, json={"data": [{"name": "other-ws"}], "pagination": {"total_pages": 1}}),
        ]
    )

    assert await assistant._authorized_workspace("my-ws", {"authorization": "a"}, session_id) == "my-ws"

    with pytest.raises(HTTPException) as excinfo:
        await assistant._authorized_workspace("my-ws", {"authorization": "b"}, session_id)
    assert excinfo.value.status_code == 404


@pytest.mark.asyncio
async def test_authorized_workspace_rejects_unknown_workspace(respx_mock):
    respx_mock.get(_WORKSPACES_LIST_URL).mock(
        return_value=httpx.Response(200, json={"data": [{"name": "team-a"}], "pagination": {"total_pages": 1}})
    )

    with pytest.raises(HTTPException) as excinfo:
        await assistant._authorized_workspace("not-a-member", {}, "sess")
    assert excinfo.value.status_code == 404


@pytest.mark.asyncio
async def test_authorized_workspace_does_not_cache_failures(respx_mock):
    route = respx_mock.get(_WORKSPACES_LIST_URL).mock(
        return_value=httpx.Response(200, json={"data": [{"name": "team-a"}], "pagination": {"total_pages": 1}})
    )

    for _ in range(2):
        with pytest.raises(HTTPException):
            await assistant._authorized_workspace("not-a-member", {}, "sess")
    # Unresolved workspaces are re-checked every message (no negative caching).
    assert route.call_count == 2


@pytest.mark.asyncio
async def test_authorized_workspace_maps_upstream_error_to_502(respx_mock):
    respx_mock.get(_WORKSPACES_LIST_URL).mock(side_effect=httpx.ConnectError("boom"))

    with pytest.raises(HTTPException) as excinfo:
        await assistant._authorized_workspace("my-ws", {}, "sess")
    assert excinfo.value.status_code == 502


def test_nemo_agent_error_detail_does_not_expose_exception_text():
    request = httpx.Request("POST", "https://platform.test/agent")
    response = httpx.Response(502, request=request)
    status_error = httpx.HTTPStatusError("private upstream detail", request=request, response=response)

    assert assistant._assistant_error_detail(status_error) == "The deployed NeMo Assistant returned HTTP 502."
    assert assistant._assistant_error_detail(RuntimeError("private stack detail")) == (
        "The deployed NeMo Assistant returned an invalid response."
    )


def test_nemo_agent_timeout_detail_explains_where_to_find_late_results():
    request = httpx.Request("POST", "https://platform.test/agent")

    assert assistant._assistant_error_detail(httpx.ReadTimeout("private detail", request=request)) == (
        "NeMo Assistant timed out before completing. The operation may still finish; "
        "check History and Intake traces before retrying."
    )


def test_parse_tool_step_input_extracts_python_repr_dict():
    payload = "**Input:**\n```json\n{'action': 'list', 'resource': 'secrets'}\n```\n**Output:** ..."
    assert assistant._parse_tool_step_input(payload) == {"action": "list", "resource": "secrets"}


def test_parse_tool_step_input_returns_empty_on_unparseable():
    assert assistant._parse_tool_step_input("no dict here") == {}
    assert assistant._parse_tool_step_input(None) == {}
    assert assistant._parse_tool_step_input("**Input:** [1, 2, 3]") == {}


def test_tool_use_stream_event_shape():
    event_type, payload = assistant._tool_use_stream_event("nemo_api", {"resource": "secrets"})
    assert event_type == "agent"
    block = json.loads(payload)["message"]["content"][0]
    assert block["type"] == "tool_use"
    assert block["name"] == "nemo_api"
    assert block["input"] == {"resource": "secrets"}


def test_tool_use_stream_event_strips_internal_session_id():
    _, payload = assistant._tool_use_stream_event(
        "ask_user_question",
        {"studio_session_id": "sess-123", "questions": [{"q": "?"}]},
    )
    block = json.loads(payload)["message"]["content"][0]
    assert "studio_session_id" not in block["input"]
    assert block["input"] == {"questions": [{"q": "?"}]}


@pytest.mark.asyncio
async def test_stream_assistant_flushes_tool_events_before_final_response(monkeypatch: pytest.MonkeyPatch):
    session_id = str(uuid.uuid4())
    entity_store = FakeEntityStore()
    conversation = AssistantConversation(
        name=f"assistant-{session_id}",
        workspace="default",
        session_id=session_id,
        owner_id="local-user",
    )
    await entity_store.create(conversation)
    invoked_messages: list[dict[str, str]] = []

    async def fake_invoke(agent_url, headers, messages, studio_session_id):
        invoked_messages.extend(messages)
        queue = assistant._session_streams[studio_session_id]
        # Two tool events queued in the same turn the invocation completes: the
        # loop can consume at most one, so the drain must flush the remainder.
        queue.put_nowait(assistant._tool_use_stream_event("nemo_api", {"resource": "secrets"}))
        queue.put_nowait(assistant._tool_use_stream_event("describe_api", {"path": "secrets"}))
        return "final answer", "model-x"

    monkeypatch.setattr(assistant, "_invoke_assistant", fake_invoke)

    frames = [
        frame
        async for frame in assistant._stream_assistant(
            session_id,
            "hello",
            "https://agent.test/x",
            {},
            "sys prompt",
            conversation,
            entity_store,
        )
    ]

    body = "".join(frames)
    first_tool = body.find("nemo_api")
    second_tool = body.find("describe_api")
    final = body.find("final answer")
    assert first_tool != -1 and second_tool != -1 and final != -1
    # Both tool-use events survive and are emitted before the final assistant message.
    assert first_tool < final
    assert second_tool < final
    contextual_message = invoked_messages[-1]["content"]
    assert "For any deploy_guardrail call for this request, pass deployment_run_id='" in contextual_message
    deployment_run_id = contextual_message.split("deployment_run_id='", 1)[1].split("'", 1)[0]
    assert str(uuid.UUID(deployment_run_id)) == deployment_run_id
    assert [message.content for message in conversation.messages] == ["hello", "final answer"]


@pytest.mark.asyncio
async def test_stream_assistant_retries_conflicted_conversation_update(monkeypatch: pytest.MonkeyPatch):
    session_id = str(uuid.uuid4())
    conversation = AssistantConversation(
        name=f"assistant-{session_id}",
        workspace="default",
        session_id=session_id,
        owner_id="local-user",
    )
    concurrent_conversation = conversation.model_copy(deep=True)
    concurrent_conversation.messages.extend(
        [
            AssistantMessage(role="user", content="remote question"),
            AssistantMessage(role="assistant", content="remote answer"),
        ]
    )

    class ConflictingEntityStore(FakeEntityStore):
        def __init__(self) -> None:
            super().__init__()
            self.update_calls = 0

        async def update(self, entity: AssistantConversation) -> AssistantConversation:
            self.update_calls += 1
            if self.update_calls == 1:
                self.entities[(concurrent_conversation.workspace, concurrent_conversation.name)] = (
                    concurrent_conversation
                )
                raise EntityConflictError("conversation was updated by another replica")
            return await super().update(entity)

    entity_store = ConflictingEntityStore()
    await entity_store.create(conversation)

    async def fake_invoke(agent_url, headers, messages, studio_session_id):
        return "local answer", "model-x"

    monkeypatch.setattr(assistant, "_invoke_assistant", fake_invoke)

    frames = [
        frame
        async for frame in assistant._stream_assistant(
            session_id,
            "local question",
            "https://agent.test/x",
            {},
            "sys prompt",
            conversation,
            entity_store,
        )
    ]

    assert "event: done" in "".join(frames)
    assert entity_store.update_calls == 2
    persisted = entity_store.entities[("default", conversation.name)]
    assert [message.content for message in persisted.messages] == [
        "remote question",
        "remote answer",
        "local question",
        "local answer",
    ]
    assert persisted.chat_artifacts.assistant_model == "model-x"


def test_assistant_request_payload_keeps_session_outside_model_messages():
    messages = [{"role": "user", "content": "hello"}]
    session_id = str(uuid.uuid4())

    payload = assistant._assistant_request_payload(messages, session_id)

    assert payload == {
        "messages": messages,
        "stream": True,
        "studio_session_id": session_id,
    }
    assert session_id not in json.dumps(payload["messages"])


def test_public_mcp_route_is_mounted_before_static_fallback():
    service = StudioService()
    app = FastAPI()
    service.configure_app(app)
    client = TestClient(app)
    session_id = str(uuid.uuid4())

    response = client.post(
        f"/studio/api/assistant/mcp/{session_id}",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
    )
    get_response = client.get(f"/studio/api/assistant/mcp/{session_id}")
    delete_response = client.delete(f"/studio/api/assistant/mcp/{session_id}")

    assert response.status_code == 200
    assert [tool["name"] for tool in response.json()["result"]["tools"]] == [
        "approval_prompt",
        "select_agent",
        "select_eval_config",
        "select_dataset_file",
        "select_model",
        "job_progress",
        "studio_link",
    ]
    assert get_response.status_code == 405
    assert get_response.headers["allow"] == "POST"
    assert delete_response.status_code == 405
    assert delete_response.headers["allow"] == "POST"


def test_assistant_routes_are_available_by_default():
    client = TestClient(StudioService().app)

    response = client.post("/v2/assistant/sessions")

    assert response.status_code == 200
    uuid.UUID(response.json()["session_id"])


def test_parse_reasoning_step_output_extracts_the_trace():
    payload = "**Input:**\n```python\nchain of thought\n```\n\n**Output:** The user asked a math question."
    assert assistant._parse_reasoning_step_output(payload) == "The user asked a math question."


def test_parse_reasoning_step_output_ignores_a_start_step():
    # The paired start step has no Output block yet.
    assert assistant._parse_reasoning_step_output("**Input:**\n```python\nchain of thought\n```") == ""
    assert assistant._parse_reasoning_step_output(None) == ""


def test_reasoning_stream_event_shape():
    event_type, payload = assistant._reasoning_stream_event("thinking out loud")
    assert event_type == "agent"
    block = json.loads(payload)["message"]["content"][0]
    assert block == {"type": "reasoning", "text": "thinking out loud"}


@pytest.mark.asyncio
async def test_invoke_assistant_relays_reasoning_despite_shared_step_id(monkeypatch: pytest.MonkeyPatch):
    """A reasoning step is a start/end pair sharing one id. The tool dedup must not
    claim that id on the start, or the end -- which carries the trace -- is dropped.
    """
    session_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    assistant._session_streams[session_id] = queue

    start = json.dumps({"id": "step-1", "name": "Reasoning: model", "payload": "**Input:**\n```python\nx\n```"})
    end = json.dumps(
        {
            "id": "step-1",
            "name": "Reasoning: model",
            "payload": "**Input:**\n```python\nx\n```\n\n**Output:** I thought about it.",
        }
    )
    lines = [
        f"intermediate_data: {start}",
        f"intermediate_data: {end}",
        'data: {"choices":[{"delta":{"content":"done"}}]}',
        "data: [DONE]",
    ]

    class _Response:
        status_code = 200

        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            for line in lines:
                yield line

    class _Stream:
        async def __aenter__(self):
            return _Response()

        async def __aexit__(self, *args):
            return False

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def stream(self, *args, **kwargs):
            return _Stream()

    monkeypatch.setattr(assistant.httpx, "AsyncClient", lambda **kwargs: _Client())

    text, _ = await assistant._invoke_assistant("https://a.test", {}, [], session_id)

    assert text == "done"
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    reasoning = [json.loads(payload) for _, payload in events]
    blocks = [b for message in reasoning for b in message["message"]["content"]]
    assert {"type": "reasoning", "text": "I thought about it."} in blocks


@pytest.mark.asyncio
async def test_invoke_assistant_emits_a_repeated_reasoning_step_once(monkeypatch: pytest.MonkeyPatch):
    """A repeated completed step must not render the same trace twice, while the
    start of the pair -- which shares its id -- must never claim that id."""
    session_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    assistant._session_streams[session_id] = queue

    start = json.dumps({"id": "step-1", "name": "Reasoning: model", "payload": "**Input:**\n```python\nx\n```"})
    end = json.dumps(
        {
            "id": "step-1",
            "name": "Reasoning: model",
            "payload": "**Input:**\n```python\nx\n```\n\n**Output:** I thought about it.",
        }
    )
    lines = [
        f"intermediate_data: {start}",
        f"intermediate_data: {end}",
        f"intermediate_data: {end}",
        'data: {"choices":[{"delta":{"content":"done"}}]}',
        "data: [DONE]",
    ]

    class _Response:
        status_code = 200

        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            for line in lines:
                yield line

    class _Stream:
        async def __aenter__(self):
            return _Response()

        async def __aexit__(self, *args):
            return False

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def stream(self, *args, **kwargs):
            return _Stream()

    monkeypatch.setattr(assistant.httpx, "AsyncClient", lambda **kwargs: _Client())

    await assistant._invoke_assistant("https://a.test", {}, [], session_id)

    blocks = []
    while not queue.empty():
        _, payload = queue.get_nowait()
        blocks.extend(json.loads(payload)["message"]["content"])
    reasoning_blocks = [b for b in blocks if b.get("type") == "reasoning"]
    assert reasoning_blocks == [{"type": "reasoning", "text": "I thought about it."}]
