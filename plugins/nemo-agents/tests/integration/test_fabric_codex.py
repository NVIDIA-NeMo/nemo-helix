# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Codex execution against a local Responses model stub."""

import asyncio
import json
from importlib.util import find_spec
from pathlib import Path

import pytest
from nemo_agents_plugin.agent_config import AgentConfig, load_agent_config
from nemo_agents_plugin.fabric.runtime import FabricInvocationRequest
from nemo_agents_plugin.fabric.session_manager import FabricSessionManager
from nemo_agents_plugin.fabric.session_registry import FabricSessionRegistry
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Request, Response

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        find_spec("openai_codex") is None,
        reason="Run make test-agents-codex to install the optional Codex harness",
    ),
]


@pytest.fixture
def codex_config(httpserver: HTTPServer, monkeypatch: pytest.MonkeyPatch) -> AgentConfig:
    monkeypatch.setenv("CODEX_TEST_API_KEY", "test-key")
    config = load_agent_config(Path(__file__).parents[2] / "examples/nemo-agent-config/agent-codex.yaml")
    model = config.models["default"]
    model.api_key_env = "CODEX_TEST_API_KEY"
    model.base_url = httpserver.url_for("/v1")

    def respond(request: Request) -> Response:
        payload = request.get_json()
        assert payload["model"] == model.model
        assert payload["stream"] is True
        message = {
            "id": "msg-test",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "hello", "annotations": []}],
        }
        response = {
            "id": "resp-test",
            "object": "response",
            "model": model.model,
            "status": "completed",
            "output": [message],
            "usage": {"input_tokens": 10, "output_tokens": 1, "total_tokens": 11},
        }
        events = [
            {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {**message, "status": "in_progress", "content": []},
            },
            {
                "type": "response.content_part.added",
                "item_id": "msg-test",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg-test",
                "output_index": 0,
                "content_index": 0,
                "delta": "hello",
            },
            {"type": "response.output_item.done", "output_index": 0, "item": message},
            {"type": "response.completed", "response": response},
        ]
        return Response(
            "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events),
            mimetype="text/event-stream",
        )

    httpserver.expect_request(
        "/v1/responses", method="POST", headers={"Authorization": "Bearer test-key"}
    ).respond_with_handler(respond)
    return config


@pytest.mark.asyncio
async def test_codex_invocation(codex_config: AgentConfig, tmp_path: Path, httpserver: HTTPServer) -> None:
    manager = FabricSessionManager(codex_config, base_dir=tmp_path, session_registry=FabricSessionRegistry())
    async with asyncio.timeout(60):
        result = await manager.invoke_once(FabricInvocationRequest(input="Say hello", timeout_seconds=45))
    httpserver.check_assertions()
    assert result.status == "succeeded", result
    assert result.response == "hello"
    assert httpserver.log


@pytest.mark.asyncio
@pytest.mark.parametrize("use_session", [False, True])
async def test_codex_relay_streaming(
    codex_config: AgentConfig, tmp_path: Path, httpserver: HTTPServer, use_session: bool
) -> None:
    manager = FabricSessionManager(codex_config, base_dir=tmp_path, session_registry=FabricSessionRegistry())
    async with asyncio.timeout(90):
        session = await manager.open_session() if use_session else None
        try:
            runtime_ids = []
            for turn in range(2):
                request = FabricInvocationRequest(
                    input="Say hello", request_id=f"codex-turn-{turn}", timeout_seconds=30
                )
                context = manager.stream_session(session, request) if session else manager.stream_once(request)
                async with context as stream:
                    records = [record async for record in stream.records()]
                    result = await stream.result()
                httpserver.check_assertions()
                assert result.status == "succeeded", result
                assert result.response == "hello"
                assert any(record.get("kind") == "scope" for record in records), records
                assert result.request_id == request.request_id
                runtime_ids.append(result.runtime_id)
            assert (runtime_ids[0] == runtime_ids[1]) is use_session
        finally:
            if session:
                await manager.close_session(session.session_id)
