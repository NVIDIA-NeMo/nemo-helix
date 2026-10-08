# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Claude Code execution against a local Anthropic model stub."""

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
        find_spec("claude_agent_sdk") is None,
        reason="Run make test-agents-claude to install the optional Claude harness",
    ),
]


@pytest.fixture
def claude_config(httpserver: HTTPServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AgentConfig:
    monkeypatch.setenv("CLAUDE_TEST_API_KEY", "test-key")
    config = load_agent_config(Path(__file__).parents[2] / "examples/nemo-agent-config/agent-claude.yaml")
    model = config.harnesses["claude"].model
    assert model is not None
    model.api_key_env = "CLAUDE_TEST_API_KEY"
    model.base_url = httpserver.url_for("").rstrip("/")
    config.environment.env.update(
        {
            "CLAUDE_CONFIG_DIR": str(tmp_path / "claude-config"),
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "ANTHROPIC_AUTH_TOKEN": "",
        }
    )

    def respond(request: Request) -> Response:
        payload = request.get_json()
        assert payload["model"] == "claude-sonnet-4-5"
        message = {
            "id": "msg-test",
            "type": "message",
            "role": "assistant",
            "model": payload["model"],
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 1},
        }
        if not payload.get("stream"):
            return Response(json.dumps(message), mimetype="application/json")
        events = [
            {"type": "message_start", "message": {**message, "content": [], "stop_reason": None}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello"}},
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
            {"type": "message_stop"},
        ]
        return Response(
            "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events),
            mimetype="text/event-stream",
        )

    httpserver.expect_request("/", method="HEAD").respond_with_data("")
    httpserver.expect_request("/v1/messages", method="POST", headers={"x-api-key": "test-key"}).respond_with_handler(
        respond
    )
    return config


@pytest.mark.asyncio
async def test_claude_invocation(claude_config: AgentConfig, tmp_path: Path, httpserver: HTTPServer) -> None:
    manager = FabricSessionManager(claude_config, base_dir=tmp_path, session_registry=FabricSessionRegistry())
    async with asyncio.timeout(60):
        result = await manager.invoke_once(FabricInvocationRequest(input="Say hello", timeout_seconds=45))
    httpserver.check_assertions()
    assert result.status == "succeeded", result
    assert result.response == "hello"
    assert httpserver.log


@pytest.mark.asyncio
@pytest.mark.parametrize("use_session", [False, True])
async def test_claude_relay_streaming(
    claude_config: AgentConfig, tmp_path: Path, httpserver: HTTPServer, use_session: bool
) -> None:
    manager = FabricSessionManager(claude_config, base_dir=tmp_path, session_registry=FabricSessionRegistry())
    async with asyncio.timeout(90):
        session = await manager.open_session() if use_session else None
        try:
            runtime_ids = []
            for turn in range(2):
                request = FabricInvocationRequest(
                    input="Say hello", request_id=f"claude-turn-{turn}", timeout_seconds=30
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
