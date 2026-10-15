# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Hermes execution against a local Chat Completions model stub."""

import asyncio
import json
import os
import re
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
        not os.environ.get("ADAPTER_PYTHON"),
        reason="Run make test-agents-hermes to install the optional Hermes harness",
    ),
]


@pytest.fixture
def hermes_config(httpserver: HTTPServer, monkeypatch: pytest.MonkeyPatch) -> AgentConfig:
    monkeypatch.setenv("HERMES_TEST_API_KEY", "test-key")
    config = load_agent_config(Path(__file__).parents[2] / "examples/nemo-agent-config/agent-hermes.yaml")
    model = config.harnesses["hermes"].model
    assert model is not None
    model.api_key_env = "HERMES_TEST_API_KEY"
    model.base_url = httpserver.url_for("/v1")
    model_name = model.model

    def respond(request: Request) -> Response:
        payload = request.get_json()
        assert payload["model"] == model_name
        if payload.get("stream"):
            chunks = [
                {"delta": {"role": "assistant", "content": "hello"}, "finish_reason": None},
                {"delta": {}, "finish_reason": "stop"},
            ]
            events = [
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": model_name,
                    "choices": [{"index": 0, **chunk}],
                }
                for chunk in chunks
            ]
            return Response(
                "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n",
                mimetype="text/event-stream",
            )
        return Response(
            json.dumps(
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "hello"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
                }
            ),
            mimetype="application/json",
        )

    # Hermes probes local endpoints for model metadata before inference.
    httpserver.expect_request(re.compile(r"/.*"), method="GET").respond_with_json({}, status=404)
    httpserver.expect_request("/api/show", method="POST").respond_with_json({}, status=404)
    httpserver.expect_request(
        "/v1/chat/completions", method="POST", headers={"Authorization": "Bearer test-key"}
    ).respond_with_handler(respond)
    return config


@pytest.mark.asyncio
async def test_hermes_invocation(hermes_config: AgentConfig, tmp_path: Path, httpserver: HTTPServer) -> None:
    manager = FabricSessionManager(hermes_config, base_dir=tmp_path, session_registry=FabricSessionRegistry())
    async with asyncio.timeout(60):
        result = await manager.invoke_once(FabricInvocationRequest(input="Say hello", timeout_seconds=45))
    httpserver.check_assertions()
    assert result.status == "succeeded", result
    assert result.response == "hello"
    assert httpserver.log


@pytest.mark.asyncio
@pytest.mark.parametrize("use_session", [False, True])
async def test_hermes_relay_streaming(
    hermes_config: AgentConfig, tmp_path: Path, httpserver: HTTPServer, use_session: bool
) -> None:
    manager = FabricSessionManager(hermes_config, base_dir=tmp_path, session_registry=FabricSessionRegistry())
    async with asyncio.timeout(90):
        session = await manager.open_session() if use_session else None
        try:
            runtime_ids = []
            for turn in range(2):
                request = FabricInvocationRequest(
                    input="Say hello", request_id=f"hermes-turn-{turn}", timeout_seconds=30
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
