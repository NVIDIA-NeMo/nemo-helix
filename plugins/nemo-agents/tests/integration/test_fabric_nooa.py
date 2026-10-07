# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real NOOA adapters against a local model stub; run with make test-agents-nooa."""

import asyncio
import json
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest
from nemo_agents_plugin.agent_config import AgentConfig
from nemo_agents_plugin.fabric.runtime import FabricInvocationRequest
from nemo_agents_plugin.fabric.session_manager import FabricSessionManager
from nemo_agents_plugin.fabric.session_registry import FabricSessionRegistry
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Request, Response

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        find_spec("nooa_cli") is None or find_spec("nooa_bench") is None,
        reason="Run make test-agents-nooa to install the optional NOOA harness",
    ),
    pytest.mark.skipif(not (3, 12) <= sys.version_info[:2] < (3, 14), reason="NOOA requires Python 3.12–3.13"),
]


@pytest.fixture(params=["coding", "bench"])
def nooa_config(request: pytest.FixtureRequest, httpserver: HTTPServer, monkeypatch: pytest.MonkeyPatch) -> AgentConfig:
    monkeypatch.setenv("NOOA_TEST_API_KEY", "test-key")
    coding = request.param == "coding"
    selection = (
        {"workflow": {"target_id": "nvidia.nooa.coding-agent"}}
        if coding
        else {"default_harness": "bench", "harnesses": {"bench": {"kind": "nooa-bench-agent"}}}
    )
    code = (
        'self.message("hello"); return_result(kind="DONE", explanation="answered")'
        if coding
        else 'return_result(solution_description="done", evidence="fixture", command_to_verify="hello")'
    )

    def model(request: Request) -> Response:
        payload = request.get_json()
        assert payload["model"] == "test-model"
        assert not payload.get("stream")
        assert any(tool["function"]["name"] == "execute_python" for tool in payload["tools"])
        return Response(
            json.dumps(
                {
                    "id": "completion-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-test",
                                        "type": "function",
                                        "function": {
                                            "name": "execute_python",
                                            "arguments": json.dumps({"code": code}),
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                }
            ),
            mimetype="application/json",
        )

    httpserver.expect_request(
        "/v1/chat/completions", method="POST", headers={"Authorization": "Bearer test-key"}
    ).respond_with_handler(model)
    return AgentConfig.model_validate(
        {
            "config_format": "nemo-agents-spec-v1",
            "name": "nooa-test",
            **selection,
            "models": {
                "default": {
                    "provider": "openai",
                    "model": "test-model",
                    "api_key_env": "NOOA_TEST_API_KEY",
                    "base_url": httpserver.url_for("/v1"),
                    "settings": {"client_type": "completion"},
                }
            },
            "environment": {"workspace": ".", "artifacts": "./artifacts"},
            "telemetry": {"enabled": False},
        }
    )


@pytest.mark.asyncio
async def test_nooa_invocation(nooa_config: AgentConfig, tmp_path: Path, httpserver: HTTPServer) -> None:
    manager = FabricSessionManager(nooa_config, base_dir=tmp_path, session_registry=FabricSessionRegistry())
    async with asyncio.timeout(45):
        result = await manager.invoke_once(FabricInvocationRequest(input="Say hello", timeout_seconds=30))
    httpserver.check_assertions()
    assert result.status == "succeeded", result
    assert result.response == "hello"


@pytest.mark.asyncio
@pytest.mark.parametrize("use_session", [False, True])
async def test_nooa_relay_streaming(
    nooa_config: AgentConfig, tmp_path: Path, httpserver: HTTPServer, use_session: bool
) -> None:
    manager = FabricSessionManager(nooa_config, base_dir=tmp_path, session_registry=FabricSessionRegistry())
    async with asyncio.timeout(90):
        session = await manager.open_session() if use_session else None
        try:
            runtime_ids = []
            for turn in range(2):
                request = FabricInvocationRequest(input="Say hello", request_id=f"nooa-turn-{turn}", timeout_seconds=30)
                context = manager.stream_session(session, request) if session else manager.stream_once(request)
                async with context as stream:
                    records = [record async for record in stream.records()]
                    result = await stream.result()
                httpserver.check_assertions()
                assert result.status == "succeeded", result
                assert result.response == "hello"
                assert result.output["telemetry"]["enabled"] is True
                assert not result.output["telemetry"].get("degraded"), result.output
                assert any(record.get("kind") == "scope" for record in records), records
                assert any(
                    record.get("metadata", {}).get("nemo_fabric_request_id") == request.request_id for record in records
                ), records
                assert result.request_id == request.request_id
                runtime_ids.append(result.runtime_id)
            assert (runtime_ids[0] == runtime_ids[1]) is use_session
        finally:
            if session:
                await manager.close_session(session.session_id)
