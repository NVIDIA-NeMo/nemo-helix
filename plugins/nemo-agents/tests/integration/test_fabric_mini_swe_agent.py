# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Mini-SWE-Agent execution against a local model stub."""

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

_MODEL_RESPONSE = {
    "id": "completion-test",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-4o-mini",
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
                            "name": "bash",
                            "arguments": json.dumps(
                                {"command": "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\nhello\\n'"}
                            ),
                        },
                    }
                ],
            },
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
}

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        find_spec("minisweagent") is None,
        reason="Run make test-agents-mini-swe-agent to install the optional harness",
    ),
]


@pytest.fixture
def mini_swe_agent_config(httpserver: HTTPServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AgentConfig:
    monkeypatch.setenv("MINI_SWE_AGENT_TEST_API_KEY", "test-key")
    config = load_agent_config(Path(__file__).parents[2] / "examples/nemo-agent-config/agent-mini-swe-agent.yaml")
    config.models["default"].provider = "openai"
    config.models["default"].model = "gpt-4o-mini"
    config.models["default"].api_key_env = "MINI_SWE_AGENT_TEST_API_KEY"
    config.models["default"].base_url = httpserver.url_for("/v1")
    config.environment.env.update(
        {
            "MSWEA_SILENT_STARTUP": "1",
            "MSWEA_GLOBAL_CONFIG_DIR": str(tmp_path / "mini-config"),
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        }
    )

    def model(request: Request) -> Response:
        payload = request.get_json()
        assert payload["model"] == "gpt-4o-mini"
        assert not payload.get("stream")
        assert any(tool["function"]["name"] == "bash" for tool in payload["tools"])
        return Response(json.dumps(_MODEL_RESPONSE), mimetype="application/json")

    httpserver.expect_request(
        "/v1/chat/completions", method="POST", headers={"Authorization": "Bearer test-key"}
    ).respond_with_handler(model)
    return config


@pytest.mark.asyncio
async def test_mini_swe_agent_invocation(
    mini_swe_agent_config: AgentConfig, tmp_path: Path, httpserver: HTTPServer
) -> None:
    manager = FabricSessionManager(mini_swe_agent_config, base_dir=tmp_path, session_registry=FabricSessionRegistry())
    async with asyncio.timeout(45):
        result = await manager.invoke_once(FabricInvocationRequest(input="Say hello", timeout_seconds=30))
    httpserver.check_assertions()
    assert result.status == "succeeded", result
    assert result.response == "hello\n"
    assert result.output["usage"]["api_calls"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("use_session", [False, True])
async def test_mini_swe_agent_relay_streaming(
    mini_swe_agent_config: AgentConfig, tmp_path: Path, httpserver: HTTPServer, use_session: bool
) -> None:
    manager = FabricSessionManager(mini_swe_agent_config, base_dir=tmp_path, session_registry=FabricSessionRegistry())
    async with asyncio.timeout(90):
        session = await manager.open_session() if use_session else None
        try:
            runtime_ids = []
            for turn in range(2):
                request = FabricInvocationRequest(
                    input=f"Say hello for turn {turn}", request_id=f"mini-swe-agent-turn-{turn}", timeout_seconds=30
                )
                context = manager.stream_session(session, request) if session else manager.stream_once(request)
                async with context as stream:
                    records = [record async for record in stream.records()]
                    result = await stream.result()
                httpserver.check_assertions()
                assert result.status == "succeeded", result
                assert result.response == "hello\n"
                assert result.output["telemetry"]["enabled"] is True
                # Fabric 0.4 reports degraded tool telemetry; keep that fix upstream.
                assert any(record.get("kind") == "scope" for record in records), records
                assert any(
                    (record.get("metadata") or {}).get("nemo_fabric_request_id") == request.request_id
                    for record in records
                ), records
                assert result.request_id == request.request_id
                calls = [req.get_json() for req, _ in httpserver.log if req.path == "/v1/chat/completions"]
                assert len(calls) == turn + 1
                messages = calls[-1]["messages"]
                assert any(f"Say hello for turn {turn}" in str(message.get("content", "")) for message in messages)
                if turn:
                    has_history = any("Say hello for turn 0" in str(message.get("content", "")) for message in messages)
                    assert has_history is use_session
                runtime_ids.append(result.runtime_id)
            assert (runtime_ids[0] == runtime_ids[1]) is use_session
        finally:
            if session:
                await manager.close_session(session.session_id)
