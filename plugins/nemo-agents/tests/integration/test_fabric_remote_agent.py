# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the packaged Remote Agent adapter through a real Fabric runtime."""

import asyncio
import json
import threading
from pathlib import Path

import httpx
import pytest
from nemo_agents_plugin.agent_config import load_agent_config
from nemo_agents_plugin.fabric.runtime import FabricInvocationRequest, FabricOneShotRequest, run_fabric_agent_once
from nemo_agents_plugin.fabric.session_manager import FabricSessionManager
from nemo_agents_plugin.fabric.session_registry import FabricSessionRegistry
from nemo_agents_plugin.fabric.streaming import extract_assistant_text_delta
from nemo_agents_plugin.fabric.translator import translate_agent_config
from nemo_fabric_collector import serve_collector
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Request, Response

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_remote_agent_invokes_http_service(
    tmp_path: Path, httpserver: HTTPServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REMOTE_AGENT_API_KEY", "test-remote-key")
    httpserver.expect_oneshot_request(
        "/v1/chat/completions",
        method="POST",
        headers={"Authorization": "Bearer test-remote-key"},
        json={
            "model": "remote-agent",
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ).respond_with_data(
        'data: {"choices":[{"delta":{"content":"Hello from remote"},"finish_reason":null}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n",
        content_type="text/event-stream",
    )
    config = load_agent_config(Path(__file__).parents[2] / "examples/nemo-agent-config/agent-remote.yaml")
    config.harnesses["remote-agent"].settings.update(base_url=httpserver.url_for("/v1"), api_type="openai-completions")
    config.environment.workspace = "."
    config.environment.artifacts = "./artifacts"

    result = await run_fabric_agent_once(
        FabricOneShotRequest(
            fabric_config=translate_agent_config(config),
            base_dir=tmp_path,
            input="Say hello",
            timeout_seconds=30,
        )
    )

    httpserver.check_assertions()
    assert result.status == "succeeded", result
    assert result.response == "Hello from remote"


@pytest.mark.asyncio
@pytest.mark.parametrize("use_session", [False, True])
async def test_remote_streams_from_shared_collector(
    tmp_path: Path, httpserver: HTTPServer, monkeypatch: pytest.MonkeyPatch, use_session: bool
) -> None:
    config = load_agent_config(Path(__file__).parents[2] / "examples/nemo-agent-config/agent-remote.yaml")
    config.harnesses["remote-agent"].settings.update(
        base_url=httpserver.url_for("/v1"), api_type="openai-completions", relay_streaming=True
    )
    config.models["default"].api_key_env = None
    config.environment.workspace = "."
    config.environment.artifacts = "./artifacts"
    received = threading.Event()
    seen_requests = []

    async with serve_collector() as collector_url:
        # The shared collector is already running. Helix must not launch another.
        def unexpected_collector(**kwargs):
            pytest.fail("Helix tried to start an embedded collector")

        monkeypatch.setattr("nemo_fabric_collector.serve_collector", unexpected_collector)
        config.telemetry.enabled = True
        config.telemetry.atof = {
            "enabled": True,
            "sinks": [{"type": "stream", "name": "nemo-fabric-stream", "url": collector_url, "transport": "ndjson"}],
        }

        def remote_service(request: Request) -> Response:
            payload = request.get_json()
            seen_requests.append(payload)
            request_id = payload["metadata"]["nemo_fabric_request_id"]
            records = [
                {
                    "kind": "scope",
                    "scope_category": "start",
                    "uuid": "unrelated",
                    "metadata": {"nemo_fabric_request_id": "another-request"},
                },
                {
                    "kind": "scope",
                    "scope_category": "end",
                    "uuid": "unrelated",
                    "category": "llm",
                    "data": {"role": "assistant", "content": "Wrong request"},
                },
                {
                    "kind": "scope",
                    "scope_category": "start",
                    "uuid": request_id,
                    "metadata": {"nemo_fabric_request_id": request_id},
                },
                {
                    "kind": "scope",
                    "scope_category": "end",
                    "uuid": request_id,
                    "category": "llm",
                    "data": {"role": "assistant", "content": "Live " + request_id},
                },
            ]
            response = httpx.post(collector_url + "/v1/atof", content="".join(json.dumps(r) + "\n" for r in records))
            response.raise_for_status()
            assert received.wait(5), "Helix did not receive the event before the HTTP response completed"
            return Response(
                'data: {"choices":[{"delta":{"content":"Final answer"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
                mimetype="text/event-stream",
            )

        httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_handler(remote_service)
        manager = FabricSessionManager(config, base_dir=tmp_path, session_registry=FabricSessionRegistry())
        session = await manager.open_session() if use_session else None
        try:
            for index in range(2):
                received.clear()
                request_id = f"turn-{index}"
                invocation = FabricInvocationRequest(input="Say hello", request_id=request_id, timeout_seconds=15)
                context = manager.stream_session(session, invocation) if session else manager.stream_once(invocation)
                async with asyncio.timeout(20), context as stream:
                    texts = []
                    async for record in stream.records():
                        text = extract_assistant_text_delta(record)
                        if text is not None:
                            texts.append(text)
                            received.set()
                    result = await stream.result()
                assert texts == ["Live " + request_id]
                assert result.status == "succeeded", result
                assert result.response == "Final answer"
            assert len(seen_requests[1]["messages"]) == (3 if use_session else 1)
            httpserver.check_assertions()
        finally:
            received.set()
            if session:
                await manager.close_session(session.session_id)
        # Closing a Helix runtime must not stop the externally owned collector.
        async with httpx.AsyncClient() as client:
            assert (await client.post(collector_url + "/v1/atof", content="")).status_code == 200
