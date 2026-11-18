# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that a cached IORails engine sends request headers to its main model, and only to it.

Request headers aren't built into engines, so the IORails main model has to get them on each call.
These tests run a real IORails engine against a fake model server and check the headers each model
received.
"""

import asyncio
import json

import pytest
from nemoguardrails import RailsConfig
from nhx.guardrails.app.handlers.utils import run_check_async, run_generate_async, stream_rails_async
from nhx.guardrails.app.services.rails.engine import IORAILS_ENGINE_ENV, create_rails_engine
from nhx.guardrails.app.utils.context_utils import set_request_default_headers_into_context
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Request, Response

_JUDGE_PROMPT = (
    "Check the conversation for unsafe content and answer in JSON.\n"
    "<BEGIN CONVERSATION>\nuser: {{ user_input }}\n<END CONVERSATION>"
)
_SELF_CHECK_PROMPT = "Should this user message be blocked? Answer Yes or No.\nUser message: {{ user_input }}\nAnswer:"


def _chat_response(content: str) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.fixture
def model_server(httpserver: HTTPServer) -> HTTPServer:
    """Fake main model and safety judge. Everything is safe and every self-check passes."""

    def handler(request: Request) -> Response:
        body = request.get_json()
        if body["model"] == "safety-judge":
            content = json.dumps({"User Safety": "safe"})
        elif body.get("stream"):
            event = {
                "id": "c1",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": "ok"}}],
            }
            return Response(f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n", content_type="text/event-stream")
        elif "Should this user message be blocked" in json.dumps(body["messages"]):
            content = "No"
        else:
            content = "main model answer"
        return Response(json.dumps(_chat_response(content)), content_type="application/json")

    httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_handler(handler)
    return httpserver


def _models(base_url: str) -> list[dict]:
    return [
        {
            "type": "main",
            "engine": "nim",
            "model": "main-model",
            "parameters": {"base_url": base_url, "default_headers": {"X-Tenant": "configured", "X-Static": "kept"}},
        },
        {
            "type": "content_safety",
            "engine": "nim",
            "model": "safety-judge",
            "parameters": {"base_url": base_url, "default_headers": {"X-Static": "judge"}},
        },
    ]


@pytest.fixture
def content_safety_config(model_server: HTTPServer) -> RailsConfig:
    return RailsConfig.from_content(
        config={
            "models": _models(model_server.url_for("/v1")),
            "rails": {"input": {"flows": ["content safety check input $model=content_safety"]}},
            "prompts": [
                {
                    "task": "content_safety_check_input $model=content_safety",
                    "content": _JUDGE_PROMPT,
                    "output_parser": "nemoguard_parse_prompt_safety",
                    "max_tokens": 50,
                }
            ],
        }
    )


@pytest.fixture
def self_check_config(model_server: HTTPServer) -> RailsConfig:
    """Self-check rails use the main model, so a ``/checks`` call reaches it."""
    return RailsConfig.from_content(
        config={
            "models": _models(model_server.url_for("/v1")),
            "rails": {"input": {"flows": ["self check input"]}},
            "prompts": [{"task": "self_check_input", "content": _SELF_CHECK_PROMPT}],
        }
    )


@pytest.fixture(autouse=True)
def _iorails(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(IORAILS_ENGINE_ENV, "true")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)


def _received(server: HTTPServer, model: str) -> list[dict[str, str]]:
    """Headers from each call ``model`` received, with lower-case names."""
    return [
        {name.lower(): value for name, value in request.headers.items()}
        for request, _ in server.log
        if request.get_json()["model"] == model
    ]


def _in_request(headers: dict[str, str]) -> None:
    """Set the request headers the way the handler does before running the rails."""
    set_request_default_headers_into_context(headers)


async def test_generation_sends_request_headers_to_the_main_model_only(
    model_server: HTTPServer, content_safety_config: RailsConfig
) -> None:
    engine = create_rails_engine(content_safety_config)
    _in_request({"x-tenant": "acme", "x-static": "kept"})

    await run_generate_async(engine, messages=[{"role": "user", "content": "hello"}])

    [main] = _received(model_server, "main-model")
    assert main["x-tenant"] == "acme"  # the request's value wins over the configured one
    assert main["x-static"] == "kept"
    # Task models keep only their configured headers.
    [judge] = _received(model_server, "safety-judge")
    assert "x-tenant" not in judge
    assert judge["x-static"] == "judge"


async def test_streaming_sends_request_headers_to_the_main_model(
    model_server: HTTPServer, content_safety_config: RailsConfig
) -> None:
    engine = create_rails_engine(content_safety_config)
    _in_request({"x-tenant": "acme"})

    chunks = [c async for c in stream_rails_async(engine, messages=[{"role": "user", "content": "hi"}], options={})]

    assert chunks == ["ok"]
    assert [call["x-tenant"] for call in _received(model_server, "main-model")] == ["acme"]


async def test_checks_send_request_headers_to_a_self_check_on_the_main_model(
    model_server: HTTPServer, self_check_config: RailsConfig
) -> None:
    engine = create_rails_engine(self_check_config)
    _in_request({"x-tenant": "acme"})

    await run_check_async(engine, messages=[{"role": "user", "content": "hello"}], rail_types=["input"])

    [main] = _received(model_server, "main-model")
    assert main["x-tenant"] == "acme"


async def test_concurrent_requests_each_send_only_their_own_headers(
    model_server: HTTPServer, content_safety_config: RailsConfig
) -> None:
    """IORails runs non-streaming work on shared queue workers. Headers must not leak between requests."""
    engine = create_rails_engine(content_safety_config)
    tenants = [f"tenant-{i}" for i in range(6)]

    async def request(tenant: str) -> None:
        _in_request({"x-tenant": tenant})  # each task has its own copy of the context
        await run_generate_async(engine, messages=[{"role": "user", "content": f"hello from {tenant}"}])

    await asyncio.gather(*(asyncio.create_task(request(t)) for t in tenants))

    sent = {
        json.dumps(request.get_json()["messages"]).split("hello from ")[1].split('"')[0]: request.headers["x-tenant"]
        for request, _ in model_server.log
        if request.get_json()["model"] == "main-model"
    }
    assert sent == {tenant: tenant for tenant in tenants}


async def test_calls_outside_a_request_send_only_configured_headers(
    model_server: HTTPServer, content_safety_config: RailsConfig
) -> None:
    engine = create_rails_engine(content_safety_config)
    _in_request({})

    await run_generate_async(engine, messages=[{"role": "user", "content": "hello"}])

    [main] = _received(model_server, "main-model")
    assert main["x-tenant"] == "configured"
