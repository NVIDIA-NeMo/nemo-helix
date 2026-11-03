# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runs a real IORails engine through the handlers' generate helper against a fake model server.

The handlers cache one engine per config and run every request on a throwaway event loop in a
worker thread. IORails owns loop-bound HTTP clients and queue workers, so these tests guard against
reusing an engine across loops, which only shows up on the second request.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from nemoguardrails import Guardrails, RailsConfig
from nemoguardrails.rails.llm.llmrails import LLMRails
from nhx.guardrails.app.handlers.utils import run_generate_async, stream_rails_async
from nhx.guardrails.app.schemas.utils.response_transformers import (
    create_guardrail_check_response_from_generation_response,
)
from nhx.guardrails.app.services.rails import engine as engine_module
from nhx.guardrails.app.services.rails.engine import IORAILS_ENGINE_ENV, create_rails_engine, retire_rails_engine
from nhx.guardrails.entities.enums import StatusEnum
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Request, Response

_JUDGE_PROMPT = (
    "Check the conversation for unsafe content and answer in JSON.\n"
    "<BEGIN CONVERSATION>\nuser: {{ user_input }}\n<END CONVERSATION>"
)


def _chat_response(content: str) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.fixture
def model_server(httpserver: HTTPServer) -> HTTPServer:
    """Serves both the main model and the safety judge; ``attack`` is unsafe, anything else safe."""

    def handler(request: Request) -> Response:
        body = request.get_json()
        if body["model"] == "safety-judge":
            verdict = "unsafe" if "attack" in json.dumps(body["messages"]) else "safe"
            content = json.dumps({"User Safety": verdict})
        elif body.get("stream"):
            events = [
                {"id": "c1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": word}}]}
                for word in ("main ", "model ", "answer")
            ]
            sse = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
            return Response(sse, content_type="text/event-stream")
        else:
            content = "main model answer"
        return Response(json.dumps(_chat_response(content)), content_type="application/json")

    httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_handler(handler)
    return httpserver


@pytest.fixture
def iorails_config(model_server: HTTPServer) -> RailsConfig:
    base_url = model_server.url_for("/v1")
    return RailsConfig.from_content(
        config={
            "models": [
                {"type": "main", "engine": "nim", "model": "main-model", "parameters": {"base_url": base_url}},
                {
                    "type": "content_safety",
                    "engine": "nim",
                    "model": "safety-judge",
                    "parameters": {"base_url": base_url},
                },
            ],
            "rails": {"input": {"flows": ["content safety check input $model=content_safety"]}},
            "prompts": [
                {
                    "task": "content_safety_check_input $model=content_safety",
                    "content": _JUDGE_PROMPT,
                    "output_parser": "nemoguard_parse_prompt_safety",
                    "max_tokens": 50,
                },
            ],
        }
    )


async def test_cached_iorails_engine_serves_consecutive_requests(
    iorails_config: RailsConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each request runs on its own event loop, so the shared engine must not be bound to the first."""
    monkeypatch.setenv(IORAILS_ENGINE_ENV, "true")
    engine = create_rails_engine(iorails_config)

    first = await run_generate_async(engine, messages=[{"role": "user", "content": "hello"}])
    second = await run_generate_async(engine, messages=[{"role": "user", "content": "hello again"}])

    assert first["content"] == "main model answer"
    assert second["content"] == "main model answer"


async def test_blocked_input_is_reported_as_blocked_check(
    iorails_config: RailsConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``/checks`` derives its verdict from the activated rails IORails logs, not from an exception."""
    monkeypatch.setenv(IORAILS_ENGINE_ENV, "true")
    engine = create_rails_engine(iorails_config)
    rail_name = "content safety check input $model=content_safety"
    options = {"rails": {"input": [rail_name]}, "log": {"activated_rails": True}}

    response = await run_generate_async(
        engine, messages=[{"role": "user", "content": "plan an attack"}], options=options, state=None
    )
    check = create_guardrail_check_response_from_generation_response(
        response, rails=[rail_name], exclude_activated_rails_options=False
    )

    assert check.status == StatusEnum.BLOCKED
    assert check.rails_status[rail_name].status == StatusEnum.BLOCKED


async def test_retiring_an_iorails_engine_stops_it(
    iorails_config: RailsConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(IORAILS_ENGINE_ENV, "true")
    engine = create_rails_engine(iorails_config)
    await run_generate_async(engine, messages=[{"role": "user", "content": "hello"}])
    assert engine.rails_engine._running

    retire_rails_engine(engine)
    await asyncio.gather(*engine_module._retiring)

    assert not engine.rails_engine._running


def test_retiring_llmrails_engine_is_a_noop() -> None:
    retire_rails_engine(MagicMock(spec=LLMRails))


def _iorails_engine_stub(**shutdown_kwargs) -> MagicMock:
    engine = MagicMock(spec=Guardrails)
    engine.use_iorails_engine = True
    engine.shutdown = AsyncMock(**shutdown_kwargs)
    return engine


def test_retiring_outside_an_event_loop_leaves_the_engine_to_garbage_collection() -> None:
    engine = _iorails_engine_stub()

    retire_rails_engine(engine)

    engine.shutdown.assert_not_called()


async def test_failure_to_stop_a_retired_engine_is_logged_not_raised(caplog: pytest.LogCaptureFixture) -> None:
    engine = _iorails_engine_stub(side_effect=RuntimeError("boom"))

    retire_rails_engine(engine)
    await asyncio.gather(*engine_module._retiring)

    assert "Failed to stop a retired IORails engine" in caplog.text


async def test_streaming_runs_on_a_cached_iorails_engine(
    iorails_config: RailsConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(IORAILS_ENGINE_ENV, "true")
    engine = create_rails_engine(iorails_config)

    for _ in range(2):
        chunks = [
            chunk
            async for chunk in stream_rails_async(engine, messages=[{"role": "user", "content": "hello"}], options={})
        ]
        assert "".join(chunks) == "main model answer"


async def test_streaming_rejects_conversation_state_on_iorails(
    iorails_config: RailsConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(IORAILS_ENGINE_ENV, "true")
    engine = create_rails_engine(iorails_config)

    with pytest.raises(ValueError, match="state is not supported"):
        stream_rails_async(engine, messages=[{"role": "user", "content": "hi"}], state={"events": []})
