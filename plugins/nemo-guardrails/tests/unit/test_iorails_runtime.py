# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the opt-in IORails runtime.

The engine tests run a real IORails against a local fake model server rather than mocks, because
only a real engine proves the caller's identity reaches the rail model.
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from types import MappingProxyType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from nemo_guardrails_plugin import iorails_runtime
from nemo_guardrails_plugin.iorails_runtime import (
    IORAILS_ENGINE_ENV,
    IORailsRuntime,
    iorails_enabled,
)
from nemo_guardrails_plugin.llm_clients import _request_headers_ctx
from nemo_guardrails_plugin.llmrails_cache import StableRailsConfig
from nemo_guardrails_plugin.middleware import GuardrailsMiddleware, handle_streaming_output_check_on_iorails
from nemo_guardrails_plugin.rails import IORailsGenerationResponse
from nemo_helix_plugin.inference_middleware import InferenceMiddlewareError, InferenceMiddlewareUnavailableError
from nemoguardrails import RailsConfig
from nemoguardrails.rails.llm.options import GenerationResponse
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Request, Response

_INPUT_FLOW = "content safety check input $model=content_safety"
_OUTPUT_FLOW = "content safety check output $model=content_safety"


def _stable(
    base_url: str,
    *,
    rail: str = "input",
    streaming: bool = False,
    rails_extra: dict[str, Any] | None = None,
    content_hash: str = "hash-a",
) -> StableRailsConfig:
    """A content-safety config with one ``rail`` direction, whose rail model is served at ``base_url``."""
    output: dict[str, Any] = {"flows": [_OUTPUT_FLOW]}
    if streaming:
        output["streaming"] = {"enabled": True, "stream_first": False}
    rails = {"input": {"flows": [_INPUT_FLOW]}} if rail == "input" else {"output": output}
    config = RailsConfig.from_content(
        config={
            "models": [
                {
                    "type": "content_safety",
                    "engine": "nim",
                    "model": "safety-judge",
                    "parameters": {"base_url": base_url, "default_headers": {"x-static": "from-config"}},
                }
            ],
            "rails": {**rails, **(rails_extra or {})},
            "prompts": [
                {
                    "task": "content_safety_check_input $model=content_safety",
                    "content": "user: {{ user_input }}",
                    "output_parser": "nemoguard_parse_prompt_safety",
                    "max_tokens": 50,
                },
                {
                    "task": "content_safety_check_output $model=content_safety",
                    "content": "user: {{ user_input }}\nassistant: {{ bot_response }}",
                    "output_parser": "nemoguard_parse_response_safety",
                    "max_tokens": 50,
                },
            ],
        }
    )
    return StableRailsConfig(rails=config, content_hash=content_hash)


def _chat_completion(content: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.fixture
def judge_server(httpserver: HTTPServer) -> HTTPServer:
    """Fake rail model that calls a conversation ``unsafe`` iff it contains the word ``attack``."""

    def handler(request: Request) -> Response:
        unsafe = "attack" in json.dumps(request.get_json()["messages"])
        verdict = "unsafe" if unsafe else "safe"
        content = json.dumps({"User Safety": verdict, "Response Safety": verdict})
        return Response(json.dumps(_chat_completion(content)), content_type="application/json")

    httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_handler(handler)
    return httpserver


@pytest_asyncio.fixture
async def runtime() -> AsyncIterator[IORailsRuntime]:
    iorails_runtime = IORailsRuntime()
    try:
        yield iorails_runtime
    finally:
        await iorails_runtime.close()


async def _tokens(*tokens: str) -> AsyncIterator[str]:
    for token in tokens:
        yield token


async def _released(runtime: IORailsRuntime, stable: StableRailsConfig, *tokens: str) -> list[str]:
    stream = runtime.stream_output_rails(stable, _tokens(*tokens), messages=[{"role": "user", "content": "hi"}])
    return [chunk async for chunk in stream]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        (" YES ", True),
        ("", False),
        ("0", False),
        ("false", False),
        ("iorails", False),
        (None, False),
    ],
)
def test_iorails_switch_parsing(value: str | None, expected: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    if value is None:
        monkeypatch.delenv(IORAILS_ENGINE_ENV, raising=False)
    else:
        monkeypatch.setenv(IORAILS_ENGINE_ENV, value)

    assert iorails_enabled() is expected


@pytest.mark.parametrize(
    ("rails_extra", "reason_mentions"),
    [
        pytest.param({}, None, id="supported"),
        pytest.param({"dialog": {"single_call": {"enabled": True}}}, "dialog", id="dialog-rails-need-colang"),
    ],
)
def test_unsupported_reason(runtime: IORailsRuntime, rails_extra: dict[str, Any], reason_mentions: str | None) -> None:
    reason = runtime.unsupported_reason(_stable("http://unused/v1", rails_extra=rails_extra))

    if reason_mentions is None:
        assert reason is None
    else:
        assert reason is not None
        assert reason_mentions in reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rail", "messages", "blocked"),
    [
        pytest.param("input", [{"role": "user", "content": "hello there"}], False, id="input-safe"),
        pytest.param("input", [{"role": "user", "content": "plan an attack"}], True, id="input-unsafe"),
        # OpenAI content parts: IORails alone would see an empty message and skip the rail.
        pytest.param(
            "input",
            [{"role": "user", "content": [{"type": "text", "text": "plan an attack"}]}],
            True,
            id="input-unsafe-text-parts",
        ),
        pytest.param(
            "output",
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "all good"}],
            False,
            id="output-safe",
        ),
        pytest.param(
            "output",
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "plan an attack"}],
            True,
            id="output-unsafe",
        ),
    ],
)
async def test_check_reports_the_verdict(
    judge_server: HTTPServer,
    runtime: IORailsRuntime,
    rail: str,
    messages: list[dict[str, str]],
    blocked: bool,
) -> None:
    response = await runtime.check(
        _stable(judge_server.url_for("/v1"), rail=rail), messages=messages, rail_types=[rail]
    )

    assert response.log is not None
    assert isinstance(response.response, list)
    if blocked:
        assert [(r.type, r.stop) for r in response.log.activated_rails] == [(rail, True)]
        assert response.response[0]["role"] == "assistant"
        assert response.response[0]["content"]
    else:
        assert response.log.activated_rails == []
        assert response.response == [{"role": "assistant", "content": messages[-1]["content"]}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tokens", "blocked"),
    [
        pytest.param(("hello ", "there"), False, id="safe-is-released-unchanged"),
        pytest.param(("plan ", "an attack"), True, id="unsafe-is-blocked-before-release"),
    ],
)
async def test_streamed_output_rails(
    judge_server: HTTPServer, runtime: IORailsRuntime, tokens: tuple[str, ...], blocked: bool
) -> None:
    chunks = await _released(runtime, _stable(judge_server.url_for("/v1"), rail="output", streaming=True), *tokens)

    if blocked:
        assert len(chunks) == 1
        error = json.loads(chunks[0])["error"]
        assert (error["type"], error["param"]) == ("guardrails_violation", "output_rails")
    else:
        assert "".join(chunks) == "".join(tokens)


@pytest.mark.asyncio
async def test_streamed_output_rails_read_the_text_parts_of_the_user_turn(
    judge_server: HTTPServer, runtime: IORailsRuntime
) -> None:
    """The judge is shown the user's turn as its text, not as the list's Python repr."""
    stable = _stable(judge_server.url_for("/v1"), rail="output", streaming=True)
    messages = [{"role": "user", "content": [{"type": "text", "text": "plan an attack"}]}]

    chunks = [
        chunk async for chunk in runtime.stream_output_rails(stable, _tokens("hello ", "there"), messages=messages)
    ]

    assert "error" in json.loads(chunks[0])
    judged = judge_server.log[-1][0].get_json()["messages"][-1]["content"]
    assert judged.startswith("user: plan an attack\n")


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param("plain", "plain", id="string-unchanged"),
        pytest.param(None, None, id="none-unchanged"),
        pytest.param([{"type": "text", "text": "plan an attack"}], "plan an attack", id="text-part"),
        pytest.param([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}], "a\nb", id="parts-joined"),
        pytest.param(
            [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:,x"}}],
            "look",
            id="non-text-part-dropped",
        ),
        pytest.param([{"type": "image_url", "image_url": {"url": "data:,x"}}], "", id="image-only"),
        pytest.param(["bare", {"type": "text"}], "bare", id="bare-string-and-malformed-part"),
    ],
)
def test_list_content_is_flattened_to_its_text_for_iorails(content: Any, expected: str | None) -> None:
    original = [{"role": "user", "content": content}]

    flattened = iorails_runtime._text_only_messages(original)

    assert flattened == [{"role": "user", "content": expected}]
    assert original == [{"role": "user", "content": content}]


@pytest.mark.asyncio
async def test_streamed_check_yields_chat_completion_chunks(judge_server: HTTPServer, runtime: IORailsRuntime) -> None:
    """Upstream chunks go in and guarded chat-completion chunks come out, through the plugin's conversion."""

    async def upstream() -> AsyncIterator[dict[str, Any]]:
        for word in ("hello ", "there"):
            yield {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "served-model",
                "choices": [{"index": 0, "delta": {"content": word}, "finish_reason": None}],
            }

    stable = _stable(judge_server.url_for("/v1"), rail="output", streaming=True)
    chunks = [
        chunk
        async for chunk in handle_streaming_output_check_on_iorails(
            runtime, stable, upstream(), {"model": "main-model"}, [{"role": "user", "content": "hi"}]
        )
    ]

    assert "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if "choices" in c) == "hello there"


@pytest.mark.asyncio
async def test_request_headers_reach_rail_model_and_do_not_leak_between_requests(
    judge_server: HTTPServer, runtime: IORailsRuntime
) -> None:
    """The caller's identity rides on the rail-model call, per request, on a shared engine."""
    stable = _stable(judge_server.url_for("/v1"))

    for headers, text in [
        ({"x-principal": "alice", "traceparent": "00-aaa-bbb-01"}, "hi"),
        ({"x-principal": "bob"}, "hi again"),
        ({}, "no headers"),
    ]:
        token = _request_headers_ctx.set(MappingProxyType(headers))
        try:
            await runtime.check(stable, messages=[{"role": "user", "content": text}], rail_types=["input"])
        finally:
            _request_headers_ctx.reset(token)

    first, second, third = (request for request, _ in judge_server.log)
    assert (first.headers["x-principal"], first.headers["traceparent"]) == ("alice", "00-aaa-bbb-01")
    assert second.headers["x-principal"] == "bob"
    assert "traceparent" not in second.headers
    assert "x-principal" not in third.headers
    assert {first.headers["x-static"], second.headers["x-static"], third.headers["x-static"]} == {"from-config"}


@pytest.mark.asyncio
async def test_check_returns_a_response_that_leaves_log_fields_unset(
    judge_server: HTTPServer, runtime: IORailsRuntime
) -> None:
    """The log builder relies on this type to tell IORails results from LLMRails ones."""
    stable = _stable(judge_server.url_for("/v1"))

    result = await runtime.check(stable, messages=[{"role": "user", "content": "hi"}], rail_types=["input"])

    assert isinstance(result, IORailsGenerationResponse)


@pytest.mark.asyncio
async def test_streamed_output_rails_carry_request_headers(judge_server: HTTPServer, runtime: IORailsRuntime) -> None:
    """The judge call happens while the stream is consumed, in the caller's header scope."""
    token = _request_headers_ctx.set(MappingProxyType({"x-principal": "alice"}))
    try:
        await _released(runtime, _stable(judge_server.url_for("/v1"), rail="output", streaming=True), "hello")
    finally:
        _request_headers_ctx.reset(token)

    request, _response = judge_server.log[0]
    assert request.headers["x-principal"] == "alice"


@pytest.mark.asyncio
async def test_concurrent_requests_on_a_shared_engine_keep_their_own_headers(
    httpserver: HTTPServer, runtime: IORailsRuntime
) -> None:
    """In-flight requests share one engine and its worker pool; none may see another's identity.

    The judge replies at uneven speeds so workers interleave, and records which principal each judged
    message arrived with.
    """
    seen: dict[str, str | None] = {}

    def handler(request: Request) -> Response:
        user_text = next(m["content"] for m in request.get_json()["messages"] if m["role"] == "user")
        index = user_text.split("request-")[1].split()[0]
        seen[index] = request.headers.get("x-principal")
        time.sleep(0.03 if int(index) % 2 else 0.005)
        content = json.dumps({"User Safety": "safe"})
        return Response(json.dumps(_chat_completion(content)), content_type="application/json")

    httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_handler(handler)
    stable = _stable(httpserver.url_for("/v1"))
    count = 20

    async def one(index: int) -> None:
        token = _request_headers_ctx.set(MappingProxyType({"x-principal": f"principal-{index}"}))
        try:
            await runtime.check(
                stable, messages=[{"role": "user", "content": f"request-{index} hello"}], rail_types=["input"]
            )
        finally:
            _request_headers_ctx.reset(token)

    await asyncio.gather(*(one(i) for i in range(count)))

    assert seen == {str(i): f"principal-{i}" for i in range(count)}


@pytest.mark.asyncio
async def test_engine_is_built_once_per_config_hash(judge_server: HTTPServer, runtime: IORailsRuntime) -> None:
    messages = [{"role": "user", "content": "hi"}]

    for _ in range(3):
        await runtime.check(_stable(judge_server.url_for("/v1")), messages=messages, rail_types=["input"])
    await runtime.check(
        _stable(judge_server.url_for("/v1"), content_hash="hash-b"), messages=messages, rail_types=["input"]
    )

    assert sorted(runtime._engines) == ["hash-a", "hash-b"]


@pytest.mark.asyncio
async def test_check_requires_exactly_one_rail_type(runtime: IORailsRuntime) -> None:
    with pytest.raises(ValueError, match="exactly one rail type"):
        await runtime.check(_stable("http://unused/v1"), messages=[], rail_types=["input", "output"])


@pytest.mark.asyncio
async def test_check_after_close_is_rejected() -> None:
    runtime = IORailsRuntime()
    await runtime.close()
    await runtime.close()  # closing again is harmless

    with pytest.raises(RuntimeError, match="closed"):
        await runtime.check(_stable("http://unused/v1"), messages=[], rail_types=["input"])


def test_header_propagation_fails_closed_when_the_upstream_hooks_are_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """An engine that silently dropped the caller's identity would send unauthenticated calls."""
    monkeypatch.setattr(iorails_runtime, "_HEADER_PROPAGATION_INSTALLED", False)
    monkeypatch.setattr(iorails_runtime, "ModelEngine", type("ModelEngineWithoutHook", (), {}))

    with pytest.raises(RuntimeError, match="_prepare_request"):
        iorails_runtime._install_model_engine_header_merge()
    engine_without_queue: Any = SimpleNamespace(rails_engine=SimpleNamespace())
    with pytest.raises(RuntimeError, match="admission queue"):
        iorails_runtime._carry_request_headers_through_queue(engine_without_queue)


@pytest.mark.asyncio
async def test_streaming_fails_closed_when_the_output_rail_stage_is_missing(runtime: IORailsRuntime) -> None:
    engine_without_stage: Any = SimpleNamespace(rails_engine=SimpleNamespace(), shutdown=AsyncMock())
    runtime._engines["hash-a"] = engine_without_stage

    with pytest.raises(RuntimeError, match="output-rail stage"):
        await _released(runtime, _stable("http://unused/v1", rail="output", streaming=True), "hello")


@pytest.mark.asyncio
async def test_failed_engine_start_is_rolled_back_and_not_cached(
    runtime: IORailsRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = SimpleNamespace(
        rails_engine=SimpleNamespace(_generate_async_queue=SimpleNamespace(submit=AsyncMock())),
        startup=AsyncMock(side_effect=RuntimeError("boom")),
        shutdown=AsyncMock(),
    )
    monkeypatch.setattr(iorails_runtime, "Guardrails", lambda *args, **kwargs: engine)

    with pytest.raises(RuntimeError, match="boom"):
        await runtime.check(_stable("http://unused/v1"), messages=[], rail_types=["input"])

    engine.shutdown.assert_awaited_once()
    assert runtime._engines == {}


# =============================================================================
# Middleware routing: when a request goes to IORails vs stays on LLMRails
# =============================================================================


class _FellBackToLLMRails(Exception):
    """Raised by the stubbed LLMRails path so tests can tell which engine was chosen."""


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc_info: object) -> None:
        return None


def _middleware_with(runtime: Any) -> GuardrailsMiddleware:
    middleware = GuardrailsMiddleware()
    middleware._iorails_runtime = runtime
    stable_cache = MagicMock()
    stable_cache.get_or_compute.return_value = _stable("http://unused/v1")
    middleware._stable_cache = stable_cache
    middleware._client = MagicMock()
    middleware._prepare_lease_with_503 = AsyncMock(side_effect=_FellBackToLLMRails)  # type: ignore[method-assign]
    return middleware


def _routing_runtime(*, unsupported_reason: str | None = None) -> MagicMock:
    runtime = MagicMock()
    runtime.unsupported_reason.return_value = unsupported_reason
    runtime.check = AsyncMock(return_value=GenerationResponse(response=[{"role": "assistant", "content": "ok"}]))
    return runtime


async def _run_rails(middleware: GuardrailsMiddleware, **overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "messages": [{"role": "user", "content": "hi"}],
        "rail_types": ["input"],
        "user_log_options": None,
        "error_msg": "failed",
    }
    request_body = overrides.pop("request_body", {"model": "main-model"})
    kwargs.update(overrides)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("nemo_guardrails_plugin.middleware.platform_headers_context", lambda client: _NullContext())
        return await middleware._run_rails(MagicMock(request_nemo_client=None), MagicMock(), request_body, {}, **kwargs)


@pytest.mark.asyncio
async def test_supported_request_runs_on_iorails() -> None:
    runtime = _routing_runtime()

    response = await _run_rails(_middleware_with(runtime))

    assert response.response[0]["content"] == "ok"
    runtime.check.assert_awaited_once()


@pytest.mark.asyncio
async def test_log_options_do_not_move_a_request_off_iorails() -> None:
    """The caller's log options are ignored (and warned about), not a reason to switch engines."""
    runtime = _routing_runtime()
    options = {"llm_calls": True, "stats": True}

    await _run_rails(_middleware_with(runtime), user_log_options=options)

    runtime.check.assert_awaited_once()
    runtime.warn_ignored_log_options.assert_called_once_with(options)


def test_ignored_log_options_are_warned_once_each(caplog: pytest.LogCaptureFixture) -> None:
    runtime = IORailsRuntime()

    with caplog.at_level(logging.WARNING, logger=iorails_runtime.logger.name):
        runtime.warn_ignored_log_options({"llm_calls": True, "stats": False})
        runtime.warn_ignored_log_options({"llm_calls": True, "internal_events": True})
        runtime.warn_ignored_log_options(None)

    first, second = (record.getMessage() for record in caplog.records)
    assert first.endswith("llm_calls")
    assert second.endswith("internal_events")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("iorails_on", "unsupported_reason", "overrides"),
    [
        pytest.param(False, None, {}, id="iorails-off"),
        pytest.param(True, "rails outside the IORails-supported set", {}, id="unsupported-config"),
        pytest.param(True, None, {"rail_types": ["input", "output"]}, id="several-rail-families"),
    ],
)
async def test_request_stays_on_llmrails(
    iorails_on: bool, unsupported_reason: str | None, overrides: dict[str, Any]
) -> None:
    runtime = _routing_runtime(unsupported_reason=unsupported_reason) if iorails_on else None

    with pytest.raises(_FellBackToLLMRails):
        await _run_rails(_middleware_with(runtime), **overrides)

    if runtime is not None:
        runtime.check.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("request_body", [{}, {"model": 7}])
async def test_request_without_model_is_rejected_on_iorails_like_on_llmrails(request_body: dict[str, Any]) -> None:
    """Engine choice must not change which requests are accepted."""
    runtime = _routing_runtime()

    with pytest.raises(InferenceMiddlewareError) as exc_info:
        await _run_rails(_middleware_with(runtime), request_body=request_body)

    assert exc_info.value.status_code == 400
    runtime.check.assert_not_awaited()


@pytest.mark.asyncio
async def test_iorails_failure_maps_to_503() -> None:
    runtime = _routing_runtime()
    runtime.check.side_effect = RuntimeError("upstream exploded")

    with pytest.raises(InferenceMiddlewareUnavailableError, match="failed"):
        await _run_rails(_middleware_with(runtime))
