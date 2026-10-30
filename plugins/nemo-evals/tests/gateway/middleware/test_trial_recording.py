# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each call made with a trial token is recorded in that trial, without touching the call."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from importlib.metadata import entry_points
from typing import Any
from unittest.mock import MagicMock

import pytest
from nemo_evals.gateway import tokens as tokens_module
from nemo_evals.gateway.middleware import trial_recording as trial_recording_module
from nemo_evals.gateway.middleware.trial_recording import (
    MIDDLEWARE_NAME,
    TrialRecordingConfig,
    TrialRecordingMiddleware,
)
from nemo_evals.gateway.policy import ModelAccess
from nemo_evals.gateway.recorder import CallRecorder
from nemo_evals.gateway.spans import (
    GATEWAY_CALL_ATTRIBUTE,
    OPENINFERENCE_SPAN_KIND,
    trial_root_span_id,
    trial_trace_id,
)
from nemo_evals.gateway.tokens import TrialClaims, TrialTokenReader, unsigned_token
from nemo_evals.intake.mapping import SESSION_ID_ATTRIBUTE, TEST_CASE_NAME_ATTRIBUTE
from nemo_helix_plugin.client.oidc import decode_jwt_claims
from nemo_helix_plugin.inference_middleware import (
    InferenceMiddlewareCacheAccessor,
    InferenceMiddlewareContext,
    InferenceRequest,
    InferenceResponse,
    ModelProviderInferenceTarget,
    NemoInferenceMiddleware,
)
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, Span

NOW = datetime.now(UTC)
TRIAL = TrialClaims(
    session="run-1:hello__abc",
    intake_workspace="evals",
    evaluation="run-1",
    task="hello",
    model_access=ModelAccess.OPEN,
)
TOKEN = unsigned_token(TRIAL)


def _value(value: Any) -> Any:
    return getattr(value, value.WhichOneof("value"))


def _attributes(span: Span) -> dict[str, Any]:
    return {attribute.key: _value(attribute.value) for attribute in span.attributes}


def _spans(batches: list[tuple[str, list[ResourceSpans]]]) -> list[tuple[str, Span]]:
    return [
        (workspace, span)
        for workspace, batch in batches
        for resource in batch
        for scope in resource.scope_spans
        for span in scope.spans
    ]


class Signed:
    """Stands in for the platform's signature check: trusts the claims of exactly the tokens it was given."""

    def __init__(self, *tokens: str) -> None:
        self._tokens = set(tokens)
        self.asked = 0

    async def __call__(self, token: str) -> Mapping[str, Any] | None:
        self.asked += 1
        return decode_jwt_claims(token) if token in self._tokens else None


class Sink:
    def __init__(self) -> None:
        self.sent: list[tuple[str, list[ResourceSpans]]] = []

    async def __call__(self, workspace: str, spans: list[ResourceSpans]) -> int:
        self.sent.append((workspace, spans))
        return 0


def _reader(*tokens: str) -> TrialTokenReader:
    return TrialTokenReader(Signed(*tokens))


def _middleware(reader: TrialTokenReader, sink: Sink, *, routed: bool = False) -> TrialRecordingMiddleware:
    """The middleware, with every model served by a provider that calls models back as its caller when ``routed``."""
    middleware = TrialRecordingMiddleware()
    middleware._recorder = CallRecorder(reader, sink, flush_interval_sec=60)
    cache = MagicMock(spec=InferenceMiddlewareCacheAccessor)
    cache.get_inference_url_and_model.return_value = ModelProviderInferenceTarget(
        model_provider_gateway_url="http://igw/provider", served_model_name="route", forwards_caller_credential=routed
    )
    middleware._inject_cache(cache)
    return middleware


def _ctx(headers: dict[str, str], path: str = "v1/chat/completions") -> InferenceMiddlewareContext:
    body = {"model": "default/agent-model", "messages": [{"role": "user", "content": "Hello"}]}
    return InferenceMiddlewareContext(
        request_id="req-1",
        virtual_model_name="agent-model",
        workspace="models",
        original_request=InferenceRequest(body=body, headers=headers, path=path),
        proxied_request=InferenceRequest(body={**body, "model": "meta/llama-3.3-70b"}, headers=headers, path=path),
    )


async def _call(middleware: TrialRecordingMiddleware, ctx: InferenceMiddlewareContext, result: Any) -> Any:
    config = TrialRecordingConfig()
    await middleware.process_request(ctx, ctx.original_request, config)
    response = await middleware.process_response(ctx, InferenceResponse(result=result, headers={}), config)
    return response.result


async def _stream(*chunks: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
    for chunk in chunks:
        yield chunk


def test_the_middleware_is_registered_for_the_gateway() -> None:
    entries = [e for e in entry_points(group="nemo.inference_middleware") if e.name == MIDDLEWARE_NAME]

    assert entries
    assert all(entry.load() is TrialRecordingMiddleware for entry in entries)


def test_the_recording_middleware_leaves_access_to_the_access_check() -> None:
    assert TrialRecordingMiddleware.check_access is NemoInferenceMiddleware.check_access


@pytest.mark.parametrize("auth_enabled", [True, False], ids=["auth on", "auth off"])
async def test_the_gateway_trusts_signed_tokens_with_auth_on_and_unsigned_ones_with_it_off(
    monkeypatch: pytest.MonkeyPatch, auth_enabled: bool
) -> None:
    """With auth on, an unsigned token naming a trial is ignored, even when the gateway accepted it."""
    monkeypatch.setattr(tokens_module, "platform_auth_enabled", lambda: auth_enabled)

    async def nothing_is_signed(token: str) -> None:
        return None

    monkeypatch.setattr(tokens_module, "verified_access_key_claims", nothing_is_signed)
    middleware = TrialRecordingMiddleware()
    monkeypatch.setattr(middleware, "_get_platform_client", lambda hook: object())
    monkeypatch.setattr(trial_recording_module, "intake_sink", lambda client: Sink())
    await middleware.on_startup()
    try:
        assert middleware._recorder is not None
        assert (await middleware._recorder._trials.trial(TOKEN) == TRIAL) is not auth_enabled
    finally:
        await middleware.on_shutdown()


async def test_a_call_with_a_token_naming_no_trial_is_never_checked() -> None:
    """Most callers aren't trials, so only tokens that claim a trial reach the signature check."""
    verify, sink = Signed(TOKEN), Sink()
    middleware = _middleware(TrialTokenReader(verify), sink)

    await _call(middleware, _ctx({"authorization": "Bearer users-own-key"}), {"usage": {"prompt_tokens": 1}})
    assert middleware._recorder is not None
    await middleware._recorder.flush()

    assert verify.asked == 0
    assert sink.sent == []


async def test_a_call_made_with_a_trial_token_is_recorded_in_the_trial() -> None:
    sink = Sink()
    middleware = _middleware(_reader(TOKEN), sink)
    body = {"choices": [], "usage": {"prompt_tokens": 120, "completion_tokens": 5}}

    assert await _call(middleware, _ctx({"authorization": f"Bearer {TOKEN}"}), body) is body
    assert middleware._recorder is not None
    await middleware._recorder.flush()

    [(workspace, span)] = _spans(sink.sent)
    assert workspace == "evals"
    assert span.trace_id.hex() == trial_trace_id(TRIAL)
    assert span.parent_span_id.hex() == trial_root_span_id(TRIAL)
    assert span.name == "chat.completions"
    attributes = _attributes(span)
    assert attributes[SESSION_ID_ATTRIBUTE] == TRIAL.session
    assert attributes[TEST_CASE_NAME_ATTRIBUTE] == "hello"
    assert attributes[GATEWAY_CALL_ATTRIBUTE] is True
    assert attributes["gen_ai.usage.input_tokens"] == 120
    assert attributes["gen_ai.usage.output_tokens"] == 5
    assert attributes["gen_ai.request.model"] == "meta/llama-3.3-70b"
    assert attributes["nemo.virtual_model"] == "models/agent-model"


CALLER_TRACE = "01a182ecbd457d118e02830498804285"
CALLER_SPAN = "9fe2c6bed0c1c48c"


async def test_a_call_that_sends_a_traceparent_is_recorded_under_the_callers_span() -> None:
    """An agent tracing its own model calls gets the gateway's span in its trace, as that call's child."""
    sink = Sink()
    middleware = _middleware(_reader(TOKEN), sink)
    headers = {
        "authorization": f"Bearer {TOKEN}",
        "traceparent": f"00-{CALLER_TRACE}-{CALLER_SPAN}-01",
        "tracestate": "relay=1",
    }

    await _call(middleware, _ctx(headers), {"usage": {"prompt_tokens": 1}})
    assert middleware._recorder is not None
    await middleware._recorder.flush()

    [(workspace, span)] = _spans(sink.sent)
    assert (span.trace_id.hex(), span.parent_span_id.hex(), span.trace_state) == (CALLER_TRACE, CALLER_SPAN, "relay=1")
    assert workspace == "evals"
    assert _attributes(span)[SESSION_ID_ATTRIBUTE] == TRIAL.session


@pytest.mark.parametrize(
    "traceparent",
    [
        pytest.param(f"00-{CALLER_TRACE}-{CALLER_SPAN}", id="missing flags"),
        pytest.param(f"00-{'0' * 32}-{CALLER_SPAN}-01", id="zero trace id"),
        pytest.param(f"00-{CALLER_TRACE}-{'0' * 16}-01", id="zero span id"),
        pytest.param(f"ff-{CALLER_TRACE}-{CALLER_SPAN}-01", id="invalid version"),
        pytest.param(f"00-{CALLER_TRACE}-{CALLER_SPAN}-01-extra", id="version 00 with extra fields"),
        pytest.param(f"00-{CALLER_TRACE.upper()}-{CALLER_SPAN}-01", id="uppercase"),
        pytest.param("not-a-traceparent", id="garbage"),
    ],
)
async def test_a_call_with_an_invalid_traceparent_is_recorded_in_the_trial_trace(traceparent: str) -> None:
    sink = Sink()
    middleware = _middleware(_reader(TOKEN), sink)

    await _call(middleware, _ctx({"authorization": f"Bearer {TOKEN}", "traceparent": traceparent}), {})
    assert middleware._recorder is not None
    await middleware._recorder.flush()

    [(_, span)] = _spans(sink.sent)
    assert (span.trace_id.hex(), span.parent_span_id.hex()) == (trial_trace_id(TRIAL), trial_root_span_id(TRIAL))


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param(
            {"authorization": f"Bearer {unsigned_token(TRIAL.model_copy(update={'task': 'other'}))}"},
            id="a trial token the platform didn't sign",
        ),
        pytest.param({}, id="no credential"),
    ],
)
async def test_a_call_without_a_trusted_trial_token_is_not_recorded(headers: dict[str, str]) -> None:
    sink = Sink()
    middleware = _middleware(_reader(TOKEN), sink)

    await _call(middleware, _ctx(headers), {"usage": {"prompt_tokens": 1}})
    assert middleware._recorder is not None
    await middleware._recorder.flush()

    assert sink.sent == []


async def test_a_streamed_call_is_recorded_with_its_usage_once_the_caller_has_it_all() -> None:
    sink = Sink()
    middleware = _middleware(_reader(TOKEN), sink)
    chunks = [
        {
            "type": "message_start",
            "message": {"model": "routed-target", "usage": {"input_tokens": 10, "cache_read_input_tokens": 30}},
        },
        {"type": "message_delta", "usage": {"output_tokens": 15}},
    ]

    stream = await _call(middleware, _ctx({"x-api-key": TOKEN}, path="v1/messages"), _stream(*chunks))
    assert middleware._recorder is not None
    assert middleware._recorder._queue.empty()
    assert [chunk async for chunk in stream] == chunks
    await middleware._recorder.flush()

    [(_, span)] = _spans(sink.sent)
    attributes = _attributes(span)
    assert span.name == "messages"
    assert attributes["gen_ai.usage.input_tokens"] == 40
    assert attributes["gen_ai.usage.output_tokens"] == 15
    assert attributes["gen_ai.response.model"] == "routed-target"
    assert "status" not in attributes


async def test_a_caller_that_stops_reading_mid_stream_is_recorded_as_cancelled() -> None:
    sink = Sink()
    middleware = _middleware(_reader(TOKEN), sink)

    stream = await _call(middleware, _ctx({"authorization": f"Bearer {TOKEN}"}), _stream({"a": 1}, {"b": 2}))
    assert await anext(stream) == {"a": 1}
    await stream.aclose()
    assert middleware._recorder is not None
    await middleware._recorder.flush()

    [(_, span)] = _spans(sink.sent)
    assert _attributes(span)["status"] == "cancelled"


async def test_the_span_starts_when_the_gateway_received_the_call() -> None:
    sink = Sink()
    middleware = _middleware(_reader(TOKEN), sink)
    ctx = _ctx({"authorization": f"Bearer {TOKEN}"})
    config = TrialRecordingConfig()

    await middleware.process_request(ctx, ctx.original_request, config)
    received = ctx.state(MIDDLEWARE_NAME).get("started_at")
    await middleware.process_response(ctx, InferenceResponse(result={}, headers={}), config)
    assert middleware._recorder is not None
    await middleware._recorder.flush()

    [(_, span)] = _spans(sink.sent)
    assert span.start_time_unix_nano == int(received.timestamp() * 1_000_000_000)


async def test_a_call_to_a_router_that_calls_back_as_the_trial_is_a_routing_span_its_callbacks_nest_under() -> None:
    """The router's own calls carry the trial's usage; recording the routed call as a model call would count it twice."""
    sink = Sink()
    middleware = _middleware(_reader(TOKEN), sink, routed=True)
    ctx = _ctx({"authorization": f"Bearer {TOKEN}"})
    config = TrialRecordingConfig()
    original = ctx.original_request

    forwarded = await middleware.process_request(
        ctx, InferenceRequest(body=dict(original.body), headers=dict(original.headers), path=original.path), config
    )
    await middleware.process_response(
        ctx, InferenceResponse(result={"usage": {"prompt_tokens": 7}}, headers={}), config
    )
    assert middleware._recorder is not None
    await middleware._recorder.flush()

    [(_, span)] = _spans(sink.sent)
    attributes = _attributes(span)
    assert forwarded.headers["traceparent"] == f"00-{trial_trace_id(TRIAL)}-{span.span_id.hex()}-01"
    assert attributes[OPENINFERENCE_SPAN_KIND] == "CHAIN"
    assert GATEWAY_CALL_ATTRIBUTE not in attributes
    assert span.parent_span_id.hex() == trial_root_span_id(TRIAL)


async def test_a_call_to_an_ordinary_provider_keeps_the_callers_traceparent() -> None:
    sink = Sink()
    middleware = _middleware(_reader(TOKEN), sink)
    caller = "00-01a182ecbd457d118e02830498804285-9fe2c6bed0c1c48c-01"
    ctx = _ctx({"authorization": f"Bearer {TOKEN}", "traceparent": caller})

    forwarded = await middleware.process_request(ctx, ctx.original_request, TrialRecordingConfig())

    assert forwarded.headers["traceparent"] == caller


async def test_config_lists_the_virtual_models_a_router_routes_to() -> None:
    config = await TrialRecordingMiddleware().validate_middleware_config(
        "trial_recording", {"routes_to": ["ws/strong", "ws/judge"]}
    )

    assert config.routes_to == ("ws/strong", "ws/judge")


@pytest.mark.parametrize("ref", ["strong", "ws/", "ws/strong/extra"])
async def test_config_refuses_a_route_that_is_not_a_workspace_and_name(ref: str) -> None:
    with pytest.raises(ValueError, match="routes_to entry"):
        await TrialRecordingMiddleware().validate_middleware_config("trial_recording", {"routes_to": [ref]})
