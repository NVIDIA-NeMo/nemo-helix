# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral checks for inference gateway request metrics."""

import asyncio
import gzip
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from unittest.mock import Mock, create_autospec

import aiohttp
import httpx
import pytest
from fastapi import FastAPI
from multidict import CIMultiDict, CIMultiDictProxy
from nhx.core.inference_gateway.api import metrics, request_metrics_middleware
from nhx.core.inference_gateway.api.request_metrics_middleware import InferenceRequestMetricsMiddleware
from nhx.core.inference_gateway.service import InferenceGatewayService
from opentelemetry.metrics import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import ClientDisconnect, Request
from starlette.responses import Response, StreamingResponse
from starlette.types import Message

WORKSPACE_PATH = "/v2/workspaces/e2e-test"
OPENAI_PATH = f"{WORKSPACE_PATH}/openai/-"
CHAT_PATH = f"{OPENAI_PATH}/v1/chat/completions"
MODEL = "meta_llama-3.2-1b-instruct"
CHAT_MESSAGES = [{"role": "user", "content": "hello"}]
FIRST_CHAT_OUTPUT = {"choices": [{"delta": {"content": "first"}}]}


@dataclass
class ObservedMetrics:
    requests: Mock
    first_output: Mock
    clock: Mock

    def assert_request(self, protocol: str, streaming: bool, outcome: str, first_output_s: float | None = None):
        attributes = {"protocol": protocol, "streaming": streaming, "outcome": outcome}
        self.requests.add.assert_called_once_with(1, attributes=attributes)
        if first_output_s is None:
            self.first_output.record.assert_not_called()
        else:
            self.first_output.record.assert_called_once_with(first_output_s, attributes=attributes)


@pytest.fixture
def observed_metrics(monkeypatch: pytest.MonkeyPatch) -> ObservedMetrics:
    observed = ObservedMetrics(
        requests=create_autospec(Counter, instance=True),
        first_output=create_autospec(Histogram, instance=True),
        clock=Mock(return_value=100.0),
    )
    monkeypatch.setattr(metrics, "requests_counter", observed.requests)
    monkeypatch.setattr(metrics, "time_to_first_useful_output", observed.first_output)
    monkeypatch.setattr(request_metrics_middleware, "monotonic", observed.clock)
    return observed


def sse_frame(payload: dict, *, named: bool = False) -> bytes:
    event = f"event: {payload['type']}\n" if named else ""
    return f"{event}data: {json.dumps(payload)}\n\n".encode()


def set_upstream_sse(mock_proxy_response, body: AsyncIterator[bytes]):
    mock_proxy_response.headers = CIMultiDictProxy(CIMultiDict({"content-type": "text/event-stream"}))
    mock_proxy_response.content.iter_any = Mock(return_value=body)


@pytest.mark.parametrize(
    ("path", "protocol", "body", "frames", "named"),
    [
        pytest.param(
            "v1/chat/completions",
            "openai_chat",
            {"messages": CHAT_MESSAGES},
            [
                {"choices": [{"delta": {"role": "assistant", "content": ""}}]},
                FIRST_CHAT_OUTPUT,
                {"choices": [{"delta": {"content": " second"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ],
            False,
            id="chat-completions",
        ),
        pytest.param(
            "v1/completions",
            "openai_completions",
            {"prompt": "hello"},
            [
                {"choices": [{"text": "", "finish_reason": None}]},
                {"choices": [{"text": "first", "finish_reason": None}]},
                {"choices": [{"text": " second", "finish_reason": None}]},
                {"choices": [{"text": "", "finish_reason": "stop"}]},
            ],
            False,
            id="completions",
        ),
        pytest.param(
            "v1/responses",
            "openai_responses",
            {"input": "hello"},
            [
                {"type": "response.created", "response": {"status": "in_progress"}},
                {"type": "response.output_text.delta", "delta": "first"},
                {"type": "response.output_text.delta", "delta": " second"},
                {"type": "response.completed", "response": {"status": "completed"}},
            ],
            True,
            id="responses",
        ),
        pytest.param(
            "v1/messages",
            "anthropic_messages",
            {"messages": CHAT_MESSAGES, "max_tokens": 16},
            [
                {"type": "ping"},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "first"}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " second"}},
                {"type": "message_stop"},
            ],
            True,
            id="anthropic-messages",
        ),
    ],
)
def test_real_gateway_times_first_useful_stream_output_once_at_completion(
    observed_metrics, client, mock_proxy_response, path, protocol, body, frames, named
):
    async def upstream():
        for elapsed, payload in enumerate(frames, 1):
            observed_metrics.requests.add.assert_not_called()
            observed_metrics.first_output.record.assert_not_called()
            observed_metrics.clock.return_value = 100.0 + elapsed
            frame = sse_frame(payload, named=named)
            midpoint = len(frame) // 2
            yield frame[:midpoint]
            yield frame[midpoint:]
        if not named:
            yield b"data: [DONE]\n\n"

    set_upstream_sse(mock_proxy_response, upstream())
    response = client.post(f"{OPENAI_PATH}/{path}", json={"model": MODEL, "stream": True, **body})

    assert response.status_code == 200
    wire = b"".join(sse_frame(payload, named=named) for payload in frames)
    if not named:
        wire += b"data: [DONE]\n\n"
    assert response.content == wire
    observed_metrics.assert_request(protocol, True, "success", first_output_s=2.0)


@pytest.mark.parametrize("failure", ["provider-error-event", "broken-upstream-body"])
async def test_real_gateway_stream_failure_does_not_count_as_http_200_success(
    observed_metrics, app, mock_proxy_response, failure
):
    async def upstream():
        observed_metrics.clock.return_value = 103.0
        yield sse_frame(FIRST_CHAT_OUTPUT)
        observed_metrics.clock.return_value = 107.0
        if failure == "broken-upstream-body":
            raise aiohttp.ClientPayloadError("upstream body ended early")
        yield sse_frame({"error": {"message": "provider failed after first output"}})
        yield b"data: [DONE]\n\n"

    set_upstream_sse(mock_proxy_response, upstream())
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(CHAT_PATH, json={"model": MODEL, "stream": True, "messages": CHAT_MESSAGES})

    assert response.status_code == 200
    assert response.content.startswith(sse_frame(FIRST_CHAT_OUTPUT))
    observed_metrics.assert_request("openai_chat", True, "stream_error", first_output_s=3.0)


@pytest.mark.parametrize(
    ("frames", "outcome", "first_output_s"),
    [
        pytest.param([], "incomplete", None, id="empty-stream-with-generated-done"),
        pytest.param([FIRST_CHAT_OUTPUT], "incomplete", 2.0, id="data-then-eof-with-generated-done"),
        pytest.param(
            [{"choices": [{"delta": {}, "finish_reason": "stop"}]}],
            "success",
            None,
            id="finished-without-useful-output",
        ),
        pytest.param(
            [{"choices": [{"delta": {}, "finish_reason": "content_filter"}]}],
            "policy_blocked",
            None,
            id="content-filter",
        ),
    ],
)
def test_real_gateway_uses_finish_evidence_and_never_times_a_control_frame(
    observed_metrics, client, mock_proxy_response, frames, outcome, first_output_s
):
    async def upstream():
        observed_metrics.clock.return_value = 102.0
        for payload in frames:
            yield sse_frame(payload)

    set_upstream_sse(mock_proxy_response, upstream())
    response = client.post(CHAT_PATH, json={"model": MODEL, "stream": True, "messages": CHAT_MESSAGES})

    assert response.status_code == 200
    assert response.content.endswith(b"data: [DONE]\n\n")
    observed_metrics.assert_request("openai_chat", True, outcome, first_output_s)


def test_real_gateway_does_not_measure_the_openai_model_catalog(observed_metrics, client):
    model_list = client.get(f"{OPENAI_PATH}/v1/models")
    assert model_list.status_code == 200
    assert model_list.json()["object"] == "list"

    model = client.get(f"{OPENAI_PATH}/v1/models/{MODEL}")
    assert model.status_code == 200
    assert model.json()["object"] == "model"

    observed_metrics.requests.add.assert_not_called()
    observed_metrics.first_output.record.assert_not_called()


@pytest.mark.parametrize(
    ("path", "protocol", "request_body", "reply", "outcome"),
    [
        pytest.param(
            "chat/completions",
            "openai_chat",
            {"messages": CHAT_MESSAGES},
            {"error": {"type": "guardrails_violation", "message": "output blocked"}},
            "policy_blocked",
            id="guardrails-violation",
        ),
        pytest.param(
            "responses",
            "openai_responses",
            {"input": "hello"},
            {"object": "response", "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
            "incomplete",
            id="incomplete-response",
        ),
    ],
)
def test_real_gateway_uses_unary_response_outcome_when_http_is_200(
    observed_metrics, client, mock_proxy_response, path, protocol, request_body, reply, outcome
):
    mock_proxy_response._body = [json.dumps(reply).encode()]
    response = client.post(f"{OPENAI_PATH}/v1/{path}", json={"model": MODEL, **request_body})

    assert response.status_code == 200
    assert response.json() == reply
    observed_metrics.assert_request(protocol, False, outcome)


@pytest.mark.parametrize(
    ("content_encoding", "outcome", "first_output_s"),
    [("identity", "success", 2.0), ("gzip", "unobserved", None)],
)
async def test_provider_passthrough_only_observes_readable_sse(
    observed_metrics, content_encoding, outcome, first_output_s
):
    output = sse_frame({"type": "response.output_text.delta", "delta": "first"}, named=True).replace(b"\n", b"\r")
    completed = sse_frame({"type": "response.completed"}, named=True).replace(b"\n", b"\r")
    body = output + completed

    async def upstream():
        observed_metrics.clock.return_value = 101.0
        if content_encoding == "identity":
            yield output[:-1]
            observed_metrics.clock.return_value = 102.0
            yield output[-1:] + completed
        else:
            yield gzip.compress(body)

    response = StreamingResponse(
        upstream(), media_type="text/event-stream", headers={"content-encoding": content_encoding}
    )
    transport = httpx.ASGITransport(app=InferenceRequestMetricsMiddleware(response))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        result = await client.get(f"{WORKSPACE_PATH}/provider/private-provider/-/v1/responses/private-response")

    assert result.status_code == 200
    assert result.headers["content-encoding"] == content_encoding
    assert result.content == body
    observed_metrics.assert_request("openai_responses", True, outcome, first_output_s)


async def test_provider_passthrough_does_not_count_raw_done_as_completion(observed_metrics):
    response = Response(b"data: [DONE]\n\n", media_type="text/event-stream")
    transport = httpx.ASGITransport(app=InferenceRequestMetricsMiddleware(response))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        result = await client.post(f"{WORKSPACE_PATH}/provider/private-provider/-/v1/chat/completions")

    assert result.status_code == 200
    assert result.content == b"data: [DONE]\n\n"
    observed_metrics.assert_request("openai_chat", True, "incomplete")


@pytest.mark.parametrize("asgi_version", ["2.3", "2.4"])
async def test_stream_disconnect_records_one_cancelled_outcome(observed_metrics, asgi_version):
    first_frame_sent = asyncio.Event()

    async def upstream():
        observed_metrics.clock.return_value = 102.0
        yield sse_frame(FIRST_CHAT_OUTPUT)
        if asgi_version == "2.3":
            await asyncio.Future()
        yield sse_frame({"choices": [{"delta": {"content": "never sent"}}]})

    response = StreamingResponse(upstream(), media_type="text/event-stream")
    app = InferenceRequestMetricsMiddleware(response)
    scope = {"type": "http", "method": "POST", "path": CHAT_PATH, "asgi": {"spec_version": asgi_version}}

    async def receive() -> Message:
        await first_frame_sent.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        if message["type"] == "http.response.start":
            assert message["status"] == 200
        elif first_frame_sent.is_set():
            raise OSError("client disconnected")
        else:
            assert message["body"] == sse_frame(FIRST_CHAT_OUTPUT)
            first_frame_sent.set()

    async with asyncio.timeout(2):
        if asgi_version == "2.4":
            with pytest.raises(ClientDisconnect):
                await app(scope, receive, send)
        else:
            await app(scope, receive, send)

    assert first_frame_sent.is_set()
    observed_metrics.assert_request("openai_chat", True, "cancelled", first_output_s=2.0)


async def test_relay_cannot_complete_a_stream_after_the_caller_disconnects(observed_metrics):
    first_frame_sent = asyncio.Event()
    caller_disconnected = asyncio.Event()
    delivered: list[Message] = []
    first_frame = sse_frame(FIRST_CHAT_OUTPUT)

    async def upstream():
        yield first_frame
        await asyncio.Future()

    async def passthrough(request: Request, call_next: RequestResponseEndpoint) -> Response:
        return await call_next(request)

    response = StreamingResponse(upstream(), media_type="text/event-stream")
    app = InferenceRequestMetricsMiddleware(BaseHTTPMiddleware(response, dispatch=passthrough))
    scope = {"type": "http", "method": "POST", "path": CHAT_PATH, "asgi": {"spec_version": "2.3"}}

    async def receive() -> Message:
        await first_frame_sent.wait()
        caller_disconnected.set()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        # Uvicorn's ASGI 2.3 send returns without writing after a caller close.
        if caller_disconnected.is_set():
            return
        delivered.append(message)
        if message["type"] == "http.response.body":
            observed_metrics.clock.return_value = 102.0
            first_frame_sent.set()

    async with asyncio.timeout(2):
        await app(scope, receive, send)

    assert delivered[0]["status"] == 200
    assert delivered[1:] == [{"type": "http.response.body", "body": first_frame, "more_body": True}]
    observed_metrics.assert_request("openai_chat", True, "cancelled", first_output_s=2.0)


@pytest.mark.parametrize(
    ("status", "outcome"),
    [(200, "success"), (307, "redirect"), (403, "client_error"), (424, "server_error"), (429, "rate_limited")],
)
async def test_prefixed_gateway_proxy_routes_use_bounded_attributes_and_skip_other_services(
    observed_metrics, status, outcome
):
    prefix = "/apis/inference-gateway"
    app = FastAPI()
    InferenceGatewayService().configure_app(app)

    @app.post("/{path:path}")
    async def downstream() -> Response:
        observed_metrics.clock.return_value = 104.0
        return Response(b"provider reply", status_code=status)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for unrelated in (
            CHAT_PATH,
            f"/apis/another-service{CHAT_PATH}",
            f"/apis/inference-gateway-other{CHAT_PATH}",
            f"{prefix}/health/ready",
        ):
            assert (await client.post(unrelated)).status_code == status
        observed_metrics.clock.return_value = 100.0
        assert (
            await client.post(
                f"{prefix}{WORKSPACE_PATH}/provider/private-provider/-/custom/private-operation",
                params={"private": "not-a-metric-attribute"},
            )
        ).status_code == status

    observed_metrics.assert_request("other", False, outcome, first_output_s=4.0 if status == 200 else None)
