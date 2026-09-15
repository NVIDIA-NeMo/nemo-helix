# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference Gateway proxy endpoints: prepared-request shape and on-the-wire path encoding."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from nemo_platform_plugin.client.client import NemoClient
from nemo_platform_plugin.client.types import BinaryContent, PreparedRequest
from nemo_platform_plugin.inference_gateway import endpoints
from nemo_platform_plugin.inference_gateway.client import InferenceGatewayClient
from nemo_platform_plugin.inference_gateway.types import JsonBody, OpenAIModel, OpenAIModelList, ProviderReadyResponse

BASE = "/apis/inference-gateway/v2/workspaces/{workspace}"


def _json_body(prepared: PreparedRequest) -> dict:
    assert isinstance(prepared.content, bytes)
    return json.loads(prepared.content)


# ---------------------------------------------------------------------------
# prepared requests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("endpoint", "http_method", "surface"),
    [
        (endpoints.provider_get, "GET", "provider"),
        (endpoints.provider_post, "POST", "provider"),
        (endpoints.provider_put, "PUT", "provider"),
        (endpoints.provider_patch, "PATCH", "provider"),
        (endpoints.provider_delete, "DELETE", "provider"),
        (endpoints.model_get, "GET", "model"),
        (endpoints.model_post, "POST", "model"),
        (endpoints.model_put, "PUT", "model"),
        (endpoints.model_patch, "PATCH", "model"),
        (endpoints.model_delete, "DELETE", "model"),
    ],
)
def test_named_proxy_routes(endpoint: Any, http_method: str, surface: str) -> None:
    kwargs: dict[str, Any] = {"workspace": "ws", "name": "nvidia-build", "trailing_uri": "v1/chat/completions"}
    if http_method in {"POST", "PUT", "PATCH"}:
        kwargs["body"] = JsonBody({"model": "m"})

    prepared = endpoint(**kwargs)

    assert prepared.method == http_method
    assert prepared.path_template == f"{BASE}/{surface}/{{name}}/-/{{trailing_uri}}"
    assert prepared.path_params == {"workspace": "ws", "name": "nvidia-build", "trailing_uri": "v1/chat/completions"}
    if "body" in kwargs:
        assert _json_body(prepared) == {"model": "m"}
        assert prepared.content_type == "application/json"
    else:
        assert prepared.content is None


@pytest.mark.parametrize(
    ("endpoint", "http_method"),
    [(endpoints.openai_get, "GET"), (endpoints.openai_post, "POST")],
)
def test_openai_proxy_routes(endpoint: Any, http_method: str) -> None:
    kwargs: dict[str, Any] = {"workspace": "ws", "trailing_uri": "v1/models"}
    if http_method == "POST":
        kwargs["body"] = JsonBody({"model": "default/vm", "messages": []})

    prepared = endpoint(**kwargs)

    assert prepared.method == http_method
    assert prepared.path_template == f"{BASE}/openai/-/{{trailing_uri}}"
    assert prepared.path_params == {"workspace": "ws", "trailing_uri": "v1/models"}
    assert prepared.response_type is Any


@pytest.mark.parametrize("endpoint", [endpoints.stream_provider, endpoints.stream_model])
def test_named_stream_routes_return_raw_bytes(endpoint: Any) -> None:
    prepared = endpoint(name="n", trailing_uri="v1/chat/completions", body=JsonBody({"stream": True}))

    assert prepared.method == "POST"
    assert prepared.response_type is BinaryContent
    assert _json_body(prepared) == {"stream": True}


def test_stream_openai_returns_raw_bytes() -> None:
    prepared = endpoints.stream_openai(trailing_uri="v1/chat/completions", body=JsonBody({"stream": True}))

    assert prepared.method == "POST"
    assert prepared.path_template == f"{BASE}/openai/-/{{trailing_uri}}"
    assert prepared.response_type is BinaryContent


def test_provider_ready() -> None:
    prepared = endpoints.provider_ready(name="nvidia-build")

    assert prepared.method == "GET"
    assert prepared.path_template == f"{BASE}/provider/{{name}}/ready"
    assert prepared.path_params == {"name": "nvidia-build"}
    assert prepared.response_type is ProviderReadyResponse


def test_openai_model_listing_routes() -> None:
    listing = endpoints.list_openai_models(workspace="ws")
    single = endpoints.get_openai_model(workspace="ws", name="default/vm")

    assert listing.path_template == f"{BASE}/openai/-/v1/models"
    assert listing.response_type is OpenAIModelList
    assert single.path_template == f"{BASE}/openai/-/v1/models/{{name}}"
    assert single.path_params == {"workspace": "ws", "name": "default/vm"}
    assert single.response_type is OpenAIModel


def test_workspace_defaults_to_the_client_workspace() -> None:
    prepared = endpoints.openai_get(trailing_uri="v1/models")

    assert "workspace" not in prepared.path_params


def test_json_body_serializes_arbitrary_objects() -> None:
    prepared = endpoints.openai_post(trailing_uri="v1/embeddings", body=JsonBody({"input": ["a", "b"], "n": 1}))

    assert _json_body(prepared) == {"input": ["a", "b"], "n": 1}


# ---------------------------------------------------------------------------
# on the wire
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self, response: httpx.Response | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.response = response or httpx.Response(200, json={"ok": True})

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.response


def _client(recorder: _Recorder) -> InferenceGatewayClient:
    return InferenceGatewayClient(
        base_url="http://test",
        workspace="default",
        http_client=httpx.Client(transport=httpx.MockTransport(recorder)),
    )


def test_trailing_uri_slashes_are_percent_encoded_on_the_wire() -> None:
    """The gateway declares ``{trailing_uri:path}`` and decodes before routing.

    Every path parameter is encoded with ``quote(safe="")``, so the caller-supplied
    provider path travels as one segment. This pins the wire form the server is
    known to accept; ``request.url.path`` would decode it and pass vacuously.
    """
    recorder = _Recorder()

    _client(recorder).openai_post(trailing_uri="v1/chat/completions", body=JsonBody({"model": "m"}))

    assert recorder.requests[0].url.raw_path == (
        b"/apis/inference-gateway/v2/workspaces/default/openai/-/v1%2Fchat%2Fcompletions"
    )


def test_named_route_encodes_both_name_and_trailing_uri() -> None:
    recorder = _Recorder()

    _client(recorder).provider_get(name="my provider", trailing_uri="v1/models")

    assert recorder.requests[0].method == "GET"
    assert recorder.requests[0].url.raw_path == (
        b"/apis/inference-gateway/v2/workspaces/default/provider/my%20provider/-/v1%2Fmodels"
    )


def test_untyped_proxy_response_is_returned_as_parsed_json() -> None:
    recorder = _Recorder(httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]}))

    data = _client(recorder).openai_post(trailing_uri="v1/chat/completions", body=JsonBody({"model": "m"})).data()

    assert data == {"choices": [{"message": {"content": "hi"}}]}


def test_stream_openai_yields_the_raw_sse_bytes() -> None:
    sse = b'data: {"choices":[{"delta":{"content":"h"}}]}\n\ndata: [DONE]\n\n'
    recorder = _Recorder(
        httpx.Response(200, stream=httpx.ByteStream(sse), headers={"content-type": "text/event-stream"})
    )

    response = _client(recorder).stream_openai(trailing_uri="v1/chat/completions", body=JsonBody({"stream": True}))
    with response.stream() as chunks:
        assert b"".join(chunks) == sse

    assert recorder.requests[0].headers["content-type"] == "application/json"
    assert json.loads(recorder.requests[0].content) == {"stream": True}


def test_list_openai_models_is_typed_and_keeps_extra_fields() -> None:
    recorder = _Recorder(
        httpx.Response(200, json={"object": "list", "data": [{"id": "default/vm", "object": "model", "extra": 1}]})
    )

    models = _client(recorder).list_openai_models().data()

    assert isinstance(models, OpenAIModelList)
    assert models.data[0].id == "default/vm"
    assert models.data[0].model_dump()["extra"] == 1


def test_from_client_shares_transport_and_workspace() -> None:
    recorder = _Recorder(httpx.Response(200, json={"ready": True}))
    base = NemoClient(
        base_url="http://test", workspace="ws", http_client=httpx.Client(transport=httpx.MockTransport(recorder))
    )

    InferenceGatewayClient.from_client(base).provider_ready(name="p")

    assert recorder.requests[0].url.path == "/apis/inference-gateway/v2/workspaces/ws/provider/p/ready"
