# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""System One (decision model) requests are typed, parsed, and refused on non-System One routes."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, AsyncIterator, cast
from unittest.mock import Mock

import pytest
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from nemo_helix_plugin.inference_middleware import (
    BackendFormat,
    InferenceMiddlewareContext,
    InferenceRequest,
    InferenceResponse,
    NemoInferenceMiddleware,
)
from nemo_helix_plugin.inference_middleware_models import VirtualModel, VirtualModelInferenceConfig
from nemo_helix_plugin.inference_middleware_types import (
    SystemOneChoiceAnswer,
    SystemOneNoulAnswer,
    SystemOneOtherAnswer,
    SystemOneRequest,
    SystemOneResponse,
)
from nemo_helix_plugin.models.types import ModelProvider, ModelProviderStatus, ServedModelMapping
from nhx.core.inference_gateway.api.middleware_registry import MiddlewareRegistry, ResolvedMiddlewareCall
from nhx.core.inference_gateway.api.model_cache import ModelCache, ModelProviderInfo
from nhx.core.inference_gateway.api.proxy import virtual_model_proxy
from nhx.core.inference_gateway.api.typed_request import parse_typed_request
from nhx.core.inference_gateway.api.typed_response import TypedResponseStream, parse_typed_response

WORKSPACE = "default"
ENTITY = "nvidia-nvidia-nemotron-decide"
ENTITY_ID = f"{WORKSPACE}/{ENTITY}"


def _systemone_request() -> dict[str, Any]:
    return {
        "model": ENTITY_ID,
        "state": {"x": "y"},
        "questions": {
            "q": {"type": "noul", "instructions": "Is x y?"},
            "c": {"type": "choice", "instructions": "Sky color?", "criteria": {"red": "red", "blue": "blue"}},
        },
    }


def _systemone_response() -> dict[str, Any]:
    return {
        "model": "nemotron-decide",
        "answers": {
            "q": {"type": "noul", "noul": 0.98},
            "c": {
                "type": "choice",
                "choice": "blue",
                "confidence": 0.99,
                "probabilities": {"red": 0.01, "blue": 0.99},
            },
            "s": {"type": "score", "score": 7},
        },
        "usage": {"input_tokens": 121, "output_tokens": 1, "decision_positions": 1},
        "metrics": {"seconds": 0.017},
    }


@pytest.mark.parametrize("state", [{"x": "y"}, "x is y", ["x is y", "z"]])
def test_systemone_request_is_typed(state: object):
    body = _systemone_request()
    body["state"] = state

    typed = parse_typed_request("/v1/systemone", body)

    assert typed is not None
    typed = cast(SystemOneRequest, typed)
    assert typed["state"] == state
    assert typed["questions"]["c"]["type"] == "choice"


def test_systemone_request_without_questions_is_untyped():
    body = _systemone_request()
    del body["questions"]

    assert parse_typed_request("v1/systemone", body) is None


def test_systemone_response_is_typed_per_answer_kind():
    parsed = parse_typed_response(BackendFormat.SYSTEMONE, _systemone_response())

    assert isinstance(parsed, SystemOneResponse)
    assert isinstance(parsed.answers["q"], SystemOneNoulAnswer)
    assert parsed.answers["q"].noul == 0.98
    assert isinstance(parsed.answers["c"], SystemOneChoiceAnswer)
    assert parsed.answers["c"].choice == "blue"
    assert isinstance(parsed.answers["s"], SystemOneOtherAnswer)
    assert parsed.usage.input_tokens == 121
    assert parsed.usage.output_tokens == 1


def test_systemone_response_dump_is_lossless():
    """IGW serializes typed_body in place of the raw body, so the dump must match what the backend sent."""
    payload = _systemone_response()
    parsed = parse_typed_response(BackendFormat.SYSTEMONE, payload)

    assert parsed is not None
    assert parsed.model_dump(mode="json") == payload


def test_systemone_response_without_usage_is_untyped():
    payload = _systemone_response()
    del payload["usage"]

    assert parse_typed_response(BackendFormat.SYSTEMONE, payload) is None


@pytest.mark.asyncio
async def test_systemone_stream_has_no_typed_view_and_keeps_raw_chunks():
    async def _chunks() -> AsyncIterator[dict[str, Any]]:
        yield {"a": 1}

    stream = TypedResponseStream(BackendFormat.SYSTEMONE, _chunks())

    assert [chunk async for chunk in stream.typed_chunks()] == []
    assert [chunk async for chunk in stream.raw_chunks()] == [{"a": 1}]


def _model_cache() -> ModelCache:
    model_cache = ModelCache()
    model_cache.update_model_info(
        ModelProviderInfo(
            model_provider=ModelProvider(
                workspace=WORKSPACE,
                name="nvidia-inference",
                host_url="http://hub.local/v1",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                served_models=[
                    ServedModelMapping(model_entity_id=ENTITY_ID, served_model_name="nvidia/nvidia/nemotron-decide")
                ],
                status=ModelProviderStatus.READY,
            )
        )
    )
    model_cache.rebuild_model_entity_map()
    return model_cache


def _virtual_model(backend_format: BackendFormat | None) -> VirtualModel:
    return VirtualModel(
        name=ENTITY,
        workspace=WORKSPACE,
        default_model_entity=ENTITY_ID,
        models=[VirtualModelInferenceConfig(model=ENTITY_ID, backend_format=backend_format)],
    )


def _request() -> Request:
    request = Mock(spec=Request)
    request.method = "POST"
    request.headers = {"content-type": "application/json"}
    request.query_params = {}
    return request


@pytest.mark.asyncio
async def test_chat_request_to_systemone_model_is_refused_before_proxying(mock_proxy_client):
    with pytest.raises(HTTPException) as exc_info:
        await virtual_model_proxy(
            request=_request(),
            workspace=WORKSPACE,
            vm_name=ENTITY,
            virtual_model=_virtual_model(BackendFormat.SYSTEMONE),
            trailing_uri="v1/chat/completions",
            json_body={"model": ENTITY, "messages": [{"role": "user", "content": "hi"}]},
            http_client=mock_proxy_client,
            model_cache=_model_cache(),
            registry=MiddlewareRegistry(plugins={}),
        )

    assert exc_info.value.status_code == 400
    assert "serves only POST /v1/systemone" in str(exc_info.value.detail)
    mock_proxy_client.request.assert_not_called()


@pytest.mark.asyncio
async def test_systemone_request_rewritten_to_chat_by_middleware_is_proxied(mock_proxy_client):
    class _ToChat(NemoInferenceMiddleware):
        async def process_request(
            self, ctx: InferenceMiddlewareContext, request: InferenceRequest, middleware_config: object
        ) -> InferenceRequest:
            return InferenceRequest(body=request.body, headers=request.headers, path="v1/chat/completions")

    registry = MiddlewareRegistry(plugins={"to-chat": _ToChat()})
    registry.request_middleware_calls[(WORKSPACE, ENTITY)] = [
        ResolvedMiddlewareCall(plugin_name="to-chat", config_type="t", resolved_config={})
    ]

    await virtual_model_proxy(
        request=_request(),
        workspace=WORKSPACE,
        vm_name=ENTITY,
        virtual_model=_virtual_model(None),
        trailing_uri="v1/systemone",
        json_body=_systemone_request(),
        http_client=mock_proxy_client,
        model_cache=_model_cache(),
        registry=registry,
    )

    mock_proxy_client.request.assert_called_once()


@pytest.mark.parametrize("backend_format", [BackendFormat.SYSTEMONE, BackendFormat.OPENAI_CHAT, None])
@pytest.mark.asyncio
async def test_systemone_request_gets_typed_systemone_response(
    mock_proxy_client, mock_proxy_response, backend_format: BackendFormat | None
):
    """The /v1/systemone path decides the response format, even for an entity still marked OPENAI_CHAT."""
    seen: list[tuple[BackendFormat | None, object]] = []

    class _Recorder(NemoInferenceMiddleware):
        async def process_response(
            self, ctx: InferenceMiddlewareContext, response: InferenceResponse, middleware_config: object
        ) -> InferenceResponse:
            seen.append((ctx.backend_format, response.typed_body))
            return response

    registry = MiddlewareRegistry(plugins={"recorder": _Recorder()})
    registry.response_middleware_calls[(WORKSPACE, ENTITY)] = [
        ResolvedMiddlewareCall(plugin_name="recorder", config_type="t", resolved_config={})
    ]
    mock_proxy_response._body = [json.dumps(_systemone_response()).encode()]

    response = await virtual_model_proxy(
        request=_request(),
        workspace=WORKSPACE,
        vm_name=ENTITY,
        virtual_model=_virtual_model(backend_format),
        trailing_uri="v1/systemone",
        json_body=_systemone_request(),
        http_client=mock_proxy_client,
        model_cache=_model_cache(),
        registry=registry,
    )

    assert isinstance(response, StreamingResponse)
    assert response.status_code == 200
    [(seen_format, typed_body)] = seen
    assert seen_format is BackendFormat.SYSTEMONE
    assert isinstance(typed_body, SystemOneResponse)
