# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for proxy functionality."""

import asyncio
import gzip
import json
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import aiohttp
import anthropic.types as anthropic_types
import httpx
import openai.types.chat as openai_chat_types
import pytest
import pytest_asyncio
from aiohttp import ClientError
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from multidict import CIMultiDict, CIMultiDictProxy
from nemo_platform.types.inference import ModelProvider, ServedModelMapping
from nemo_platform.types.inference.virtual_model import VirtualModel as SDKVirtualModel
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.inference_middleware import (
    BackendFormat,
    ImmediateResponse,
    InferenceMiddlewareContext,
    InferenceRequest,
    InferenceResponse,
    NemoInferenceMiddleware,
)
from nmp.core.inference_gateway.api.middleware_registry import (
    MiddlewareRegistry,
    ResolvedMiddlewareCall,
    build_inference_response,
)
from nmp.core.inference_gateway.api.model_cache import ModelCache, ModelProviderInfo
from nmp.core.inference_gateway.api.proxy import (
    NextRequestInfo,
    _build_inference_response_with_annotations,
    _parse_sse_stream,
    _rewrite_model_field,
    _rewrite_model_field_in_error_body,
    _rewrite_model_field_in_stream,
    build_next_request,
    fetch_proxy_response,
    normalize_proxy_url,
    proxy_request,
    stream_response_result,
    virtual_model_proxy,
)
from pytest_httpserver import HTTPServer


@pytest.fixture
def mock_request():
    """Create a mock FastAPI request."""
    request = Mock(spec=Request)
    request.method = "POST"
    request.headers = {"host": "localhost:8080", "accept": "application/json"}
    request.url.scheme = "http"
    request.body = AsyncMock(return_value=b'{"test": "data"}')
    request.query_params = {"param1": "value1"}
    return request


@pytest_asyncio.fixture
async def next_request_info(mock_request):
    return await build_next_request(mock_request, "http://example.com", "api")


async def _read_streaming_response(response: StreamingResponse) -> bytes:
    chunks: list[bytes] = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, str):
            chunks.append(chunk.encode())
        else:
            chunks.append(bytes(chunk))
    return b"".join(chunks)


def _json_request_body(result: NextRequestInfo) -> dict[str, Any]:
    assert result.body is not None
    return json.loads(result.body)


@pytest.mark.asyncio
async def test_build_next_request(mock_request):
    """Test building next request info for POST request with body."""
    mock_request.headers["host"] = "should_not_be_proxied"
    result = await build_next_request(mock_request, "http://example.com", "api")

    assert result.body == b'{"test": "data"}'
    assert result.method == "POST"
    assert result.url == "http://example.com/api"
    assert "host" not in result.headers
    assert "x-forwarded-proto" not in result.headers
    assert "x-forwarded-host" not in result.headers
    assert "x-forwarded-for" not in result.headers


# Below was claudes work that I need to fix up massively
@pytest.mark.asyncio
async def test_proxy_request_success(mock_proxy_client, next_request_info):
    response = await proxy_request(mock_proxy_client, next_request_info)
    assert isinstance(response, StreamingResponse)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_stream_response_result_prefers_typed_non_streaming_payload():
    typed = openai_chat_types.ChatCompletion.model_validate(
        {
            "id": "typed",
            "object": "chat.completion",
            "created": 1,
            "model": "llama",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "typed"}, "finish_reason": "stop"}],
        }
    )

    response = await stream_response_result(
        InferenceResponse(result={"id": "raw"}, headers={}, typed_body=typed),
        200,
        {},
    )

    body = await _read_streaming_response(response)
    assert json.loads(body)["id"] == "typed"


@pytest.mark.asyncio
async def test_stream_response_result_strips_stale_upstream_body_headers():
    """Re-serialized middleware output does not retain upstream body framing."""
    response = await stream_response_result(
        {"id": "decoded"},
        200,
        {
            "content-length": "123",
            "content-encoding": "gzip",
            "transfer-encoding": "chunked",
            "x-upstream-header": "preserved",
        },
    )

    assert "content-length" not in response.headers
    assert "content-encoding" not in response.headers
    assert "transfer-encoding" not in response.headers
    assert response.headers["content-type"] == "application/json"
    assert response.headers["x-upstream-header"] == "preserved"
    assert json.loads(await _read_streaming_response(response)) == {"id": "decoded"}


def test_response_annotations_do_not_overwrite_typed_body_fields():
    typed = openai_chat_types.ChatCompletion.model_validate(
        {
            "id": "typed",
            "object": "chat.completion",
            "created": 1,
            "model": "llama",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "typed"}, "finish_reason": "stop"}],
        }
    )
    response = InferenceResponse(
        result={"id": "raw"},
        headers={},
        typed_body=typed,
        response_body_annotations={
            "id": "annotation-wins",
            "guardrails_data": {"config_ids": ["default/safety-config"]},
        },
    )

    annotated_response = _build_inference_response_with_annotations(response)

    assert response.typed_body is typed
    assert response.result == {"id": "raw"}
    assert annotated_response.typed_body is None
    assert annotated_response.response_body_annotations == {}
    assert annotated_response.result == {
        **typed.model_dump(mode="json"),
        "guardrails_data": {"config_ids": ["default/safety-config"]},
    }


def test_response_annotations_noop_without_annotations():
    typed = openai_chat_types.ChatCompletion.model_validate(
        {
            "id": "typed",
            "object": "chat.completion",
            "created": 1,
            "model": "llama",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "typed"}, "finish_reason": "stop"}],
        }
    )
    response = InferenceResponse(result={"id": "raw"}, headers={}, typed_body=typed)

    annotated_response = _build_inference_response_with_annotations(response)

    assert annotated_response is response
    assert annotated_response.typed_body is typed


@pytest.mark.asyncio
async def test_response_annotations_noop_for_streaming():
    async def raw_chunks():
        yield {
            "id": "raw-chunk",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "llama",
            "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
        }

    response = build_inference_response(raw_chunks(), {}, BackendFormat.OPENAI_CHAT)
    response.response_body_annotations["guardrails_data"] = {"config_ids": ["default/safety-config"]}

    response = _build_inference_response_with_annotations(response)

    body = (await _read_streaming_response(await stream_response_result(response, 200, {}))).decode()
    assert "guardrails_data" not in body
    assert 'data: {"id": "raw-chunk"' in body


@pytest.mark.asyncio
async def test_stream_response_result_uses_raw_stream_when_typed_stream_is_present():
    async def raw_chunks():
        yield {"id": "raw-chunk", "vendor_passthrough": "raw"}

    async def typed_chunks():
        yield openai_chat_types.ChatCompletionChunk.model_validate(
            {
                "id": "typed-chunk",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "llama",
                "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
            }
        )

    response = await stream_response_result(
        InferenceResponse(result=raw_chunks(), headers={}, typed_body=typed_chunks()),
        200,
        {},
    )

    body = (await _read_streaming_response(response)).decode()
    assert 'data: {"id": "raw-chunk", "vendor_passthrough": "raw"}' in body
    assert "typed-chunk" not in body
    assert "data: [DONE]" in body


@pytest.mark.asyncio
async def test_stream_response_result_replays_raw_stream_with_unrecognised_chunks():
    """Unrecognised chunks (malformed shape, vendor extras, Anthropic ``ping``,
    etc.) are skipped from the typed view but always appear in the
    raw stream — wire-level serialization must not be filtered by the typed
    Union's coverage."""

    async def raw_chunks():
        yield {
            "id": "chunk-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "llama",
            "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
        }
        yield {"id": "invalid"}  # malformed → skipped from typed view
        yield {"vendor_passthrough": "raw-tail"}  # unrecognised shape → skipped

    response_envelope = build_inference_response(raw_chunks(), {}, BackendFormat.OPENAI_CHAT)
    typed_stream_before = response_envelope.typed_body
    assert typed_stream_before is not None
    typed_stream = cast(AsyncIterator[Any], response_envelope.typed_body)

    typed_chunks = [chunk async for chunk in typed_stream]
    # Only the valid chunk comes through the typed view.
    assert len(typed_chunks) == 1
    # typed_body is NOT nulled — the typed view stays useful.
    assert response_envelope.typed_body is typed_stream_before

    response = await stream_response_result(response_envelope, 200, {})

    body = (await _read_streaming_response(response)).decode()
    # All three raw chunks are re-emitted via raw_chunks() for serialization.
    assert 'data: {"id": "chunk-1"' in body
    assert 'data: {"id": "invalid"}' in body
    assert 'data: {"vendor_passthrough": "raw-tail"}' in body
    assert "data: [DONE]" in body


@pytest.mark.asyncio
async def test_virtual_model_proxy_typed_context_defaults_unset_backend_format_to_openai_chat(mock_proxy_client):
    seen_backend_formats: list[BackendFormat | None] = []

    class _ContextPlugin(NemoInferenceMiddleware):
        async def process_response(
            self,
            ctx: InferenceMiddlewareContext,
            response: InferenceResponse,
            middleware_config: object,
        ) -> InferenceResponse:
            seen_backend_formats.append(ctx.backend_format)
            return response

    workspace = "e2e-test"
    vm_name = "my-router"
    model_entity_id = f"{workspace}/meta_llama-3.2-1b-instruct"
    served_model_name = "meta/llama-3.2-1b-instruct"

    model_cache = ModelCache()
    model_cache.update_model_info(
        ModelProviderInfo(
            model_provider=ModelProvider(
                workspace="default",
                name="nim",
                host_url="http://nim.local",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                served_models=[
                    ServedModelMapping(
                        model_entity_id=model_entity_id,
                        served_model_name=served_model_name,
                    )
                ],
                status="READY",
            )
        )
    )
    model_cache.rebuild_model_entity_map()

    registry = MiddlewareRegistry(plugins={"test-plugin": _ContextPlugin()})
    registry.response_middleware_calls[(workspace, vm_name)] = [
        ResolvedMiddlewareCall(plugin_name="test-plugin", config_type="t", resolved_config={})
    ]

    request = Mock(spec=Request)
    request.method = "POST"
    request.headers = {"content-type": "application/json"}
    request.query_params = {}

    response = await virtual_model_proxy(
        request=request,
        workspace=workspace,
        vm_name=vm_name,
        virtual_model=SDKVirtualModel(
            id=f"{workspace}/{vm_name}",
            entity_id=f"{workspace}/{vm_name}",
            name=vm_name,
            workspace=workspace,
            parent=workspace,
            db_version=1,
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            default_model_entity=model_entity_id,
        ),
        trailing_uri="v1/chat/completions",
        json_body={"model": vm_name, "messages": [{"role": "user", "content": "hi"}]},
        http_client=mock_proxy_client,
        model_cache=model_cache,
        registry=registry,
    )

    assert isinstance(response, StreamingResponse)
    await _read_streaming_response(response)

    assert response.status_code == 200
    assert seen_backend_formats == [BackendFormat.OPENAI_CHAT]


@pytest.mark.asyncio
async def test_virtual_model_proxy_adds_request_nemo_client_to_middleware_context(mock_proxy_client):
    seen_clients: list[AsyncNemoClient | None] = []

    class _ContextPlugin(NemoInferenceMiddleware):
        async def process_request(
            self,
            ctx: InferenceMiddlewareContext,
            request: InferenceRequest,
            middleware_config: object,
        ) -> ImmediateResponse:
            seen_clients.append(ctx.request_nemo_client)
            return ImmediateResponse(data={"ok": True})

    workspace = "e2e-test"
    vm_name = "my-router"
    registry = MiddlewareRegistry(plugins={"test-plugin": _ContextPlugin()})
    registry.request_middleware_calls[(workspace, vm_name)] = [
        ResolvedMiddlewareCall(plugin_name="test-plugin", config_type="t", resolved_config={})
    ]

    request = Mock(spec=Request)
    request.method = "POST"
    request.headers = {"content-type": "application/json"}
    request.query_params = {}

    async with httpx.AsyncClient() as http_client:
        request_nemo_client = AsyncNemoClient(
            base_url="http://platform.test",
            default_headers={"traceparent": "00-request"},
            http_client=http_client,
        )

        response = await virtual_model_proxy(
            request=request,
            workspace=workspace,
            vm_name=vm_name,
            virtual_model=SDKVirtualModel(
                id=f"{workspace}/{vm_name}",
                entity_id=f"{workspace}/{vm_name}",
                name=vm_name,
                workspace=workspace,
                parent=workspace,
                db_version=1,
                created_at="2026-01-01T00:00:00Z",
                updated_at="2026-01-01T00:00:00Z",
            ),
            trailing_uri="v1/chat/completions",
            json_body={"model": vm_name, "messages": [{"role": "user", "content": "hi"}]},
            http_client=mock_proxy_client,
            model_cache=ModelCache(),
            registry=registry,
            request_nemo_client=request_nemo_client,
        )

    assert isinstance(response, StreamingResponse)
    body = json.loads(await _read_streaming_response(response))
    assert body == {"ok": True}
    assert seen_clients == [request_nemo_client]
    mock_proxy_client.request.assert_not_called()


@pytest.mark.asyncio
async def test_virtual_model_proxy_injects_request_context_annotations_without_response_middleware(
    mock_proxy_client,
    mock_proxy_response,
):
    class _RequestAnnotatingPlugin(NemoInferenceMiddleware):
        async def process_request(
            self,
            ctx: InferenceMiddlewareContext,
            request: InferenceRequest,
            middleware_config: object,
        ) -> InferenceRequest:
            ctx.response_body_annotations["guardrails_data"] = {"config_ids": ["ws/input-only"]}
            return request

    mock_proxy_response.read = AsyncMock(
        return_value=json.dumps(
            {
                "id": "raw",
                "object": "chat.completion",
                "created": 1,
                "model": "llama",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "raw"},
                        "finish_reason": "stop",
                    }
                ],
            }
        ).encode()
    )

    workspace = "e2e-test"
    vm_name = "request-annotation-router"
    model_entity_id = f"{workspace}/meta_llama-3.2-1b-instruct"
    served_model_name = "meta/llama-3.2-1b-instruct"

    model_cache = ModelCache()
    model_cache.update_model_info(
        ModelProviderInfo(
            model_provider=ModelProvider(
                workspace="default",
                name="nim",
                host_url="http://nim.local",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                served_models=[
                    ServedModelMapping(
                        model_entity_id=model_entity_id,
                        served_model_name=served_model_name,
                    )
                ],
                status="READY",
            )
        )
    )
    model_cache.rebuild_model_entity_map()

    registry = MiddlewareRegistry(plugins={"request-plugin": _RequestAnnotatingPlugin()})
    registry.request_middleware_calls[(workspace, vm_name)] = [
        ResolvedMiddlewareCall(plugin_name="request-plugin", config_type="t", resolved_config={})
    ]

    request = Mock(spec=Request)
    request.method = "POST"
    request.headers = {"content-type": "application/json"}
    request.query_params = {}

    response = await virtual_model_proxy(
        request=request,
        workspace=workspace,
        vm_name=vm_name,
        virtual_model=SDKVirtualModel(
            id=f"{workspace}/{vm_name}",
            entity_id=f"{workspace}/{vm_name}",
            name=vm_name,
            workspace=workspace,
            parent=workspace,
            db_version=1,
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            default_model_entity=model_entity_id,
        ),
        trailing_uri="v1/chat/completions",
        json_body={"model": vm_name, "messages": [{"role": "user", "content": "hi"}]},
        http_client=mock_proxy_client,
        model_cache=model_cache,
        registry=registry,
    )

    body = json.loads(await _read_streaming_response(cast(StreamingResponse, response)))

    assert body["id"] == "raw"
    assert body["guardrails_data"] == {"config_ids": ["ws/input-only"]}


@pytest.mark.parametrize(
    "middleware_order",
    [
        ("translate-plugin", "annotate-plugin"),
        ("annotate-plugin", "translate-plugin"),
    ],
)
@pytest.mark.asyncio
async def test_virtual_model_proxy_injects_annotations_after_response_middleware_ordering(
    mock_proxy_client,
    mock_proxy_response,
    middleware_order: tuple[str, str],
):
    class _AnnotatingPlugin(NemoInferenceMiddleware):
        async def process_response(
            self,
            ctx: InferenceMiddlewareContext,
            response: InferenceResponse,
            middleware_config: object,
        ) -> InferenceResponse:
            response.response_body_annotations["guardrails_data"] = {"config_ids": ["default/safety-config"]}
            return response

    class _TranslatePlugin(NemoInferenceMiddleware):
        async def process_response(
            self,
            ctx: InferenceMiddlewareContext,
            response: InferenceResponse,
            middleware_config: object,
        ) -> InferenceResponse:
            assert isinstance(response.typed_body, openai_chat_types.ChatCompletion)
            response.typed_body = openai_chat_types.ChatCompletion.model_validate(
                {
                    "id": "translated",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "llama",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "translated"},
                            "finish_reason": "stop",
                        }
                    ],
                }
            )
            response.result = response.typed_body.model_dump(mode="json")
            return response

    mock_proxy_response.read = AsyncMock(
        return_value=json.dumps(
            {
                "id": "raw",
                "object": "chat.completion",
                "created": 1,
                "model": "llama",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "raw"},
                        "finish_reason": "stop",
                    }
                ],
            }
        ).encode()
    )

    workspace = "e2e-test"
    vm_name = "annotation-router"
    model_entity_id = f"{workspace}/meta_llama-3.2-1b-instruct"
    served_model_name = "meta/llama-3.2-1b-instruct"

    model_cache = ModelCache()
    model_cache.update_model_info(
        ModelProviderInfo(
            model_provider=ModelProvider(
                workspace="default",
                name="nim",
                host_url="http://nim.local",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                served_models=[
                    ServedModelMapping(
                        model_entity_id=model_entity_id,
                        served_model_name=served_model_name,
                    )
                ],
                status="READY",
            )
        )
    )
    model_cache.rebuild_model_entity_map()

    registry = MiddlewareRegistry(
        plugins={
            "annotate-plugin": _AnnotatingPlugin(),
            "translate-plugin": _TranslatePlugin(),
        }
    )
    registry.response_middleware_calls[(workspace, vm_name)] = [
        ResolvedMiddlewareCall(plugin_name=plugin_name, config_type="t", resolved_config={})
        for plugin_name in middleware_order
    ]

    request = Mock(spec=Request)
    request.method = "POST"
    request.headers = {"content-type": "application/json"}
    request.query_params = {}

    response = await virtual_model_proxy(
        request=request,
        workspace=workspace,
        vm_name=vm_name,
        virtual_model=SDKVirtualModel(
            id=f"{workspace}/{vm_name}",
            entity_id=f"{workspace}/{vm_name}",
            name=vm_name,
            workspace=workspace,
            parent=workspace,
            db_version=1,
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            default_model_entity=model_entity_id,
        ),
        trailing_uri="v1/chat/completions",
        json_body={"model": vm_name, "messages": [{"role": "user", "content": "hi"}]},
        http_client=mock_proxy_client,
        model_cache=model_cache,
        registry=registry,
    )

    body = json.loads(await _read_streaming_response(cast(StreamingResponse, response)))

    assert body["id"] == "translated"
    assert body["choices"][0]["message"]["content"] == "translated"
    assert body["guardrails_data"] == {"config_ids": ["default/safety-config"]}


@pytest.mark.asyncio
async def test_virtual_model_proxy_with_unresolved_provider_secret_returns_424(mock_proxy_client):
    workspace = "e2e-test"
    vm_name = "secure-router"
    model_entity_id = f"{workspace}/secure-model"

    model_cache = ModelCache()
    model_cache.update_model_info(
        ModelProviderInfo(
            model_provider=ModelProvider(
                workspace=workspace,
                name="secure-provider",
                host_url="http://secure.local",
                api_key_secret_name="missing-secret",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                served_models=[
                    ServedModelMapping(
                        model_entity_id=model_entity_id,
                        served_model_name="secure-v1",
                    )
                ],
                status="READY",
            ),
            secret_value=None,
        )
    )
    model_cache.rebuild_model_entity_map()

    request = Mock(spec=Request)
    request.method = "POST"
    request.headers = {"content-type": "application/json"}
    request.query_params = {}

    with pytest.raises(HTTPException) as exc_info:
        await virtual_model_proxy(
            request=request,
            workspace=workspace,
            vm_name=vm_name,
            virtual_model=SDKVirtualModel(
                id=f"{workspace}/{vm_name}",
                entity_id=f"{workspace}/{vm_name}",
                name=vm_name,
                workspace=workspace,
                parent=workspace,
                db_version=1,
                created_at="2026-01-01T00:00:00Z",
                updated_at="2026-01-01T00:00:00Z",
                default_model_entity=model_entity_id,
            ),
            trailing_uri="v1/chat/completions",
            json_body={"model": vm_name, "messages": [{"role": "user", "content": "hi"}]},
            http_client=mock_proxy_client,
            model_cache=model_cache,
            registry=MiddlewareRegistry(),
        )

    assert exc_info.value.status_code == 424
    assert "secret not found or unreachable" in exc_info.value.detail
    mock_proxy_client.request.assert_not_called()


@pytest.mark.asyncio
async def test_virtual_model_proxy_preserves_request_middleware_backend_format_for_immediate_response():
    seen_typed_ids: list[str] = []

    class _ImmediateAnthropicPlugin(NemoInferenceMiddleware):
        async def process_request(
            self,
            ctx: InferenceMiddlewareContext,
            request: InferenceRequest,
            middleware_config: object,
        ) -> ImmediateResponse:
            ctx.backend_format = BackendFormat.ANTHROPIC_MESSAGES
            return ImmediateResponse(
                data={
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hello"}],
                    "model": "claude",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )

    class _ResponsePlugin(NemoInferenceMiddleware):
        async def process_response(
            self,
            ctx: InferenceMiddlewareContext,
            response: InferenceResponse,
            middleware_config: object,
        ) -> InferenceResponse:
            assert ctx.backend_format is BackendFormat.ANTHROPIC_MESSAGES
            assert isinstance(response.typed_body, anthropic_types.Message)
            seen_typed_ids.append(response.typed_body.id)
            return response

    workspace = "e2e-test"
    vm_name = "immediate-router"
    registry = MiddlewareRegistry(
        plugins={
            "request-plugin": _ImmediateAnthropicPlugin(),
            "response-plugin": _ResponsePlugin(),
        }
    )
    registry.request_middleware_calls[(workspace, vm_name)] = [
        ResolvedMiddlewareCall(plugin_name="request-plugin", config_type="t", resolved_config={})
    ]
    registry.response_middleware_calls[(workspace, vm_name)] = [
        ResolvedMiddlewareCall(plugin_name="response-plugin", config_type="t", resolved_config={})
    ]

    request = Mock(spec=Request)
    request.method = "POST"
    request.headers = {"content-type": "application/json"}
    request.query_params = {}

    response = await virtual_model_proxy(
        request=request,
        workspace=workspace,
        vm_name=vm_name,
        virtual_model=SDKVirtualModel(
            id=f"{workspace}/{vm_name}",
            entity_id=f"{workspace}/{vm_name}",
            name=vm_name,
            workspace=workspace,
            parent=workspace,
            db_version=1,
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        ),
        trailing_uri="v1/messages",
        json_body={"model": vm_name, "messages": [{"role": "user", "content": "hi"}]},
        http_client=Mock(),
        model_cache=ModelCache(),
        registry=registry,
    )

    body = json.loads(await _read_streaming_response(cast(StreamingResponse, response)))

    assert body["id"] == "msg_1"
    assert seen_typed_ids == ["msg_1"]


@pytest.mark.asyncio
async def test_virtual_model_proxy_preserves_immediate_response_annotations_through_response_middleware():
    class _ImmediatePlugin(NemoInferenceMiddleware):
        async def process_request(
            self,
            ctx: InferenceMiddlewareContext,
            request: InferenceRequest,
            middleware_config: object,
        ) -> ImmediateResponse:
            return ImmediateResponse(
                data={
                    "id": "raw",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "llama",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "blocked"},
                            "finish_reason": "content_filter",
                        }
                    ],
                },
                response_body_annotations={"guardrails_data": {"config_ids": ["ws/cfg"]}},
            )

    class _ResponsePlugin(NemoInferenceMiddleware):
        async def process_response(
            self,
            ctx: InferenceMiddlewareContext,
            response: InferenceResponse,
            middleware_config: object,
        ) -> InferenceResponse:
            assert isinstance(response.typed_body, openai_chat_types.ChatCompletion)
            response.typed_body.id = "translated"
            return response

    workspace = "e2e-test"
    vm_name = "immediate-annotation-router"
    registry = MiddlewareRegistry(
        plugins={
            "request-plugin": _ImmediatePlugin(),
            "response-plugin": _ResponsePlugin(),
        }
    )
    registry.request_middleware_calls[(workspace, vm_name)] = [
        ResolvedMiddlewareCall(plugin_name="request-plugin", config_type="t", resolved_config={})
    ]
    registry.response_middleware_calls[(workspace, vm_name)] = [
        ResolvedMiddlewareCall(plugin_name="response-plugin", config_type="t", resolved_config={})
    ]

    request = Mock(spec=Request)
    request.method = "POST"
    request.headers = {"content-type": "application/json"}
    request.query_params = {}

    response = await virtual_model_proxy(
        request=request,
        workspace=workspace,
        vm_name=vm_name,
        virtual_model=SDKVirtualModel(
            id=f"{workspace}/{vm_name}",
            entity_id=f"{workspace}/{vm_name}",
            name=vm_name,
            workspace=workspace,
            parent=workspace,
            db_version=1,
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        ),
        trailing_uri="v1/chat/completions",
        json_body={"model": vm_name, "messages": [{"role": "user", "content": "hi"}]},
        http_client=Mock(),
        model_cache=ModelCache(),
        registry=registry,
    )

    body = json.loads(await _read_streaming_response(cast(StreamingResponse, response)))

    assert body["id"] == "translated"
    assert body["guardrails_data"] == {"config_ids": ["ws/cfg"]}


@pytest.mark.asyncio
async def test_virtual_model_proxy_post_response_receives_canonical_non_streaming_response(
    mock_proxy_client,
    mock_proxy_response,
):
    seen_typed_ids: list[str] = []
    seen_annotations: list[dict[str, Any]] = []
    post_response_called = asyncio.Event()

    class _ResponsePlugin(NemoInferenceMiddleware):
        async def process_response(
            self,
            ctx: InferenceMiddlewareContext,
            response: InferenceResponse,
            middleware_config: object,
        ) -> InferenceResponse:
            assert isinstance(response.typed_body, openai_chat_types.ChatCompletion)
            response.response_body_annotations["guardrails_data"] = {"config_ids": ["ws/cfg"]}
            return response

    class _PostResponsePlugin(NemoInferenceMiddleware):
        async def process_response(
            self,
            ctx: InferenceMiddlewareContext,
            response: InferenceResponse,
            middleware_config: object,
        ) -> InferenceResponse:
            assert isinstance(response.typed_body, openai_chat_types.ChatCompletion)
            seen_typed_ids.append(response.typed_body.id)
            seen_annotations.append(dict(response.response_body_annotations))
            post_response_called.set()
            return response

    mock_proxy_response.read = AsyncMock(
        return_value=json.dumps(
            {
                "id": "chatcmpl-post",
                "object": "chat.completion",
                "created": 1,
                "model": "llama",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            }
        ).encode()
    )
    workspace = "e2e-test"
    vm_name = "post-router"
    model_entity_id = f"{workspace}/meta_llama-3.2-1b-instruct"
    served_model_name = "meta/llama-3.2-1b-instruct"

    model_cache = ModelCache()
    model_cache.update_model_info(
        ModelProviderInfo(
            model_provider=ModelProvider(
                workspace="default",
                name="nim",
                host_url="http://nim.local",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                served_models=[
                    ServedModelMapping(
                        model_entity_id=model_entity_id,
                        served_model_name=served_model_name,
                    )
                ],
                status="READY",
            )
        )
    )
    model_cache.rebuild_model_entity_map()

    registry = MiddlewareRegistry(
        plugins={
            "response-plugin": _ResponsePlugin(),
            "post-response-plugin": _PostResponsePlugin(),
        }
    )
    registry.response_middleware_calls[(workspace, vm_name)] = [
        ResolvedMiddlewareCall(plugin_name="response-plugin", config_type="t", resolved_config={})
    ]
    registry.post_response_middleware_calls[(workspace, vm_name)] = [
        ResolvedMiddlewareCall(plugin_name="post-response-plugin", config_type="t", resolved_config={})
    ]

    request = Mock(spec=Request)
    request.method = "POST"
    request.headers = {"content-type": "application/json"}
    request.query_params = {}

    response = await virtual_model_proxy(
        request=request,
        workspace=workspace,
        vm_name=vm_name,
        virtual_model=SDKVirtualModel(
            id=f"{workspace}/{vm_name}",
            entity_id=f"{workspace}/{vm_name}",
            name=vm_name,
            workspace=workspace,
            parent=workspace,
            db_version=1,
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            default_model_entity=model_entity_id,
        ),
        trailing_uri="v1/chat/completions",
        json_body={"model": vm_name, "messages": [{"role": "user", "content": "hi"}]},
        http_client=mock_proxy_client,
        model_cache=model_cache,
        registry=registry,
    )

    body = json.loads(await _read_streaming_response(cast(StreamingResponse, response)))
    await asyncio.wait_for(post_response_called.wait(), timeout=1)

    assert body["id"] == "chatcmpl-post"
    assert body["guardrails_data"] == {"config_ids": ["ws/cfg"]}
    assert seen_typed_ids == ["chatcmpl-post"]
    assert seen_annotations == [{"guardrails_data": {"config_ids": ["ws/cfg"]}}]


@pytest.mark.asyncio
async def test_virtual_model_proxy_broken_vm_returns_503_without_running_middleware(mock_proxy_client):
    """A VM in :attr:`MiddlewareRegistry.broken_vms` must 503 before any middleware runs.

    This is the fail-closed counterpart to the deletion-detection plumbing:
    once the polling loop has noticed that a referenced ``config_id`` is gone,
    the proxy must refuse to serve requests for that VM rather than silently
    bypassing the (now-evicted) middleware chain. Recovery is automatic on the
    next successful re-resolve, which is exercised end-to-end in
    ``test_virtual_model_cache.test_refresh_marks_vm_broken_when_config_deleted_and_clears_on_recreate``.
    """
    request_plugin_called = False
    response_plugin_called = False

    class _ShouldNotRunRequestPlugin(NemoInferenceMiddleware):
        async def process_request(self, ctx, request, middleware_config):
            nonlocal request_plugin_called
            request_plugin_called = True
            return request

    class _ShouldNotRunResponsePlugin(NemoInferenceMiddleware):
        async def process_response(self, ctx, response, middleware_config):
            nonlocal response_plugin_called
            response_plugin_called = True
            return response

    workspace = "ws"
    vm_name = "broken-vm"

    model_cache = ModelCache()
    registry = MiddlewareRegistry(
        plugins={
            "req": _ShouldNotRunRequestPlugin(),
            "resp": _ShouldNotRunResponsePlugin(),
        }
    )
    # Seed both phase dicts to confirm the proxy does NOT consult them — a
    # broken VM short-circuits before any registry lookup.
    registry.request_middleware_calls[(workspace, vm_name)] = [
        ResolvedMiddlewareCall(plugin_name="req", config_type="t", resolved_config={})
    ]
    registry.response_middleware_calls[(workspace, vm_name)] = [
        ResolvedMiddlewareCall(plugin_name="resp", config_type="t", resolved_config={})
    ]
    registry.broken_vms.add((workspace, vm_name))

    request = Mock(spec=Request)
    request.method = "POST"
    request.headers = {"content-type": "application/json"}
    request.query_params = {}

    with pytest.raises(HTTPException) as exc_info:
        await virtual_model_proxy(
            request=request,
            workspace=workspace,
            vm_name=vm_name,
            virtual_model=SDKVirtualModel(
                id=f"{workspace}/{vm_name}",
                entity_id=f"{workspace}/{vm_name}",
                name=vm_name,
                workspace=workspace,
                parent=workspace,
                db_version=1,
                created_at="2026-01-01T00:00:00Z",
                updated_at="2026-01-01T00:00:00Z",
                default_model_entity=f"{workspace}/missing",
            ),
            trailing_uri="v1/chat/completions",
            json_body={"messages": [{"role": "user", "content": "hi"}]},
            http_client=mock_proxy_client,
            model_cache=model_cache,
            registry=registry,
        )

    assert exc_info.value.status_code == 503
    assert "Middleware configuration unavailable" in exc_info.value.detail
    assert f"{workspace}/{vm_name}" in exc_info.value.detail
    # Critical: no middleware (and no backend call) was attempted.
    assert request_plugin_called is False
    assert response_plugin_called is False
    mock_proxy_client.request.assert_not_called()


@pytest.mark.asyncio
async def test_proxy_request_client_error(mock_proxy_client, next_request_info):
    mock_proxy_client.request = AsyncMock(side_effect=ClientError("Connection failed"))

    with pytest.raises(HTTPException) as exc_info:
        await proxy_request(mock_proxy_client, next_request_info)

    assert "Backend networking error" in str(exc_info.value.detail)
    assert exc_info.value.status_code == 502


@pytest.mark.asyncio
async def test_proxy_request_streaming_closes_connection(mock_proxy_client, mock_proxy_response, next_request_info):
    response = await proxy_request(mock_proxy_client, next_request_info)

    # Consume the streaming response to trigger cleanup
    async for _ in response.body_iterator:
        pass

    mock_proxy_response.close.assert_called_once()


@pytest.mark.asyncio
async def test_build_next_request_with_default_extra_body(mock_request):
    """Test that default_extra_body provides defaults that can be overridden by request."""
    # Request body has some fields
    mock_request.body = AsyncMock(return_value=b'{"model": "gpt-4", "temperature": 0.9}')

    # Provider has default_extra_body - these can be overridden by request
    default_extra_body = {"stream": True, "max_tokens": 100, "temperature": 0.5}

    result = await build_next_request(mock_request, "http://example.com", "api", default_extra_body=default_extra_body)

    result_body = _json_request_body(result)

    # Request body fields should take precedence over default_extra_body
    assert result_body["temperature"] == 0.9  # From request, not default_extra_body
    assert result_body["model"] == "gpt-4"  # From request

    # Default extra body fields should be included when not in request
    assert result_body["stream"] is True  # From default_extra_body
    assert result_body["max_tokens"] == 100  # From default_extra_body


@pytest.mark.asyncio
async def test_build_next_request_with_required_extra_body(mock_request):
    """Test that required_extra_body cannot be overridden by request."""
    # Request body tries to set stream=False
    mock_request.body = AsyncMock(return_value=b'{"model": "gpt-4", "stream": false}')

    # Provider has required_extra_body - these CANNOT be overridden
    required_extra_body = {"stream": True, "organization": "required-org"}

    result = await build_next_request(
        mock_request, "http://example.com", "api", required_extra_body=required_extra_body
    )

    result_body = _json_request_body(result)

    # Required extra body fields should take precedence over request
    assert result_body["stream"] is True  # From required_extra_body, overrides request's false
    assert result_body["organization"] == "required-org"  # From required_extra_body

    # Request fields not in required_extra_body are preserved
    assert result_body["model"] == "gpt-4"  # From request


@pytest.mark.asyncio
async def test_build_next_request_request_headers_override(mock_request):
    """request_headers replaces request.headers as the header source.

    This is how middleware-modified InferenceRequest.headers reach the backend:
    the caller passes modified_request.headers instead of letting build_next_request
    read from the raw FastAPI Request object.
    """
    # Original request has one set of headers
    mock_request.headers = {"x-original": "original-value", "content-type": "application/json"}

    # Middleware has added a custom header and removed another
    middleware_headers = {
        "x-original": "original-value",
        "x-middleware-added": "plugin-value",
        "content-type": "application/json",
    }

    result = await build_next_request(
        mock_request,
        "http://example.com",
        "api",
        request_headers=middleware_headers,
    )

    # Middleware-added header must reach the backend
    assert result.headers["x-middleware-added"] == "plugin-value"
    # Original header still present
    assert result.headers["x-original"] == "original-value"


@pytest.mark.asyncio
async def test_build_next_request_request_headers_override_drops_sensitive_headers(mock_request):
    """Headers in REQUEST_HEADERS_TO_DROP are still filtered even from request_headers override."""
    middleware_headers = {
        "x-custom": "keep-me",
        "host": "should-be-dropped",
        "authorization": "Bearer client-token-that-must-be-dropped",
    }

    result = await build_next_request(
        mock_request,
        "http://example.com",
        "api",
        request_headers=middleware_headers,
    )

    assert result.headers["x-custom"] == "keep-me"
    assert "host" not in result.headers
    assert "authorization" not in result.headers  # provider auth injected separately


@pytest.mark.asyncio
async def test_build_next_request_drops_x_api_key(mock_request):
    """A client-supplied x-api-key is dropped so it never reaches the upstream.

    The platform injects provider auth from Secrets; forwarding a client placeholder
    x-api-key (e.g. "not-used") makes Anthropic/Bedrock backends 401. It must be
    stripped like ``authorization``.
    """
    mock_request.headers = {
        "content-type": "application/json",
        "x-api-key": "not-used",
    }

    result = await build_next_request(mock_request, "http://example.com", "v1/messages")

    assert "x-api-key" not in result.headers  # provider auth injected separately
    assert result.headers["content-type"] == "application/json"


@pytest.mark.asyncio
async def test_build_next_request_request_headers_override_drops_x_api_key(mock_request):
    """x-api-key is filtered from a request_headers override too (the model-scoped path)."""
    middleware_headers = {
        "x-custom": "keep-me",
        "x-api-key": "not-used",
    }

    result = await build_next_request(
        mock_request,
        "http://example.com",
        "v1/messages",
        request_headers=middleware_headers,
    )

    assert result.headers["x-custom"] == "keep-me"
    assert "x-api-key" not in result.headers  # provider auth injected separately


@pytest.mark.asyncio
async def test_build_next_request_request_headers_none_falls_back_to_request(mock_request):
    """When request_headers is None (the default), request.headers is used as before."""
    mock_request.headers = {"x-from-request": "yes", "content-type": "application/json"}

    result = await build_next_request(
        mock_request,
        "http://example.com",
        "api",
        request_headers=None,
    )

    assert result.headers["x-from-request"] == "yes"


@pytest.mark.asyncio
async def test_build_next_request_with_default_extra_headers(mock_request):
    """Test that default_extra_headers provides defaults that can be overridden by request."""
    # Request has some headers
    mock_request.headers = {
        "content-type": "application/json",
        "x-custom-header": "request-value",
    }

    # Provider has default_extra_headers - these can be overridden by request
    default_extra_headers = {
        "X-Provider": "test-provider",
        "X-Organization": "test-org",
        "X-Custom-Header": "provider-value",
    }

    result = await build_next_request(
        mock_request, "http://example.com", "api", default_extra_headers=default_extra_headers
    )

    # Request headers should take precedence over default_extra_headers (case-insensitive)
    assert result.headers["x-custom-header"] == "request-value"  # From request, not default

    # Default extra headers should be included when not in request
    assert result.headers["x-provider"] == "test-provider"  # From default_extra_headers
    assert result.headers["x-organization"] == "test-org"  # From default_extra_headers


@pytest.mark.asyncio
async def test_build_next_request_with_required_extra_headers(mock_request):
    """Test that required_extra_headers cannot be overridden by request."""
    # Request has some headers
    mock_request.headers = {
        "content-type": "application/json",
        "x-custom-header": "request-value",
    }

    # Provider has required_extra_headers - these CANNOT be overridden
    required_extra_headers = {
        "X-Required": "required-value",
        "X-Custom-Header": "required-override",  # Should override request value
    }

    result = await build_next_request(
        mock_request, "http://example.com", "api", required_extra_headers=required_extra_headers
    )

    # Required extra headers should take precedence over request headers
    assert result.headers["x-custom-header"] == "required-override"  # From required, overrides request
    assert result.headers["x-required"] == "required-value"  # From required_extra_headers


@pytest.mark.asyncio
async def test_build_next_request_with_all_extra_fields(mock_request):
    """Test that default and required extra_body and extra_headers work together."""
    mock_request.body = AsyncMock(return_value=b'{"model": "gpt-4", "temperature": 0.9}')
    mock_request.headers = {"content-type": "application/json", "x-request": "from-request"}

    # default_extra_* provides fallbacks, required_extra_* enforces values
    default_extra_body = {"max_tokens": 100, "temperature": 0.5}  # temperature will be overridden by request
    default_extra_headers = {"X-Default": "default-val"}
    required_extra_body = {"stream": True}  # stream cannot be overridden
    required_extra_headers = {"X-Required": "required-val"}

    result = await build_next_request(
        mock_request,
        "http://example.com",
        "api",
        default_extra_body=default_extra_body,
        default_extra_headers=default_extra_headers,
        required_extra_body=required_extra_body,
        required_extra_headers=required_extra_headers,
    )

    result_body = _json_request_body(result)
    # Merge order: default_extra_body < request < required_extra_body
    assert result_body["model"] == "gpt-4"  # From request
    assert result_body["temperature"] == 0.9  # From request (overrides default)
    assert result_body["max_tokens"] == 100  # From default (not in request or required)
    assert result_body["stream"] is True  # From required (enforced)

    # Check header merging: default_extra_headers < request < required_extra_headers
    assert result.headers["x-default"] == "default-val"
    assert result.headers["x-request"] == "from-request"
    assert result.headers["x-required"] == "required-val"


@pytest.mark.asyncio
async def test_build_next_request_without_extra_fields(mock_request):
    """Test that build_next_request works without extra_body and extra_headers."""
    mock_request.body = AsyncMock(return_value=b'{"model": "gpt-4"}')

    result = await build_next_request(mock_request, "http://example.com", "api")

    result_body = _json_request_body(result)
    assert result_body == {"model": "gpt-4"}  # Only request body, no extras


@pytest.mark.asyncio
async def test_build_next_request_with_none_extra_fields(mock_request):
    """Test that build_next_request handles None values for extra_body and extra_headers."""
    mock_request.body = AsyncMock(return_value=b'{"model": "gpt-4"}')

    result = await build_next_request(
        mock_request,
        "http://example.com",
        "api",
        default_extra_body=None,
        default_extra_headers=None,
        required_extra_body=None,
        required_extra_headers=None,
    )

    result_body = _json_request_body(result)
    assert result_body == {"model": "gpt-4"}  # Only request body, no extras


@pytest.mark.asyncio
async def test_build_next_request_with_empty_extra_fields(mock_request):
    """Test that build_next_request handles empty dicts for extra_body and extra_headers."""
    mock_request.body = AsyncMock(return_value=b'{"model": "gpt-4"}')
    mock_request.headers = {"content-type": "application/json"}

    result = await build_next_request(
        mock_request,
        "http://example.com",
        "api",
        default_extra_body={},
        default_extra_headers={},
        required_extra_body={},
        required_extra_headers={},
    )

    result_body = _json_request_body(result)
    assert result_body == {"model": "gpt-4"}  # Only request body, no extras


@pytest.mark.asyncio
async def test_build_next_request_with_non_json_body_and_extra_body(mock_request):
    """Test that extra_body is ignored for non-JSON request bodies."""
    # Non-JSON body (e.g., plain text or binary)
    mock_request.body = AsyncMock(return_value=b"plain text data")

    default_extra_body = {"stream": True}
    required_extra_body = {"temperature": 0.7}

    result = await build_next_request(
        mock_request,
        "http://example.com",
        "api",
        default_extra_body=default_extra_body,
        required_extra_body=required_extra_body,
    )

    # Body should be preserved as-is (raw bytes) since it's not JSON
    assert result.body == b"plain text data"


@pytest.mark.asyncio
async def test_build_next_request_required_enforces_values(mock_request):
    """Test that required_extra_body enforces values even when request tries to override."""
    # Request body tries to set distillable_text=False
    mock_request.body = AsyncMock(return_value=b'{"model": "gpt-4", "enforce_distillable_text": false}')

    # Provider's required setting - this is the security use case
    required_extra_body = {"enforce_distillable_text": True}

    result = await build_next_request(
        mock_request, "http://example.com", "api", required_extra_body=required_extra_body
    )

    result_body = _json_request_body(result)

    # Required takes precedence - user cannot bypass this security setting
    assert result_body["enforce_distillable_text"] is True  # From required, not request
    assert result_body["model"] == "gpt-4"  # Request field preserved


@pytest.mark.parametrize(
    "host_url,trailing_uri,expected",
    [
        # No /v1 in host - keep trailing_uri as-is
        ("https://nim.internal:8000", "v1/chat/completions", "https://nim.internal:8000/v1/chat/completions"),
        ("https://localhost:8080", "v1/chat/completions", "https://localhost:8080/v1/chat/completions"),
        # Host ends with /v1 and trailing_uri starts with v1/ - strip duplicate
        ("https://api.openai.com/v1", "v1/chat/completions", "https://api.openai.com/v1/chat/completions"),
        ("https://api.openai.com/v1", "v1/models", "https://api.openai.com/v1/models"),
        ("https://api.openai.com/v1/", "v1/chat/completions", "https://api.openai.com/v1/chat/completions"),
        ("https://api.openai.com/v1//", "v1/chat/completions", "https://api.openai.com/v1/chat/completions"),
        # Host ends with /v1 but trailing_uri doesn't start with v1/ - no change
        ("https://api.openai.com/v1", "chat/completions", "https://api.openai.com/v1/chat/completions"),
        ("https://api.openai.com/v1", "", "https://api.openai.com/v1/"),
        # v1 in hostname should not affect normalization
        ("https://v1.api.example.com/v1", "v1/chat/completions", "https://v1.api.example.com/v1/chat/completions"),
        # trailing_uri exactly "v1" (no slash) - don't strip
        ("https://api.openai.com/v1", "v1", "https://api.openai.com/v1/v1"),
        # trailing_uri with leading slash should have slash stripped
        ("https://nim.internal:8000", "/v1/chat/completions", "https://nim.internal:8000/v1/chat/completions"),
        ("https://api.openai.com/v1", "/v1/chat/completions", "https://api.openai.com/v1/chat/completions"),
        ("https://localhost:8080", "/chat/completions", "https://localhost:8080/chat/completions"),
    ],
)
def test_normalize_proxy_url(host_url, trailing_uri, expected):
    """Test URL normalization handles duplicate /v1 paths and leading slashes correctly."""
    assert normalize_proxy_url(host_url, trailing_uri) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [401, 403, 404])
async def test_proxy_request_wraps_certain_errors_in_502(mock_proxy_client, next_request_info, status_code):
    """Test that certain backend errors (401/403/404) are wrapped in 502."""
    import aiohttp

    mock_response = Mock(spec=aiohttp.ClientResponse)
    mock_response.status = status_code
    mock_response.closed = False
    mock_response.headers = CIMultiDict(
        {"content-type": "application/json", "content-length": "24", "retry-after": "30"}
    )
    mock_response.read = AsyncMock(return_value=b'{"error": "model not found on backend"}')
    mock_proxy_client.request = AsyncMock(return_value=mock_response)

    with pytest.raises(HTTPException) as exc_info:
        await proxy_request(mock_proxy_client, next_request_info)

    assert exc_info.value.status_code == 502
    assert f"Backend returned {status_code}" in exc_info.value.detail
    assert "model not found on backend" in exc_info.value.detail
    assert exc_info.value.headers is not None
    assert exc_info.value.headers.get("retry-after") == "30"
    assert "content-type" not in exc_info.value.headers
    assert "content-length" not in exc_info.value.headers


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 422, 429, 500, 502, 503])
async def test_proxy_request_passes_through_other_backend_errors(mock_proxy_client, next_request_info, status_code):
    """Test that other backend errors (429, 422, 5xx, etc.) pass through with original status."""
    import aiohttp

    mock_response = Mock(spec=aiohttp.ClientResponse)
    mock_response.status = status_code
    mock_response.closed = False
    mock_response.headers = CIMultiDict(
        {"content-type": "application/json", "content-length": "24", "retry-after": "30"}
    )
    mock_response.read = AsyncMock(return_value=b'{"error": "rate limit exceeded"}')
    mock_proxy_client.request = AsyncMock(return_value=mock_response)

    with pytest.raises(HTTPException) as exc_info:
        await proxy_request(mock_proxy_client, next_request_info)

    assert exc_info.value.status_code == status_code
    assert "rate limit exceeded" in exc_info.value.detail
    assert exc_info.value.headers is not None
    assert exc_info.value.headers.get("retry-after") == "30"
    # The upstream body's framing headers describe a body we're not forwarding
    # (FastAPI generates its own JSON error body), so they must not leak through.
    assert "content-length" not in exc_info.value.headers
    assert "content-type" not in exc_info.value.headers


@pytest.mark.asyncio
async def test_proxy_request_drops_headers_named_by_connection_on_error(mock_proxy_client, next_request_info):
    """A backend can name additional hop-by-hop headers via Connection; those must not forward either."""
    import aiohttp

    mock_response = Mock(spec=aiohttp.ClientResponse)
    mock_response.status = 503
    mock_response.closed = False
    mock_response.headers = CIMultiDict(
        {
            "connection": "x-backend-only",
            "x-backend-only": "internal-secret",
            "retry-after": "30",
        }
    )
    mock_response.read = AsyncMock(return_value=b'{"error": "unavailable"}')
    mock_proxy_client.request = AsyncMock(return_value=mock_response)

    with pytest.raises(HTTPException) as exc_info:
        await proxy_request(mock_proxy_client, next_request_info)

    assert exc_info.value.headers is not None
    assert exc_info.value.headers.get("retry-after") == "30"
    assert "connection" not in exc_info.value.headers
    assert "x-backend-only" not in exc_info.value.headers


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [200, 201, 204, 301, 302])
async def test_proxy_request_passes_through_success_responses(
    mock_proxy_client, mock_proxy_response, next_request_info, status_code
):
    """Test that success responses (2xx/3xx) are passed through unchanged."""
    mock_proxy_response.status = status_code

    response = await proxy_request(mock_proxy_client, next_request_info)

    assert isinstance(response, StreamingResponse)
    assert response.status_code == status_code


# =============================================================================
# Compression Passthrough Tests
# =============================================================================


@pytest.mark.asyncio
async def test_accept_encoding_forwarded_to_upstream(mock_request):
    """Test that client's Accept-Encoding header is forwarded to upstream."""
    mock_request.headers = {
        "host": "localhost:8080",
        "accept-encoding": "gzip, deflate, br",
    }

    result = await build_next_request(mock_request, "http://example.com", "v1/models")

    assert "accept-encoding" in result.headers
    assert result.headers["accept-encoding"] == "gzip, deflate, br"


@pytest.mark.asyncio
async def test_compressed_response_bytes_passed_through(mock_proxy_client, mock_proxy_response, next_request_info):
    """Test that compressed response bytes pass through unchanged.

    This verifies the gateway doesn't decompress responses, allowing compressed
    data to flow transparently from upstream to client.
    """
    from multidict import CIMultiDict, CIMultiDictProxy

    # Simulate a gzip-compressed response from upstream
    original_data = b'{"data": "test"}'
    compressed_data = gzip.compress(original_data)

    mock_proxy_response.headers = CIMultiDictProxy(
        CIMultiDict(
            [
                ("content-type", "application/json"),
                ("content-encoding", "gzip"),
            ]
        )
    )

    async def compressed_chunk_iterator():
        yield compressed_data

    mock_proxy_response.content.iter_chunked = Mock(return_value=compressed_chunk_iterator())

    response = await proxy_request(mock_proxy_client, next_request_info)

    # Verify Content-Encoding header is preserved
    assert response.headers.get("content-encoding") == "gzip"

    # Verify compressed bytes pass through unchanged
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)
    received_data = b"".join(chunks)

    assert received_data == compressed_data
    assert gzip.decompress(received_data) == original_data


@pytest.mark.asyncio
async def test_fetch_proxy_response_decodes_gzip_json(httpserver: HTTPServer):
    """The middleware request overrides a non-decompressing session for gzip JSON."""
    expected = {"id": "response", "choices": [{"message": {"content": "ok"}}]}
    httpserver.expect_request("/gzip-json").respond_with_data(
        gzip.compress(json.dumps(expected).encode()),
        headers={"content-encoding": "gzip"},
        content_type="application/json",
    )
    request_info = NextRequestInfo(
        url=httpserver.url_for("/gzip-json"),
        body=None,
        headers=CIMultiDict({"accept-encoding": "gzip"}),
        method="GET",
        query_params={},
    )

    async with aiohttp.ClientSession(auto_decompress=False) as http_client:
        result, _, status_code = await fetch_proxy_response(http_client, request_info)

    assert status_code == 200
    assert result == expected


@pytest.mark.asyncio
async def test_fetch_proxy_response_reports_payload_decoding_failure(
    mock_proxy_client, mock_proxy_response, next_request_info
):
    """aiohttp payload decoding failures are reported as gateway processing errors."""
    mock_proxy_response.read = AsyncMock(side_effect=aiohttp.ClientPayloadError("invalid gzip body"))

    with pytest.raises(HTTPException) as exc_info:
        await fetch_proxy_response(mock_proxy_client, next_request_info)

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "Inference Gateway could not decode the upstream response body."


@pytest.mark.asyncio
async def test_fetch_proxy_response_reports_json_processing_failure(
    mock_proxy_client, mock_proxy_response, next_request_info
):
    """A JSON parsing failure describes gateway processing instead of blaming the backend."""
    mock_proxy_response.read = AsyncMock(return_value=b"not-json")

    with pytest.raises(HTTPException) as exc_info:
        await fetch_proxy_response(mock_proxy_client, next_request_info)

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == (
        "Inference Gateway could not parse the upstream response as JSON for inference middleware processing."
    )


@pytest.mark.asyncio
async def test_fetch_proxy_response_forwards_retry_after_on_backend_error(
    mock_proxy_client, mock_proxy_response, next_request_info
):
    """A 429/503 from the backend keeps its Retry-After header so the evaluator's resilience layer can read it."""
    mock_proxy_response.status = 429
    mock_proxy_response.headers = CIMultiDict(
        [("content-type", "application/json"), ("content-length", "24"), ("retry-after", "30")]
    )
    mock_proxy_response.read = AsyncMock(return_value=b'{"error": "rate limited"}')

    with pytest.raises(HTTPException) as exc_info:
        await fetch_proxy_response(mock_proxy_client, next_request_info)

    assert exc_info.value.status_code == 429
    assert exc_info.value.headers is not None
    assert exc_info.value.headers.get("retry-after") == "30"
    assert "backend returned" not in exc_info.value.detail.lower()
    # The upstream body's framing headers describe a body we're not forwarding
    # (FastAPI generates its own JSON error body), so they must not leak through.
    assert "content-length" not in exc_info.value.headers
    assert "content-type" not in exc_info.value.headers


@pytest.mark.asyncio
async def test_fetch_proxy_response_decodes_gzip_sse(httpserver: HTTPServer):
    """The middleware path keeps streaming while aiohttp decodes gzip SSE."""
    expected = [
        {"id": "chunk-1", "choices": [{"delta": {"content": "hello"}}]},
        {"id": "chunk-2", "choices": [{"delta": {"content": " world"}}]},
    ]
    encoded = gzip.compress(
        ("\n".join(f"data: {json.dumps(chunk)}" for chunk in expected) + "\ndata: [DONE]\n").encode()
    )
    httpserver.expect_request("/gzip-sse").respond_with_data(
        encoded,
        headers={"content-encoding": "gzip"},
        content_type="text/event-stream",
    )
    request_info = NextRequestInfo(
        url=httpserver.url_for("/gzip-sse"),
        body=None,
        headers=CIMultiDict({"accept-encoding": "gzip"}),
        method="GET",
        query_params={},
    )

    async with aiohttp.ClientSession(auto_decompress=False) as http_client:
        result, _, status_code = await fetch_proxy_response(http_client, request_info)
        assert status_code == 200
        assert not isinstance(result, dict)
        parsed = [chunk async for chunk in result]

    assert parsed == expected


# ---------------------------------------------------------------------------
# _parse_sse_stream tests
# ---------------------------------------------------------------------------


def _make_sse_response(*lines: str, encoding: str = "utf-8") -> Mock:
    """Build a mock aiohttp.ClientResponse whose content yields *lines* as bytes.

    Each element of *lines* is one raw SSE text line (without trailing newline).
    The mock's ``content.iter_any()`` yields them as separate byte chunks, simulating
    the arbitrary fragmentation produced by ``aiohttp.StreamReader.iter_any()``.
    """
    raw = "\n".join(lines) + "\n"
    chunks = [raw.encode(encoding)]

    async def _iter_any():
        for chunk in chunks:
            yield chunk

    response = Mock()
    response.content = Mock()
    response.content.iter_any = Mock(return_value=_iter_any())
    response.close = Mock()
    response.closed = False
    response.connection = None
    return response


def _make_sse_response_fragmented(*lines: str) -> Mock:
    """Same as _make_sse_response but fragments output byte-by-byte to stress the buffer."""
    raw = "\n".join(lines) + "\n"

    async def _iter_any():
        for byte in raw.encode():
            yield bytes([byte])

    response = Mock()
    response.content = Mock()
    response.content.iter_any = Mock(return_value=_iter_any())
    response.close = Mock()
    response.closed = False
    response.connection = None
    return response


@pytest.mark.asyncio
async def test_parse_sse_basic():
    """Each data: line is decoded and yielded as a dict."""
    chunk1 = {"id": "c1", "choices": [{"delta": {"content": "Hello"}}]}
    chunk2 = {"id": "c1", "choices": [{"delta": {"content": " world"}}]}
    response = _make_sse_response(
        f"data: {json.dumps(chunk1)}",
        f"data: {json.dumps(chunk2)}",
        "data: [DONE]",
    )
    results = [chunk async for chunk in _parse_sse_stream(response)]
    assert results == [chunk1, chunk2]


@pytest.mark.asyncio
async def test_parse_sse_done_terminates_early():
    """data: [DONE] stops the stream; subsequent lines are not yielded."""
    chunk1 = {"id": "c1", "choices": [{"delta": {"content": "hi"}}]}
    after_done = {"id": "c2", "choices": [{"delta": {"content": "ignored"}}]}
    response = _make_sse_response(
        f"data: {json.dumps(chunk1)}",
        "data: [DONE]",
        f"data: {json.dumps(after_done)}",
    )
    results = [chunk async for chunk in _parse_sse_stream(response)]
    assert results == [chunk1]


@pytest.mark.asyncio
async def test_parse_sse_skips_non_data_lines():
    """Comment lines, event names, and blank lines are silently skipped."""
    chunk1 = {"id": "c1", "choices": [{"delta": {"content": "ok"}}]}
    response = _make_sse_response(
        ": this is a comment",
        "event: message",
        "",
        f"data: {json.dumps(chunk1)}",
        "data: [DONE]",
    )
    results = [chunk async for chunk in _parse_sse_stream(response)]
    assert results == [chunk1]


@pytest.mark.asyncio
async def test_parse_sse_skips_malformed_json():
    """data: lines with invalid JSON are silently skipped; valid ones still arrive."""
    chunk1 = {"id": "c1", "choices": [{"delta": {"content": "ok"}}]}
    response = _make_sse_response(
        "data: {not valid json",
        f"data: {json.dumps(chunk1)}",
        "data: [DONE]",
    )
    results = [chunk async for chunk in _parse_sse_stream(response)]
    assert results == [chunk1]


@pytest.mark.asyncio
async def test_parse_sse_handles_crlf_line_endings():
    """\\r\\n line endings (some providers) are handled correctly."""
    chunk1 = {"id": "c1", "choices": [{"delta": {"content": "hi"}}]}
    raw = f"data: {json.dumps(chunk1)}\r\ndata: [DONE]\r\n"

    async def _iter_any():
        yield raw.encode()

    response = Mock()
    response.content = Mock()
    response.content.iter_any = Mock(return_value=_iter_any())
    response.close = Mock()
    response.connection = None

    results = [chunk async for chunk in _parse_sse_stream(response)]
    assert results == [chunk1]


@pytest.mark.asyncio
async def test_parse_sse_handles_lines_split_across_read_boundaries():
    """SSE lines that are split across iter_any() chunks are reassembled correctly."""
    chunk1 = {"id": "c1", "choices": [{"delta": {"content": "hello"}}]}
    response = _make_sse_response_fragmented(
        f"data: {json.dumps(chunk1)}",
        "data: [DONE]",
    )
    results = [chunk async for chunk in _parse_sse_stream(response)]
    assert results == [chunk1]


@pytest.mark.asyncio
async def test_parse_sse_empty_stream_yields_nothing():
    """An empty response (no data: lines before [DONE]) yields no chunks."""
    response = _make_sse_response("data: [DONE]")
    results = [chunk async for chunk in _parse_sse_stream(response)]
    assert results == []
    response.close.assert_called_once()


# ---------------------------------------------------------------------------
# AIRCORE-???: response model-field rewrite (served_model_name -> entity ref)
# ---------------------------------------------------------------------------
#
# After proxying, the user-facing response body must surface the model entity
# reference, not the upstream's served_model_name. The pipeline rewrites the
# field both for buffered (dict) responses and for each chunk of an SSE stream,
# *before* response middleware runs so plugins observe the entity-keyed view.


def test_rewrite_model_field_top_level_match():
    """Standard OpenAI Chat / Completions / non-streaming Anthropic shape."""
    payload = {"id": "x", "model": "served-name", "choices": []}
    _rewrite_model_field(payload, "served-name", "ws/entity")
    assert payload["model"] == "ws/entity"


def test_rewrite_model_field_top_level_no_match_passthrough():
    """A different value at ``model`` is left alone (strict equality)."""
    payload = {"id": "x", "model": "something-else", "choices": []}
    _rewrite_model_field(payload, "served-name", "ws/entity")
    assert payload["model"] == "something-else"


def test_rewrite_model_field_absent_field_is_noop():
    """A chunk without a ``model`` field is a no-op (e.g. Anthropic ping/delta events)."""
    payload = {"type": "ping"}
    _rewrite_model_field(payload, "served-name", "ws/entity")
    assert payload == {"type": "ping"}


def test_rewrite_model_field_nested_message_anthropic_message_start():
    """Anthropic ``message_start`` carries ``model`` inside the embedded ``Message``."""
    payload = {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "served-name",
            "content": [],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    }
    _rewrite_model_field(payload, "served-name", "ws/entity")
    assert payload["message"]["model"] == "ws/entity"


def test_rewrite_model_field_nested_response_openai_responses_event():
    """OpenAI Responses API events carry ``model`` inside the embedded ``Response``."""
    payload = {"type": "response.created", "response": {"id": "resp_1", "model": "served-name"}}
    _rewrite_model_field(payload, "served-name", "ws/entity")
    assert payload["response"]["model"] == "ws/entity"


def test_rewrite_model_field_handles_non_dict_gracefully():
    """A non-dict payload (string, list, None) is a no-op rather than a crash."""
    _rewrite_model_field("not-a-dict", "served-name", "ws/entity")  # type: ignore[arg-type]
    _rewrite_model_field(None, "served-name", "ws/entity")  # type: ignore[arg-type]
    _rewrite_model_field([{"model": "served-name"}], "served-name", "ws/entity")  # type: ignore[arg-type]
    # Nothing to assert — we just need it to not raise.


def test_rewrite_model_field_partial_match_only():
    """Equality only — substring matches must not trigger a rewrite."""
    payload = {"model": "served-name-suffixed"}
    _rewrite_model_field(payload, "served-name", "ws/entity")
    assert payload["model"] == "served-name-suffixed"


# ---------------------------------------------------------------------------
# Error-body rewrite: served_model_name must not leak in 4xx/5xx error cases,
# matching the happy-path rewrite. Covers structured JSON error bodies and
# free-form plain-text bodies.
# ---------------------------------------------------------------------------


def test_rewrite_error_body_json_structured_model_field():
    """A structured ``model`` field in a JSON error body is rewritten."""
    body = json.dumps({"error": {"message": "boom"}, "model": "model-1"})
    out = _rewrite_model_field_in_error_body(body, "model-1", "default/model-1")
    assert json.loads(out)["model"] == "default/model-1"


def test_rewrite_error_body_no_double_application_when_restored_contains_served():
    """Regression: the served name is applied exactly once even when the restored id
    *contains* it (the common ``model-1`` -> ``default/model-1`` case). A parse-then-
    blanket-replace approach would re-hit the already-rewritten value and produce
    ``default/default/model-1``; a single pass over the original must not.

    Covers a served name appearing BOTH as a structured ``model`` field and inside a
    free-form error message string in the same body.
    """
    served, restored = "model-1", "default/model-1"
    body = json.dumps({"model": served, "error": {"message": f"{served} is unavailable"}})
    out = _rewrite_model_field_in_error_body(body, served, restored)

    parsed = json.loads(out)  # still valid JSON
    assert parsed["model"] == restored
    assert parsed["error"]["message"] == f"{restored} is unavailable"
    # The doubled-prefix mangling must not appear anywhere.
    assert "default/default/" not in out


def test_rewrite_error_body_json_model_in_message_string():
    """OpenAI-shaped 'model not found' errors carry the served name inside the
    error *message* string, not a structured field; it must still be scrubbed."""
    body = json.dumps(
        {
            "error": {
                "message": "The model `model-1` does not exist or you do not have access to it.",
                "type": "invalid_request_error",
                "code": "model_not_found",
            }
        }
    )
    out = _rewrite_model_field_in_error_body(body, "model-1", "default/model-1")
    parsed = json.loads(out)  # substring swap kept the body valid JSON
    assert "default/model-1" in parsed["error"]["message"]
    assert "`model-1`" not in parsed["error"]["message"]


def test_rewrite_error_body_plain_text_substring_replace():
    """A non-JSON plain-text error body is handled by substring replacement."""
    body = "Upstream error: model model-1 is overloaded, retry later."
    out = _rewrite_model_field_in_error_body(body, "model-1", "default/model-1")
    assert out == "Upstream error: model default/model-1 is overloaded, retry later."


def test_rewrite_error_body_empty_is_noop():
    """An empty error body is returned unchanged."""
    assert _rewrite_model_field_in_error_body("", "model-1", "default/model-1") == ""


def test_rewrite_error_body_empty_served_name_is_noop():
    """An empty *served_model_name* is a no-op: ``str.replace("", ...)`` would otherwise
    splice the restored id between every character and corrupt the whole body."""
    body = json.dumps({"error": {"message": "boom"}})
    assert _rewrite_model_field_in_error_body(body, "", "default/model-1") == body


def test_rewrite_error_body_no_match_passthrough():
    """A body that never mentions the served name is returned unchanged."""
    body = json.dumps({"error": {"message": "rate limit exceeded"}})
    assert _rewrite_model_field_in_error_body(body, "model-1", "default/model-1") == body


def test_rewrite_error_body_skips_when_name_has_json_unsafe_char():
    """If a model name isn't a plain entity-ref/model id (e.g. it carries a quote or
    brace), the blind substring swap could corrupt a JSON body — so the scrub is
    skipped entirely and the body (served name included) is returned untouched.
    """
    body = json.dumps({"model": 'weird"name', "error": {"message": 'weird"name failed'}})
    # served name contains a double-quote -> outside ENTITY_REF_PATTERN -> skip
    assert _rewrite_model_field_in_error_body(body, 'weird"name', "default/model-1") == body
    # restored id malformed (contains a brace) -> also skip
    body2 = json.dumps({"model": "model-1"})
    assert _rewrite_model_field_in_error_body(body2, "model-1", "default/{model-1}") == body2


def test_rewrite_error_body_scalar_json_string_is_scrubbed():
    """A JSON scalar (a bare quoted string body) is still scrubbed by the single pass."""
    body = json.dumps("model-1 is unavailable")
    out = _rewrite_model_field_in_error_body(body, "model-1", "default/model-1")
    assert out == json.dumps("default/model-1 is unavailable")


def _error_response(status: int, body: bytes, *, content_type: str = "application/json") -> Mock:
    """Build a mock aiohttp error response returning *body* with *status*."""
    resp = Mock(spec=aiohttp.ClientResponse)
    resp.status = status
    resp.closed = False
    resp.headers = CIMultiDict({"content-type": content_type})
    resp.read = AsyncMock(return_value=body)
    return resp


@pytest.mark.asyncio
async def test_fetch_proxy_response_rewrites_openai_model_not_found_400(mock_proxy_client, next_request_info):
    """OpenAI 400 model-not-found: the served name is scrubbed from the error detail."""
    body = json.dumps(
        {
            "error": {
                "message": "The model `served-name` does not exist or you do not have access to it.",
                "type": "invalid_request_error",
                "code": "model_not_found",
            }
        }
    ).encode()
    mock_proxy_client.request = AsyncMock(return_value=_error_response(400, body))

    with pytest.raises(HTTPException) as exc_info:
        await fetch_proxy_response(
            mock_proxy_client,
            next_request_info,
            served_model_name="served-name",
            restored_model_id="ws/entity",
        )

    assert exc_info.value.status_code == 400
    assert "served-name" not in exc_info.value.detail
    assert "ws/entity" in exc_info.value.detail


@pytest.mark.asyncio
async def test_fetch_proxy_response_rewrites_anthropic_error_400(mock_proxy_client, next_request_info):
    """Anthropic 400: the served name is scrubbed from the Anthropic-shaped error body."""
    body = json.dumps(
        {
            "type": "error",
            "error": {
                "type": "not_found_error",
                "message": "model: served-name",
            },
        }
    ).encode()
    mock_proxy_client.request = AsyncMock(return_value=_error_response(400, body))

    with pytest.raises(HTTPException) as exc_info:
        await fetch_proxy_response(
            mock_proxy_client,
            next_request_info,
            served_model_name="served-name",
            restored_model_id="ws/entity",
        )

    assert exc_info.value.status_code == 400
    assert "served-name" not in exc_info.value.detail
    assert "ws/entity" in exc_info.value.detail


@pytest.mark.asyncio
async def test_fetch_proxy_response_rewrites_rate_limit_429(mock_proxy_client, next_request_info):
    """429 rate-limit: served name scrubbed from the message, status passed through."""
    body = json.dumps(
        {"error": {"message": "Rate limit reached for model served-name", "type": "rate_limit_error"}}
    ).encode()
    mock_proxy_client.request = AsyncMock(return_value=_error_response(429, body))

    with pytest.raises(HTTPException) as exc_info:
        await fetch_proxy_response(
            mock_proxy_client,
            next_request_info,
            served_model_name="served-name",
            restored_model_id="ws/entity",
        )

    assert exc_info.value.status_code == 429
    assert "served-name" not in exc_info.value.detail
    assert "ws/entity" in exc_info.value.detail


@pytest.mark.asyncio
async def test_fetch_proxy_response_rewrites_plain_text_5xx(mock_proxy_client, next_request_info):
    """OpenAI plain-text 5xx: substring replacement scrubs the served name."""
    body = b"Internal server error while serving model served-name; please retry."
    mock_proxy_client.request = AsyncMock(return_value=_error_response(500, body, content_type="text/plain"))

    with pytest.raises(HTTPException) as exc_info:
        await fetch_proxy_response(
            mock_proxy_client,
            next_request_info,
            served_model_name="served-name",
            restored_model_id="ws/entity",
        )

    assert exc_info.value.status_code == 500
    assert "served-name" not in exc_info.value.detail
    assert "ws/entity" in exc_info.value.detail


@pytest.mark.asyncio
async def test_fetch_proxy_response_rewrites_401_wrapped_in_502(mock_proxy_client, next_request_info):
    """A 401 is wrapped as 502; the served name is still scrubbed from the wrapped detail."""
    body = json.dumps({"error": {"message": "invalid key for served-name"}}).encode()
    mock_proxy_client.request = AsyncMock(return_value=_error_response(401, body))

    with pytest.raises(HTTPException) as exc_info:
        await fetch_proxy_response(
            mock_proxy_client,
            next_request_info,
            served_model_name="served-name",
            restored_model_id="ws/entity",
        )

    assert exc_info.value.status_code == 502
    assert "served-name" not in exc_info.value.detail
    assert "ws/entity" in exc_info.value.detail


@pytest.mark.asyncio
async def test_fetch_proxy_response_error_rewrite_noop_without_names(mock_proxy_client, next_request_info):
    """With no served/restored names supplied, the error body is passed through verbatim."""
    body = json.dumps({"error": {"message": "served-name is down"}}).encode()
    mock_proxy_client.request = AsyncMock(return_value=_error_response(500, body))

    with pytest.raises(HTTPException) as exc_info:
        await fetch_proxy_response(mock_proxy_client, next_request_info)

    assert exc_info.value.status_code == 500
    assert "served-name" in exc_info.value.detail


@pytest.mark.asyncio
async def test_virtual_model_proxy_scrubs_served_name_on_upstream_error(mock_proxy_client, mock_proxy_response):
    """End-to-end error path: an upstream 400 flows through virtual_model_proxy and the
    HTTPException detail carries the entity ref, never the served_model_name. This covers
    the streaming-request case too: a non-2xx status is detected before the SSE branch,
    so a request that asked for ``stream: True`` but got an error is scrubbed the same way.
    """
    workspace = "e2e-test"
    vm_name = "error-rewrite-router"
    model_entity_id = f"{workspace}/meta_llama-3.2-1b-instruct"
    served_model_name = "meta/llama-3.2-1b-instruct"

    error_body = json.dumps(
        {
            "error": {
                "message": f"The model `{served_model_name}` does not exist or you do not have access to it.",
                "type": "invalid_request_error",
                "code": "model_not_found",
            }
        }
    ).encode()
    mock_proxy_response.status = 400
    mock_proxy_response.read = AsyncMock(return_value=error_body)

    model_cache = ModelCache()
    model_cache.update_model_info(
        ModelProviderInfo(
            model_provider=ModelProvider(
                workspace="default",
                name="nim",
                host_url="http://nim.local",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                served_models=[
                    ServedModelMapping(
                        model_entity_id=model_entity_id,
                        served_model_name=served_model_name,
                    )
                ],
                status="READY",
            )
        )
    )
    model_cache.rebuild_model_entity_map()

    request = Mock(spec=Request)
    request.method = "POST"
    request.headers = {"content-type": "application/json"}
    request.query_params = {}

    with pytest.raises(HTTPException) as exc_info:
        await virtual_model_proxy(
            request=request,
            workspace=workspace,
            vm_name=vm_name,
            virtual_model=SDKVirtualModel(
                id=f"{workspace}/{vm_name}",
                entity_id=f"{workspace}/{vm_name}",
                name=vm_name,
                workspace=workspace,
                parent=workspace,
                db_version=1,
                created_at="2026-01-01T00:00:00Z",
                updated_at="2026-01-01T00:00:00Z",
                default_model_entity=model_entity_id,
            ),
            trailing_uri="v1/chat/completions",
            json_body={"model": vm_name, "stream": True, "messages": [{"role": "user", "content": "hi"}]},
            http_client=mock_proxy_client,
            model_cache=model_cache,
            registry=MiddlewareRegistry(),
        )

    assert exc_info.value.status_code == 400
    assert served_model_name not in exc_info.value.detail
    assert model_entity_id in exc_info.value.detail


@pytest.mark.asyncio
async def test_rewrite_model_field_in_stream_openai_chunks_every_chunk():
    """Every OpenAI ChatCompletionChunk in a stream gets its top-level ``model`` rewritten."""

    async def _src() -> AsyncIterator[dict[str, Any]]:
        for delta in ["he", "ll", "o"]:
            yield {
                "id": "chunk",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "served-name",
                "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
            }

    chunks: list[dict[str, Any]] = []
    async for chunk in _rewrite_model_field_in_stream(_src(), "served-name", "ws/entity"):
        chunks.append(chunk)

    assert len(chunks) == 3
    assert all(chunk["model"] == "ws/entity" for chunk in chunks)


@pytest.mark.asyncio
async def test_rewrite_model_field_in_stream_anthropic_message_start_only():
    """Anthropic streams: only ``message_start.message.model`` is rewritten; other event
    types (``content_block_*``, ``message_delta``, ``message_stop``, ``ping``) have no
    ``model`` field and pass through untouched.
    """
    source_chunks: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "served-name",
                "content": [],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        },
        {"type": "ping"},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ]

    async def _src() -> AsyncIterator[dict[str, Any]]:
        for chunk in source_chunks:
            yield chunk

    out: list[dict[str, Any]] = []
    async for chunk in _rewrite_model_field_in_stream(_src(), "served-name", "ws/entity"):
        out.append(chunk)

    # message_start carries the rewrite.
    assert out[0]["message"]["model"] == "ws/entity"
    # Every other event type is byte-for-byte equal to its source (except message_start
    # which we mutated in place above).
    for src_chunk, out_chunk in zip(source_chunks[1:], out[1:], strict=True):
        assert out_chunk == src_chunk


@pytest.mark.asyncio
async def test_virtual_model_proxy_rewrites_response_model_non_streaming(mock_proxy_client, mock_proxy_response):
    """End-to-end: a non-streaming OpenAI response from the upstream has the served-model
    name rewritten to the qualified workspace/entity reference before it reaches the
    user. The body the user sees never contains ``served_model_name``.
    """
    workspace = "e2e-test"
    vm_name = "rewrite-router"
    model_entity_id = f"{workspace}/meta_llama-3.2-1b-instruct"
    served_model_name = "meta/llama-3.2-1b-instruct"

    mock_proxy_response.read = AsyncMock(
        return_value=json.dumps(
            {
                "id": "resp",
                "object": "chat.completion",
                "created": 1,
                "model": served_model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
            }
        ).encode()
    )

    model_cache = ModelCache()
    model_cache.update_model_info(
        ModelProviderInfo(
            model_provider=ModelProvider(
                workspace="default",
                name="nim",
                host_url="http://nim.local",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                served_models=[
                    ServedModelMapping(
                        model_entity_id=model_entity_id,
                        served_model_name=served_model_name,
                    )
                ],
                status="READY",
            )
        )
    )
    model_cache.rebuild_model_entity_map()

    request = Mock(spec=Request)
    request.method = "POST"
    request.headers = {"content-type": "application/json"}
    request.query_params = {}

    response = await virtual_model_proxy(
        request=request,
        workspace=workspace,
        vm_name=vm_name,
        virtual_model=SDKVirtualModel(
            id=f"{workspace}/{vm_name}",
            entity_id=f"{workspace}/{vm_name}",
            name=vm_name,
            workspace=workspace,
            parent=workspace,
            db_version=1,
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            default_model_entity=model_entity_id,
        ),
        trailing_uri="v1/chat/completions",
        json_body={"model": vm_name, "messages": [{"role": "user", "content": "hi"}]},
        http_client=mock_proxy_client,
        model_cache=model_cache,
        registry=MiddlewareRegistry(),
    )

    body = json.loads(await _read_streaming_response(cast(StreamingResponse, response)))
    assert body["model"] == model_entity_id
    assert body["model"] != served_model_name


@pytest.mark.asyncio
async def test_virtual_model_proxy_rewrites_response_model_openai_streaming(mock_proxy_client, mock_proxy_response):
    """End-to-end OpenAI streaming: every emitted chunk's ``model`` field is rewritten."""
    workspace = "e2e-test"
    vm_name = "rewrite-router-stream"
    model_entity_id = f"{workspace}/meta_llama-3.2-1b-instruct"
    served_model_name = "meta/llama-3.2-1b-instruct"

    sse_chunks = [
        {
            "id": "chunk-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": served_model_name,
            "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
        },
        {
            "id": "chunk-2",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": served_model_name,
            "choices": [{"index": 0, "delta": {"content": "!"}, "finish_reason": "stop"}],
        },
    ]
    sse_payload = ("\n".join(f"data: {json.dumps(chunk)}" for chunk in sse_chunks) + "\ndata: [DONE]\n").encode()

    async def _iter_any():
        yield sse_payload

    mock_proxy_response.headers = CIMultiDictProxy(CIMultiDict([("content-type", "text/event-stream")]))
    mock_proxy_response.content.iter_any = Mock(return_value=_iter_any())

    model_cache = ModelCache()
    model_cache.update_model_info(
        ModelProviderInfo(
            model_provider=ModelProvider(
                workspace="default",
                name="nim",
                host_url="http://nim.local",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                served_models=[
                    ServedModelMapping(
                        model_entity_id=model_entity_id,
                        served_model_name=served_model_name,
                    )
                ],
                status="READY",
            )
        )
    )
    model_cache.rebuild_model_entity_map()

    request = Mock(spec=Request)
    request.method = "POST"
    request.headers = {"content-type": "application/json"}
    request.query_params = {}

    response = await virtual_model_proxy(
        request=request,
        workspace=workspace,
        vm_name=vm_name,
        virtual_model=SDKVirtualModel(
            id=f"{workspace}/{vm_name}",
            entity_id=f"{workspace}/{vm_name}",
            name=vm_name,
            workspace=workspace,
            parent=workspace,
            db_version=1,
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            default_model_entity=model_entity_id,
        ),
        trailing_uri="v1/chat/completions",
        json_body={
            "model": vm_name,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
        http_client=mock_proxy_client,
        model_cache=model_cache,
        registry=MiddlewareRegistry(),
    )

    body = (await _read_streaming_response(cast(StreamingResponse, response))).decode()

    # The wire-level body never carries the served name.
    assert served_model_name not in body
    # Every original chunk has been re-emitted with the entity ref instead.
    assert body.count(f'"model": "{model_entity_id}"') == len(sse_chunks)
    assert "data: [DONE]" in body


@pytest.mark.asyncio
async def test_virtual_model_proxy_rewrites_response_model_anthropic_streaming(mock_proxy_client, mock_proxy_response):
    """Anthropic streaming: the ``message_start`` event has its embedded ``message.model``
    rewritten; ``ping``/``content_block_*``/``message_delta``/``message_stop`` events
    carry no ``model`` field and pass through unchanged.
    """
    workspace = "e2e-test"
    vm_name = "anthropic-rewrite-router"
    model_entity_id = f"{workspace}/claude-sonnet"
    served_model_name = "claude-3-5-sonnet-20241022"

    sse_events: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": served_model_name,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        },
        {"type": "ping"},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    ]
    sse_payload = ("\n".join(f"data: {json.dumps(ev)}" for ev in sse_events) + "\ndata: [DONE]\n").encode()

    async def _iter_any():
        yield sse_payload

    mock_proxy_response.headers = CIMultiDictProxy(CIMultiDict([("content-type", "text/event-stream")]))
    mock_proxy_response.content.iter_any = Mock(return_value=_iter_any())

    model_cache = ModelCache()
    model_cache.update_model_info(
        ModelProviderInfo(
            model_provider=ModelProvider(
                workspace="default",
                name="anthropic",
                host_url="http://anthropic.local",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                served_models=[
                    ServedModelMapping(
                        model_entity_id=model_entity_id,
                        served_model_name=served_model_name,
                    )
                ],
                status="READY",
            )
        )
    )
    model_cache.rebuild_model_entity_map()

    request = Mock(spec=Request)
    request.method = "POST"
    request.headers = {"content-type": "application/json"}
    request.query_params = {}

    response = await virtual_model_proxy(
        request=request,
        workspace=workspace,
        vm_name=vm_name,
        virtual_model=SDKVirtualModel(
            id=f"{workspace}/{vm_name}",
            entity_id=f"{workspace}/{vm_name}",
            name=vm_name,
            workspace=workspace,
            parent=workspace,
            db_version=1,
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            default_model_entity=model_entity_id,
        ),
        trailing_uri="v1/messages",
        json_body={
            "model": vm_name,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
        http_client=mock_proxy_client,
        model_cache=model_cache,
        registry=MiddlewareRegistry(),
    )

    body = (await _read_streaming_response(cast(StreamingResponse, response))).decode()

    # Served name is fully scrubbed from the wire. Entity ref appears exactly once
    # (only on message_start; other events have no ``model`` field).
    assert served_model_name not in body
    assert body.count(f'"model": "{model_entity_id}"') == 1
    # Pass-through events that carry no ``model`` field still appear in the output.
    assert '"type": "ping"' in body
    assert '"type": "content_block_delta"' in body
    assert '"type": "message_stop"' in body


@pytest.mark.asyncio
async def test_virtual_model_proxy_rewrites_response_model_to_post_middleware_entity(
    mock_proxy_client, mock_proxy_response
):
    """When request middleware rewrites ``body["model"]`` to a different entity ref,
    the response rewrite uses the *post-middleware* entity ref, not the original VM
    name or ``default_model_entity`` value.
    """

    class _RoutingPlugin(NemoInferenceMiddleware):
        async def process_request(
            self,
            ctx: InferenceMiddlewareContext,
            request: InferenceRequest,
            middleware_config: object,
        ) -> InferenceRequest:
            return InferenceRequest(
                body={**request.body, "model": "e2e-test/post-middleware-entity"},
                headers=request.headers,
                path=request.path,
            )

    workspace = "e2e-test"
    vm_name = "routing-vm"
    routed_entity_id = "e2e-test/post-middleware-entity"
    served_model_name = "post-middleware-served-name"

    mock_proxy_response.read = AsyncMock(
        return_value=json.dumps(
            {
                "id": "resp",
                "object": "chat.completion",
                "created": 1,
                "model": served_model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
            }
        ).encode()
    )

    model_cache = ModelCache()
    model_cache.update_model_info(
        ModelProviderInfo(
            model_provider=ModelProvider(
                workspace="default",
                name="nim",
                host_url="http://nim.local",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                served_models=[
                    ServedModelMapping(
                        model_entity_id=routed_entity_id,
                        served_model_name=served_model_name,
                    )
                ],
                status="READY",
            )
        )
    )
    model_cache.rebuild_model_entity_map()

    registry = MiddlewareRegistry(plugins={"routing-plugin": _RoutingPlugin()})
    registry.request_middleware_calls[(workspace, vm_name)] = [
        ResolvedMiddlewareCall(plugin_name="routing-plugin", config_type="t", resolved_config={})
    ]

    request = Mock(spec=Request)
    request.method = "POST"
    request.headers = {"content-type": "application/json"}
    request.query_params = {}

    response = await virtual_model_proxy(
        request=request,
        workspace=workspace,
        vm_name=vm_name,
        virtual_model=SDKVirtualModel(
            id=f"{workspace}/{vm_name}",
            entity_id=f"{workspace}/{vm_name}",
            name=vm_name,
            workspace=workspace,
            parent=workspace,
            db_version=1,
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            default_model_entity=None,
        ),
        trailing_uri="v1/chat/completions",
        json_body={"model": vm_name, "messages": [{"role": "user", "content": "hi"}]},
        http_client=mock_proxy_client,
        model_cache=model_cache,
        registry=registry,
    )

    body = json.loads(await _read_streaming_response(cast(StreamingResponse, response)))
    # Response rewrite uses the entity ref the request middleware routed to,
    # not the VM name and not any unresolved default.
    assert body["model"] == routed_entity_id
    assert body["model"] != vm_name
    assert body["model"] != served_model_name


@pytest.mark.asyncio
async def test_virtual_model_proxy_response_rewrite_runs_before_response_middleware(
    mock_proxy_client, mock_proxy_response
):
    """Response middleware sees the entity-keyed view: by the time ``process_response``
    runs, ``response.result["model"]`` has already been rewritten from ``served_model_name``
    to the qualified entity ref. This matches the request side, where plugins observe
    entity refs in body["model"] rather than served names.
    """
    seen_models: list[str] = []

    class _CapturingResponsePlugin(NemoInferenceMiddleware):
        async def process_response(
            self,
            ctx: InferenceMiddlewareContext,
            response: InferenceResponse,
            middleware_config: object,
        ) -> InferenceResponse:
            assert isinstance(response.result, dict)
            seen_models.append(response.result.get("model", "<missing>"))
            return response

    workspace = "e2e-test"
    vm_name = "capture-router"
    model_entity_id = f"{workspace}/meta_llama-3.2-1b-instruct"
    served_model_name = "meta/llama-3.2-1b-instruct"

    mock_proxy_response.read = AsyncMock(
        return_value=json.dumps(
            {
                "id": "resp",
                "object": "chat.completion",
                "created": 1,
                "model": served_model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
            }
        ).encode()
    )

    model_cache = ModelCache()
    model_cache.update_model_info(
        ModelProviderInfo(
            model_provider=ModelProvider(
                workspace="default",
                name="nim",
                host_url="http://nim.local",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                served_models=[
                    ServedModelMapping(
                        model_entity_id=model_entity_id,
                        served_model_name=served_model_name,
                    )
                ],
                status="READY",
            )
        )
    )
    model_cache.rebuild_model_entity_map()

    registry = MiddlewareRegistry(plugins={"capture-plugin": _CapturingResponsePlugin()})
    registry.response_middleware_calls[(workspace, vm_name)] = [
        ResolvedMiddlewareCall(plugin_name="capture-plugin", config_type="t", resolved_config={})
    ]

    request = Mock(spec=Request)
    request.method = "POST"
    request.headers = {"content-type": "application/json"}
    request.query_params = {}

    await virtual_model_proxy(
        request=request,
        workspace=workspace,
        vm_name=vm_name,
        virtual_model=SDKVirtualModel(
            id=f"{workspace}/{vm_name}",
            entity_id=f"{workspace}/{vm_name}",
            name=vm_name,
            workspace=workspace,
            parent=workspace,
            db_version=1,
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            default_model_entity=model_entity_id,
        ),
        trailing_uri="v1/chat/completions",
        json_body={"model": vm_name, "messages": [{"role": "user", "content": "hi"}]},
        http_client=mock_proxy_client,
        model_cache=model_cache,
        registry=registry,
    )

    assert seen_models == [model_entity_id], (
        f"Response middleware should have observed the entity-rewritten model "
        f"({model_entity_id!r}), but saw {seen_models}."
    )
