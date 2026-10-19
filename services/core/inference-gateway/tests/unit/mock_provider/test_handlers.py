# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for mock_provider handlers module."""

import json
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import Request
from nhx.core.inference_gateway.api.mock_provider.handlers import handle_mock_request
from nhx.core.inference_gateway.api.mock_provider.responses import (
    MOCK_RESPONSE_HEADER,
    MOCK_RESPONSE_MAP_HEADER,
    MOCK_STATUS_HEADER,
    get_call_tracker,
    reset_call_counts,
)
from starlette.datastructures import State


@pytest.fixture
def mock_app_state():
    """Create a mock app.state for testing."""
    return State()


@pytest.fixture
def mock_request(mock_app_state):
    """Create a mock FastAPI request with app.state."""
    request = Mock(spec=Request)
    request.method = "POST"
    request.headers = {}
    # Set up app.state for call tracking
    request.app = Mock()
    request.app.state = mock_app_state
    return request


@pytest.fixture(autouse=True)
def reset_call_counts_fixture(mock_app_state):
    """Reset call counts before and after each test."""
    reset_call_counts(mock_app_state)
    yield
    reset_call_counts(mock_app_state)


@pytest.mark.asyncio
async def test_handle_mock_request_with_header(mock_request):
    """Test handling request with X-Mock-Response header."""
    mock_response = {"id": "chatcmpl-mock", "choices": []}
    mock_request.headers = {MOCK_RESPONSE_HEADER: json.dumps(mock_response)}

    response = await handle_mock_request(
        request=mock_request,
        trailing_uri="v1/chat/completions",
    )

    assert response.status_code == 200
    assert json.loads(response.body) == mock_response


@pytest.mark.asyncio
async def test_handle_mock_request_with_provider_defaults(mock_request):
    """Test handling request with provider default_extra_headers."""
    mock_response = {"provider": "response"}
    default_extra_headers = {MOCK_RESPONSE_HEADER: json.dumps(mock_response)}

    response = await handle_mock_request(
        request=mock_request,
        trailing_uri="v1/chat/completions",
        default_extra_headers=default_extra_headers,
    )

    assert response.status_code == 200
    assert json.loads(response.body) == mock_response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("trailing_uri", "expected_response"),
    [
        ("v1/health/ready", {"status": "ready"}),
        ("v1/health/live", {"status": "live"}),
        ("health/ready", {"status": "ready"}),
        ("health/live", {"status": "live"}),
        ("/v1/health/ready", {"status": "ready"}),  # leading slash normalized
    ],
)
async def test_handle_mock_request_smart_default_health(mock_request, trailing_uri, expected_response):
    """Test handling request with smart defaults for health endpoints."""
    mock_request.method = "GET"

    response = await handle_mock_request(
        request=mock_request,
        trailing_uri=trailing_uri,
    )

    assert response.status_code == 200
    assert json.loads(response.body) == expected_response


@pytest.mark.asyncio
async def test_handle_mock_request_smart_default_models(mock_request):
    """Test handling request with smart default for models endpoint."""
    mock_request.method = "GET"

    response = await handle_mock_request(
        request=mock_request,
        trailing_uri="v1/models",
    )

    assert response.status_code == 200
    body = json.loads(response.body)
    assert body["object"] == "list"
    assert len(body["data"]) > 0
    assert body["data"][0]["id"] == "mock-model"


@pytest.mark.asyncio
async def test_handle_mock_request_no_response_returns_400(mock_request):
    """Test that missing mock response returns 400 error."""
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await handle_mock_request(
            request=mock_request,
            trailing_uri="v1/chat/completions",
        )

    assert exc_info.value.status_code == 400
    assert "Mock provider mode is enabled but no mock response is configured" in exc_info.value.detail
    assert "X-Mock-Response" in exc_info.value.detail


@pytest.mark.asyncio
async def test_handle_mock_request_custom_status(mock_request):
    """Test handling request with custom status code."""
    mock_response = {"error": "rate limited"}
    mock_request.headers = {
        MOCK_RESPONSE_HEADER: json.dumps(mock_response),
        MOCK_STATUS_HEADER: "429",
    }

    response = await handle_mock_request(
        request=mock_request,
        trailing_uri="v1/chat/completions",
    )

    assert response.status_code == 429
    assert json.loads(response.body) == mock_response


@pytest.mark.asyncio
async def test_handle_mock_request_invalid_json_returns_400(mock_request):
    """Test that invalid JSON in header returns 400 error."""
    from fastapi import HTTPException

    mock_request.headers = {MOCK_RESPONSE_HEADER: "not valid json"}

    with pytest.raises(HTTPException) as exc_info:
        await handle_mock_request(
            request=mock_request,
            trailing_uri="v1/chat/completions",
        )

    assert exc_info.value.status_code == 400
    assert "Invalid JSON" in exc_info.value.detail


@pytest.mark.asyncio
async def test_handle_mock_request_header_priority_over_provider(mock_request):
    """Test that request header takes priority over provider defaults."""
    request_response = {"source": "request"}
    provider_response = {"source": "provider"}

    mock_request.headers = {MOCK_RESPONSE_HEADER: json.dumps(request_response)}
    default_extra_headers = {MOCK_RESPONSE_HEADER: json.dumps(provider_response)}

    response = await handle_mock_request(
        request=mock_request,
        trailing_uri="v1/chat/completions",
        default_extra_headers=default_extra_headers,
    )

    assert json.loads(response.body) == request_response


# Tests for dynamic per-model responses via X-Mock-Response-Map header


@pytest.mark.asyncio
async def test_dynamic_responses_exact_model_match(mock_request):
    """Test dynamic responses with exact model match."""
    response_map = {
        "model-a": [{"response_body": {"content": "response-a"}, "response_code": 200}],
        "model-b": [{"response_body": {"content": "response-b"}, "response_code": 200}],
    }
    default_extra_headers = {MOCK_RESPONSE_MAP_HEADER: json.dumps(response_map)}
    mock_request.json = AsyncMock(return_value={"model": "model-a"})

    response = await handle_mock_request(
        request=mock_request,
        trailing_uri="v1/chat/completions",
        default_extra_headers=default_extra_headers,
    )

    assert response.status_code == 200
    assert json.loads(response.body) == {"content": "response-a"}


@pytest.mark.asyncio
async def test_dynamic_responses_wildcard_fallback(mock_request):
    """Test dynamic responses with wildcard fallback."""
    response_map = {
        "*": [{"response_body": {"content": "wildcard-response"}, "response_code": 200}],
    }
    default_extra_headers = {MOCK_RESPONSE_MAP_HEADER: json.dumps(response_map)}
    mock_request.json = AsyncMock(return_value={"model": "any-model"})

    response = await handle_mock_request(
        request=mock_request,
        trailing_uri="v1/chat/completions",
        default_extra_headers=default_extra_headers,
    )

    assert response.status_code == 200
    assert json.loads(response.body) == {"content": "wildcard-response"}


@pytest.mark.asyncio
async def test_dynamic_responses_sequential_calls(mock_request):
    """Test that sequential calls return different responses."""
    response_map = {
        "model-a": [
            {"response_body": {"content": "first"}, "response_code": 200},
            {"response_body": {"content": "second"}, "response_code": 200},
            {"response_body": {"content": "third"}, "response_code": 200},
        ],
    }
    default_extra_headers = {MOCK_RESPONSE_MAP_HEADER: json.dumps(response_map)}
    mock_request.json = AsyncMock(return_value={"model": "model-a"})

    # First call
    response1 = await handle_mock_request(
        request=mock_request,
        trailing_uri="v1/chat/completions",
        default_extra_headers=default_extra_headers,
    )
    assert json.loads(response1.body) == {"content": "first"}

    # Second call
    response2 = await handle_mock_request(
        request=mock_request,
        trailing_uri="v1/chat/completions",
        default_extra_headers=default_extra_headers,
    )
    assert json.loads(response2.body) == {"content": "second"}

    # Third call
    response3 = await handle_mock_request(
        request=mock_request,
        trailing_uri="v1/chat/completions",
        default_extra_headers=default_extra_headers,
    )
    assert json.loads(response3.body) == {"content": "third"}

    # Fourth call - should clamp to last
    response4 = await handle_mock_request(
        request=mock_request,
        trailing_uri="v1/chat/completions",
        default_extra_headers=default_extra_headers,
    )
    assert json.loads(response4.body) == {"content": "third"}


@pytest.mark.asyncio
async def test_dynamic_responses_call_counts_per_model(mock_request):
    """Test that call counts are tracked per model."""
    response_map = {
        "ws1/model-a": [
            {"response_body": {"content": "ws1-1"}, "response_code": 200},
            {"response_body": {"content": "ws1-2"}, "response_code": 200},
        ],
        "ws2/model-a": [
            {"response_body": {"content": "ws2-1"}, "response_code": 200},
            {"response_body": {"content": "ws2-2"}, "response_code": 200},
        ],
    }
    default_extra_headers = {MOCK_RESPONSE_MAP_HEADER: json.dumps(response_map)}

    # Call with workspace1/model-a
    mock_request.json = AsyncMock(return_value={"model": "ws1/model-a"})
    response1 = await handle_mock_request(
        request=mock_request,
        trailing_uri="v1/chat/completions",
        default_extra_headers=default_extra_headers,
    )
    assert json.loads(response1.body) == {"content": "ws1-1"}

    # Call with workspace2/model-a
    mock_request.json = AsyncMock(return_value={"model": "ws2/model-a"})
    response2 = await handle_mock_request(
        request=mock_request,
        trailing_uri="v1/chat/completions",
        default_extra_headers=default_extra_headers,
    )
    assert json.loads(response2.body) == {"content": "ws2-1"}

    # Call workspace1/model-a again - should increment counter
    mock_request.json = AsyncMock(return_value={"model": "ws1/model-a"})
    response3 = await handle_mock_request(
        request=mock_request,
        trailing_uri="v1/chat/completions",
        default_extra_headers=default_extra_headers,
    )
    assert json.loads(response3.body) == {"content": "ws1-2"}


@pytest.mark.asyncio
async def test_dynamic_responses_invalid_json_ignored(mock_request):
    """Test that invalid JSON in config header throws an error."""
    from fastapi import HTTPException

    default_extra_headers = {
        MOCK_RESPONSE_MAP_HEADER: "not valid json",
    }
    mock_request.json = AsyncMock(return_value={"model": "model-a"})

    # Should fall through to 400 error since no other response configured
    with pytest.raises(HTTPException) as exc_info:
        await handle_mock_request(
            request=mock_request,
            trailing_uri="v1/chat/completions",
            default_extra_headers=default_extra_headers,
        )

    assert exc_info.value.status_code == 400


def test_reset_call_counts(mock_app_state):
    """Test that reset_call_counts clears all tracking state."""
    # Get tracker and add some state
    tracker = get_call_tracker(mock_app_state)
    tracker.get_and_increment("model-a")
    tracker.get_and_increment("model-a")
    tracker.get_and_increment("model-b")

    assert tracker.get_count("model-a") == 2
    assert tracker.get_count("model-b") == 1

    reset_call_counts(mock_app_state)

    assert tracker.get_count("model-a") == 0
    assert tracker.get_count("model-b") == 0


# Tests for streaming responses


@pytest.mark.asyncio
async def test_handle_mock_request_streaming_returns_streaming_response(mock_request):
    """Test that streaming requests return StreamingResponse."""
    from fastapi.responses import StreamingResponse

    mock_response = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "test-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello world"}, "finish_reason": "stop"}],
    }
    default_extra_headers = {MOCK_RESPONSE_HEADER: json.dumps(mock_response)}
    mock_request.json = AsyncMock(return_value={"stream": True})

    response = await handle_mock_request(
        request=mock_request,
        trailing_uri="v1/chat/completions",
        default_extra_headers=default_extra_headers,
    )

    # Verify correct response type and headers for streaming
    assert isinstance(response, StreamingResponse)
    assert response.status_code == 200
    assert response.media_type == "text/event-stream"
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["connection"] == "keep-alive"


@pytest.mark.asyncio
async def test_handle_mock_request_non_streaming_returns_json_response(mock_request):
    """Test that non-streaming requests return JSONResponse."""
    from fastapi.responses import JSONResponse

    mock_response = {"id": "chatcmpl-test", "choices": []}
    default_extra_headers = {MOCK_RESPONSE_HEADER: json.dumps(mock_response)}
    mock_request.json = AsyncMock(return_value={"stream": False})

    response = await handle_mock_request(
        request=mock_request,
        trailing_uri="v1/chat/completions",
        default_extra_headers=default_extra_headers,
    )

    # Verify correct response type for non-streaming
    assert isinstance(response, JSONResponse)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_streaming_response_returns_valid_chunks(mock_request):
    """Test that streaming correctly converts response content to streamed chunks."""
    # Mock response that should be streamed
    original_content = "Hello world"

    mock_response = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "test-model",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": original_content}, "finish_reason": "stop"}
        ],
    }
    default_extra_headers = {MOCK_RESPONSE_HEADER: json.dumps(mock_response)}
    mock_request.json = AsyncMock(return_value={"stream": True})

    response = await handle_mock_request(
        request=mock_request,
        trailing_uri="v1/chat/completions",
        default_extra_headers=default_extra_headers,
    )

    # Read all chunks from the stream
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)

    stream_content = b"".join(chunks).decode("utf-8")
    lines = [line for line in stream_content.strip().split("\n\n") if line and line.startswith("data: ")]

    # Reconstruct content from SSE chunks (excluding final chunk with finish_reason and [DONE] marker)
    reconstructed_content = ""
    for line in lines[:-2]:  # Process all chunks except final chunk + [DONE]
        chunk_json = json.loads(line.replace("data: ", ""))
        delta = chunk_json["choices"][0].get("delta", {})
        if "content" in delta:
            reconstructed_content += delta["content"]

    # Verify the streaming pipeline correctly preserved the original content
    assert reconstructed_content == original_content

    # Verify stream terminates properly with [DONE]
    assert lines[-1] == "data: [DONE]"


@pytest.mark.asyncio
async def test_streaming_response_empty_content(mock_request):
    """Test that streaming response handles empty content gracefully."""
    mock_response = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "test-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
    }
    default_extra_headers = {MOCK_RESPONSE_HEADER: json.dumps(mock_response)}
    mock_request.json = AsyncMock(return_value={"stream": True})

    response = await handle_mock_request(
        request=mock_request,
        trailing_uri="v1/chat/completions",
        default_extra_headers=default_extra_headers,
    )

    # Verify streaming works without error for empty content
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)

    stream_content = b"".join(chunks).decode("utf-8")

    # Basic sanity check: should still produce valid SSE with [DONE]
    assert "data: [DONE]" in stream_content
    assert len(chunks) > 0
